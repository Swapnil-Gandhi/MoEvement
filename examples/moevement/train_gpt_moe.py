# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""GPT-MoE example: small transformer with MoE FFN + MoEvement.

Second canonical example next to ``train_cifar_moe.py``.  Cifar
exercises CV-shape MoE (conv backbone + token-grid routing); this
script exercises transformer-shape MoE (token-embedding +
self-attention + MoE-FFN), which is closer to typical real-world
MoEvement workloads.

Architecture (1 embedding + 4 transformer blocks + 1 head = 6 layers,
PipelineModule splits 3/3 across PP=2):

- ``GPTEmbedding``: token + position lookup -> (B, S, H)
- ``GPTBlock`` × 4: LayerNorm -> causal self-attention -> residual ->
  LayerNorm -> MoE FFN -> residual.  The standard GPT decoder block
  with the dense FFN swapped for ``deepspeed.moe.layer.MoE``.
- ``GPTHead``: LayerNorm -> linear projection to vocab logits.

Loss is causal next-token CE: ``F.cross_entropy(logits.view(-1, V),
targets.view(-1))``.  Data is byte-level TinyShakespeare by default
(downloaded once to ``$MOEV_DATA_DIR`` or
``~/.cache/moevement/tinyshakespeare``); pass ``--fake-data`` to skip
the download and use synthetic random tokens (smoke / no-network).
The recovery contract is independent of data realism — the example
demonstrates wiring on a real corpus, not convergence.

Same engine config + log_iter as ``train_cifar_moe.py`` (imported
from ``_common.py``).  Same perf-instrumentation flags
(--measure-elapsed, --gas, --hidden, --profile, etc.) so MoEvement
overhead can be measured on transformer shape too.

**Launch:**

```
deepspeed --num_gpus=4 examples/moevement/train_gpt_moe.py \\
    --num-iters 50
```

Related:
- ``train_cifar_moe.py`` — CV-shape canonical example.
- ``resume_after_fault.py`` — works against either example's
  checkpoints (the recovery contract is model-independent).
- ``docs/moevement/README.md`` — feature overview.
"""

import argparse
import os
import threading
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
import deepspeed.comm as dist
from deepspeed.moe.layer import MoE
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.pipe import LayerSpec, PipelineModule
from deepspeed.utils import RepeatingLoader

from _common import EP_SIZE, HIDDEN, NUM_EXPERTS, engine_config, log_iter

# Workload constants — small enough for a 4×A100 dev box, transformer-
# shaped enough to exercise MoE all_to_all + per-token routing.
# VOCAB_SIZE=256 matches the byte-level tokenization of the real
# TinyShakespeare data (every UTF-8 byte is a token); the synthetic
# fake-data path uses the same vocab so model dims line up.
VOCAB_SIZE = 256
SEQ_LEN = 32
NUM_HEADS = 4
NUM_GPT_LAYERS = 4
BATCH = 4

# Karpathy's char-rnn TinyShakespeare corpus.  ~1 MB plain text, the
# canonical small-GPT real dataset (nanoGPT uses the same).  Stable
# URL since 2015.
TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class GPTEmbedding(nn.Module):
    """Token + position embeddings producing (B, S, hidden) activations."""

    def __init__(self, vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN, hidden=HIDDEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_seq_len, hidden)

    def forward(self, x):
        # x: (B, S) int64 token IDs.  Cast to long defensively in case the
        # pipeline send-recv path delivered a wider dtype.
        x = x.long()
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.tok_emb(x) + self.pos_emb(positions)


class GPTBlock(nn.Module):
    """LayerNorm -> causal self-attention -> LayerNorm -> MoE FFN, both residual.

    Standard pre-norm GPT decoder layout.  The dense FFN is replaced
    with a top-1 ``deepspeed.moe.layer.MoE`` so MoEvement has an
    operator to snapshot per window.
    """

    def __init__(self,
                 hidden=HIDDEN,
                 num_heads=NUM_HEADS,
                 num_experts=NUM_EXPERTS,
                 ep_size=EP_SIZE,
                 max_seq_len=SEQ_LEN):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        # MoE wraps a 2-layer FFN expert with GELU between, the canonical
        # transformer FFN shape.  4× expansion is the standard ratio.
        self.moe = MoE(
            hidden_size=hidden,
            expert=nn.Sequential(
                nn.Linear(hidden, 4 * hidden),
                nn.GELU(),
                nn.Linear(4 * hidden, hidden),
            ),
            num_experts=num_experts,
            k=1,
            ep_size=ep_size,
        )
        # Pre-build the causal mask once.  Registered as a buffer so it
        # follows the module's device.  S×S is tiny (32×32 here); the
        # alternative (build on every forward) would also be cheap but
        # adds Python overhead inside the per-microbatch hot path.
        mask = nn.Transformer.generate_square_subsequent_mask(max_seq_len)
        self.register_buffer("_attn_mask", mask, persistent=False)

    def forward(self, x):
        h = self.ln1(x)
        # is_causal=True with attn_mask is the supported pattern in torch
        # 2.6; the mask shapes the kernel selection.  need_weights=False
        # avoids materializing the full attention map.
        attn_out, _ = self.attn(h, h, h, attn_mask=self._attn_mask, is_causal=True, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        moe_out, _aux, _z = self.moe(h)
        return x + moe_out


class GPTHead(nn.Module):
    """LayerNorm -> linear projection -> (B, S, vocab_size) logits."""

    def __init__(self, hidden=HIDDEN, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.ln(x))


def _gpt_loss_fn(logits, targets):
    """Causal next-token CE; reshapes (B, S, V) and (B, S) into 2-D / 1-D."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())


def build_gpt_pipeline_model(num_stages=2, hidden=HIDDEN):
    """Assemble the GPT-MoE PipelineModule.

    Layer count: 1 embedding + 4 GPT blocks + 1 head = 6 layers.
    ``partition_method='uniform'`` splits 6 layers as 3/3 across PP=2 so
    every stage carries multiple MoE layers (engine guard requires at
    least one MoE per stage; an unweighted partition would otherwise
    bunch the embedding's parameter-heavy table on one side and starve
    the other stage of MoE).
    """
    layers = [
        LayerSpec(GPTEmbedding, VOCAB_SIZE, SEQ_LEN, hidden),
        *[LayerSpec(GPTBlock, hidden, NUM_HEADS, NUM_EXPERTS, EP_SIZE, SEQ_LEN) for _ in range(NUM_GPT_LAYERS)],
        LayerSpec(GPTHead, hidden, VOCAB_SIZE),
    ]
    return PipelineModule(
        layers=layers,
        num_stages=num_stages,
        loss_fn=_gpt_loss_fn,
        partition_method="uniform",
    )


def gpt_data_iter(batch_size=BATCH, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, fake_data=False, data_dir=None, seed=123):
    """Yield ``(token_ids, target_ids)`` micro-batches for the pipe engine.

    Two paths, mirroring ``_common.cifar_data_iter``:

    - ``fake_data=True``: synthetic random ``(B, S)`` int64 tensors via
      ``RepeatingLoader`` over a single sample.  Identical input every
      iter — used by the smoke test and any no-network run.  Determinism
      + no external dependency.
    - ``fake_data=False`` (default): byte-level TinyShakespeare.  The
      raw text is downloaded once to ``data_dir`` (default
      ``$MOEV_DATA_DIR`` or ``~/.cache/moevement/tinyshakespeare``) and
      tokenized as bytes — every UTF-8 byte is a token in [0, 255], so
      no tokenizer dependency.  Each batch is a fresh random window of
      ``seq_len`` consecutive tokens with the target shifted by one
      (next-token prediction).  Same vocab size as the synthetic path,
      so the model's embedding dims line up.
    """
    if fake_data:
        gen = torch.Generator().manual_seed(seed)
        sample = (
            torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen, dtype=torch.long),
            torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen, dtype=torch.long),
        )
        return iter(RepeatingLoader([sample]))

    import urllib.request
    if data_dir is None:
        data_dir = os.environ.get("MOEV_DATA_DIR", os.path.expanduser("~/.cache/moevement/tinyshakespeare"))
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(txt_path):
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, txt_path)
    with open(txt_path, "rb") as f:
        raw = f.read()
    # Each byte is a token id in [0, 255]; vocab_size=256 covers the lot
    # without needing a learned tokenizer.
    tokens = torch.tensor(list(raw), dtype=torch.long)
    n_tokens = tokens.numel()
    gen = torch.Generator().manual_seed(seed)

    def _generator():
        while True:
            starts = torch.randint(0, n_tokens - seq_len - 1, (batch_size, ), generator=gen)
            x = torch.stack([tokens[s:s + seq_len] for s in starts.tolist()])
            y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts.tolist()])
            yield (x, y)

    return _generator()


def build_engine(config=None, skip_initial_broadcast=False, hidden=HIDDEN):
    """Wire a DeepSpeed PipelineEngine over the GPT-MoE model."""
    if config is None:
        config = engine_config()
    model = build_gpt_pipeline_model(hidden=hidden)
    param_group = {"params": list(model.parameters()), "name": "moevement_gpt_params"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=config["optimizer"]["params"]["lr"])
    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        optimizer=optimizer,
        dist_init_required=False,
        skip_initial_broadcast=skip_initial_broadcast,
    )
    return engine


def _start_idle_probe(mode):
    """Optional side-thread diagnostic probe (parity with cifar example)."""
    if mode == "off":
        return None

    stop = threading.Event()
    if mode == "pin_read_d2h":
        idle_pinned = torch.empty(1024 * 1024, dtype=torch.float32, pin_memory=True)
        idle_gpu_src = torch.zeros(1024 * 1024, dtype=torch.float32, device="cuda")
        idle_stream = torch.cuda.Stream()  #ignore-cuda
        idle_event = torch.cuda.Event()  #ignore-cuda

        def worker():
            while not stop.is_set():
                with torch.cuda.stream(idle_stream):  #ignore-cuda
                    idle_pinned.copy_(idle_gpu_src, non_blocking=True)
                idle_event.record(idle_stream)
                idle_event.synchronize()
                _ = idle_pinned.sum()
                stop.wait(0.001)
    elif mode == "cuda_event_only":
        idle_stream = torch.cuda.Stream()  #ignore-cuda
        idle_event = torch.cuda.Event()  #ignore-cuda

        def worker():
            while not stop.is_set():
                idle_event.record(idle_stream)
                idle_event.synchronize()
                stop.wait(0.001)
    elif mode == "cuda_sync_only":

        def worker():
            while not stop.is_set():
                torch.cuda.synchronize()  #ignore-cuda  drains every stream
                stop.wait(0.001)
    else:
        raise ValueError(f"unknown --idle-thread mode: {mode!r}")

    threading.Thread(target=worker, daemon=True, name="gpt_idle_probe").start()
    return stop


def _print_marker_breakdown(prof, rank, hidden, batch, gas, replication_factor):
    """moevement/* marker breakdown on rank 0; mirrors the cifar helper."""
    agg_count = defaultdict(int)
    agg_total_us = defaultdict(float)
    extra_keys = {"c10d::all_reduce_", "nccl:all_reduce", "ncclAllReduce", "aten::zeros", "aten::copy_"}
    for evt in prof.key_averages():
        if evt.count <= 0:
            continue
        if evt.key.startswith("moevement/"):
            agg_count[evt.key] += evt.count
            agg_total_us[evt.key] += evt.cpu_time_total
        elif evt.key in extra_keys:
            agg_count[evt.key] += evt.count
            agg_total_us[evt.key] += evt.cpu_time_total

    if rank != 0:
        return
    print()
    print("=== MoEvement marker breakdown (rank 0) ===")
    print(f"workload: gpt hidden={hidden} batch={batch} gas={gas} "
          f"replication_factor={replication_factor}")
    print()
    print(f"{'marker':<35} {'count':>6} {'total_ms':>12} {'mean_us':>10}")
    print("-" * 70)
    for name in sorted(agg_count):
        count = agg_count[name]
        total_us = agg_total_us[name]
        mean_us = total_us / count if count else 0
        print(f"{name:<35} {count:>6d} {total_us / 1000:>12.3f} {mean_us:>10.0f}")

    print()
    print("=== Top-30 CPU events (rank 0) ===")
    print(f"{'event':<45} {'count':>6} {'total_ms':>12} {'mean_us':>10}")
    print("-" * 80)
    top = sorted(prof.key_averages(), key=lambda e: e.cpu_time_total, reverse=True)[:30]
    for evt in top:
        name = evt.key[:44]
        total_us = evt.cpu_time_total
        mean_us = total_us / evt.count if evt.count else 0
        print(f"{name:<45} {evt.count:>6d} {total_us / 1000:>12.3f} {mean_us:>10.0f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-iters", type=int, default=50, help="Training iterations after warmup.")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Iters to run before --measure-elapsed starts the global timer (e.g. 10).",
    )
    parser.add_argument("--gas", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--hidden", type=int, default=0, help="HIDDEN override (0 keeps _common.HIDDEN=256).")
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="Use synthetic random (B, S) token sequences instead of downloading "
        "TinyShakespeare.  Identical input every iter — used by the smoke test "
        "and any no-network run.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Where to cache TinyShakespeare (default: $MOEV_DATA_DIR or "
        "~/.cache/moevement/tinyshakespeare).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to write checkpoints into; collective save, must be reachable from all ranks.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Write a checkpoint every N iters (and at end) when --save-dir is set.",
    )
    parser.add_argument(
        "--persist-to-disk",
        action="store_true",
        help="Opt-in: write the MoEvement bundle to disk inside engine.save_checkpoint.  "
        "Default off — peer-pull is the primary recovery target.  Set this for "
        "Whole-Cluster Restart compatibility.",
    )
    parser.add_argument(
        "--measure-elapsed",
        action="store_true",
        help="Wrap the iter loop with barrier+t0 / cuda.sync+barrier+elapsed and print "
        "rank0 stats + total_elapsed_s.",
    )
    parser.add_argument(
        "--idle-thread",
        default="off",
        choices=("off", "pin_read_d2h", "cuda_event_only", "cuda_sync_only"),
        help="Spawn a side-thread diagnostic probe.",
    )
    parser.add_argument(
        "--disable-moevement",
        action="store_true",
        help="Run with moevement.enabled=False — stock DeepSpeed PP+ZeRO+FP16 floor.",
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=-1,
        help="Override moevement.replication_factor.  -1 keeps _common default of 1.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="torch.profiler around the iter loop with marker breakdown on rank 0.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by the DeepSpeed launcher.")
    return parser.parse_args()


def main():
    args = parse_args()

    deepspeed.init_distributed(dist_backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  #ignore-cuda

    torch.manual_seed(42)

    config = engine_config()
    if args.disable_moevement:
        config["moevement"]["enabled"] = False
    if args.replication_factor >= 0:
        config["moevement"]["replication_factor"] = args.replication_factor
    if args.persist_to_disk:
        config["moevement"]["persist_to_disk"] = True
    if args.gas != 1:
        config["gradient_accumulation_steps"] = args.gas
        config["train_batch_size"] = config["train_micro_batch_size_per_gpu"] * 2 * args.gas

    hidden = args.hidden if args.hidden > 0 else HIDDEN
    engine = build_engine(config=config, hidden=hidden)
    if rank == 0:
        if args.disable_moevement:
            print("[rank 0] MoEvement DISABLED (stock DeepSpeed pipe + ZeRO-1 + FP16, model=gpt-moe)", flush=True)
        else:
            coord = engine.moevement_coordinator
            print(
                f"[rank 0] MoEvement engine ready (w_sparse={coord.scheduler.w_sparse}, "
                f"replication_factor={coord.config.replication_factor}, model=gpt-moe)",
                flush=True,
            )

    idle_stop = _start_idle_probe(args.idle_thread)
    if rank == 0 and idle_stop is not None:
        print(f"[rank 0] idle-thread probe: mode={args.idle_thread}", flush=True)

    data_iter = gpt_data_iter(fake_data=args.fake_data, data_dir=args.data_dir)
    if rank == 0:
        src = "synthetic" if args.fake_data else "TinyShakespeare (byte-level)"
        print(f"[rank 0] data={src}", flush=True)

    for warm_step in range(1, args.warmup_iters + 1):
        loss = engine.train_batch(data_iter=data_iter)
        if rank == 0:
            log_iter(rank, warm_step, float(loss) if loss is not None else float("nan"), engine)

    if args.measure_elapsed:
        torch.cuda.synchronize()  #ignore-cuda
        dist.barrier()
        total_t0 = time.perf_counter()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    iter_times = []
    prof = None
    profile_ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) if args.profile else None)
    if profile_ctx is not None:
        profile_ctx.__enter__()
        prof = profile_ctx
    try:
        last_step = args.warmup_iters + args.num_iters
        for step in range(args.warmup_iters + 1, last_step + 1):
            t0 = time.perf_counter()
            loss = engine.train_batch(data_iter=data_iter)
            torch.cuda.synchronize()  #ignore-cuda
            iter_times.append(time.perf_counter() - t0)
            if rank == 0:
                log_iter(rank, step, float(loss) if loss is not None else float("nan"), engine)
            if args.save_dir and args.save_every > 0 and (step % args.save_every == 0 or step == last_step):
                tag = f"step_{step}"
                engine.save_checkpoint(args.save_dir, tag=tag)
                if rank == 0:
                    print(f"[rank 0] saved checkpoint: {args.save_dir}/{tag}", flush=True)
    finally:
        if profile_ctx is not None:
            profile_ctx.__exit__(None, None, None)

    if args.measure_elapsed:
        torch.cuda.synchronize()  #ignore-cuda
        dist.barrier()
        total_elapsed = time.perf_counter() - total_t0
        if rank == 0:
            sorted_ms = sorted(t * 1000 for t in iter_times)
            avg_ms = sum(sorted_ms) / len(sorted_ms)
            median_ms = sorted_ms[len(sorted_ms) // 2]
            tag = "DISABLED" if args.disable_moevement else "ENABLED"
            print(
                f"[gpt] moevement={tag} idle_thread={args.idle_thread} "
                f"rank0_iter_ms: avg={avg_ms:.2f} median={median_ms:.2f} "
                f"min={sorted_ms[0]:.2f} max={sorted_ms[-1]:.2f}",
                flush=True,
            )
            print(
                f"[gpt] moevement={tag} idle_thread={args.idle_thread} "
                f"total_elapsed_s={total_elapsed:.3f} num_iters={args.num_iters}",
                flush=True,
            )

    if prof is not None:
        rep_factor = config["moevement"]["replication_factor"] if not args.disable_moevement else -1
        _print_marker_breakdown(prof, rank, hidden=hidden, batch=BATCH, gas=args.gas, replication_factor=rep_factor)

    if idle_stop is not None:
        idle_stop.set()

    if rank == 0:
        print("[rank 0] gpt-moe training complete", flush=True)


if __name__ == "__main__":
    main()
