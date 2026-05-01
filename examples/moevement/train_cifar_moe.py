# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Canonical MoEvement training example: CIFAR-10 + MoE pipeline.

The model + data + engine config live in ``_common.py``; this script
drives the training loop and exposes perf-instrumentation flags for
measuring overhead.

Real CV-shape MoEvement workload modelled on
DeepSpeedExamples/training/cifar with PP=2 DP=2 + MoEvement enabled.
The conv stem produces a 64-spatial-token sequence per image, then 4
MoE blocks route those tokens, then a mean-pool classifier predicts
the 10 CIFAR classes.

The other example scripts (``resume_after_fault.py``,
``run_with_survivor_supervisor.py``) share the same model + data via
``_common.py``, so the recovery story is uniform across the suite.

**Launch:**

```
deepspeed --num_gpus=4 examples/moevement/train_cifar_moe.py \\
    --num-iters 50
```

Pass ``--fake-data`` to skip the torchvision CIFAR-10 download
(synthetic ``(B, 3, 32, 32)`` tensors of the same shape — used by the
smoke test and any no-network run).  ``--data-dir`` overrides the
cache location (default ``$MOEV_DATA_DIR`` or
``~/.cache/moevement/cifar10``).

**Perf-instrumentation flags (all default off):**

- ``--measure-elapsed``: barrier-synced global wallclock around the
  iter loop, printed on rank 0 with the per-iter median.
- ``--warmup-iters N``: untimed iters before --measure-elapsed starts.
- ``--gas N``: override gradient_accumulation_steps for scale tests.
- ``--hidden N``: override HIDDEN (FC + MoE width).
- ``--disable-moevement``: stock DeepSpeed PP+ZeRO+FP16 floor anchor
  (no per-window snapshot, no replication, no scheduler).
- ``--replication-factor N``: override moevement.replication_factor;
  use 0 for snapshot-only (f=0), 1 for snapshot + 1-peer gloo (f=1).
- ``--profile``: torch.profiler around the iter loop with a
  moevement/* marker breakdown printed on rank 0.  Diagnostic only —
  re-run without --profile for production timing.
- ``--idle-thread {pin_read_d2h, cuda_event_only, cuda_sync_only}``:
  side-thread probe (diagnostic only).

Related:
- ``resume_after_fault.py`` — Whole-Cluster Restart on this model.
- ``run_with_survivor_supervisor.py`` — DP Cascade + peer-pull on this
  model.
"""

import argparse
import os
import threading
import time
from collections import defaultdict

import torch

import deepspeed
import deepspeed.comm as dist

from _common import BATCH, HIDDEN, build_engine, cifar_data_iter, engine_config, log_iter


def _start_idle_probe(mode):
    """Optional side-thread diagnostic probe.

    Variants:
    - ``pin_read_d2h``: GPU->pinned D2H + event sync + CPU sum
    - ``cuda_event_only``: event record + sync on a dummy stream
    - ``cuda_sync_only``: torch.cuda.synchronize() in a loop  #ignore-cuda
    Returns the stop event so the caller can shut down the daemon, or
    None when ``mode == "off"``.
    """
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

    threading.Thread(target=worker, daemon=True, name="cifar_idle_probe").start()
    return stop


def _print_marker_breakdown(prof, rank, hidden, batch, gas, replication_factor):
    """Aggregate per-marker stats across all events with a moevement/* name.

    Surfaces moevement/* trace_ranges plus a few collective / copy
    events so callers can attribute the f=0 overhead without parsing
    the chrome trace JSON.  Mirrors profile_run.py's helper but
    parameterised for the cifar workload.
    """
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
    print(f"workload: cifar hidden={hidden} batch={batch} gas={gas} "
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
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="Number of training iterations (after warmup).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Iters to run before --measure-elapsed starts the global timer.  "
        "Pin to >= 5 when measuring, so JIT / dynamo / NCCL bootstrap doesn't "
        "swamp the first iter (~9 s on this box).",
    )
    parser.add_argument(
        "--gas",
        type=int,
        default=1,
        help="Gradient accumulation steps (overrides engine_config).  Bump to ~16 "
        "when measuring overhead so per-iter compute is large enough that the "
        "MoEvement work is a meaningful fraction of the iter.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=0,
        help="Override HIDDEN (FC + MoE width) for measurement.  Default 0 keeps "
        "_common.py's HIDDEN=256.  Use ~2048 to match profile_run.py scale.",
    )
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="Use synthetic (B, 3, 32, 32) tensors instead of downloading CIFAR-10.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Where to cache CIFAR-10 (default: $MOEV_DATA_DIR or ~/.cache/moevement/cifar10).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to write checkpoints into.  Set to enable disk-checkpointing; "
        "must be accessible from all ranks.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Write a checkpoint every N iters (and at end) when --save-dir is set.  "
        "0 disables saving.",
    )
    parser.add_argument(
        "--persist-to-disk",
        action="store_true",
        help="Opt-in: write the MoEvement bundle (sparse snapshot + upstream logs) "
        "to disk inside engine.save_checkpoint.  Default off — peer-pull from a "
        "surviving DP peer's pinned host snapshot is MoEvement's primary recovery "
        "path, so the disk bundle is only needed for Whole-Cluster Restart "
        "(resume_after_fault.py).  Setting this flag flips moevement.persist_to_disk "
        "to True.",
    )
    parser.add_argument(
        "--measure-elapsed",
        action="store_true",
        help="Wrap the iter loop with barrier+t0 / cuda.sync+barrier+elapsed and "
        "print a global wallclock summary on rank 0.  Use with --num-iters >= 30 "
        "for a stable measurement.",
    )
    parser.add_argument(
        "--idle-thread",
        default="off",
        choices=("off", "pin_read_d2h", "cuda_event_only", "cuda_sync_only"),
        help="Spawn a side-thread diagnostic probe.  Off by default.",
    )
    parser.add_argument(
        "--disable-moevement",
        action="store_true",
        help="Run with the MoEvement coordinator disabled (sets moevement.enabled=False "
        "in the engine config).  Disabled means no per-window snapshot D2H, no "
        "replication, no scheduler — i.e. stock DeepSpeed PP+ZeRO+FP16 with "
        "zero fault-tolerance during these iters.  Use as the floor anchor for "
        "MoEvement overhead measurement.",
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=-1,
        help="Override moevement.replication_factor in the engine config.  -1 keeps "
        "_common.py's default of 1.  Set to 0 to disable peer replication "
        "(snapshot-only path) so the disabled / f=0 / f=1 overhead breakdown can "
        "be measured.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap the measured iter loop with torch.profiler and print a "
        "moevement/* marker breakdown + top-30 CPU events on rank 0.  Use to "
        "attribute MoEvement overhead per stage; numbers are diagnostic, not "
        "wallclock — re-run without --profile for production timing.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Populated automatically by the DeepSpeed launcher.",
    )
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
        # train_batch_size = micro_batch * dp_world * gas; engine_config's
        # default fixes this at micro * dp_world (gas=1), so update it
        # consistently when the user bumps gas.
        config["train_batch_size"] = config["train_micro_batch_size_per_gpu"] * 2 * args.gas

    hidden = args.hidden if args.hidden > 0 else HIDDEN
    engine = build_engine(config=config, hidden=hidden)
    if rank == 0:
        src = "synthetic" if args.fake_data else "torchvision CIFAR-10"
        if args.disable_moevement:
            print(f"[rank 0] MoEvement DISABLED (stock DeepSpeed pipe + ZeRO-1 + FP16, data={src})", flush=True)
        else:
            coord = engine.moevement_coordinator
            print(
                f"[rank 0] MoEvement engine ready (w_sparse={coord.scheduler.w_sparse}, "
                f"replication_factor={coord.config.replication_factor}, data={src})",
                flush=True,
            )

    idle_stop = _start_idle_probe(args.idle_thread)
    if rank == 0 and idle_stop is not None:
        print(f"[rank 0] idle-thread probe: mode={args.idle_thread}", flush=True)

    data_iter = cifar_data_iter(batch_size=BATCH, fake_data=args.fake_data, data_dir=args.data_dir)

    # Run warmup iters BEFORE starting the global timer so JIT / dynamo /
    # NCCL bootstrap don't dominate the measured wallclock.  The first
    # iter on this box compiles ~9 s of dynamo graphs; subsequent iters
    # at this scale (HIDDEN=256, BATCH=4) run in tens of ms.
    for warm_step in range(1, args.warmup_iters + 1):
        loss = engine.train_batch(data_iter=data_iter)
        if rank == 0:
            log_iter(rank, warm_step, float(loss) if loss is not None else float("nan"), engine)

    if args.measure_elapsed:
        # Drain warmup work + barrier so total_elapsed starts from a
        # clean cross-rank state.  Mirrors the profile_run.py pattern.
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
            # Disk checkpointing.  Collective: every rank must hit it.
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
                f"[cifar] moevement={tag} idle_thread={args.idle_thread} "
                f"rank0_iter_ms: avg={avg_ms:.2f} median={median_ms:.2f} "
                f"min={sorted_ms[0]:.2f} max={sorted_ms[-1]:.2f}",
                flush=True,
            )
            print(
                f"[cifar] moevement={tag} idle_thread={args.idle_thread} "
                f"total_elapsed_s={total_elapsed:.3f} num_iters={args.num_iters}",
                flush=True,
            )

    if prof is not None:
        rep_factor = config["moevement"]["replication_factor"] if not args.disable_moevement else -1
        _print_marker_breakdown(prof, rank, hidden=hidden, batch=BATCH, gas=args.gas, replication_factor=rep_factor)

    if idle_stop is not None:
        idle_stop.set()

    if rank == 0:
        print("[rank 0] cifar-moe training complete", flush=True)


if __name__ == "__main__":
    main()
