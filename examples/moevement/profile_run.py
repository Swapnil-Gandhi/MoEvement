# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Developer tool: profile MoEvement on a configurable MoE model.

Used to refresh perf baselines when ``w_sparse`` / ``num_active``,
hidden / num_experts, or any snapshot-path optimisation changes.  Not
part of the user-facing examples — it builds its own model rather than
reusing ``_common.py``'s toy shapes so the audit can be re-grounded
against realistic scale.

Workload knobs (env vars):

- ``MOEV_PROFILE_HIDDEN``         (default 256)
- ``MOEV_PROFILE_NUM_EXPERTS``    (default 4)
- ``MOEV_PROFILE_NUM_MOE_LAYERS`` (default 4)
- ``MOEV_PROFILE_EP_SIZE``        (default 2)
- ``MOEV_PROFILE_BATCH``          (default 4)
- ``MOEV_PROFILE_SEQ_LEN``        (default 32)
- ``MOEV_PROFILE_NUM_ITERS``      (default 15; profile-active iters)
- ``MOEV_PROFILE_WARMUP_ITERS``   (default 3)
- ``MOEV_PROFILE_TRACE_PATH``     (default /tmp/moev_profile_trace.json)

Launch (4xA100, PP=2 DP=2):

    MOEV_PROFILE_HIDDEN=2048 MOEV_PROFILE_NUM_EXPERTS=8 \\
        deepspeed --num_gpus=4 examples/moevement/profile_run.py
"""

import os
import time
from collections import defaultdict

import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.moe.layer import MoE
from deepspeed.pipe import LayerSpec, PipelineModule
from deepspeed.utils import RepeatingLoader


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_str(name, default):
    return os.environ.get(name, default)


HIDDEN = _env_int("MOEV_PROFILE_HIDDEN", 256)
NUM_EXPERTS = _env_int("MOEV_PROFILE_NUM_EXPERTS", 4)
NUM_MOE_LAYERS = _env_int("MOEV_PROFILE_NUM_MOE_LAYERS", 4)
EP_SIZE = _env_int("MOEV_PROFILE_EP_SIZE", 2)
BATCH = _env_int("MOEV_PROFILE_BATCH", 16)
GRAD_ACCUM_STEPS = _env_int("MOEV_PROFILE_GAS", 1)
SEQ_LEN = _env_int("MOEV_PROFILE_SEQ_LEN", 32)
NUM_CLASSES = 32
NUM_ITERS = _env_int("MOEV_PROFILE_NUM_ITERS", 15)
WARMUP_ITERS = _env_int("MOEV_PROFILE_WARMUP_ITERS", 3)
TRACE_PATH = _env_str("MOEV_PROFILE_TRACE_PATH", "/tmp/moev_profile_trace.json")
MOEVEMENT_DISABLED = _env_int("MOEV_PROFILE_DISABLE", 0) == 1
REPLICATION_FACTOR = _env_int("MOEV_PROFILE_REPLICATION_FACTOR", 1)
PCIE_BANDWIDTH_GBS = float(os.environ.get("MOEV_PROFILE_PCIE_GBS", "24.0"))
UPSTREAM_LOGGING = _env_int("MOEV_PROFILE_UPSTREAM_LOGGING", 1) == 1
W_SPARSE_OVERRIDE = _env_int("MOEV_PROFILE_W_SPARSE", 0)
SNAPSHOT_OVERLAP_TARGET = float(os.environ.get("MOEV_PROFILE_OVERLAP_TARGET", "1.0"))
# Override replication_queue_max_outstanding (default 256 in config.py).
# Useful for production-scale runs where the default grow_on_miss=258
# applies to GPU staging buffers and OOMs at HIDDEN=4096+.  0 means
# don't override.
QUEUE_MAX_OUTSTANDING_OVERRIDE = _env_int("MOEV_PROFILE_QUEUE_MAX", 0)
# Per-iter loss/scale/overflow logging for f0-vs-f1 trajectory comparison.
# When set, last-stage DP rank 0 emits one CSV line per iter to stderr
# tagged "[per-iter]" so it can be greped out of the run log.
PER_ITER_LOG = _env_int("MOEV_PROFILE_PER_ITER_LOG", 0) == 1
# Pin torch + cuda RNG so model init + dropout draws are bit-identical
# across runs.  Only useful when comparing two configs head-to-head.
DETERMINISTIC_INIT = _env_int("MOEV_PROFILE_DETERMINISTIC_INIT", 0) == 1


def _timing_stats(values):
    """avg / median / min / max in milliseconds for a list of seconds."""
    sorted_ms = sorted(v * 1000 for v in values)
    return {
        "avg": sum(sorted_ms) / len(sorted_ms),
        "median": sorted_ms[len(sorted_ms) // 2],
        "min": sorted_ms[0],
        "max": sorted_ms[-1],
    }


def _gather_iter_times(iter_times):
    """All_gather a per-rank list of float seconds; return shape (W, N) cpu tensor.

    Single end-of-run gather instead of per-iter all_reduce.  Per-iter
    cross-rank syncs add collective overhead that inflates the global
    wallclock without ending up in any per-iter measurement window.
    Gather once at the end and reconstruct any cross-rank view
    (max-per-iter, per-rank median, ...) in pure Python.
    """
    if torch.cuda.is_available():  #ignore-cuda
        device = torch.device("cuda", torch.cuda.current_device())  #ignore-cuda
    else:
        device = torch.device("cpu")
    local = torch.tensor(iter_times, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.stack(gathered).cpu()


class _Embed(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return self.lin(x)


class _MoEBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.moe = MoE(hidden_size=HIDDEN,
                       expert=nn.Linear(HIDDEN, HIDDEN),
                       num_experts=NUM_EXPERTS,
                       k=1,
                       ep_size=EP_SIZE)

    def forward(self, x):
        out, _aux, _z = self.moe(x)
        return out


class _Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


def _build_model():
    layers = [
        LayerSpec(_Embed),
        *[LayerSpec(_MoEBlock) for _ in range(NUM_MOE_LAYERS)],
        LayerSpec(_Head),
    ]
    return PipelineModule(layers=layers, num_stages=2, loss_fn=nn.CrossEntropyLoss())


def _engine_config():
    # Let DeepSpeed derive ``train_batch_size`` from
    # ``micro_batch × dp_world × grad_acc`` so the harness scales
    # across single-host (4 GPUs, DP=2) and multi-host (8+ GPUs,
    # DP=4+) topologies without manual overrides.
    return {
        "train_micro_batch_size_per_gpu": BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "steps_per_print": 100,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "torch_adam": True,
            },
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1
        },
        "pipeline": {
            "activation_checkpoint_interval": 0
        },
        "moevement": {
            "enabled":
            not MOEVEMENT_DISABLED,
            "pcie_bandwidth_gbs":
            PCIE_BANDWIDTH_GBS,  # Env-tunable (MOEV_PROFILE_PCIE_GBS). Lets the scheduler choose w_sparse organically.
            "initial_iter_time_sec":
            0.05,
            "replication_factor":
            REPLICATION_FACTOR,
            "upstream_logging":
            UPSTREAM_LOGGING,
            "w_sparse_override":
            W_SPARSE_OVERRIDE,
            "snapshot_overlap_target":
            SNAPSHOT_OVERLAP_TARGET,
            **({
                "replication_queue_max_outstanding": QUEUE_MAX_OUTSTANDING_OVERRIDE
            } if QUEUE_MAX_OUTSTANDING_OVERRIDE > 0 else {}),
        },
    }


def _data_iter():
    gen = torch.Generator().manual_seed(42)
    sample = (
        torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.float16, generator=gen),
        torch.randint(0, NUM_CLASSES, (BATCH, ), generator=gen),
    )
    return iter(RepeatingLoader([sample]))


def _print_marker_breakdown(prof, rank):
    """Aggregate per-marker stats across all events with a moevement/* name."""
    # key_averages can return multiple rows per marker (CPU vs CUDA
    # activity); sum into one row per marker name.
    agg_count = defaultdict(int)
    agg_total_us = defaultdict(float)
    # Also surface NCCL all_reduce + a few collective-related rows so a
    # caller can split snap_active into (all_reduce / alloc / copy / etc.)
    # without parsing the trace JSON.
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
    print(f"=== MoEvement marker breakdown (rank 0) ===")
    print(f"workload: HIDDEN={HIDDEN} NUM_EXPERTS={NUM_EXPERTS} NUM_MOE_LAYERS={NUM_MOE_LAYERS} "
          f"EP_SIZE={EP_SIZE} BATCH={BATCH} SEQ_LEN={SEQ_LEN}")
    print(f"profile iters: {NUM_ITERS} (warmup {WARMUP_ITERS})")
    print()
    print(f"{'marker':<35} {'count':>6} {'total_ms':>12} {'mean_us':>10}")
    print("-" * 70)
    for name in sorted(agg_count):
        count = agg_count[name]
        total_us = agg_total_us[name]
        mean_us = total_us / count if count else 0
        print(f"{name:<35} {count:>6d} {total_us / 1000:>12.3f} {mean_us:>10.0f}")

    # Top-30 global events by cpu_time_total — catches MoEvement-
    # attributable cost that lives outside moevement/* trace_ranges
    # (e.g., aten::narrow / aten::contiguous / optimizer ops) so we
    # can diagnose iter-time overhead that the curated filter hides.
    print()
    print(f"=== Top-30 CPU events (rank 0) ===")
    print(f"{'event':<45} {'count':>6} {'total_ms':>12} {'mean_us':>10}")
    print("-" * 80)
    top = sorted(prof.key_averages(), key=lambda e: e.cpu_time_total, reverse=True)[:30]
    for evt in top:
        name = evt.key[:44]
        total_us = evt.cpu_time_total
        mean_us = total_us / evt.count if evt.count else 0
        print(f"{name:<45} {evt.count:>6d} {total_us / 1000:>12.3f} {mean_us:>10.0f}")


def main():
    deepspeed.init_distributed()
    rank = dist.get_rank()

    if DETERMINISTIC_INIT:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)  #ignore-cuda

    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    model = _build_model()
    param_group = {"params": list(model.parameters()), "name": "moevement_profile_params"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        config=_engine_config(),
        model=model,
        optimizer=optimizer,
        dist_init_required=False,
    )
    data = _data_iter()

    # Diagnostic probe: spawn a daemon background thread to test whether
    # the f1 cheap-regime speedup is an artifact of *any* CPU thread
    # being active alongside the training thread, or whether it
    # specifically requires a CPU read of pinned post-D2H pages.
    # Variants:
    #   sleep        — thread sleeps in a loop (tests "thread existence")
    #   busy         — thread runs a busy CPU loop (tests "any CPU activity")
    #   pin_read     — thread periodically reads a pre-allocated pinned tensor
    #                  (tests "any pinned-mem read, decoupled from D2H")
    #   pin_read_d2h — thread periodically issues GPU→pinned D2H +
    #                  event.synchronize + CPU read.  Reproduces the
    #                  pinned-page-read mechanism in disabled mode (no
    #                  MoEvement) — tests whether the post-D2H read is
    #                  MoEvement-specific or generic.
    #   cuda_event_only — record + synchronize a CUDA event on a dummy
    #                     stream periodically; NO D2H, NO memory.  Tests
    #                     whether driver-thread / event-queue activity
    #                     alone is enough to mask the disabled tax.
    #   cuda_sync_only — call torch.cuda.synchronize() periodically.  #ignore-cuda
    #                    Tests whether any cuda driver activity at all
    #                    is enough.
    # Default off.  Empirical only — not a fix.
    idle_mode = os.environ.get("MOEV_PROFILE_IDLE_THREAD", "off")
    idle_stop = None
    if idle_mode != "off":
        import threading
        idle_stop = threading.Event()
        if idle_mode == "pin_read":
            # Pre-allocate a pinned scratch tensor that the background
            # thread can read.  Decoupled from any D2H so it tests the
            # read-pinned-pages mechanism in isolation.
            idle_pinned = torch.empty(16 * 1024 * 1024, dtype=torch.float32, pin_memory=True)
        if idle_mode == "pin_read_d2h":
            # Small pinned dest + GPU source on a dedicated stream so
            # the probe doesn't serialize with the training streams.
            # Sized small enough to keep the probe cheap but large
            # enough to span multiple pinned pages.
            idle_pinned = torch.empty(1024 * 1024, dtype=torch.float32, pin_memory=True)
            idle_gpu_src = torch.zeros(1024 * 1024, dtype=torch.float32, device="cuda")
            idle_stream = torch.cuda.Stream()  #ignore-cuda
            idle_event = torch.cuda.Event()  #ignore-cuda
        if idle_mode == "cuda_event_only":
            # No memory at all — just an event on a dummy side stream.
            # Tests whether driver-thread / event-queue activity alone
            # reproduces the speedup.
            idle_stream = torch.cuda.Stream()  #ignore-cuda
            idle_event = torch.cuda.Event()  #ignore-cuda

        def _idle_worker():
            if idle_mode == "sleep":
                while not idle_stop.is_set():
                    idle_stop.wait(0.001)
            elif idle_mode == "busy":
                # Pure-Python counter holds the GIL and starves NCCL
                # init / proxy threads.  Use an in-place tensor op that
                # releases the GIL during compute, then yield briefly.
                scratch = torch.empty(1024, dtype=torch.float32)
                while not idle_stop.is_set():
                    scratch.add_(1.0)
                    idle_stop.wait(0.0001)
            elif idle_mode == "pin_read":
                while not idle_stop.is_set():
                    _ = float(idle_pinned[0].item())  # forces a CPU read of pinned page
                    idle_stop.wait(0.001)
            elif idle_mode == "pin_read_d2h":
                # Tight loop: D2H on dedicated stream → record event →
                # event.synchronize() → CPU sum (touches the pinned pages
                # the D2H just wrote).  Mirrors the f0+fix worker's
                # post-D2H touch but in disabled mode (so no MoEvement
                # state, no replication, no engine integration).
                while not idle_stop.is_set():
                    with torch.cuda.stream(idle_stream):  #ignore-cuda
                        idle_pinned.copy_(idle_gpu_src, non_blocking=True)
                    idle_event.record(idle_stream)
                    idle_event.synchronize()
                    _ = idle_pinned.sum()
                    idle_stop.wait(0.001)
            elif idle_mode == "cuda_event_only":
                # Record + synchronize an event on a dummy side stream.
                # No memory ops, no D2H — strips pin_read_d2h down to
                # just the driver/event-queue activity.
                while not idle_stop.is_set():
                    idle_event.record(idle_stream)
                    idle_event.synchronize()
                    idle_stop.wait(0.001)
            elif idle_mode == "cuda_sync_only":
                # torch.cuda.synchronize() drains every stream in the  #ignore-cuda
                # current device.  Coarsest cuda driver activity probe.
                while not idle_stop.is_set():
                    torch.cuda.synchronize()  #ignore-cuda
                    idle_stop.wait(0.001)

        threading.Thread(target=_idle_worker, daemon=True, name="moev_idle_probe").start()
        if rank == 0:
            print(f"[profile_run] idle-thread probe: mode={idle_mode}", flush=True)

    # Warmup (outside profiler so JIT / first-call costs don't pollute).
    for _ in range(WARMUP_ITERS):
        engine.train_batch(data_iter=data)
    # Drain warmup work + barrier so total_elapsed below starts from a
    # clean cross-rank state.  Without this the first-iter timer
    # absorbs leftover JIT / NCCL bootstrap from any straggler rank.
    if torch.cuda.is_available():  #ignore-cuda
        torch.cuda.synchronize()  #ignore-cuda
    dist.barrier()

    skip_profiler = int(os.environ.get("MOEV_PROFILE_SKIP_PROFILER", "0")) == 1
    if rank == 0:
        msg = f"[profile_run] warmup done; capturing {NUM_ITERS} iters"
        if skip_profiler:
            print(f"{msg} (profiler bypassed; clean iter_ms only)")
        else:
            print(f"{msg} into trace at {TRACE_PATH}")

    iter_times = []
    # Per-iter logging fires only on last-stage DP-rank-0 (the rank that
    # holds the loss tensor in pipe-parallel).  Stderr-tagged so it can
    # be greped out of the run log.
    log_per_iter = (PER_ITER_LOG and engine.is_last_stage() and engine.grid.get_data_parallel_rank() == 0)
    # CUDA work is async — ``train_batch`` returns when the CPU side is
    # done queuing kernels, not when the GPU has executed them.  Without
    # an explicit ``cuda.synchronize`` before the timer stop, the iter
    # timer measures CPU queuing latency, not wallclock per iter, and
    # comparisons across configs that differ in GPU-side queue depth
    # (e.g. factor=0 vs factor=1, where factor=1's boundary
    # ``synchronize()`` drains the side stream while factor=0 lets it
    # accumulate) become incomparable — one config's queued work spills
    # into the next iter's timer window.  Sync once per iter under the
    # timer to capture true wallclock.  Gated by MOEV_PROFILE_TIMER_SYNC
    # (default 1) so the f0-with-sync vs f0-without-sync A/B can run
    # from the same harness.
    timer_sync = int(os.environ.get("MOEV_PROFILE_TIMER_SYNC", "1")) == 1
    # Wrap the iter loop in the CUDA profiler start/stop API so an outer
    # `nsys profile --capture-range=cudaProfilerApi` records only steady
    # state, not init / first-iter JIT.  Off by default — opt-in for
    # nsys runs only.
    nsys_range = int(os.environ.get("MOEV_PROFILE_NSYS_RANGE", "0")) == 1

    def _emit_per_iter(idx, loss_obj, elapsed_ms, cpu_ms=None, sync_ms=None):
        if not log_per_iter:
            return
        if loss_obj is not None and hasattr(loss_obj, "item"):
            loss_val = float(loss_obj.item())
        else:
            loss_val = float("nan")
        cur_scale = getattr(engine.optimizer, "cur_scale", "?")
        overflow = getattr(engine.optimizer, "overflow", "?")
        breakdown = ""
        if cpu_ms is not None and sync_ms is not None:
            breakdown = f" cpu_ms={cpu_ms:.2f} sync_ms={sync_ms:.2f}"
        print(
            f"[per-iter] iter={idx} loss={loss_val:.6f} "
            f"scale={cur_scale} overflow={overflow} iter_ms={elapsed_ms:.2f}{breakdown}",
            flush=True)

    # End-to-end wallclock around the iter loop.  Iter-times stay rank-
    # local in the loop; we all_gather once at the end and reconstruct
    # cross-rank views (per-iter max, per-rank median, total elapsed)
    # in Python.  Single end-of-run collective avoids the all-reduce
    # overhead that the prior per-iter MAX approach added to
    # global_elapsed without landing inside any per-iter window.
    total_t0 = time.perf_counter()
    if skip_profiler:
        prof = None
        if nsys_range:
            torch.cuda.profiler.start()  #ignore-cuda
        for i in range(NUM_ITERS):
            t0 = time.perf_counter()
            loss = engine.train_batch(data_iter=data)
            t_cpu_done = time.perf_counter()
            if timer_sync:
                torch.cuda.synchronize()  #ignore-cuda
            t_sync_done = time.perf_counter()
            elapsed = t_sync_done - t0
            iter_times.append(elapsed)
            cpu_ms = (t_cpu_done - t0) * 1000
            sync_ms = (t_sync_done - t_cpu_done) * 1000
            _emit_per_iter(i, loss, elapsed * 1000, cpu_ms=cpu_ms, sync_ms=sync_ms)
        if nsys_range:
            torch.cuda.profiler.stop()  #ignore-cuda
    else:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
        ) as prof:
            for i in range(NUM_ITERS):
                t0 = time.perf_counter()
                loss = engine.train_batch(data_iter=data)
                t_cpu_done = time.perf_counter()
                if timer_sync:
                    torch.cuda.synchronize()  #ignore-cuda
                t_sync_done = time.perf_counter()
                elapsed = t_sync_done - t0
                iter_times.append(elapsed)
                cpu_ms = (t_cpu_done - t0) * 1000
                sync_ms = (t_sync_done - t_cpu_done) * 1000
                _emit_per_iter(i, loss, elapsed * 1000, cpu_ms=cpu_ms, sync_ms=sync_ms)

    # Drain any pending CUDA work then barrier-sync so total_elapsed
    # reflects the slowest rank's true end-to-end wallclock.
    if torch.cuda.is_available():  #ignore-cuda
        torch.cuda.synchronize()  #ignore-cuda
    dist.barrier()
    total_elapsed = time.perf_counter() - total_t0

    # One collective gather of per-rank iter_times + total_elapsed so
    # rank 0 can report a multi-rank view without per-iter chatter.
    all_iter_times = _gather_iter_times(iter_times)
    all_total_elapsed = _gather_iter_times([total_elapsed]).view(-1)

    if rank == 0:
        if prof is not None:
            prof.export_chrome_trace(TRACE_PATH)
        rank0_stats = _timing_stats(iter_times)
        max_per_iter = all_iter_times.max(dim=0).values.tolist()
        global_stats = _timing_stats(max_per_iter)
        rank_medians = [_timing_stats(row.tolist())["median"] for row in all_iter_times]
        total_ms = [v.item() * 1000 for v in all_total_elapsed]
        tag = "DISABLED" if MOEVEMENT_DISABLED else "ENABLED"
        print(f"[profile_run] moevement={tag}  rank0_iter_ms: "
              f"avg={rank0_stats['avg']:.2f} median={rank0_stats['median']:.2f} "
              f"min={rank0_stats['min']:.2f} max={rank0_stats['max']:.2f}")
        print(f"[profile_run] moevement={tag}  all_rank_max_iter_ms: "
              f"avg={global_stats['avg']:.2f} median={global_stats['median']:.2f} "
              f"min={global_stats['min']:.2f} max={global_stats['max']:.2f}")
        print(f"[profile_run] moevement={tag}  total_elapsed_ms: "
              f"rank0={total_ms[0]:.2f} max_rank={max(total_ms):.2f} min_rank={min(total_ms):.2f}")
        print(f"[profile_run] moevement={tag}  per_rank_median_ms: {[round(v, 2) for v in rank_medians]}")
        # Per-iter timings + drop-first-N summary so distribution
        # can be inspected without re-running.  Gated by env to keep
        # the default summary terse.
        if int(os.environ.get("MOEV_PROFILE_DUMP_ITERS", "0")) == 1:
            in_order_ms = [t * 1000 for t in iter_times]
            print(f"[profile_run] per-iter ms: {[round(v, 1) for v in in_order_ms]}")
            print(f"[profile_run] all-rank max per-iter ms: {[round(v * 1000, 1) for v in max_per_iter]}")
            for drop in (10, 20, 30):
                if len(in_order_ms) <= drop + 2:
                    continue
                tail = sorted(in_order_ms[drop:])
                print(f"[profile_run] drop_first={drop} (n={len(tail)}): "
                      f"avg={sum(tail) / len(tail):.2f} median={tail[len(tail) // 2]:.2f} "
                      f"min={tail[0]:.2f} max={tail[-1]:.2f}")
    if prof is not None:
        _print_marker_breakdown(prof, rank)

    # Receive-side audit: at f1 the gloo ring sends snapshots peer-to-peer.
    # Wait for any in-flight replication futures to drain before reading
    # — _received_snapshots is populated only after the recv calls complete,
    # and the worker thread may still be mid-send at end-of-loop.
    coord = getattr(engine, "moevement_coordinator", None)
    if coord is not None and getattr(coord, "snapshot_engine", None) is not None:
        from concurrent.futures import wait as _futures_wait
        in_flight = list(getattr(coord, "_replication_futures", []))
        if in_flight:
            _futures_wait(in_flight, timeout=30)
            print(f"[recv-audit] rank={rank} drained {len(in_flight)} futures", flush=True)
        recv = getattr(coord.snapshot_engine, "_received_snapshots", {})
        recv_summary = {peer: len(ops) for peer, ops in recv.items()}
        print(f"[recv-audit] rank={rank} received_from_peers={recv_summary}", flush=True)


if __name__ == "__main__":
    main()
