# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Worker entrypoint for the multi-host emulated-fault MoEvement recovery test.

Invoked by the deepspeed launcher across both hosts.  Same shape as the
single-host SIGKILL test — pre-fault training, recovery, post-recovery
weight equivalence — except the "fault" is emulated by calling
``simulate_rank_failure`` on the victim rank in place rather than
sending SIGKILL.  The WORLD stays intact, no rendezvous is needed,
peer-pull and the cascade-recovery path otherwise run unchanged.

What this does NOT validate (covered by the single-host SIGKILL test):
the launcher's fault detection, ``engine.survivor_rendezvous``, the
WORLD-destroy + reinit codepath, ``skip_initial_broadcast=True`` on
the spare's engine build.

What this DOES validate that single-host can't:
peer-pull wire over real cross-host gloo TCP, the SD-O4 streaming-
recovery ``max(pull, replay)`` invariant under realistic wire latency.
"""

import argparse
import os
import sys
import time

import torch

# Worker is launched as the deepspeed launcher's target script.  Its own
# directory (tests/unit/moevement) is automatically on sys.path; the
# tests/ root needs to be added so ``from unit.moevement._foo import ...``
# resolves the same way it does under pytest.
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_WORKER_DIR, "..", "..")))

from unit.moevement._fault_inject import simulate_rank_failure
from unit.moevement._recovery_test_helpers import (
    build_engine,
    data_iter,
    engine_config,
)


def _log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fault-at-iter",
        type=int,
        default=9,
        help="Iter at which rank 0 simulates failure (in-place state clear).  "
        "Must be large enough for at least one finalize_window to have fired so "
        "the donor (rank 1) holds a persisted shard.",
    )
    parser.add_argument(
        "--streaming-recovery",
        action="store_true",
        help="Use SD-O4 streaming peer-pull protocol instead of bulk.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-3,
        help="Relative tolerance for post-recovery weight equivalence — matches "
        "the relaxed bound in test_engine_integration.py for FP16 round-trip "
        "noise from update_lp_params after each active-op restore.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-3,
        help="Absolute tolerance for post-recovery weight equivalence.",
    )
    # ``--local_rank`` is injected by ``deepspeed.launcher.launch``; we
    # don't use it (LOCAL_RANK lands in the env via the launcher too)
    # but argparse must accept it or the worker exits with code 2.
    parser.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    import deepspeed
    import deepspeed.comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.moevement.recovery_helpers import (
        run_as_spare,
        run_as_survivor,
        run_until_recovered,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    _log(
        rank, f"init_distributed world_size={world_size} master={os.environ.get('MASTER_ADDR')}:"
        f"{os.environ.get('MASTER_PORT')}")
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
    get_accelerator().set_device(int(os.environ.get("LOCAL_RANK", rank)) % get_accelerator().device_count())

    # Data-major topology so the launcher's host-by-rank allocation
    # (host A: ranks 0,1; host B: ranks 2,3) yields DP groups {0,2} and
    # {1,3} that BOTH cross hosts.  Default ``PipeDataParallelTopology``
    # (axes=['pipe', 'data']) keeps DP intra-host, which makes
    # peer-pull go over loopback even multi-host — defeats the SD-O4
    # ``max(pull, replay)`` invariant validation.  Reversing the axes
    # to ['data', 'pipe'] flips DP cross-host, PP intra-host.
    from deepspeed.runtime.pipe.topology import ProcessTopology
    topology = ProcessTopology(axes=["data", "pipe"], dims=[2, 2])

    config = engine_config(streaming_recovery=args.streaming_recovery)
    torch.manual_seed(42)
    engine = build_engine(config, skip_initial_broadcast=False, topology=topology)
    coord = engine.moevement_coordinator
    w_sparse = coord.scheduler.w_sparse
    _log(rank, f"engine built (PipelineEngine={type(engine).__name__}, w_sparse={w_sparse}); topology axes=['data', 'pipe']")

    # Pre-fault training.  Same data RNG seed as the single-host test
    # so the recovering DP group's replay sees the identical sample
    # sequence — without this the spare's post-zero RNG state would
    # diverge from the donor's.
    n_iters = args.fault_at_iter
    torch.manual_seed(123)
    di = data_iter()
    for _ in range(n_iters):
        engine.train_batch(data_iter=di)

    reference = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}

    # Drain in-flight replication so rank 1's persisted shard for
    # rank 0 has fully landed before the fault.  Same quiescence step
    # the in-process rebuild test uses.
    while coord._replication_futures:
        coord._replication_futures.popleft().result(timeout=30.0)
    dist.barrier()
    _log(rank, f"pre-fault training done (n_iters={n_iters})")

    # FAULT INJECTION (in-place, no actual kill).  Only rank 0 calls
    # ``simulate_rank_failure``; other ranks observe the resulting
    # ``coord._recovering=True`` indirectly through the world handshake
    # in the next ``recovery_barrier`` round.
    if rank == 0:
        _log(rank, "simulating rank failure (in-place state clear, GPU intact)")
        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)

    # Recovery setup.  No ``survivor_rendezvous`` — WORLD is intact
    # because no process actually died.  Under the data-major topology
    # rank 0's DP peer is rank 2 (cross-host), so rank 2 serves and
    # rank 0 pulls; ranks 1 and 3 are the OTHER DP group (also at
    # different stages — see launch.py:179 global_rank_mapping) and
    # pause via recovery_barrier without participating in peer-pull.
    if rank == 0:
        pull_t0 = time.perf_counter()
        ok = run_as_spare(engine, coord, peer_pull_source_rank=2)
        pull_dt_ms = (time.perf_counter() - pull_t0) * 1000
        assert ok, "peer-pull returned False — donor reported no shard"
        _log(rank, f"[RECOVERY-TIMING] load_sparse_from_peer: {pull_dt_ms:.1f} ms")
    elif rank == 2:
        serve_t0 = time.perf_counter()
        run_as_survivor(
            engine,
            coord,
            victim_rank=0,
            peer_pull_source_rank=2,
            rendezvous=False,
        )
        serve_dt_ms = (time.perf_counter() - serve_t0) * 1000
        _log(rank, f"[RECOVERY-TIMING] serve_sparse_snapshot_to_peer: {serve_dt_ms:.1f} ms")

    dist.barrier()

    # Replay until recovery clears on every rank.  Under the data-major
    # topology, recovering DP group is {0, 2} (replays the persisted
    # window + catch-up) and the unaffected DP group is {1, 3} (pauses
    # inside ``recovery_barrier``, abandons its iter, then resumes).
    max_replay_iters = max(16, w_sparse * 3)
    torch.manual_seed(123)
    di = data_iter()
    replay_t0 = time.perf_counter()
    consumed = run_until_recovered(engine, coord, di, max_replay_iters)
    replay_dt_ms = (time.perf_counter() - replay_t0) * 1000
    _log(
        rank, f"[RECOVERY-TIMING] replay loop: {replay_dt_ms:.1f} ms across {consumed} iters "
        f"({replay_dt_ms / max(1, consumed):.1f} ms/iter)")

    # Post-recovery equivalence.  All four ranks should have weights
    # matching their pre-fault references — DP peers converge to the
    # same shard, so rank 0's reference (zeroed by the fault) is
    # restored to rank 1's pre-fault values via peer-pull.
    post = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
    for name, restored in post.items():
        torch.testing.assert_close(
            restored,
            reference[name],
            rtol=args.rtol,
            atol=args.atol,
            msg=lambda msg, n=name: (f"rank {rank} param {n} post-recovery diverged from "
                                     f"pre-fault reference; {msg}"),
        )

    assert engine.global_steps == n_iters, (f"rank {rank} global_steps={engine.global_steps} != "
                                            f"n_iters={n_iters} post-recovery")
    _log(rank, f"PASS (global_steps={engine.global_steps}, replay_iters={consumed})")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        import traceback
        rank = int(os.environ.get("RANK", -1))
        print(f"[rank {rank}] FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
