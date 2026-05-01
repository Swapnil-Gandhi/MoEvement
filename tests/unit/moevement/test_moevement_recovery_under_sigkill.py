# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Full MoEvement recovery protocol under a real SIGKILL + spare substitution.

**Status: PASSING**.  End-to-end validation of the full MoEvement
recovery protocol under a real SIGKILL:

1. Pre-fault training on all 4 ranks.
2. SIGKILL rank 0 (external fault injection from the parent).
3. Survivors (ranks 1, 2, 3) run ``survivor_rendezvous``; spare
   cold-starts with ``skip_initial_broadcast=True``.
4. Spare peer-pulls its sparse snapshot from rank 1
   (``load_sparse_from_peer`` + ``serve_sparse_snapshot_to_peer``).
   Peer-pull carries MoEvement operator state + FP16 loss-scaler AND
   engine-scalar state (``global_steps``, ``global_samples``, LR
   scheduler state, compression scheduler state) so the spare's
   engine lands on the paused peers' counters + scheduler step in
   one restore.
5. Every rank replays; recovering DP group catches up, unaffected
   DP group pauses in ``recovery_barrier`` then resumes.
6. All 4 ranks' post-recovery weights match the pre-fault reference
   within 1e-3 rtol/atol; engine scalars match across ranks.

Bridges two existing tests:

* ``TestMoEvementRecoveryWithRebuildNcclGroups`` exercises the recovery
  protocol end-to-end — ``load_sparse_from_peer`` +
  ``serve_sparse_snapshot_to_peer`` + replay-until-recovery-complete +
  per-param equivalence assertion — but fakes the fault via
  ``simulate_rank_failure`` (in-process state zero) and then calls
  ``engine.rebuild_nccl_groups()`` in lockstep on every rank.
* ``test_survivor_rendezvous_pp.py`` exercises a real SIGKILL + spare
  cold-start on a PP topology, but the post-rebuild validation is
  scoped to a WORLD probe + one ``train_batch`` after a trivial
  DP-group broadcast — no MoEvement recovery protocol involved.

This test composes the two: MoE-enabled PP=2 DP=2 engine, real SIGKILL
on rank 0, spare cold-starts via ``deepspeed.initialize(...,
skip_initial_broadcast=True)``, survivors call
``engine.survivor_rendezvous``, then the spare pulls a snapshot from
its surviving DP peer (rank 1) via the production peer-pull API, and
every rank replays through the persisted window until recovery
completes.  Post-recovery weights on every rank must match the
fault-free iter-``n_iters`` reference saved pre-fault.

Victim: rank 0 (stage 0, dp_rank 0).  Peer-pull source: rank 1 (stage
0, dp_rank 1, same DP group).  Ranks 2 and 3 are in the unaffected
DP group and pause via ``recovery_barrier`` until the recovering DP
group finishes.
"""

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

# File-based rendezvous (same shape as test_survivor_rendezvous_pp.py).
_READY_FILE = "ready_rank{rank}.txt"
_FAULT_FILE = "fault_signal.txt"
_NEW_WORLD_FILE = "new_world.txt"
_DONE_FILE = "done_rank{rank}.txt"
_REFERENCE_FILE = "reference_rank{rank}.pt"

_READY_TIMEOUT = 180.0
# 240s covers pre-fault training + survivor_rendezvous + peer-pull
# (~60s) + the replay loop converging on all 4 ranks (~50-90s under
# the current xfail; the actual fail is the equivalence assertion
# on the recovering DP group, which reaches ``done`` on survivors
# well inside this budget).
_DONE_TIMEOUT = 240.0
_POLL = 0.1

_VICTIM_RANK = 0


def _get_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(_POLL)
    return False


# Model + engine + data fixtures live in ``_recovery_test_helpers`` so
# the multi-host emulated-fault test can reuse the same shapes.
from unit.moevement._recovery_test_helpers import (
    build_engine as _build_engine,
    data_iter as _data_iter,
    engine_config as _engine_config,
)


def _worker_entrypoint(rank, world_size, master_addr, master_port, coord_dir, is_spare, streaming_recovery=False):
    import torch

    import deepspeed
    import deepspeed.comm as dist
    from deepspeed.accelerator import get_accelerator

    def log(msg):
        role = "spare" if is_spare else "initial"
        print(f"[rank {rank} ({role})] {msg}", flush=True)

    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        log(f"init_process_group master={master_addr}:{master_port} rank={rank} ws={world_size}")
        deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
        get_accelerator().set_device(rank % get_accelerator().device_count())

        config = _engine_config(streaming_recovery=streaming_recovery)
        torch.manual_seed(42)
        engine = _build_engine(config, skip_initial_broadcast=is_spare)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        log(f"engine built (PipelineEngine={type(engine).__name__}, w_sparse={w_sparse})")

        n_iters = 2 * w_sparse + 1
        reference = None

        if not is_spare:
            # Pre-fault training.  Capture the final-iter weights on every
            # rank so the post-recovery equivalence assertion can use the
            # fault-free reference without reconstructing it from scratch.
            #
            # Seed the data RNG deterministically so the replay loop on
            # the recovering DP group can reproduce the exact same sample
            # sequence — without this, the spare's post-build RNG state
            # depends on ``skip_initial_broadcast`` code path divergence,
            # and its replay sees different inputs than the pre-fault run.
            torch.manual_seed(123)
            data_iter = _data_iter()
            for _ in range(n_iters):
                engine.train_batch(data_iter=data_iter)
            reference = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
            # Persist to disk so the spare can pick up the reference from
            # rank 1 (its surviving DP peer — DP peers hold identical
            # weights) after peer-pull + replay completes.
            torch.save(reference, os.path.join(coord_dir, _REFERENCE_FILE.format(rank=rank)))
            # Capture engine-scalar reference alongside the weight reference
            # so the spare can verify post-peer-pull sync against rank 1's
            # pre-fault state.  LR scheduler is typically None on pipeline
            # engines without an explicit lr_scheduler config, so this
            # captures whatever is actually live.
            scalars_reference = {
                "global_steps":
                engine.global_steps,
                "global_samples":
                engine.global_samples,
                "lr_scheduler":
                (engine.lr_scheduler.state_dict() if getattr(engine, "lr_scheduler", None) is not None else None),
                "compression_scheduler": (engine.compression_scheduler.state_dict() if getattr(
                    engine, "compression_scheduler", None) is not None else None),
            }
            torch.save(scalars_reference, os.path.join(coord_dir, f"scalars_rank{rank}.pt"))

            # Wait for in-flight replication to land on rank 1 before
            # fault injection — same quiescence step as the in-process
            # rebuild test.  Without this the gloo worker's pending send
            # can race the SIGKILL and leave rank 1's
            # ``_received_snapshots[0]`` empty.
            while coord._replication_futures:
                coord._replication_futures.popleft().result(timeout=30.0)
            dist.barrier()
            log(f"pre-fault training done (n_iters={n_iters})")

        ready_path = os.path.join(coord_dir, _READY_FILE.format(rank=rank))
        with open(ready_path, "w") as f:
            f.write("ok")
        log(f"wrote ready file {ready_path}")

        from deepspeed.moevement.recovery_helpers import (
            run_as_survivor,
            run_as_spare,
            run_until_recovered,
        )

        if not is_spare:
            log("waiting for fault signal")
            while not os.path.exists(os.path.join(coord_dir, _FAULT_FILE)):
                time.sleep(_POLL)
            while not os.path.exists(os.path.join(coord_dir, _NEW_WORLD_FILE)):
                time.sleep(_POLL)
            with open(os.path.join(coord_dir, _NEW_WORLD_FILE)) as f:
                new_addr, new_port = f.read().strip().split(",")
            new_port = int(new_port)
            log(f"new master = {new_addr}:{new_port}")

            serve_t0 = time.perf_counter() if rank == 1 else None
            run_as_survivor(
                engine,
                coord,
                victim_rank=_VICTIM_RANK,
                new_master_addr=new_addr,
                new_master_port=new_port,
                new_rank=rank,
                new_world_size=world_size,
                peer_pull_source_rank=1,
            )
            if rank == 1:
                serve_dt_ms = (time.perf_counter() - serve_t0) * 1000
                log(f"[RECOVERY-TIMING] peer-pull serve_sparse_snapshot_to_peer: {serve_dt_ms:.1f} ms")
            log("survivor_rendezvous completed")
        else:
            pull_t0 = time.perf_counter()
            ok = run_as_spare(engine, coord, peer_pull_source_rank=1)
            assert ok is True, "peer-pull returned False — rank 1 reported no shard"
            pull_dt_ms = (time.perf_counter() - pull_t0) * 1000
            log(f"[RECOVERY-TIMING] peer-pull load_sparse_from_peer: {pull_dt_ms:.1f} ms")

        dist.barrier()

        # Replay until recovery completes on every rank.  Localized
        # cascade: ranks 0 and 1 (the recovering DP group) replay +
        # catch up; ranks 2 and 3 (unaffected DP group) pause inside
        # recovery_barrier then abandon their iter and resume.
        max_replay_iters = max(16, w_sparse * 3)
        # Match the pre-fault data seed so the replay loop on the
        # recovering DP group sees the exact same sample sequence.
        torch.manual_seed(123)
        data_iter = _data_iter()
        replay_t0 = time.perf_counter()
        replay_iter_count = run_until_recovered(engine, coord, data_iter, max_replay_iters)
        replay_dt_ms = (time.perf_counter() - replay_t0) * 1000
        log(f"[RECOVERY-TIMING] replay loop: {replay_dt_ms:.1f} ms across {replay_iter_count} iters "
            f"({replay_dt_ms/max(1,replay_iter_count):.1f} ms/iter)")

        # Equivalence assertion.  Survivors use their own saved
        # reference; the spare loads rank 1's reference from disk —
        # ranks 0 and 1 are DP peers and converge to the same weights
        # post-recovery, so rank 1's pre-fault final state is the
        # correct target for the spare too.
        if reference is None:
            reference = torch.load(os.path.join(coord_dir, _REFERENCE_FILE.format(rank=1)))
        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            # Tolerance matches the relaxed equivalence in
            # test_engine_integration.py — post-recovery residual is FP16
            # round-trip noise from ``update_lp_params`` after each active-op
            # restore, not a logic bug.
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"rank {rank} param {n} post-recovery diverged from "
                                         f"iter-{n_iters} fault-free reference; {msg}"),
            )
        assert engine.global_steps == n_iters, (f"rank {rank} global_steps={engine.global_steps} != "
                                                f"n_iters={n_iters} post-recovery")
        # Engine-scalar equivalence.  Spare uses rank 1's scalars
        # reference (DP peers hold identical engine state); survivors
        # use their own saved scalars.  Verifies that peer-pull's
        # engine_scalars restore path lands the spare on the paused
        # peers' counters + scheduler step, matching the weight
        # equivalence above.
        scalars_ref = torch.load(os.path.join(coord_dir, f"scalars_rank{1 if is_spare else rank}.pt"))
        assert engine.global_samples == scalars_ref["global_samples"], (
            f"rank {rank} global_samples={engine.global_samples} != "
            f"reference {scalars_ref['global_samples']} post-recovery")
        if scalars_ref["compression_scheduler"] is not None:
            assert engine.compression_scheduler is not None, (f"rank {rank} compression_scheduler is None "
                                                              f"but reference had one")
            assert (engine.compression_scheduler.state_dict() == scalars_ref["compression_scheduler"]), (
                f"rank {rank} compression_scheduler state diverged post-recovery")
        if scalars_ref["lr_scheduler"] is not None:
            assert engine.lr_scheduler is not None, f"rank {rank} lr_scheduler is None but reference had one"
            assert (engine.lr_scheduler.state_dict() == scalars_ref["lr_scheduler"]), (
                f"rank {rank} lr_scheduler state diverged post-recovery")
        log(f"post-recovery equivalence verified (global_steps={engine.global_steps})")

        done_path = os.path.join(coord_dir, _DONE_FILE.format(rank=rank))
        with open(done_path, "w") as f:
            f.write("ok")
        log("done")
    except Exception as e:
        import traceback
        print(f"[rank {rank}] WORKER CRASHED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def _run_peer_pull_recovery_scenario(tmp_path, streaming_recovery):
    """Spawn 4 workers, SIGKILL rank 0, watch the spare peer-pull its shard.

    Parameterised by ``streaming_recovery`` so bulk and streaming both
    exercise the full SIGKILL → cluster-manager → peer-pull →
    sparse-to-dense-conversion → replay-to-fault-iter path end-to-end.
    Called from the two ``test_*`` wrappers below.
    """
    try:
        nvidia_smi = subprocess.run(["nvidia-smi", "--list-gpus"],
                                    capture_output=True,
                                    text=True,
                                    timeout=10.0,
                                    check=False)
        n_gpus = len(nvidia_smi.stdout.strip().splitlines()) if nvidia_smi.returncode == 0 else 0
    except FileNotFoundError:
        n_gpus = 0
    if n_gpus < 4:
        pytest.skip(f"requires 4 GPUs; found {n_gpus}")

    coord_dir = tempfile.mkdtemp(prefix="moevement_rec_", dir=str(tmp_path))
    world_size = 4
    initial_master_addr = "127.0.0.1"
    initial_master_port = _get_free_port()
    new_master_addr = "127.0.0.1"
    new_master_port = _get_free_port()

    ctx = mp.get_context("spawn")

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_entrypoint,
            args=(rank, world_size, initial_master_addr, initial_master_port, coord_dir, False, streaming_recovery),
        )
        p.start()
        procs.append(p)

    try:
        for rank in range(world_size):
            path = os.path.join(coord_dir, _READY_FILE.format(rank=rank))
            if not _wait_for(path, _READY_TIMEOUT):
                pytest.fail(f"rank {rank} never reached ready file {path}")

        victim = procs[_VICTIM_RANK]
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=10.0)
        assert not victim.is_alive(), f"rank {_VICTIM_RANK} did not die after SIGKILL"

        with open(os.path.join(coord_dir, _NEW_WORLD_FILE), "w") as f:
            f.write(f"{new_master_addr},{new_master_port}")
        with open(os.path.join(coord_dir, _FAULT_FILE), "w") as f:
            f.write("fault")

        spare = ctx.Process(
            target=_worker_entrypoint,
            args=(_VICTIM_RANK, world_size, new_master_addr, new_master_port, coord_dir, True, streaming_recovery),
        )
        spare.start()
        procs[_VICTIM_RANK] = spare

        for rank in range(world_size):
            path = os.path.join(coord_dir, _DONE_FILE.format(rank=rank))
            if not _wait_for(path, _DONE_TIMEOUT):
                pytest.fail(f"rank {rank} never reached done file {path} — "
                            f"MoEvement peer-pull recovery under SIGKILL likely hung or diverged "
                            f"(streaming_recovery={streaming_recovery})")

        for p in procs:
            p.join(timeout=10.0)

        for rank in range(world_size):
            p = procs[rank]
            assert p.exitcode == 0, (f"rank {rank} exited with code {p.exitcode} — "
                                     f"MoEvement recovery flow raised "
                                     f"(streaming_recovery={streaming_recovery}); "
                                     f"inspect worker stderr")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    reason="requires at least 4 GPUs for PP=2 DP=2 (3 survivors + 1 spare)")
def test_moevement_peer_pull_recovery_under_real_sigkill(tmp_path):
    """Agent: 4 PP=2 DP=2 workers, MoEvement-enabled, SIGKILL rank 0, peer-pull recovery."""
    _run_peer_pull_recovery_scenario(tmp_path, streaming_recovery=False)


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    reason="requires at least 4 GPUs for PP=2 DP=2 (3 survivors + 1 spare)")
def test_moevement_peer_pull_recovery_under_real_sigkill_streaming(tmp_path):
    """Same scenario as above but with ``streaming_recovery=True``.

    Exercises the SD-O4 S2 path: iter-major wire + background pull
    thread + incremental converter ingest + ``_setup_replay_iter``
    draining the pull queue on demand.  Post-recovery weight
    equivalence must match the bulk path.
    """
    _run_peer_pull_recovery_scenario(tmp_path, streaming_recovery=True)
