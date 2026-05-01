# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""PP=2 DP=2 end-to-end SIGKILL + survivor-rendezvous canary for PipelineEngine.

Extends ``test_survivor_rendezvous.py`` (DeepSpeedEngine + ZeRO-1) to
``PipelineEngine`` on a ``PipelineModule`` with ``num_stages=2`` on a
world of 4 (⇒ PP=2 DP=2).  Same SIGKILL + cold-start-with-
``skip_initial_broadcast=True`` pattern; this canary exercises the
PipelineEngine code path so any regression in the kwarg plumbing or
the PP-specific rebuild handshake surfaces as a hard failure.

Post-rebuild the canary goes beyond the WORLD probe: ranks do a
DP-group broadcast from the surviving DP peer (to sync the spare's
fresh-init stage-0 weights with its DP peer's pre-fault state), then
run one ``engine.train_batch`` on the rebuilt groups.  This exercises
the full post-rebuild pipeline-schedule + optimizer-step path and
surfaced the Layer 7 bug (``relink_all_dp_refs`` didn't refresh
``self.model_parallel_group`` — ZeRO's FP16-overflow-check allreduce
hit the torn-down PG and raised ``TypeError: Invariant encountered``);
fix landed alongside this canary extension.

Victim: rank 0 (stage 0, dp_rank 0).  Spare cold-starts as rank 0 on
the new master.  Ranks 1 (stage 0, dp_rank 1), 2 (stage 1, dp_rank 0),
3 (stage 1, dp_rank 1) survive.

**Prior xfail causes, now resolved.**  When first wired up, this
canary cycled through three distinct PP-specific blockers, all
fixed in-tree:

1. **``PipelineModule.__init__`` world-clone ``new_group`` drift.**
   ``PipelineModule`` called ``dist.new_group(ranks=range(world_size))``
   to build ``self.world_group`` — identical shape to the Layer 1
   ``_clone_world_group`` bug.  Survivor-rebuild path does not re-issue
   this call (rebuild reuses the default PG) but cold-started spare
   does, drifting the ``_world.group_count`` counter.  Fixed by
   switching to ``dist.get_world_group()`` (the WORLD sentinel —
   no ``new_group`` call, no counter).  See ``pipe/module.py:163``
   comment.

2. **``PipelineEngine.__init__`` param-count allreduce.**  After the
   counter-drift fix, spare stalled in
   ``dist.all_reduce(params_tensor, group=model_parallel_group)``
   inside ``PipelineEngine.__init__`` — a purely diagnostic collective
   (aggregates per-stage param counts for a logging line) that
   survivors ran pre-fault on the OLD ``model_parallel_group`` and
   did NOT re-run on the rebuilt one.  Fixed by gating on
   ``self._skip_initial_broadcast`` and falling back to per-stage
   counts.

3. **``PipelineEngine.__init__`` adjacent-stage "send a 0" handshake.**
   After gating (2), spare stalled in the pipeline-communicator
   init p2p handshake (``p2p.send(self.loss, next_stage)`` /
   ``p2p.recv(...)``).  Same structural pattern as (2): handshake ran
   pre-fault on the OLD pp group, not re-run on the rebuilt one.
   Fixed by the same ``_skip_initial_broadcast`` gate.

Shared pattern across (2) and (3): any init-time collective in
``PipelineEngine.__init__`` on a sub-group will stall on the
recovery-mode spare because survivors already paid the handshake on
the torn-down group and won't repay it on the rebuilt group.  Gate
on ``_skip_initial_broadcast``; content-bearing collectives that
the recovery protocol will supply via snapshot/peer-pull don't need
a survivor-side mirror.
"""

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

_READY_FILE = "ready_rank{rank}.txt"
_FAULT_FILE = "fault_signal.txt"
_NEW_WORLD_FILE = "new_world.txt"
_DONE_FILE = "done_rank{rank}.txt"

_READY_TIMEOUT = 120.0
_DONE_TIMEOUT = 180.0
_POLL = 0.1

_HIDDEN = 16
_BATCH = 2
_SEQ = 4

# Victim rank — hard-coded so the post-rebuild DP weight-sync broadcast
# can pick a non-victim source within the spare's DP group.  The
# supervisor / test-agent plumbing above also kills this rank.
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


# Module-scope layer classes: ``PipelineModule`` / ``LayerSpec``
# instantiate these inside each rank's own import, so they must be
# picklable + importable at module scope (``mp.spawn`` re-imports this
# file in every worker).

import torch.nn as nn  # noqa: E402


class _Embed(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, x):
        return self.lin(x)


class _Hidden(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, x):
        return self.lin(x)


class _Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


def _build_pipeline_module():
    from deepspeed.pipe import PipelineModule, LayerSpec
    layers = [
        LayerSpec(_Embed),
        LayerSpec(_Hidden),
        LayerSpec(_Hidden),
        LayerSpec(_Head),
    ]
    return PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss())


def _data_iter():
    import torch
    # fp16 config ⇒ inputs must match model dtype.  Label shape must
    # match what ``_Head`` returns: mean over dim=1 ⇒ (batch, hidden).
    sample = (torch.randn(_BATCH, _SEQ, _HIDDEN, dtype=torch.float16), torch.randn(_BATCH,
                                                                                   _HIDDEN,
                                                                                   dtype=torch.float16))

    class _Repeating:

        def __init__(self, s):
            self.s = s

        def __iter__(self):
            return self

        def __next__(self):
            return self.s

    return _Repeating(sample)


def _worker_entrypoint(rank, world_size, master_addr, master_port, coord_dir, is_spare):
    """Runs in each worker process — mirrors the DeepSpeedEngine canary."""
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
        device_name = get_accelerator().device_name()
        get_accelerator().set_device(rank % get_accelerator().device_count())

        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 2,
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
        }
        model = _build_pipeline_module()
        engine, *_ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
            # Recovery-mode spare skips the initial broadcast: its weights
            # load via snapshot/peer-pull (or in this canary, are simply
            # unused because the post-rebuild validation is a WORLD probe,
            # not another train_batch).  Survivors keep the flag off so
            # their normal pre-fault ``_broadcast_model`` runs on the
            # old comm as usual.
            skip_initial_broadcast=is_spare,
        )
        log(f"engine built (PipelineEngine={type(engine).__name__})")

        if not is_spare:
            engine.train_batch(data_iter=_data_iter())
            log("pre-fault train_batch done")

        ready_path = os.path.join(coord_dir, _READY_FILE.format(rank=rank))
        with open(ready_path, "w") as f:
            f.write("ok")
        log(f"wrote ready file {ready_path}")

        if not is_spare:
            log("waiting for fault signal")
            while not os.path.exists(os.path.join(coord_dir, _FAULT_FILE)):
                time.sleep(_POLL)
            log("fault signalled — reading new world config")

            while not os.path.exists(os.path.join(coord_dir, _NEW_WORLD_FILE)):
                time.sleep(_POLL)
            with open(os.path.join(coord_dir, _NEW_WORLD_FILE)) as f:
                new_addr, new_port = f.read().strip().split(",")
            new_port = int(new_port)
            log(f"new master = {new_addr}:{new_port}")

            # Drain pre-fault CUDA work before the rendezvous.  Not
            # actually required for correctness post-fix (Layer 3 was
            # op-order mismatch, not CUDA-stream residue), but keeps the
            # rendezvous timing deterministic and mirrors the sibling
            # test.
            get_accelerator().synchronize()

            engine.survivor_rendezvous(
                new_master_addr=new_addr,
                new_master_port=new_port,
                new_rank=rank,
                new_world_size=world_size,
            )
            log("survivor_rendezvous completed")

        # Post-rebuild WORLD-level probe: confirms the default PG is
        # rewired across survivors + spare.
        probe = torch.ones(1, device=device_name) * (rank + 1)
        dist.all_reduce(probe)
        result = probe.item()
        expected = sum(r + 1 for r in range(world_size))
        assert result == expected, (f"rank {rank} post-rebuild WORLD allreduce got {result}, "
                                    f"expected {expected}; PP-topology rebuild failed to rewire the default PG")
        log(f"post-rebuild WORLD allreduce OK, sum={result}")

        # Survivor-to-spare weight sync within the spare's DP group.
        # The spare cold-started with ``skip_initial_broadcast=True``, so
        # its stage-0 weights are fresh random init — not the pre-fault
        # trained state its surviving DP peer (rank 1) holds.  Bring the
        # DP group back into agreement by broadcasting from the surviving
        # peer.  Unaffected DP groups (ranks 2 & 3 at stage 1) already
        # agree internally; they broadcast from their min rank as a
        # symmetric no-op so each DP group performs the same op count on
        # its rebuilt ``dp_proc_group``.  This is option (A) in the task
        # plan — a minimum proof that the post-rebuild engine is
        # actually usable for training.  Options (B) disk-checkpoint and
        # (C) MoEvement peer-pull are separate milestones.
        dp_ranks = engine.grid.dp_group
        dp_pg = engine.grid.dp_proc_group
        src_rank = min(r for r in dp_ranks if r != _VICTIM_RANK)
        for p in engine.module.parameters():
            dist.broadcast(p.data, src=src_rank, group=dp_pg)
        log(f"post-rebuild DP weight sync done (dp_ranks={dp_ranks}, src={src_rank})")

        # Post-rebuild train_batch: exercises the pipeline schedule +
        # optimizer step on the rebuilt groups.  Success criterion is
        # that every rank returns without raising — op-order asymmetry
        # or stale PG references would deadlock or error here.
        engine.train_batch(data_iter=_data_iter())
        log("post-rebuild train_batch OK")

        done_path = os.path.join(coord_dir, _DONE_FILE.format(rank=rank))
        with open(done_path, "w") as f:
            f.write(f"{result:.0f}")
        log("done")
    except Exception as e:
        import traceback
        print(f"[rank {rank}] WORKER CRASHED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    reason="requires at least 4 GPUs for PP=2 DP=2 (3 survivors + 1 spare)")
def test_survivor_rendezvous_pp2_dp2_under_real_sigkill(tmp_path):
    """Agent: spawn 4 PP=2 DP=2 workers, SIGKILL rank 0, rendezvous with spare."""
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

    coord_dir = tempfile.mkdtemp(prefix="survivor_rdv_pp_", dir=str(tmp_path))
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
            args=(rank, world_size, initial_master_addr, initial_master_port, coord_dir, False),
        )
        p.start()
        procs.append(p)

    try:
        for rank in range(world_size):
            path = os.path.join(coord_dir, _READY_FILE.format(rank=rank))
            if not _wait_for(path, _READY_TIMEOUT):
                pytest.fail(f"rank {rank} never reached ready file {path}")

        victim = procs[0]
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=10.0)
        assert not victim.is_alive(), "rank 0 did not die after SIGKILL"

        with open(os.path.join(coord_dir, _NEW_WORLD_FILE), "w") as f:
            f.write(f"{new_master_addr},{new_master_port}")
        with open(os.path.join(coord_dir, _FAULT_FILE), "w") as f:
            f.write("fault")

        spare = ctx.Process(
            target=_worker_entrypoint,
            args=(0, world_size, new_master_addr, new_master_port, coord_dir, True),
        )
        spare.start()
        procs[0] = spare

        for rank in range(world_size):
            path = os.path.join(coord_dir, _DONE_FILE.format(rank=rank))
            if not _wait_for(path, _DONE_TIMEOUT):
                pytest.fail(f"rank {rank} never reached done file {path} — "
                            f"PP survivor_rendezvous or spare cold-start likely hung")

        for p in procs:
            p.join(timeout=10.0)

        for rank in range(world_size):
            p = procs[rank]
            assert p.exitcode == 0, (f"rank {rank} exited with code {p.exitcode} — "
                                     f"PP survivor_rendezvous flow raised; inspect the worker stderr")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
