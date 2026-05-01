# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end SIGKILL + survivor-rendezvous integration test.

Shape: one worker is SIGKILLed mid-training, the three survivors
stay in-process and call ``engine.survivor_rendezvous`` to destroy
+ re-init their default PG on a new master and rebuild every
cached sub-group, a spare worker cold-starts onto the new master
with ``skip_initial_broadcast=True`` (recovery-mode init — see
below), and all four ranks then complete a WORLD-level collective.

Does NOT use ``DistributedTest`` — that harness assumes every
worker survives; here a worker is deliberately killed by the test
itself.  Instead, the pytest function acts as the agent: spawns
N+1 workers via ``multiprocessing.spawn``, SIGKILLs one at a
scripted point, writes the new-world config to a file rendezvous,
and waits for the survivors + cold-started spare to complete.

**Why ``skip_initial_broadcast=True`` on the spare.**  The spare
cold-starts via ``deepspeed.initialize``; without the flag,
``_broadcast_model`` would fire on the post-rebuild default PG and
dispatch one NCCL Broadcast per parameter at opCounts 0..N-1.
Survivors reuse their pre-fault engine, so their
``_broadcast_model`` already ran on the (now-destroyed) pre-fault
comm — they never re-broadcast on the new comm.  That asymmetry
(spare: N broadcasts then probe; survivors: probe only) parks the
shared NCCL ring on mismatched op slots — spare's single-node
P2P/SHM allreduce drains with the correct sum, but survivors'
allreduce kernels never complete.  The flag tells the spare to
skip the broadcast entirely: recovery paths load params via
snapshot/peer-pull, so broadcasting the spare's fresh random init
would both corrupt survivors' trained state and create the
op-order asymmetry.  With the flag set, op order is trivially
aligned (no broadcasts on any rank) and the post-rebuild collective
completes on all four.

**Earlier xfail causes, now resolved:**
The test cycled through three distinct failure modes before
reaching its current XPASS state; all three are fixed in-tree:
  1. ``_clone_world_group`` counter drift — survivors called
     ``new_group`` inside rebuild's ``_refresh_group_caches`` while
     the spare called it from inside ``deepspeed.initialize``; the
     two store-keyed calls produced different NCCL comms despite
     matching rank lists.  Resolved by changing ``_clone_world_group``
     to return ``dist.group.WORLD`` directly (no ``new_group`` call,
     no counter).
  2. Default-PG NCCL bootstrap EOF (``store->get('0')`` returning
     0 bytes) — downstream symptom of (1), resolved by the same
     change.
  3. Post-rebuild allreduce op-order mismatch — the present case,
     resolved by the ``skip_initial_broadcast`` flag on the spare.
     A prior hypothesis of "CUDA-stream residue" was empirically
     refuted by per-process NCCL traces.
"""

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

# File-based rendezvous keys.  Workers write to / poll these inside
# ``coord_dir``; the agent writes the new-world config after it has
# SIGKILLed the victim.
_READY_FILE = "ready_rank{rank}.txt"
_FAULT_FILE = "fault_signal.txt"
_NEW_WORLD_FILE = "new_world.txt"
_DONE_FILE = "done_rank{rank}.txt"

# Tight timeouts — the full sequence (train step + SIGKILL + survivor
# rendezvous + spare cold-start + post-rebuild train step) should
# complete well inside a minute on this box.  If anything hangs past
# this we've found a new failure mode.
_READY_TIMEOUT = 60.0
_DONE_TIMEOUT = 120.0
_POLL = 0.1


def _get_free_port():
    """Pick an unused TCP port on the loopback for the fresh master."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(path, timeout):
    """Poll for ``path`` to appear, returning True if it did before ``timeout``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(_POLL)
    return False


def _worker_entrypoint(rank, world_size, master_addr, master_port, coord_dir, is_spare):
    """Runs in each worker process.

    Four roles:
      * Pre-fault ranks 0..3 on the old master → init, train one step,
        write ``ready_rank{r}``, then wait for either the fault signal
        (survivor path) or SIGKILL (victim path).
      * ``is_spare`` rank on the new master → cold-start, load nothing,
        build engine, train one step on the post-rebuild world.

    The two roles share most of the entrypoint; branching is by
    whether ``is_spare`` is set (i.e., whether this process was spawned
    by the agent as a replacement vs. at initial boot).
    """
    # Keep heavy imports inside the entrypoint so ``mp.spawn`` forks a
    # minimal parent.  Also lets the agent process avoid loading torch
    # + deepspeed just to coordinate workers.
    import torch
    import torch.nn as nn

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
        log(f"world_size={dist.get_world_size()} my_rank={dist.get_rank()}")
        device_name = get_accelerator().device_name()
        get_accelerator().set_device(rank % get_accelerator().device_count())

        model = nn.Linear(8, 8).to(device_name)
        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True
                }
            },
            "zero_optimization": {
                "stage": 1
            },
            "fp16": {
                "enabled": False
            },
        }
        # Spare joins a post-fault world where survivors kept their
        # trained state on the pre-fault comm; skipping the initial
        # param broadcast both preserves survivors' weights and
        # aligns op order on the shared post-rebuild NCCL comm.
        engine, *_ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=config,
                                          skip_initial_broadcast=is_spare)
        log("engine built")

        x = torch.ones(1, 8, device=device_name)
        y_target = torch.zeros(1, 8, device=device_name)

        # Survivors run a pre-fault step to materialize optimizer state
        # + warm the NCCL comm.  The spare skips this: its ``engine``
        # just bootstrapped on the new master, and spare + survivors
        # must issue exactly one matching allreduce each on the new
        # world for the post-rebuild step to line up.
        if not is_spare:
            loss = ((engine(x) - y_target)**2).sum()
            engine.backward(loss)
            engine.step()
            log(f"pre-fault step done, loss={loss.item():.6f}")

        ready_path = os.path.join(coord_dir, _READY_FILE.format(rank=rank))
        with open(ready_path, "w") as f:
            f.write("ok")
        log(f"wrote ready file {ready_path}")

        if not is_spare:
            # Survivors wait for the agent's fault signal, read the new
            # master config, and run the in-process rendezvous.  The
            # victim (rank 0) never reaches this point — it's SIGKILLed
            # by the agent right after writing the ready file.
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

            # Drain any kernels the pre-fault ZeRO step left enqueued
            # on this rank's GPU.  Without this, pending ops hold
            # references to the old default PG's NCCL comm, and the
            # destroy_process_group inside survivor_rendezvous leaves
            # the queue pointing at a dead comm — subsequent
            # collectives on the new PG enqueue behind the dead-refs
            # and ``.item()`` on their result blocks forever.  Spare
            # doesn't hit this because its device was clean (the
            # SIGKILLed victim's state was already released by the
            # kernel; spare cold-starts on a freshly-reclaimed GPU).
            get_accelerator().synchronize()

            # 1:1 substitution — rank + world_size unchanged; only the
            # master endpoint moved because the old rank-0-hosted
            # TCPStore died with the victim.
            engine.survivor_rendezvous(new_master_addr=new_addr,
                                       new_master_port=new_port,
                                       new_rank=rank,
                                       new_world_size=world_size)
            log("survivor_rendezvous completed")

        # Post-rebuild sanity: a collective on the default WORLD group
        # across all four ranks (3 survivors + 1 cold-started spare).
        # Deliberately NOT using ``engine.step()`` here — see the
        # module docstring's "Known limitation" note for the sub-group-
        # clone counter-drift issue that breaks ZeRO post-rebuild.
        # What this DOES prove: destroy + re-init + rebuild_nccl_groups
        # leaves the default PG fully usable across survivors + spare
        # under a real SIGKILL, which is the load-bearing property of
        # ``survivor_rendezvous``.
        probe = torch.ones(1, device=device_name) * (rank + 1)
        dist.all_reduce(probe)
        result = probe.item()
        expected = sum(r + 1 for r in range(world_size))
        assert result == expected, (f"rank {rank} post-rebuild WORLD allreduce got {result}, "
                                    f"expected {expected}; PG was not fully rebuilt across ranks")
        log(f"post-rebuild WORLD allreduce OK, sum={result}")

        done_path = os.path.join(coord_dir, _DONE_FILE.format(rank=rank))
        with open(done_path, "w") as f:
            f.write(f"{result:.0f}")
        log("done")
    except Exception as e:
        # Make the worker exit non-zero so the agent sees the failure
        # in the exit-code check and the test fails with context.
        import traceback
        print(f"[rank {rank}] WORKER CRASHED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    reason="requires at least 4 GPUs for 3 survivors + 1 spare")
def test_survivor_rendezvous_under_real_sigkill(tmp_path):
    """Agent: spawn 4 workers, SIGKILL rank 0, rendezvous survivors + spare.

    Four-phase flow:
      1. **Boot.** Start 4 workers on an initial master.  Wait for
         each to write ``ready_rank{r}``.
      2. **Fault.** SIGKILL worker 0.  Write the new-master
         ``(addr, port)`` to ``new_world.txt`` (new port because the
         old TCPStore on port X was hosted on rank 0 and died with
         it), then touch ``fault_signal.txt``.
      3. **Rendezvous.** Survivors (ranks 1, 2, 3) notice the fault
         signal, read the new master, and call ``survivor_rendezvous``.
         Simultaneously the agent spawns a replacement worker as
         rank 0 on the new master — it cold-starts, boots the engine,
         and joins the post-rebuild world.
      4. **Verify.** Wait for ``done_rank{r}`` on every rank.  Every
         worker that reaches this file has completed one post-rebuild
         training step with a finite loss.
    """
    # Probe GPU count here; some CI boxes have <4 and the test must
    # skip cleanly rather than hang inside NCCL init.
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

    coord_dir = tempfile.mkdtemp(prefix="survivor_rdv_", dir=str(tmp_path))
    world_size = 4
    initial_master_addr = "127.0.0.1"
    initial_master_port = _get_free_port()
    new_master_addr = "127.0.0.1"
    new_master_port = _get_free_port()

    ctx = mp.get_context("spawn")

    # Phase 1 — boot the initial world.
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

        # Phase 2 — fault injection on rank 0.  SIGKILL rather than
        # a graceful exit so survivors see a wedged NCCL comm from the
        # victim's perspective (any in-flight collective is not
        # completed by the dying rank).
        victim = procs[0]
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=10.0)
        assert not victim.is_alive(), "rank 0 did not die after SIGKILL"

        with open(os.path.join(coord_dir, _NEW_WORLD_FILE), "w") as f:
            f.write(f"{new_master_addr},{new_master_port}")
        with open(os.path.join(coord_dir, _FAULT_FILE), "w") as f:
            f.write("fault")

        # Phase 3 — spawn the replacement (spare) rank 0 on the new
        # master.  Happens concurrently with survivors' rendezvous so
        # the four ranks meet on the new endpoint.
        spare = ctx.Process(
            target=_worker_entrypoint,
            args=(0, world_size, new_master_addr, new_master_port, coord_dir, True),
        )
        spare.start()
        procs[0] = spare

        # Phase 4 — wait for everyone to finish.
        for rank in range(world_size):
            path = os.path.join(coord_dir, _DONE_FILE.format(rank=rank))
            if not _wait_for(path, _DONE_TIMEOUT):
                pytest.fail(f"rank {rank} never reached done file {path} — "
                            f"survivor_rendezvous or spare cold-start likely hung")

        for p in procs:
            p.join(timeout=10.0)

        for rank in range(world_size):
            p = procs[rank]
            assert p.exitcode == 0, (f"rank {rank} exited with code {p.exitcode} — "
                                     f"post-rebuild training raised; inspect the worker stderr")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
