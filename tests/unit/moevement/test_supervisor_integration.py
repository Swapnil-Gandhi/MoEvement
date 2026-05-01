# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end test for :class:`SurvivorSupervisor`.

Exercises the full supervisor-driven fault-recovery loop with no
test-side orchestration: the supervisor itself spawns workers, detects
the victim's death, writes the new-master config to its TCPStore, and
spawns the replacement with ``DS_SURVIVOR_IS_SPARE=1``.  The worker
entrypoint is a tiny Python program written to a tempfile, using
``WorkerProbe`` as a real user would.

This is the missing piece between ``test_survivor_rendezvous.py`` (which
plays agent itself via ``mp.spawn`` + filesystem rendezvous) and
production — with this wired up, a real launcher script is one
``subprocess.Popen(... SurvivorSupervisor ...)`` away.

Scope matches the supervisor's v1 scope: 1 fault, 1:1 substitution,
world_size unchanged.  Uses the plain ``nn.Linear`` model (same shape
as the DeepSpeedEngine canary) — the PP-topology variant of this
integration test is a followup.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_WORKER_SOURCE = textwrap.dedent('''
    import os
    import signal
    import sys
    import time

    import torch
    import torch.nn as nn

    import deepspeed
    import deepspeed.comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.launcher.survivor_supervisor import WorkerProbe

    # Victim: rank 0 self-kills after one pre-fault step.  Keeps the
    # test hermetic — the supervisor alone is responsible for detecting
    # the death and spawning a spare.
    _VICTIM_RANK = 0

    def main():
        probe = WorkerProbe()
        rank = probe.rank
        world_size = probe.world_size

        deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
        get_accelerator().set_device(rank % get_accelerator().device_count())
        device = get_accelerator().device_name()

        model = nn.Linear(8, 8).to(device)
        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 1e-3, "torch_adam": True}},
            "zero_optimization": {"stage": 1},
            "fp16": {"enabled": False},
        }
        engine, *_ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
            skip_initial_broadcast=probe.is_spare(),
        )

        # Pre-fault step — skipped on the spare, which cold-starts into
        # a post-fault world.  Survivors need this to materialize
        # optimizer state + warm the NCCL comm on the old master.
        if not probe.is_spare():
            x = torch.ones(1, 8, device=device)
            y = torch.zeros(1, 8, device=device)
            loss = ((engine(x) - y) ** 2).sum()
            engine.backward(loss)
            engine.step()

        # Victim: self-destruct here.  Supervisor's Popen.poll() observes
        # the non-zero exit (SIGKILL) and kicks off the rendezvous flow.
        if rank == _VICTIM_RANK and not probe.is_spare():
            os.kill(os.getpid(), signal.SIGKILL)

        # Survivors + spare: poll for the fault signal with a bounded
        # wait.  The spare sees no fault (it's the replacement, not a
        # survivor), so its probe.check_for_fault returns None and it
        # proceeds directly to the post-rebuild collective.
        deadline = time.time() + 60.0
        if not probe.is_spare():
            while time.time() < deadline:
                fault = probe.check_for_fault()
                if fault is not None:
                    new_addr, new_port, ws = fault
                    get_accelerator().synchronize()
                    engine.survivor_rendezvous(
                        new_master_addr=new_addr,
                        new_master_port=new_port,
                        new_rank=rank,
                        new_world_size=ws,
                    )
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"rank {rank} never observed fault signal")

        # Post-rebuild probe: same shape as the survivor canaries.
        probe_tensor = torch.ones(1, device=device) * (rank + 1)
        dist.all_reduce(probe_tensor)
        expected = sum(r + 1 for r in range(world_size))
        assert probe_tensor.item() == expected, (
            f"rank {rank} got {probe_tensor.item()}, expected {expected}"
        )

        probe.signal_done()

    if __name__ == "__main__":
        try:
            main()
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
''')


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    reason="requires at least 4 GPUs for 3 survivors + 1 spare")
def test_supervisor_drives_end_to_end_recovery(tmp_path):
    """Supervisor spawns workers, rank 0 self-kills, spare joins, all reach done."""
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

    # Persist the worker source as an on-disk script so the supervisor
    # can Popen it via ``python <path>``.  Living in tmp_path keeps the
    # test self-cleaning.
    worker_path = tmp_path / "supervisor_worker.py"
    worker_path.write_text(_WORKER_SOURCE)

    # Forward the test harness's Python env vars to the worker so
    # ``DS_IGNORE_CUDA_DETECTION`` and friends propagate.  The supervisor
    # inherits ``os.environ`` into each worker via its ``_spawn_worker``.
    # We don't need to touch anything here beyond pointing at the right
    # interpreter.
    worker_cmd = [sys.executable, str(worker_path)]

    # Import the supervisor module inside the test, not at module scope
    # — avoids touching torch / deepspeed during test collection on
    # machines without CUDA.
    from deepspeed.launcher.survivor_supervisor import SurvivorSupervisor

    sup = SurvivorSupervisor(worker_cmd=worker_cmd, world_size=4)
    exit_code = sup.run(timeout_sec=180.0)
    assert exit_code == 0, f"supervisor returned {exit_code} (0=success, 1=timeout, 2=second-fault)"
