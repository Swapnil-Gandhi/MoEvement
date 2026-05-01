# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Smoke tests for the MoEvement user-facing examples.

These tests invoke ``examples/moevement/*.py`` end-to-end via the
``deepspeed`` launcher on a real 4-GPU box.  The unit tests in
``test_sparse_checkpoint.py`` and friends pin product correctness;
these smoke tests catch "someone renamed a product API and didn't
update the example" — a failure class the unit tests can't see.

Each test is a ``subprocess.run`` that asserts the example exits 0
and greps stdout for one expected marker.  Runtime is bounded so the
suite stays fast (deepspeed init dominates).

Skipped when fewer than 4 GPUs are visible (e.g., CI without a
multi-GPU runner — known coverage gap on single-GPU boxes).
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

_EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/moevement"))


def _gpu_count():
    """Visible GPU count, 0 if torch / CUDA isn't available on this box."""
    try:
        import torch
        if not torch.cuda.is_available():  #ignore-cuda
            return 0
        return torch.cuda.device_count()  #ignore-cuda
    except Exception:
        return 0


requires_4_gpus = pytest.mark.skipif(_gpu_count() < 4, reason="MoEvement examples require 4 GPUs (PP=2 DP=2 topology)")


def _run_example(script, args, cwd=_EXAMPLES_DIR, timeout_sec=240):
    """Launch ``script`` via the ``deepspeed`` CLI with 4 GPUs; return (returncode, stdout+stderr)."""
    env = os.environ.copy()
    # If the conda env doesn't have nvcc, the example dodges FusedAdam
    # via ``torch_adam=True`` but the MoEvement engine's import-time
    # op-registration still probes CUDA availability.  Set this env var
    # in CI / dev-box environments without nvcc.
    env.setdefault("DS_IGNORE_CUDA_DETECTION", "1")
    cmd = [shutil.which("deepspeed") or "deepspeed", "--num_gpus=4", script, *args]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_sec)
    return proc.returncode, proc.stdout + proc.stderr


def _run_supervisor_example(script, args, cwd=_EXAMPLES_DIR, timeout_sec=360):
    """Launch the supervisor example via plain python (its own launcher).

    The ``run_with_survivor_supervisor.py`` example is NOT compatible with
    the ``deepspeed`` CLI because ``SurvivorSupervisor`` IS the launcher
    — it spawns its own worker subprocesses and owns the rendezvous
    side-channel.  Running it under the deepspeed CLI would create two
    competing launchers and the worker env would be ambiguous.
    """
    env = os.environ.copy()
    env.setdefault("DS_IGNORE_CUDA_DETECTION", "1")
    cmd = [sys.executable, script, *args]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_sec)
    return proc.returncode, proc.stdout + proc.stderr


@requires_4_gpus
def test_resume_after_fault_smoke():
    """``resume_after_fault.py`` picks up from a saved ckpt, replays, continues.

    Two-step: first produce a checkpoint via ``train_cifar_moe.py``
    (the canonical training example, with ``--fake-data`` so the smoke
    test doesn't hit network), then resume from it.  The expected marker
    is the recovery-active banner on rank 0 — without it, either
    ``load_checkpoint`` silently skipped MoEvement's bundle or the
    bundle was absent.

    Runs in a tempdir so parallel smoke-test runs don't stomp on each
    other's checkpoint state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "ckpt")

        # Step 1: produce a checkpoint.  20 iters is the floor that puts
        # a real window into ``_persisted_snapshots``: the example's
        # pcie_bandwidth_gbs=1e-6 forces ``w_sparse > 1``, and the
        # persisted window lags one ``finalize_window`` cycle behind the
        # live snapshots (window 0 doesn't reach ``_persisted`` until a
        # later iter-boundary fires).  At 10 iters the bundle is empty;
        # the resume path runs but has nothing to replay, which defeats
        # the smoke-test's job of pinning the replay contract.
        rc, out = _run_example(
            "train_cifar_moe.py",
            [
                "--fake-data",
                "--save-dir",
                save_dir,
                "--num-iters",
                "20",
                "--save-every",
                "20",
                # MoEvement bundle is opt-in (default off — peer-pull is the
                # primary recovery target); turn it on so resume has a bundle
                # to replay from.
                "--persist-to-disk",
            ],
        )
        assert rc == 0, f"training step failed rc={rc}; tail:\n{out[-1500:]}"
        assert os.path.isdir(os.path.join(save_dir, "step_20"))

        # Step 2: resume from it.
        rc, out = _run_example(
            "resume_after_fault.py",
            ["--load-dir", save_dir, "--num-iters", "15"],
        )
        assert rc == 0, f"resume step failed rc={rc}; tail:\n{out[-1500:]}"
        assert "MoEvement recovery active" in out, ("expected recovery-active banner — the bundle may not have "
                                                    f"loaded; tail:\n{out[-1500:]}")
        assert "[replay]" in out, f"expected at least one [replay] log line; tail:\n{out[-1500:]}"
        assert "[rank 0] resume complete" in out, f"resume did not finish cleanly; tail:\n{out[-1500:]}"


@requires_4_gpus
def test_train_cifar_moe_smoke():
    """``train_cifar_moe.py`` runs 5 iters with ``--fake-data``, exits 0, prints CV banner.

    ``--fake-data`` skips the torchvision CIFAR-10 download (the smoke
    test box may be offline) and substitutes synthetic ``(B, 3, 32, 32)``
    fp16 tensors that match the real shape.  The CV-shaped model and
    pipeline behaviour are exercised identically; what's deferred is
    the ``torchvision.datasets.CIFAR10`` codepath, which is upstream
    code we don't own.

    Five iters is the floor that proves the conv backbone + 2× MoE
    blocks + classifier all run cleanly through the pipeline (rank-0
    forward, rank-1 forward, backward, optim) without a stage-mismatch
    or shape error.
    """
    rc, out = _run_example(
        "train_cifar_moe.py",
        ["--fake-data", "--num-iters", "5"],
    )
    assert rc == 0, f"train_cifar_moe.py exited {rc}; output tail:\n{out[-2000:]}"
    assert "data=synthetic" in out, ("expected synthetic-data banner on rank 0; "
                                     f"output tail:\n{out[-2000:]}")
    assert "[rank 0] cifar-moe training complete" in out, (f"cifar-moe training did not finish; "
                                                           f"output tail:\n{out[-2000:]}")


@requires_4_gpus
def test_train_gpt_moe_smoke():
    """``train_gpt_moe.py`` runs 5 iters with ``--fake-data``, exits 0, prints GPT banner.

    Mirrors ``test_train_cifar_moe_smoke`` for the GPT-MoE example.
    ``--fake-data`` skips the TinyShakespeare download and uses
    synthetic random ``(B, S)`` int64 tokens of the same shape, so the
    smoke test runs offline.  Five iters proves the embedding + 4
    transformer blocks (each LN -> attn -> LN -> MoE-FFN) + head all
    run cleanly through the pipeline (rank-0 forward, rank-1 forward,
    backward, optim) without a stage-mismatch or shape error.
    """
    rc, out = _run_example(
        "train_gpt_moe.py",
        ["--fake-data", "--num-iters", "5"],
    )
    assert rc == 0, f"train_gpt_moe.py exited {rc}; output tail:\n{out[-2000:]}"
    assert "model=gpt-moe" in out, ("expected gpt-moe model banner on rank 0; "
                                    f"output tail:\n{out[-2000:]}")
    assert "[rank 0] gpt-moe training complete" in out, (f"gpt-moe training did not finish; "
                                                         f"output tail:\n{out[-2000:]}")


@requires_4_gpus
def test_run_with_survivor_supervisor_smoke():
    """SurvivorSupervisor demo: SIGKILL → peer-pull → recovery → post-fault training.

    Exercises the advertised DP Cascade recovery mode end-to-end:

    - 4 workers spawned by ``SurvivorSupervisor`` (not the deepspeed CLI
      — the supervisor IS the launcher here).
    - Rank 0 self-SIGKILLs at a known iter; supervisor detects, spawns
      spare with ``DS_SURVIVOR_IS_SPARE=1``.
    - Survivors call ``survivor_rendezvous``; rank 1 serves peer-pull;
      spare peer-pulls.
    - Every rank drives the replay loop until recovery completes; then
      6 post-recovery training iters.
    - All 4 ranks call ``signal_done``; supervisor exits 0.

    Three expected markers pin the full sequence — omitting any of them
    means the supervisor / peer-pull / replay path broke silently:
    ``SELF-SIGKILL`` (fault fires), ``peer-pull complete`` (spare
    loaded state), ``recovery converged`` (replay drove both DP groups
    through the window).

    Runtime ~90-120s on a 4xA100 box — compile warmup + SIGKILL round-
    trip + replay of a 18-iter window.  Timeout bumped to 360s for CI.
    """
    rc, out = _run_supervisor_example(
        "run_with_survivor_supervisor.py",
        ["--fault-at-iter", "18", "--post-recovery-iters", "6"],
    )
    assert rc == 0, f"supervisor example exited {rc}; output tail:\n{out[-2500:]}"
    assert "SELF-SIGKILL" in out, f"victim did not self-kill; tail:\n{out[-2500:]}"
    assert "peer-pull complete" in out, f"spare did not peer-pull; tail:\n{out[-2500:]}"
    assert "recovery converged" in out, f"recovery did not converge; tail:\n{out[-2500:]}"
    assert "[rank 0 (spare)] post-recovery step=1" in out, (f"spare did not run post-recovery "
                                                            f"training iters; tail:\n{out[-2500:]}")
