# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Multi-host emulated-fault MoEvement recovery test.

Invokes the deepspeed launcher to spawn 4 PP=2 DP=2 workers across
two hosts (the local machine + ``10.3.0.11``).  Rank 0 simulates
failure in place via ``simulate_rank_failure`` at a configured fault
iter; the donor (rank 1) serves its persisted shard; the rest of the
recovery flow runs unchanged.

The "emulated fault" model trades launcher fault-detection coverage
(already exercised by the single-host SIGKILL test
``test_moevement_recovery_under_sigkill.py``) for cross-host wire
exercise — peer-pull traverses the real gloo TCP network rather than
loopback, which is what the SD-O4 streaming-recovery
``max(pull, replay)`` invariant claims about.

Skips automatically when ``10.3.0.11`` isn't reachable.
"""

import os
import socket
import subprocess
import sys

import pytest

_HOST_B = "10.3.0.11"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_HOSTFILE = os.path.join(_REPO_ROOT, "examples", "moevement", "hostfile")
_WORKER = os.path.join(os.path.dirname(__file__), "_multihost_worker.py")


def _can_reach_host_b(timeout=2.0):
    """Quick TCP probe of host B's SSH port."""
    try:
        with socket.create_connection((_HOST_B, 22), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _run_emulated_fault_scenario(tmp_path, streaming_recovery):
    if not os.path.exists(_HOSTFILE):
        pytest.skip(f"hostfile {_HOSTFILE} not found")
    if not _can_reach_host_b():
        pytest.skip(f"multi-host test requires {_HOST_B} reachable")

    # The deepspeed launcher honors a hostfile only when present at the
    # given path on the LAUNCHER host; the per-host workers are launched
    # via pdsh / ssh and inherit MASTER_ADDR/MASTER_PORT from the
    # launcher's own elections.  Use the conda env's binary so this
    # works under pytest invocation (which may not have ``deepspeed``
    # ahead of the system shim on PATH).
    deepspeed_bin = os.path.join(os.path.dirname(sys.executable), "deepspeed")
    if not os.path.exists(deepspeed_bin):
        pytest.skip(f"deepspeed launcher binary not found at {deepspeed_bin}")

    cmd = [
        deepspeed_bin,
        "--hostfile",
        _HOSTFILE,
        "--num_gpus",
        "2",
        "--num_nodes",
        "2",
        "--master_addr",
        "10.3.0.10",
        _WORKER,
    ]
    if streaming_recovery:
        cmd.append("--streaming-recovery")

    env = os.environ.copy()
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    # The launcher SSH's to host B and runs the worker; both hosts
    # need the conda env's binaries on PATH.  Setting it via the
    # launcher's environment-passing path (``.deepspeed_env`` on each
    # host) is the documented mechanism, but for the test we rely on
    # the user's per-host setup having that file in place.

    log_path = tmp_path / "multihost.log"
    with open(log_path, "w") as f:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=420,
        )

    log_content = log_path.read_text()
    if result.returncode != 0:
        # Trim to the last ~6KB so the failure reason is visible
        # without flooding pytest's output for a multi-host setup
        # that may emit lots of init noise.
        tail = log_content[-6000:] if len(log_content) > 6000 else log_content
        pytest.fail(
            f"deepspeed multi-host run exited with code {result.returncode}\n\n"
            f"--- last 6KB of launcher log ---\n{tail}",
            pytrace=False,
        )

    # Verify all 4 ranks reported PASS — defends against a worker
    # silently exiting 0 without reaching the assertion block.
    expected = {f"[rank {r}] PASS" for r in range(4)}
    missing = {marker for marker in expected if marker not in log_content}
    if missing:
        tail = log_content[-6000:] if len(log_content) > 6000 else log_content
        pytest.fail(
            f"deepspeed multi-host run returned 0 but missing PASS markers: {missing}\n\n"
            f"--- last 6KB of launcher log ---\n{tail}",
            pytrace=False,
        )


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="multi-host test requires GPUs on both hosts")
def test_moevement_emulated_fault_multihost_bulk(tmp_path):
    """4-rank PP=2 DP=2 across 2 hosts, emulated fault on rank 0, bulk peer-pull."""
    _run_emulated_fault_scenario(tmp_path, streaming_recovery=False)


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="multi-host test requires GPUs on both hosts")
def test_moevement_emulated_fault_multihost_streaming(tmp_path):
    """Same as above with SD-O4 streaming peer-pull protocol enabled.

    This is the test that actually exercises the streaming-recovery
    ``max(pull, replay)`` invariant under cross-host wire latency —
    single-host loopback has ~zero wire time, so the streaming path's
    benefit is invisible there.  Multi-host is where the design's
    projected savings (4-37% on gloo TCP) become measurable.
    """
    _run_emulated_fault_scenario(tmp_path, streaming_recovery=True)
