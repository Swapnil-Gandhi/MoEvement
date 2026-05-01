# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fault-injection helpers for MoEvement tests.

Simulating a rank failure in-process (without actually killing the
process) gives us a testable surface for the recovery path.  The
alternative — multiprocess-kill + external respawn — needs a cluster
manager and would have to live outside ``DistributedTest``'s pool
abstraction.

The helper leaves topology, process groups, and the coordinator's
scheduler config intact so the surrounding test harness can keep
calling collectives in lockstep with other ranks.  What we drop is
strictly the local *state* a fresh replacement process wouldn't have:
persisted snapshots, in-flight buffers, received-from-peer caches,
and the coordinator's upstream-log buffers on the failed rank.
"""

__all__ = ["simulate_rank_failure"]


def simulate_rank_failure(coord, model=None, zero_model_weights=False):
    """Simulate this rank's local MoEvement state being lost to a crash.

    Intended to be called on the "failing" rank only, in the middle of
    a distributed test.  Non-failing ranks should continue their normal
    train loop; they'll observe the failed rank re-join at the next
    world-handshake collective in ``recovery_barrier`` and ship replay
    logs via ``_pp_log_transfer``.

    Args:
        coord: A ``MoEvementCoordinator`` bound to a live engine.
        model: The engine's top-level module.  Only required when
            ``zero_model_weights=True``.
        zero_model_weights: When ``True``, writes zeros over every
            parameter's ``.data``.  Mimics the "fresh replacement rank
            with no trained weights" scenario so recovery has
            something meaningful to restore; leave it ``False`` for
            tests that only want to exercise log transfer and window
            replay.

    Leaves intact:
        * ``coord._dp_group`` / ``coord._pp_group`` (process groups
          stay alive — we're not killing the process).
        * ``coord._stage_id`` / ``coord._num_stages`` / topology.
        * ``coord.scheduler`` config (the replacement rank would load
          the same config from the DeepSpeed JSON).

    Flips:
        * ``coord._recovering = True`` so the next iteration's
          ``recovery_barrier`` sees this rank as recovering.
    """
    engine = coord.snapshot_engine
    engine._persisted_snapshots.clear()
    engine._in_flight_snapshots.clear()
    engine._snapshots.clear()
    # ``_pending_gpu_staging`` holds device-side scratch buffers waiting
    # for D2H to drain; a crashed process would lose them, and keeping
    # them around could let a subsequent ``synchronize`` try to release
    # buffers that no longer correspond to any live snapshot.
    engine._pending_gpu_staging = []
    engine._received_snapshots.clear()
    engine._received_metadata.clear()
    engine._received_window_start.clear()
    if engine._received_flat_buffers:
        for flat in engine._received_flat_buffers:
            engine._pool.release(flat)
        engine._received_flat_buffers = []

    if coord.upstream_logger is not None:
        coord.upstream_logger._logs.clear()
        coord.upstream_logger._received_logs.clear()

    # Converter carries FROZEN/ACTIVE operator state from a prior
    # in-progress recovery; a fresh rank wouldn't have that either.
    if hasattr(coord, "converter") and coord.converter is not None:
        coord.converter.clear()

    if zero_model_weights:
        if model is None:
            raise ValueError("zero_model_weights=True requires model=")
        for p in model.parameters():
            p.data.zero_()

    coord._recovering = True
