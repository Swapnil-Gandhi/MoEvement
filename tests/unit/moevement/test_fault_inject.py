# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for the ``simulate_rank_failure`` helper.

Single-rank tests — the helper operates on a coordinator's in-memory
state, not on distributed collectives, so the harness doesn't need a
pool.  A future recovery E2E test will compose this helper with a real
``DistributedTest`` engine.
"""

import torch
import torch.nn as nn

from deepspeed.moevement.config import MoEvementConfig
from deepspeed.moevement.coordinator import MoEvementCoordinator
from deepspeed.moevement.sparse_snapshot import OperatorSnapshot
from deepspeed.moevement.upstream_logging import LogEntry

from unit.moevement._fault_inject import simulate_rank_failure


def _seeded_coord():
    """Build a coordinator with non-empty local state to exercise the helper.

    The state here mimics what a running rank accumulates across a
    training window: a couple of persisted snapshots, a received-from-peer
    cache entry, a logged activation, and a replication flat buffer.
    The helper's job is to drop all of these; each test asserts on a
    different slice of the drop.
    """
    coord = MoEvementCoordinator(MoEvementConfig())

    # Persisted snapshots (prior window's state, the thing peer
    # replication also holds mirrors of).
    snap_a = OperatorSnapshot("op_a", iteration=1, is_active=True)
    snap_a.add_tensor("params.weight", torch.ones(4))
    coord.snapshot_engine._persisted_snapshots[(1, "op_a")] = snap_a
    snap_b = OperatorSnapshot("op_b", iteration=1, is_active=False)
    snap_b.add_tensor("compute_weights.weight", torch.ones(4))
    coord.snapshot_engine._persisted_snapshots[(1, "op_b")] = snap_b

    # Received-from-peer state: in-memory mirror of a DP peer's shard.
    coord.snapshot_engine._received_snapshots[1] = {(1, "op_a"): {"params.weight": torch.zeros(4)}}
    coord.snapshot_engine._received_metadata[1] = {(1, "op_a"): {"is_active": True}}
    coord.snapshot_engine._received_window_start[1] = 1

    # Upstream log buffers (shipped at stage boundaries, kept for replay).
    coord.upstream_logger._logs[(0, 0)].append(
        LogEntry(iteration=0, micro_batch_id=0, stage_id=0, direction="activation", tensor=torch.ones(4)))
    coord.upstream_logger._received_logs[(0, 0)].append(
        LogEntry(iteration=0, micro_batch_id=0, stage_id=0, direction="activation", tensor=torch.ones(4)))
    return coord


class TestSimulateRankFailure:

    def test_clears_persisted_snapshots_and_peer_mirror(self):
        """Persisted + received-peer state both get dropped.

        A crashed rank loses both its own persisted snapshots and its
        cached mirrors of peer shards — the new process starts cold on
        both axes.  Flipping only one would leave inconsistent state:
        peer-pull would race the half-cleared mirror.
        """
        coord = _seeded_coord()
        simulate_rank_failure(coord)

        assert coord.snapshot_engine._persisted_snapshots == {}
        assert coord.snapshot_engine._received_snapshots == {}
        assert coord.snapshot_engine._received_metadata == {}
        assert coord.snapshot_engine._received_window_start == {}

    def test_clears_upstream_logs(self):
        """Both ``_logs`` (outgoing) and ``_received_logs`` (incoming) drop.

        Upstream logs are the replay fuel for cascade recovery.  A fresh
        replacement rank has neither the logs it was accumulating nor
        any it had received from neighbours; dropping both keeps the
        replay machinery from replaying stale state.
        """
        coord = _seeded_coord()
        assert coord.upstream_logger._logs  # sanity
        assert coord.upstream_logger._received_logs

        simulate_rank_failure(coord)

        assert len(coord.upstream_logger._logs) == 0
        assert len(coord.upstream_logger._received_logs) == 0

    def test_flips_recovering_flag(self):
        """``_recovering`` is the gate that ``recovery_barrier`` consults.

        Without the flag flip, the rank would re-enter training on
        the normal (non-recovery) schedule even though its state is
        gone.
        """
        coord = _seeded_coord()
        assert coord._recovering is False

        simulate_rank_failure(coord)

        assert coord._recovering is True

    def test_preserves_topology_and_scheduler(self):
        """Process-group and scheduler config outlive the simulated failure.

        In real life a replacement rank rejoins the same distributed
        job — process groups remain live on the surviving peers.  Our
        in-process simulation has to match that: tests keep calling
        collectives on the same groups, so the helper must NOT drop
        them.  Scheduler config is likewise reloaded from the same
        DeepSpeed JSON by the replacement rank, so we leave it alone.
        """
        coord = _seeded_coord()
        coord._dp_group = "sentinel-dp-group"
        coord._pp_group = "sentinel-pp-group"
        coord._stage_id = 3
        coord._num_stages = 4
        coord.scheduler.w_sparse = 7

        simulate_rank_failure(coord)

        assert coord._dp_group == "sentinel-dp-group"
        assert coord._pp_group == "sentinel-pp-group"
        assert coord._stage_id == 3
        assert coord._num_stages == 4
        assert coord.scheduler.w_sparse == 7

    def test_zero_model_weights_writes_zeros(self):
        """Opt-in zeroing lets a test assert that recovery *restored* weights.

        Without this, the replacement rank's weights are whatever they
        were before the simulated failure (still the trained state), so
        any subsequent assertion that "recovery put weights back" is
        vacuous.  With the flag on, the rank starts from zeros and the
        restoration is observable.
        """
        coord = _seeded_coord()
        model = nn.Linear(4, 4)
        with torch.no_grad():
            model.weight.fill_(3.14)
            model.bias.fill_(2.71)

        simulate_rank_failure(coord, model=model, zero_model_weights=True)

        assert torch.all(model.weight.data == 0)
        assert torch.all(model.bias.data == 0)

    def test_zero_model_weights_requires_model(self):
        """Raising is better than silently no-op'ing a misconfigured test."""
        coord = _seeded_coord()
        import pytest
        with pytest.raises(ValueError, match="zero_model_weights=True requires model="):
            simulate_rank_failure(coord, zero_model_weights=True)
