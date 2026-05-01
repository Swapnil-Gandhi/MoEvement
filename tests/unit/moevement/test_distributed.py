# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Multi-rank integration tests for MoEvement collectives.

These tests exercise the real ``deepspeed.comm`` backend (gloo on CPU,
NCCL on GPU) rather than mocking it, so they catch framing and ordering
bugs that unit tests with patched ``dist`` cannot surface.
"""

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from deepspeed.moevement.config import MoEvementConfig
from deepspeed.moevement.coordinator import MoEvementCoordinator
from deepspeed.moevement.sparse_snapshot import (
    PEER_PULL_PROTOCOL_BULK,
    PEER_PULL_PROTOCOL_STREAMING,
    OperatorSnapshot,
    SparseSnapshotEngine,
)
from deepspeed.moevement.upstream_logging import LogEntry

from unit.common import DistributedTest


def _make_gloo_world_group():
    """Build a gloo subgroup spanning the whole world.

    Mirrors production: training collectives keep the default (NCCL on
    CUDA) backend, while MoEvement replication / log transfer rides a
    dedicated gloo subgroup so CPU snapshot tensors can be shipped
    without GPU transit and without racing training comm.
    """
    return dist.new_group(ranks=list(range(dist.get_world_size())), backend='gloo')


def _current_device():
    """Per-rank accelerator device (cuda:k on a CUDA box, else cpu).

    Used to pin the recovery handshake's announcement tensor to the
    same device the production coordinator uses, so the world
    ``all_gather`` runs on the default training group natively.
    """
    if get_accelerator().is_available() and get_accelerator().device_name() == 'cuda':
        return torch.device(get_accelerator().current_device_name())
    return torch.device('cpu')


class TestPeerReplicationDistributed(DistributedTest):
    """Real point-to-point round-trip on the symmetric replication ring."""

    world_size = 2

    def test_replicate_to_peers_round_trip(self):
        """Every rank sends its own shard forward; recv side is keyed by sender.

        With world_size=2 and r=1 the ring is rank 0 ↔ rank 1: each one
        both sends to and receives from the other.  Only rank 0 has a
        populated snapshot here, so rank 1 ends up with rank 0's data
        under ``_received_snapshots[0]``, and rank 0 sees an empty slot
        under ``_received_snapshots[1]``.
        """
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        if rank == 0:
            snap = OperatorSnapshot("op0", iteration=5, is_active=True)
            snap.add_tensor("params.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3))
            snap.add_tensor("optimizer.exp_avg", torch.zeros(2, 3, dtype=torch.float32))
            engine._persisted_snapshots[(5, "op0")] = snap
            engine._window_start_iteration = 5

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=torch.device("cpu"),
        )

        if rank == 1:
            # Rank 1 received rank 0's snapshot on the ring.
            metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=0)
            assert metadata is not None
            assert metadata["window_start_iteration"] == 5
            assert metadata["per_iter_active"] == {5: {"op0": True}}

            expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            torch.testing.assert_close(per_iter[5]["op0"]["params.weight"], expected)
            torch.testing.assert_close(per_iter[5]["op0"]["optimizer.exp_avg"], torch.zeros(2, 3))

    def test_replicate_to_peers_empty_is_no_op(self):
        """Empty senders emit length=0 on the ring and the receiver records an empty slot."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)
        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=torch.device("cpu"),
        )

        # Both ranks sent length=0; each one sees an empty-state entry
        # for its ring neighbour, and the accessor returns (None, None).
        other = 1 - rank
        assert engine._received_snapshots.get(other) == {}
        metadata, states = engine.get_received_snapshots_for(sender_dp_rank=other)
        assert metadata is None
        assert states is None


class TestProductionPathReplication(DistributedTest):
    """f1 round-trip via the REAL production pipeline.

    Existing replication tests construct snapshots via
    ``OperatorSnapshot.add_tensor(...)``, bypassing the
    ``snapshot_operator → _batched_d2h`` D2H code path that runs in
    production.  This class exercises the full path:
    ``snapshot_operator`` (real D2H) → ``record_pending_d2h_event``
    → ``finalize_window`` → second-window boundary → ``replicate_to_peers``
    (real clone + ``pinned_release_fn`` + gloo isend/recv).

    Catches regressions the bypass tests miss:
    - clone phase corrupting bytes
    - ``pinned_release_fn`` releasing a flat that's still being read
    - busy-flat lifecycle race against ``finalize_window``
    - layout-pack offset bugs in ``_batched_d2h``
    - FP32→FP16 promotion drift on the frozen path
    """

    world_size = 2

    def test_round_trip_via_real_d2h_frozen_cpu_input(self):
        """Frozen-op f1 round-trip with CPU input (CPU-fallback D2H path)."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        # Unique values per rank — receiver verifies content matches sender's source.
        # FP32 source; frozen path will cast to FP16 inside _batched_d2h.
        src_weight = torch.arange(8, dtype=torch.float32) + (rank * 100.0)
        op_name = f"op_from_rank_{rank}"

        # Window 0: snapshot + boundary (deferred sync).
        engine.snapshot_operator(op_name, {"weight": src_weight}, None, is_active=False, iteration=0)
        engine.synchronize()
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=1)
        engine.wait_for_pending_d2h_event()

        # Window 1: empty boundary — promotes window 0's snapshots from
        # _in_flight to _persisted (production semantics).
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=2)
        engine.wait_for_pending_d2h_event()

        # Replicate.  Goes through clone + pinned_release_fn + gloo.
        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=_current_device(),
        )

        # Verify peer received bytes matching SENDER's original source
        # (full round-trip: sender FP32 → D2H pinned FP16 → clone → gloo
        # → receiver FP16 → assert_close back at FP32).
        sender_rank = (rank - 1) % 2
        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=sender_rank)
        assert metadata is not None, f"rank {rank} got no snapshot from sender {sender_rank}"
        sender_src = torch.arange(8, dtype=torch.float32) + (sender_rank * 100.0)
        received_fp16 = per_iter[0][f"op_from_rank_{sender_rank}"]["compute_weights.weight"]
        assert received_fp16.dtype == torch.float16
        torch.testing.assert_close(received_fp16.float(), sender_src, atol=1e-2, rtol=1e-3)

    def test_round_trip_via_real_d2h_active_cpu_input(self):
        """Active-op f1 round-trip with CPU input — FP32 params + optim state."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        src_weight = torch.arange(8, dtype=torch.float32) + (rank * 100.0)
        src_exp_avg = torch.arange(8, 16, dtype=torch.float32) + (rank * 1000.0)
        op_name = f"op_from_rank_{rank}"
        params = {"weight": src_weight}
        optim_state = {"exp_avg": src_exp_avg}

        engine.snapshot_operator(op_name, params, optim_state, is_active=True, iteration=0)
        engine.synchronize()
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=1)
        engine.wait_for_pending_d2h_event()
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=2)
        engine.wait_for_pending_d2h_event()

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=_current_device(),
        )

        sender_rank = (rank - 1) % 2
        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=sender_rank)
        assert metadata is not None
        sender_src_weight = torch.arange(8, dtype=torch.float32) + (sender_rank * 100.0)
        sender_src_exp_avg = torch.arange(8, 16, dtype=torch.float32) + (sender_rank * 1000.0)
        recv_states = per_iter[0][f"op_from_rank_{sender_rank}"]
        # Active path stores at FP32 — full byte-equality.
        torch.testing.assert_close(recv_states["params.weight"], sender_src_weight)
        torch.testing.assert_close(recv_states["optimizer.exp_avg"], sender_src_exp_avg)

    def test_round_trip_via_real_d2h_gpu_input(self):
        """f1 round-trip with GPU input — exercises production async D2H path."""
        if not torch.cuda.is_available():  #ignore-cuda
            import pytest
            pytest.skip("requires CUDA for GPU async D2H path")

        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        device = _current_device()
        src_weight = (torch.arange(8, dtype=torch.float32) + (rank * 100.0)).to(device)
        op_name = f"op_from_rank_{rank}"

        engine.snapshot_operator(op_name, {"weight": src_weight}, None, is_active=False, iteration=0)
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=1)
        engine.wait_for_pending_d2h_event()
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=2)
        engine.wait_for_pending_d2h_event()
        # Force main-stream drain so the deferred-sync path's pinned
        # bytes are CPU-readable before replicate_to_peers reads them
        # via the worker (test runs replicate inline, not on a worker).
        torch.cuda.synchronize()  #ignore-cuda

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=device,
        )

        sender_rank = (rank - 1) % 2
        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=sender_rank)
        assert metadata is not None
        sender_src = torch.arange(8, dtype=torch.float32) + (sender_rank * 100.0)
        received_fp16 = per_iter[0][f"op_from_rank_{sender_rank}"]["compute_weights.weight"]
        assert received_fp16.dtype == torch.float16
        torch.testing.assert_close(received_fp16.float(), sender_src, atol=1e-2, rtol=1e-3)


class TestRingReplicationWorldSize4(DistributedTest):
    """Happy-path tests for the symmetric replication ring at world_size=4.

    Each rank seeds a uniquely-valued snapshot so a receiver can tell
    from content alone which peer sent what.  Runs the real gloo
    backend via ``DistributedTest`` — catches rank-pairing / framing
    bugs that the single-process mocked unit tests can't surface.
    """

    world_size = 4

    def test_ring_r1_every_rank_receives_its_predecessor(self):
        """With r=1, rank k receives exactly rank (k-1) mod 4's snapshot."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        op_name = f"op_from_rank_{rank}"
        snap = OperatorSnapshot(op_name, iteration=1, is_active=True)
        snap.add_tensor("params.weight", torch.full((4, ), float(rank * 10)))
        engine._persisted_snapshots[(1, op_name)] = snap
        engine._window_start_iteration = 1

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=4,
            device=torch.device("cpu"),
        )

        # Every rank sees exactly one sender — its ring predecessor.
        sender_rank = (rank - 1) % 4
        assert engine.received_senders() == [sender_rank]

        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=sender_rank)
        assert metadata is not None
        assert metadata["window_start_iteration"] == 1
        assert metadata["per_iter_active"] == {1: {f"op_from_rank_{sender_rank}": True}}

        expected = torch.full((4, ), float(sender_rank * 10))
        torch.testing.assert_close(per_iter[1][f"op_from_rank_{sender_rank}"]["params.weight"], expected)

    def test_ring_r2_each_rank_receives_from_two_predecessors(self):
        """With r=2, rank k has both (k-1) mod 4 AND (k-2) mod 4's snapshots."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=2)

        op_name = f"op_from_rank_{rank}"
        snap = OperatorSnapshot(op_name, iteration=2, is_active=True)
        snap.add_tensor("params.weight", torch.full((4, ), float(rank * 10)))
        engine._persisted_snapshots[(2, op_name)] = snap
        engine._window_start_iteration = 2

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=4,
            device=torch.device("cpu"),
        )

        expected_senders = {(rank - 1) % 4, (rank - 2) % 4}
        assert set(engine.received_senders()) == expected_senders

        for sender_rank in expected_senders:
            metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=sender_rank)
            assert metadata is not None
            assert metadata["per_iter_active"] == {2: {f"op_from_rank_{sender_rank}": True}}
            expected_weight = torch.full((4, ), float(sender_rank * 10))
            torch.testing.assert_close(per_iter[2][f"op_from_rank_{sender_rank}"]["params.weight"], expected_weight)

    def test_ring_r_clamped_to_world_size_minus_one(self):
        """replication_factor=10 on world_size=4 clamps to r=3 (full ring coverage)."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=10)

        op_name = f"op_from_rank_{rank}"
        snap = OperatorSnapshot(op_name, iteration=3, is_active=True)
        snap.add_tensor("params.weight", torch.full((4, ), float(rank * 10)))
        engine._persisted_snapshots[(3, op_name)] = snap
        engine._window_start_iteration = 3

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=4,
            device=torch.device("cpu"),
        )

        # r = min(10, world_size-1) = 3 → receive from every other rank.
        expected_senders = set(range(4)) - {rank}
        assert set(engine.received_senders()) == expected_senders


class TestMultiWindowReplication(DistributedTest):
    """State hygiene across repeated ``replicate_to_peers`` calls.

    Verifies receiver-side state rotates correctly between windows —
    prior window's data is dropped, fresh window's data replaces it,
    and pool-managed flat buffers don't leak.
    """

    world_size = 2

    def test_second_window_overrides_first(self):
        """Repeated replications refresh the received state."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        # Window 1: rank 0 sends a snapshot with value 10.0.
        if rank == 0:
            snap1 = OperatorSnapshot("op0", iteration=1, is_active=True)
            snap1.add_tensor("params.weight", torch.full((4, ), 10.0))
            engine._persisted_snapshots[(1, "op0")] = snap1
            engine._window_start_iteration = 1

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=torch.device("cpu"),
        )

        if rank == 1:
            metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=0)
            assert metadata["window_start_iteration"] == 1
            torch.testing.assert_close(per_iter[1]["op0"]["params.weight"], torch.full((4, ), 10.0))

        # Window 2: rank 0's snapshot now has value 20.0 at iter 2.  The
        # receiver must drop window 1's data and reflect window 2's —
        # this is the per-window ``_received_snapshots.clear()`` behaviour.
        if rank == 0:
            engine._persisted_snapshots.clear()
            snap2 = OperatorSnapshot("op0", iteration=2, is_active=True)
            snap2.add_tensor("params.weight", torch.full((4, ), 20.0))
            engine._persisted_snapshots[(2, "op0")] = snap2
            engine._window_start_iteration = 2

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=torch.device("cpu"),
        )

        if rank == 1:
            metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=0)
            assert metadata["window_start_iteration"] == 2
            torch.testing.assert_close(per_iter[2]["op0"]["params.weight"], torch.full((4, ), 20.0))

    def test_empty_window_followed_by_populated(self):
        """Empty → populated transition doesn't leave residual state."""
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        # Window 1: both ranks empty — length=0 sentinels exchanged.
        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=torch.device("cpu"),
        )
        other = 1 - rank
        assert engine._received_snapshots.get(other) == {}

        # Window 2: rank 0 populates; rank 1 still empty.  Rank 1 should
        # now have rank 0's data, not the prior empty sentinel.
        if rank == 0:
            snap = OperatorSnapshot("op_w2", iteration=5, is_active=True)
            snap.add_tensor("params.weight", torch.ones(4))
            engine._persisted_snapshots[(5, "op_w2")] = snap
            engine._window_start_iteration = 5

        engine.replicate_to_peers(
            dp_group=gloo_group,
            dp_rank=rank,
            dp_world_size=2,
            device=torch.device("cpu"),
        )

        if rank == 1:
            metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=0)
            assert metadata is not None
            assert metadata["window_start_iteration"] == 5
            assert 5 in per_iter and "op_w2" in per_iter[5]
            torch.testing.assert_close(per_iter[5]["op_w2"]["params.weight"], torch.ones(4))

    def test_received_flat_buffers_do_not_accumulate(self):
        """Per-window pool-buffer lifecycle: old flats released, new flats allocated.

        Without per-window release of ``_received_flat_buffers``, each
        replication call would grow the list — a memory leak
        proportional to training duration.
        """
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        engine = SparseSnapshotEngine(replication_factor=1)

        if rank == 0:
            snap = OperatorSnapshot("op0", iteration=1, is_active=True)
            snap.add_tensor("params.weight", torch.full((4, ), 10.0))
            engine._persisted_snapshots[(1, "op0")] = snap
            engine._window_start_iteration = 1

        # Run 3 windows; receiver buffer count should stay constant.
        for window in range(3):
            if rank == 0:
                engine._persisted_snapshots[(1, "op0")].state_dict["params.weight"].fill_(float(window * 100))
                engine._window_start_iteration = window

            engine.replicate_to_peers(
                dp_group=gloo_group,
                dp_rank=rank,
                dp_world_size=2,
                device=torch.device("cpu"),
            )

            if rank == 1:
                # Per window, rank 1 should have exactly one flat buffer
                # (one op × one dtype group = one flat).
                assert len(engine._received_flat_buffers) == 1, \
                    f"window={window}: expected 1 received flat, got {len(engine._received_flat_buffers)}"


class TestRecoveryBarrierDistributed(DistributedTest):
    """Three-stage pipeline where the middle stage is recovering."""

    world_size = 3

    def test_middle_stage_recovery_collects_logs_from_both_neighbours(self):
        rank = dist.get_rank()

        # recovery_barrier early-returns when pp_group is None, so the
        # 3-rank test needs a concrete group; every rank is its own PP
        # stage, so the PP group spans the whole world.
        pp_group = dist.new_group(ranks=list(range(dist.get_world_size())))

        coord = MoEvementCoordinator(MoEvementConfig())
        # ``_device`` is used to place the world-handshake announcement
        # tensor; default 'cpu' would route the NCCL all_gather a CPU
        # tensor.  Pin to this rank's accelerator device so the handshake
        # runs on the native training backend.
        coord._device = _current_device()
        coord.set_pipeline_topology(
            pp_group=pp_group,
            stage_id=rank,
            num_stages=3,
            stage_to_global_fn=lambda s: s,
        )
        coord._global_step = 1
        coord._recovering = (rank == 1)

        # Populate neighbour loggers with one matching log entry each.
        if rank == 0:
            entry = LogEntry(
                iteration=0,
                micro_batch_id=0,
                stage_id=0,
                direction="activation",
                tensor=torch.tensor([1.0, 2.0, 3.0]),
            )
            coord.upstream_logger._logs[(0, 0)].append(entry)
        elif rank == 2:
            entry = LogEntry(
                iteration=0,
                micro_batch_id=0,
                stage_id=2,
                direction="gradient",
                tensor=torch.tensor([-1.0, -2.0, -3.0]),
            )
            coord.upstream_logger._logs[(0, 0)].append(entry)

        coord.recovery_barrier()

        # Rank 1 returns from the barrier holding the shipped logs; ranks
        # 0 and 2 are still inside ``_wait_for_recovery``, blocked in an
        # ``all_gather`` that needs rank 1 as a peer.  The rank-1-only
        # assertions must run first, then rank 1 participates in one more
        # handshake with ``_recovering=False`` so the paused ranks see
        # ``any_recovering=False`` and exit their wait loop.  The finally
        # block guarantees the unblock happens even if an assertion
        # fails, so a real bug surfaces as a failed test rather than a
        # teardown hang.
        try:
            if rank == 1:
                acts = coord.get_replay_activations(iteration=0, micro_batch_id=0)
                grads = coord.get_replay_gradients(iteration=0, micro_batch_id=0)

                assert acts is not None and len(acts) == 1
                torch.testing.assert_close(acts[0], torch.tensor([1.0, 2.0, 3.0]))
                assert grads is not None and len(grads) == 1
                torch.testing.assert_close(grads[0], torch.tensor([-1.0, -2.0, -3.0]))
        finally:
            if rank == 1:
                coord._recovering = False
                coord._world_recovery_handshake()

    def test_no_recovery_barrier_is_silent(self):
        """When no rank is recovering, the barrier returns without shipping anything."""
        rank = dist.get_rank()

        pp_group = dist.new_group(ranks=list(range(dist.get_world_size())))

        coord = MoEvementCoordinator(MoEvementConfig())
        coord._device = _current_device()
        coord.set_pipeline_topology(
            pp_group=pp_group,
            stage_id=rank,
            num_stages=3,
            stage_to_global_fn=lambda s: s,
        )
        coord._global_step = 1
        coord._recovering = False

        coord.recovery_barrier()

        # No rank received any logs.
        assert coord.upstream_logger.get_received_activation(0, 0) is None
        assert coord.upstream_logger.get_received_gradient(0, 0) is None


def _seed_sender_with_two_iter_window(engine, dp_rank_served, *, iterations=(10, 11)):
    """Populate ``engine._received_snapshots[dp_rank_served]`` as if the ring
    recv path had delivered a two-iter window from that sender.

    ``_serve_streaming`` walks ``_received_snapshots[sender_dp_rank]`` and
    ``_received_metadata[sender_dp_rank]``; seeding both here gives it
    something to ship without running a full ring replicate first.
    """
    engine._received_window_start[dp_rank_served] = int(iterations[0])
    engine._received_snapshots.setdefault(dp_rank_served, {})
    engine._received_metadata.setdefault(dp_rank_served, {})
    for idx, iteration in enumerate(iterations):
        op_name = f"op_{idx}"
        iter_key = (int(iteration), op_name)
        # Distinct values per iter per op so the receiver can detect
        # any iter / op cross-contamination in the streaming wire.
        engine._received_snapshots[dp_rank_served][iter_key] = {
            "params.weight": torch.full((4, ), float(iteration * 10 + idx)),
        }
        engine._received_metadata[dp_rank_served][iter_key] = {"is_active": bool(idx % 2 == 0)}


class TestPeerPullStreamingProtocol(DistributedTest):
    """End-to-end wire test for the S1 streaming protocol.

    Seeds a surviving ``SparseSnapshotEngine`` with the on-CPU state
    its ring receiver would have held at fault time, then runs
    ``serve_peer_pull_request`` / ``pull_snapshot_from_peer`` across a
    real 2-rank gloo group for both protocol versions.  The pulled
    ``(metadata, per_iter_operator_states)`` must be byte-identical
    across bulk and streaming — the whole point of S1 is that the wire
    shape changes but the callers see the same result object.
    """

    world_size = 2

    def _run_round_trip(self, protocol_version):
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()

        if rank == 0:
            # Rank 0 = surviving peer, serves the pulled shard.
            peer_engine = SparseSnapshotEngine(replication_factor=1)
            _seed_sender_with_two_iter_window(peer_engine, dp_rank_served=5, iterations=(10, 11))
            peer_engine.serve_peer_pull_request(
                requester_rank=1,
                sender_dp_rank=5,
                group=gloo_group,
                fault_iter=42,
                loss_scaler_state={"scale": 123.0},
                engine_scalars={"global_steps": 42},
                protocol_version=protocol_version,
            )
            return None
        else:
            fresh_engine = SparseSnapshotEngine(replication_factor=1)
            return fresh_engine.pull_snapshot_from_peer(
                peer_rank=0,
                group=gloo_group,
                protocol_version=protocol_version,
            )

    def test_streaming_round_trip_end_to_end(self):
        """Streaming wire delivers the seeded state to the replacement unchanged."""
        rank = dist.get_rank()
        result = self._run_round_trip(PEER_PULL_PROTOCOL_STREAMING)
        if rank == 0:
            assert result is None
            return

        metadata, states = result
        assert metadata["window_start_iteration"] == 10
        assert metadata["fault_iter"] == 42
        assert metadata["loss_scaler_state"] == {"scale": 123.0}
        assert metadata["engine_scalars"] == {"global_steps": 42}
        assert metadata["per_iter_active"] == {10: {"op_0": True}, 11: {"op_1": False}}

        # Tensor payloads distinct per iter — catches iter / op cross-delivery.
        torch.testing.assert_close(states[10]["op_0"]["params.weight"], torch.full((4, ), 100.0))
        torch.testing.assert_close(states[11]["op_1"]["params.weight"], torch.full((4, ), 111.0))

    def test_bulk_round_trip_end_to_end(self):
        """Bulk wire path still works after the protocol-version handshake addition."""
        rank = dist.get_rank()
        result = self._run_round_trip(PEER_PULL_PROTOCOL_BULK)
        if rank == 0:
            assert result is None
            return

        metadata, states = result
        assert metadata["window_start_iteration"] == 10
        assert metadata["fault_iter"] == 42
        assert metadata["per_iter_active"] == {10: {"op_0": True}, 11: {"op_1": False}}
        torch.testing.assert_close(states[10]["op_0"]["params.weight"], torch.full((4, ), 100.0))
        torch.testing.assert_close(states[11]["op_1"]["params.weight"], torch.full((4, ), 111.0))

    def test_unsupported_protocol_version_rejected_cleanly(self):
        """Server replies length=0 when handshake version is not recognized.

        The replacement path treats length=0 as "peer has nothing for me"
        and retries another peer — so an unknown-version server can't
        deadlock a roll-forward upgrade.  The server side goes through
        the coordinator's dispatch path, which is where version
        validation lives; here we simulate it at the engine seam by
        sending the bogus version on the wire and confirming the server
        emits the length=0 sentinel on its own.

        This mirrors the server-side ``coordinator.serve_sparse_snapshot_to_peer``
        unsupported-version branch without spinning up the full
        coordinator — what we're verifying is that ``length=0`` looks
        like ``(None, None)`` to the bulk-pull receiver.
        """
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()
        bogus_version = 99

        if rank == 0:
            # Emulate coordinator.serve_sparse_snapshot_to_peer on an
            # unknown version: drop the length=0 sentinel and return.
            dist.send(torch.tensor([0], dtype=torch.int64), dst=1, group=gloo_group)
        else:
            fresh_engine = SparseSnapshotEngine(replication_factor=1)
            metadata, states = fresh_engine.pull_snapshot_from_peer(peer_rank=0,
                                                                    group=gloo_group,
                                                                    protocol_version=bogus_version)
            # The pull sees length=0 on its first recv and falls through
            # to (None, None) — the same shape as "peer has no data", so
            # the caller is free to retry another peer.
            assert metadata is None and states is None
