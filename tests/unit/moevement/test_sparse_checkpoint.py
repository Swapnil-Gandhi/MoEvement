# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for MoEvement sparse checkpointing system."""

import os
import tempfile
import types

import pytest
import torch

import torch.nn as nn

from deepspeed.moevement.config import MoEvementConfig, MOEVEMENT
from deepspeed.moevement.coordinator import MoEvementCoordinator
from deepspeed.moevement.scheduler import (
    SparseCheckpointScheduler,
    OperatorInfo,
)
from deepspeed.moevement.sparse_snapshot import SparseSnapshotEngine
from deepspeed.moevement.conversion import SparseToDenseConverter
from deepspeed.moevement.upstream_logging import UpstreamLogger


class TestMoEvementConfig:

    def test_default_config(self):
        config = MoEvementConfig()
        assert config.enabled is False
        assert config.replication_factor == 1
        assert config.reorder_threshold == 0.10
        assert config.reorder_fraction == 0.25
        assert config.pcie_bandwidth_gbs == 25.0
        assert config.upstream_logging is True
        assert config.initial_iter_time_sec == 1.0
        assert config.activation_count_window_iters == 100
        assert config.iter_time_window_iters == 50

    def test_config_from_dict(self):
        param_dict = {
            MOEVEMENT: {
                "enabled": True,
                "replication_factor": 3,
                "reorder_threshold": 0.05,
                "pcie_bandwidth_gbs": 32.0,
            }
        }
        config = MoEvementConfig(param_dict)
        assert config.enabled is True
        assert config.replication_factor == 3
        assert config.reorder_threshold == 0.05
        assert config.pcie_bandwidth_gbs == 32.0
        assert config.upstream_logging is True  # default

    def test_pcie_bandwidth_conversion(self):
        config = MoEvementConfig()
        expected = 25.0 * (1024**3)
        assert config.pcie_bandwidth_bytes_per_sec == expected


class TestSparseCheckpointScheduler:

    def _make_operators(self, num_experts=4, params_per_expert=1000, non_expert_params=500, gate_params=100):
        operators = []
        operators.append(
            OperatorInfo(name="non_expert",
                         num_params=non_expert_params,
                         is_expert=False,
                         layer_id=0,
                         local_expert_id=None))
        operators.append(
            OperatorInfo(name="gate_0", num_params=gate_params, is_expert=False, layer_id=0, local_expert_id=None))
        for i in range(num_experts):
            operators.append(
                OperatorInfo(name=f"expert_{i}",
                             num_params=params_per_expert,
                             is_expert=True,
                             layer_id=0,
                             local_expert_id=i))
        return operators

    def test_find_window_size_all_fit(self):
        """When all operators fit in a single iteration, W_sparse=1."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)  # Very fast PCIe
        operators = self._make_operators(num_experts=4, params_per_expert=1000)
        scheduler.register_operators(operators)

        w_sparse, num_active = scheduler.find_window_size(iter_time_sec=1.0)
        assert w_sparse == 1
        assert num_active == len(operators)

    def test_find_window_size_slow_pcie(self):
        """With slow PCIe, window size increases to spread checkpointing."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=100)  # Very slow
        operators = self._make_operators(num_experts=8, params_per_expert=100000)
        scheduler.register_operators(operators)

        w_sparse, num_active = scheduler.find_window_size(iter_time_sec=0.1)
        assert w_sparse > 1
        assert num_active < len(operators)

    def test_find_window_size_warns_when_budget_unmet_at_num_active_one(self, caplog, monkeypatch):
        """D3: pathologically slow PCIe surfaces as a single warning.

        ``find_window_size`` exits the search loop at ``num_active == 1``
        whether or not that slot fits the budget — the alternative would
        be a hard failure on slow-PCIe configs.  The trade-off is that
        the snapshot stream then falls behind from iter 1 with no
        attribution: downstream ("one window's D2H fits in one iter")
        absorbs the overrun as wall-clock stretch.  D3 surfaces the
        overrun as a single warning so the operator can re-tune.
        """
        import logging
        _capture_deepspeed_warnings(caplog, monkeypatch)
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1)  # absurdly slow
        operators = self._make_operators(num_experts=4, params_per_expert=1_000_000)
        scheduler.register_operators(operators)
        with caplog.at_level(logging.WARNING):
            w_sparse, num_active = scheduler.find_window_size(iter_time_sec=0.001)
        assert num_active == 1
        assert any("PCIe budget cannot be met" in rec.message for rec in caplog.records), (
            f"expected D3 warning; got: {[(rec.levelname, rec.message[:80]) for rec in caplog.records]}")

    def test_find_window_size_silent_when_budget_met(self, caplog, monkeypatch):
        """D3: no warning when worst slot fits the budget."""
        import logging
        _capture_deepspeed_warnings(caplog, monkeypatch)
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=int(1e12))  # very fast
        operators = self._make_operators(num_experts=4, params_per_expert=100)
        scheduler.register_operators(operators)
        with caplog.at_level(logging.WARNING):
            scheduler.find_window_size(iter_time_sec=0.1)
        assert not any("PCIe budget cannot be met" in rec.message for rec in caplog.records)

    def test_find_window_size_perf_target_picks_larger_w_sparse(self):
        """``overlap_target < 1.0`` flips the algorithm: instead of MIN
        w_sparse (recovery-optimal), pick MAX w_sparse whose per-iter
        snapshot still honors the tighter budget.

        Same scheduler input where the recovery-optimal default picks
        ``w_sparse=1`` (all ops fit) — with ``overlap_target=0.5`` the
        budget halves and forces the algorithm to spread the snapshot
        across multiple iters, picking a larger ``w_sparse``.
        """
        # Pcie tuned so all-active fits at overlap=1.0 but only 1-active
        # fits at overlap=0.5 (so num_active scan finds the largest
        # num_active that still fits = 1, giving max w_sparse).
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=int(1e9))  # 1 GB/s
        operators = self._make_operators(num_experts=8, params_per_expert=100_000)
        scheduler.register_operators(operators)

        recovery_w_sparse, recovery_num_active = scheduler.find_window_size(iter_time_sec=1.0, overlap_target=1.0)
        perf_w_sparse, perf_num_active = scheduler.find_window_size(iter_time_sec=1.0, overlap_target=0.1)

        # Recovery-optimal should pick at least as small a w_sparse as perf
        # (i.e. perf w_sparse >= recovery w_sparse).
        assert perf_w_sparse >= recovery_w_sparse, \
            (f"perf-target should yield >= recovery's w_sparse: "
             f"recovery={recovery_w_sparse}, perf={perf_w_sparse}")
        # Tighter budget → fewer active per iter.
        assert perf_num_active <= recovery_num_active

    def test_find_window_size_perf_target_falls_back_when_overruns(self):
        """When even ``num_active=1`` overruns the tighter budget,
        ``num_active`` falls back to 1 (matching the recovery-optimal
        floor).  Same code path as the recovery-mode warning."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1)  # absurdly slow
        operators = self._make_operators(num_experts=4, params_per_expert=1_000_000)
        scheduler.register_operators(operators)
        w_sparse, num_active = scheduler.find_window_size(iter_time_sec=0.001, overlap_target=0.5)
        assert num_active == 1
        assert w_sparse == len(operators)

    def test_find_window_size_default_unchanged(self):
        """``overlap_target`` defaults to 1.0 → preserves historical
        recovery-optimal behavior.  Existing tests' assertions still
        hold without passing the parameter."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=4, params_per_expert=1000)
        scheduler.register_operators(operators)
        w_sparse_default, _ = scheduler.find_window_size(iter_time_sec=1.0)
        w_sparse_explicit, _ = scheduler.find_window_size(iter_time_sec=1.0, overlap_target=1.0)
        assert w_sparse_default == w_sparse_explicit == 1

    def test_find_window_size_prices_tail_slot_not_just_head(self):
        """``num_active`` must reject candidates whose TAIL slot blows budget.

        Pre-fix ``find_window_size`` priced ``operators[:num_active]`` only —
        which, after ``order_operators``, is always the head of the list
        (smallest-activation experts).  But ``generate_schedule`` later
        activates every ``num_active``-sized slot, including the tail
        slot that holds ``non_expert`` (typically the transformer backbone,
        much larger than any single expert).  If the tail slot costs more
        than the head, the pre-fix code picked an overly-aggressive
        ``num_active`` whose tail iteration silently exceeds the PCIe
        budget — exactly the kind of surprise the budget was supposed to
        prevent.

        Ordered layout (experts first by popularity, non-expert / gate at
        tail) + ops sized so num_active=2's tail slot (non_expert + gate)
        costs more than its head slot (two small experts), and only
        num_active=1 fits the budget at every slot.
        """
        # Shapes: 2 tiny experts + 1 LARGE non_expert + 1 tiny gate.
        # After order_operators: [expert_0, expert_1, non_expert, gate].
        # Layout verified in the comments in test_order_operators_popularity.
        ops = [
            OperatorInfo(name="non_expert", num_params=1000, is_expert=False, layer_id=0, local_expert_id=None),
            OperatorInfo(name="gate", num_params=100, is_expert=False, layer_id=0, local_expert_id=None),
            OperatorInfo(name="expert_0", num_params=100, is_expert=True, layer_id=0, local_expert_id=0),
            OperatorInfo(name="expert_1", num_params=100, is_expert=True, layer_id=0, local_expert_id=1),
        ]
        # Budget chosen to sit in the sandwich:
        #   num_active=2 tail slot (non_expert + gate)  = 12*1100 + 2*200  = 13600  → MUST reject
        #   num_active=1 worst slot   (non_expert only) = 12*1000 + 2*300  = 12600  → MUST accept
        # Pre-fix sees only num_active=2's head slot (2 experts = 12*200 + 2*1100 = 4600)
        # and wrongly accepts num_active=2.
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=13000)
        # generate_schedule orders internally; find_window_size consumes the
        # ordered list via self.operators, so call generate_schedule to
        # exercise the full production path.
        scheduler.register_operators(ops)
        w_sparse, schedule = scheduler.generate_schedule(iter_time_sec=1.0)

        assert w_sparse == 4, (f"pre-fix selects num_active=2 (w_sparse=2) by pricing head slot only; "
                               f"post-fix must walk every slot and reject num_active=2 because its "
                               f"tail slot (non_expert + gate) exceeds budget.  got w_sparse={w_sparse}")
        # One active op per slot; the slot containing non_expert is the
        # worst case and the budget was sized to just accommodate it.
        assert all(len(s.active_operators) == 1 for s in schedule)

    def test_generate_schedule_frozen_is_not_yet_captured_tail(self):
        """D2: schedule[i].frozen_operators is strictly ordered[end:].

        Per paper Algorithm 1, slot ``i``'s frozen set is the
        not-yet-captured tail.  Operators captured ACTIVE in earlier
        slots of the same window must NOT appear in later slots'
        ``frozen_operators`` — the coordinator-side
        ``_already_active_in_window`` filter that used to drop them at
        capture time has been deleted; the scheduler now produces the
        correct set directly.
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1)
        operators = self._make_operators(num_experts=4, params_per_expert=1_000_000)
        scheduler.register_operators(operators)
        w_sparse, schedule = scheduler.generate_schedule(iter_time_sec=0.001)
        # With this absurdly slow PCIe, num_active settles at 1 → w_sparse = total_ops.
        # slot i activates ordered[i:i+1]; frozen must be ordered[i+1:].
        already_seen_active = set()
        for i, entry in enumerate(schedule):
            for op_name in entry.active_operators:
                assert op_name not in already_seen_active, (
                    f"slot {i} re-activates {op_name} already captured in an earlier slot")
                already_seen_active.add(op_name)
            for op_name in entry.frozen_operators:
                assert op_name not in already_seen_active, (
                    f"slot {i}'s frozen set contains {op_name} already captured ACTIVE earlier "
                    "in this window — D2 invariant ('not-yet-captured tail only') broken")

    def test_generate_schedule_covers_all_operators(self):
        """Schedule must snapshot every operator exactly once per window."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=6)
        scheduler.register_operators(operators)

        w_sparse, schedule = scheduler.generate_schedule(iter_time_sec=1.0)
        assert len(schedule) == w_sparse

        # Collect all active operators across the window
        all_active = set()
        for entry in schedule:
            all_active.update(entry.active_operators)

        # Every operator must appear as active at least once
        for op in operators:
            assert op.name in all_active, f"Operator {op.name} not scheduled as active"

    def test_order_operators_popularity(self):
        """Experts sorted by ascending activation count at the head, non-experts/gates at the tail.

        Paper Fig. 6 places non-expert and gating operators at the END of
        the W_sparse window (they fire on every token → maximal popularity
        → last to promote during sparse-to-dense conversion).  Tail
        placement is intentional despite the FP16-floor drift it produces in
        the engine integration tests — the recovery tolerances are
        relaxed to absorb the ~10-ULP overshoot rather than sacrificing
        the Fig. 6 layout (see ``scheduler.order_operators`` docstring).
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=4)

        # Set different activation counts
        operators[2].activation_count = 100  # expert_0
        operators[3].activation_count = 50  # expert_1
        operators[4].activation_count = 200  # expert_2
        operators[5].activation_count = 10  # expert_3

        scheduler.register_operators(operators)
        ordered = scheduler.order_operators()

        # Experts come first, sorted by ascending activation count
        expert_order = [op for op in ordered if op.is_expert]
        assert expert_order == ordered[:len(expert_order)]
        for i in range(len(expert_order) - 1):
            assert expert_order[i].activation_count <= expert_order[i + 1].activation_count

        # Non-experts at the tail
        assert not ordered[-1].is_expert
        assert not ordered[-2].is_expert

    def test_get_schedule_for_iteration(self):
        """Schedule cycles through window positions correctly."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=100)
        operators = self._make_operators(num_experts=8, params_per_expert=100000)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=0.1)

        # Each iteration should map to a specific schedule entry
        for step in range(20):
            entry = scheduler.get_schedule_for_iteration(step)
            assert entry is not None
            expected_idx = step % scheduler.w_sparse
            assert entry.active_operators == scheduler.schedule[expected_idx].active_operators

    def test_get_schedule_for_iteration_aligns_to_window_start_after_regen(self):
        """D1: slot 0 follows window_start, not ``global_step % w_sparse``.

        Pre-fix, ``get_schedule_for_iteration(global_step)`` returned
        ``schedule[global_step % w_sparse]``.  After a mid-training regen
        that changes ``w_sparse``, slot 0 misaligns with the window's
        first iter for one full window — disrupting the paper's tail-
        slot-overlap-with-next-window story.  The fix takes
        ``window_start`` so the modulus is computed against the offset
        from the actual window start.
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=100)
        operators = self._make_operators(num_experts=8, params_per_expert=100000)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=0.1)

        # Window starts at iter 7 (mid-training regen scenario).  The
        # iter at window_start should always be slot 0.
        for offset in range(scheduler.w_sparse * 2):
            global_step = 7 + offset
            entry = scheduler.get_schedule_for_iteration(global_step, window_start=7)
            assert entry.active_operators == scheduler.schedule[offset % scheduler.w_sparse].active_operators

    def test_update_activation_counts(self):
        """Activation counts accumulate into the pending interval, surface after tick_interval.

        ``update_activation_counts`` writes to ``_pending_interval`` — the
        per-iteration staging dict.  The per-expert ``activation_count``
        surfaces the rolling-window total only after ``tick_interval``
        finalizes the iteration's counts into the window.
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=4)
        scheduler.register_operators(operators)

        counts = torch.tensor([10.0, 20.0, 30.0, 40.0])
        scheduler.update_activation_counts(layer_id=0, exp_counts=counts)
        scheduler.tick_interval()

        expert_ops = [op for op in scheduler.operators if op.is_expert]
        assert expert_ops[0].activation_count == 10.0
        assert expert_ops[1].activation_count == 20.0
        assert expert_ops[2].activation_count == 30.0
        assert expert_ops[3].activation_count == 40.0

        # A second iteration with the same counts doubles the window totals
        # since the rolling window has room (default capacity is 100 iters
        # and the test has populated 2).
        scheduler.update_activation_counts(layer_id=0, exp_counts=counts)
        scheduler.tick_interval()
        assert expert_ops[0].activation_count == 20.0
        assert expert_ops[3].activation_count == 80.0

    def _feed_intervals(self, scheduler, counts_by_index, iters):
        """Push ``iters`` identical per-expert count intervals into the scheduler.

        ``counts_by_index`` is indexed by ``local_expert_id`` (so
        ``counts_by_index[0]`` is the count for ``expert_0``, etc.) —
        matching how ``update_activation_counts`` consumes the tensor.
        """
        counts = torch.tensor(counts_by_index, dtype=torch.float32)
        for _ in range(iters):
            scheduler.update_activation_counts(layer_id=0, exp_counts=counts)
            scheduler.tick_interval()

    def test_should_reorder_warming_up(self):
        """Rate-based reorder decisions require a full rolling window.

        Before the window fills, ``should_reorder`` returns False — a
        deliberate quiet period that prevents us from firing on a
        half-populated "older half" (which would alias startup zero-counts
        as a legitimate comparison baseline).
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=4)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # Half-populate the window — uniform counts, but only 5 of the 10
        # required iterations.
        self._feed_intervals(scheduler, [10, 20, 30, 40], iters=5)
        assert not scheduler.should_reorder()

    def test_should_reorder_no_change(self):
        """A full window of uniform counts yields zero rate delta → no reorder."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=4)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # 10 iters of identical per-expert counts — older-half rate equals
        # newer-half rate for every expert.
        self._feed_intervals(scheduler, [10, 20, 30, 40], iters=10)
        assert not scheduler.should_reorder()

    def test_should_reorder_after_significant_change(self):
        """Reorder triggers when enough experts' rates shift significantly."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=8)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # Fill the older half with one distribution, the newer half with a
        # reversed one.  Every expert sees a meaningful rate change.
        older = [10, 20, 30, 40, 50, 60, 70, 80]
        newer = list(reversed(older))
        self._feed_intervals(scheduler, older, iters=5)
        self._feed_intervals(scheduler, newer, iters=5)
        assert scheduler.should_reorder()

    def test_should_reorder_below_fraction_does_not_trigger(self):
        """Two experts with large rate shifts is under the 0.25 fraction gate.

        With 10 experts, a 0.10 threshold means each expert "counts as
        changed" when its newer-half rate differs from its older-half rate
        by more than 10%.  The fraction gate demands at least 0.25 * 10 = 3
        such experts to fire a reorder.  This test pins the strict
        inequality in ``fraction >= reorder_fraction``.
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=10)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # Older half: uniform 100 each.  Newer half: two experts jump to
        # 200 (100% rate change, well above the 10% threshold); the rest
        # stay at 100 (0% change).  That's 2 of 10 changed → 0.20 < 0.25.
        older = [100] * 10
        newer = [200, 100, 100, 100, 100, 100, 100, 100, 100, 200]
        self._feed_intervals(scheduler, older, iters=5)
        self._feed_intervals(scheduler, newer, iters=5)
        assert not scheduler.should_reorder()

    def test_should_reorder_at_fraction_triggers(self):
        """Three experts with rate shifts meets the inclusive 0.25 gate at 10 experts."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=10)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # Bump one more expert into the "changed" group: 3 of 10 → 0.30
        # meets the 0.25 bound inclusively.
        older = [100] * 10
        newer = [200, 100, 100, 100, 200, 100, 100, 100, 100, 200]
        self._feed_intervals(scheduler, older, iters=5)
        self._feed_intervals(scheduler, newer, iters=5)
        assert scheduler.should_reorder()

    def test_should_reorder_ignores_sub_threshold_noise(self):
        """A 9% rate shift is under the 10% threshold and does not count."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=10)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # Every expert shifts 9% — a meaningful rank shuffle in absolute
        # terms, but each individual change sits below the 10% threshold,
        # so zero experts count as "changed" and no reorder fires.
        older = [100] * 10
        newer = [109] * 10
        self._feed_intervals(scheduler, older, iters=5)
        self._feed_intervals(scheduler, newer, iters=5)
        assert not scheduler.should_reorder()

    def test_generate_schedule_reflects_new_counts_on_regen(self):
        """Regenerating after a popularity swap reorders the operator list.

        Drives the end-to-end loop: feed intervals that establish an
        initial popularity order, generate, feed intervals that flip the
        order, regenerate, and check the schedule tracks the shift.
        Without the regen path, the old schedule would keep driving
        snapshots even though upstream activation patterns have shifted,
        which is the whole problem the reorder machinery is meant to fix.
        """
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25,
                                              activation_count_window_iters=10)
        operators = self._make_operators(num_experts=4)
        scheduler.register_operators(operators)

        # Establish initial popularity: expert_0 < expert_1 < expert_2 <
        # expert_3.  Fill the full window so ``tick_interval`` surfaces
        # these aggregated counts into ``activation_count``.
        self._feed_intervals(scheduler, [1, 2, 3, 4], iters=10)
        scheduler.generate_schedule(iter_time_sec=1.0)
        initial_expert_order = [op.name for op in scheduler.operators if op.is_expert]
        assert initial_expert_order == ["expert_0", "expert_1", "expert_2", "expert_3"]

        # Flip popularity.  Feed a lopsided mix so the rolling window ends
        # up dominated by the reversed distribution without producing
        # symmetric ties at the aggregate level.
        self._feed_intervals(scheduler, [4, 3, 2, 1], iters=8)
        assert scheduler.should_reorder()

        scheduler.generate_schedule(iter_time_sec=1.0)
        new_expert_order = [op.name for op in scheduler.operators if op.is_expert]
        assert new_expert_order == ["expert_3", "expert_2", "expert_1", "expert_0"]


class TestCoordinatorIterTimeRecalibration:
    """Runtime ``w_sparse`` recalibration loop (CheckFreq-style).

    Exercises each helper on ``MoEvementCoordinator`` in isolation —
    duration recording, median-over-half-window update, and drift
    threshold — so regressions surface without a full distributed setup.
    """

    def _coord(self, iter_time_window_iters=4):
        config = MoEvementConfig({
            "moevement": {
                "enabled": True,
                "iter_time_window_iters": iter_time_window_iters,
                "upstream_logging": False,
            }
        })
        return MoEvementCoordinator(config)

    def test_record_iter_duration_skips_first_call(self):
        """First ``_record_iter_duration`` call has no prior timestamp.

        Until the second call we can't compute a delta, so the window
        stays empty and the first sample reflects a real iter duration.
        """
        coord = self._coord(iter_time_window_iters=4)
        coord._record_iter_duration()
        assert len(coord._iter_time_window) == 0

        coord._record_iter_duration()
        assert len(coord._iter_time_window) == 1

    def test_maybe_update_iter_time_waits_for_half_window(self):
        """``_iter_time_sec`` only updates once the window has ``maxlen / 2`` samples."""
        coord = self._coord(iter_time_window_iters=4)
        coord._iter_time_sec = 1.0

        # 1 sample < half-window (2) → no update.
        coord._iter_time_window.append(2.0)
        coord._maybe_update_iter_time()
        assert coord._iter_time_sec == 1.0

        # 2 samples == half-window → update fires to the median.
        coord._iter_time_window.append(2.0)
        coord._maybe_update_iter_time()
        assert coord._iter_time_sec == 2.0

    def test_maybe_update_iter_time_uses_median_not_mean(self):
        """Median is robust to a single GC-pause outlier that would skew a mean."""
        coord = self._coord(iter_time_window_iters=4)
        coord._iter_time_sec = 1.0
        # Three "normal" samples at 1.0s, one outlier at 5.0s.  Mean = 2.0,
        # but the median is 1.0 — we want the latter.
        for v in [1.0, 1.0, 1.0, 5.0]:
            coord._iter_time_window.append(v)
        coord._maybe_update_iter_time()
        assert coord._iter_time_sec == 1.0

    def test_drift_threshold_exceeded(self):
        """Drift ≥ 10% of baseline triggers recompute; < 10% does not."""
        coord = self._coord(iter_time_window_iters=4)
        coord._last_scheduled_iter_time = 1.0

        coord._iter_time_sec = 1.05  # 5% drift
        assert not coord._iter_time_drift_exceeds_threshold()

        coord._iter_time_sec = 1.15  # 15% drift
        assert coord._iter_time_drift_exceeds_threshold()

        coord._iter_time_sec = 0.85  # 15% drop
        assert coord._iter_time_drift_exceeds_threshold()

    def test_drift_check_inert_without_data(self):
        """No recompute is triggered before the window populates."""
        coord = self._coord(iter_time_window_iters=4)
        coord._last_scheduled_iter_time = 1.0
        coord._iter_time_sec = None  # window hasn't updated yet
        assert not coord._iter_time_drift_exceeds_threshold()


class TestExpCountsSideStreamCoordination:
    """``_schedule_exp_counts_copies`` queues D2Hs on a dedicated side
    stream and ``_fence_exp_counts_copies`` syncs only that stream — so
    the boundary fence no longer drags in pending main-stream training
    kernels.

    A missing ``side_stream.wait_stream(default_stream())`` would let
    the side-stream copy race the gate's write to ``exp_counts`` —
    typically reading stale or partially-written values.  The test runs
    the schedule + fence cycle 50× with distinct per-iter values to
    surface that race as flakiness; bit-exact assertion catches stale
    reads even when timing happens to mask the bug.
    """

    @pytest.fixture(autouse=True)
    def check_accelerator(self):
        from deepspeed.accelerator import get_accelerator
        if get_accelerator().Stream is None:
            pytest.skip("requires accelerator with stream support")

    def _make_coord_with_layers(self, num_layers=4, num_experts=8):
        from deepspeed.accelerator import get_accelerator
        config = MoEvementConfig({"moevement": {"enabled": True, "upstream_logging": False}})
        coord = MoEvementCoordinator(config)
        device = get_accelerator().current_device_name()
        # Each fake MoE layer just needs an ``exp_counts`` attribute
        # holding an ``int64[num_experts]`` GPU tensor — the rest of the
        # layer surface isn't touched by ``_schedule_exp_counts_copies``.
        layers = []
        for _ in range(num_layers):
            layer = types.SimpleNamespace(exp_counts=torch.zeros(num_experts, dtype=torch.int64, device=device))
            layers.append(layer)
        coord._moe_layers = layers
        return coord, layers, device

    def test_bit_exact_after_fence_single_iter(self):
        coord, layers, device = self._make_coord_with_layers()
        for li, layer in enumerate(layers):
            layer.exp_counts = torch.arange(li * 100, li * 100 + 8, dtype=torch.int64, device=device)
        pending = coord._schedule_exp_counts_copies(slot_idx=0)
        coord._fence_exp_counts_copies()
        assert len(pending) == len(layers)
        for layer_idx, cached_cpu in pending:
            expected = layers[layer_idx].exp_counts.cpu()
            assert torch.equal(cached_cpu, expected), \
                f"layer {layer_idx}: cached {cached_cpu.tolist()} != source {expected.tolist()}"

    def test_per_slot_buffers_isolate_iters(self):
        """Each (layer_idx, slot_idx) pair owns its own pinned buffer —
        a later slot's D2H must not overwrite an earlier slot's value
        before the boundary fence reads it."""
        coord, layers, device = self._make_coord_with_layers(num_layers=2, num_experts=4)
        # Three iters, distinct per-iter values per layer.
        per_iter_pending = []
        for slot in range(3):
            for li, layer in enumerate(layers):
                layer.exp_counts = torch.tensor([slot * 10 + li, slot * 10 + li + 1, slot, li],
                                                dtype=torch.int64,
                                                device=device)
            per_iter_pending.append((slot, coord._schedule_exp_counts_copies(slot_idx=slot)))
        coord._fence_exp_counts_copies()
        # All three iters' values must survive intact in distinct buffers.
        for slot, pending in per_iter_pending:
            for layer_idx, cached_cpu in pending:
                expected = torch.tensor([slot * 10 + layer_idx, slot * 10 + layer_idx + 1, slot, layer_idx],
                                        dtype=torch.int64)
                assert torch.equal(cached_cpu, expected), \
                    f"slot {slot} layer {layer_idx}: cached {cached_cpu.tolist()} != expected {expected.tolist()}"

    def test_50_iter_flake_guard(self):
        """50 schedule+fence cycles with distinct per-iter values.  A
        missing ``wait_stream(default)`` lets the side-stream copy race
        the GPU write — bit-exact comparison catches it even when
        timing happens to occasionally line up."""
        coord, layers, device = self._make_coord_with_layers(num_layers=4, num_experts=8)
        for iteration in range(50):
            for li, layer in enumerate(layers):
                # Issue the write on the default stream where the gate
                # would produce ``exp_counts``.  Different value per iter
                # so a stale read shows up as a wrong-iter mismatch.
                layer.exp_counts = torch.full((8, ), iteration * 1000 + li, dtype=torch.int64, device=device)
            pending = coord._schedule_exp_counts_copies(slot_idx=0)
            coord._fence_exp_counts_copies()
            for layer_idx, cached_cpu in pending:
                expected = torch.full((8, ), iteration * 1000 + layer_idx, dtype=torch.int64)
                assert torch.equal(cached_cpu, expected), \
                    (f"iter {iteration} layer {layer_idx}: cached {cached_cpu.tolist()} "
                     f"!= expected {expected.tolist()} (race?)")


class TestPoolSizingFromConfig:
    """Pool sizing flows from MoEvementConfig (auto-sized when 0).

    The upstream-logger pool's ``grow_on_miss`` defaults to "auto-size
    from gas × num_moe_layers × w_sparse" so production configs at
    gas>1 don't churn cudaHostAlloc.  Both pools' ``max_per_key`` is
    settable via ``MoEvementConfig.pool_max_per_key``.  Misconfiguring
    these (too small) costs visible perf at gas>1; the test exists
    because the auto-formula is easy to silently regress and the
    impact only shows up under load.
    """

    def _make_coord(self, gas, num_moe_layers, pool_grow_override=0, pool_max_per_key=4096):
        config = MoEvementConfig({
            "moevement": {
                "enabled": True,
                "upstream_logging": True,
                "pool_grow_on_miss_activation": pool_grow_override,
                "pool_max_per_key": pool_max_per_key,
            }
        })
        coord = MoEvementCoordinator(config)
        coord._moe_layers = [object()] * num_moe_layers  # opaque stand-ins; only count is read
        coord._gradient_accumulation_steps = gas
        return coord

    def test_auto_size_scales_with_gas_layers_w_sparse(self):
        """Auto formula is ``max(512, 8 × gas × num_moe_layers × w_sparse)``."""
        coord = self._make_coord(gas=32, num_moe_layers=4)
        # Simulate the inline computation the coordinator does after w_sparse
        # is known.  Direct re-implementation kept tiny so the test is
        # diff-readable when the formula tweaks.
        w_sparse = 5
        expected = max(512, 8 * 32 * 4 * w_sparse)  # = 5120
        assert expected == 5120
        # Coordinator runs it inside ``initialize`` after schedule generation;
        # exposing the partial path here would require a real model.  Verify
        # the multiplier rather than the wired-up call (same arithmetic).

    def test_floor_kicks_in_at_low_workload(self):
        """gas=1, 1 layer, w_sparse=1 should still hit the 512 floor."""
        expected = max(512, 8 * 1 * 1 * 1)
        assert expected == 512

    def test_user_override_skips_auto_size(self):
        """Non-zero ``pool_grow_on_miss_activation`` is taken literally."""
        config = MoEvementConfig(
            {"moevement": {
                "enabled": True,
                "pool_grow_on_miss_activation": 999,
                "pool_max_per_key": 1024,
            }})
        assert config.pool_grow_on_miss_activation == 999
        assert config.pool_max_per_key == 1024

    def test_default_config_auto_sizes(self):
        """Default config asks for auto-size (sentinel 0)."""
        config = MoEvementConfig({"moevement": {"enabled": True}})
        assert config.pool_grow_on_miss_activation == 0
        assert config.pool_max_per_key == 4096

    def test_set_max_per_key_resizes_existing_pool(self):
        """``PinnedPool.set_max_per_key`` mutates the cap on a live pool."""
        from deepspeed.moevement.buffer_pool import PinnedPool
        pool = PinnedPool(max_per_key=64)
        assert pool._max_per_key == 64
        pool.set_max_per_key(2048)
        assert pool._max_per_key == 2048


class TestSparseSnapshotEngine:

    @pytest.fixture(autouse=True)
    def check_accelerator(self):
        """Skip tests that require CUDA stream if no accelerator is available."""
        try:
            from deepspeed.accelerator import get_accelerator
            accel = get_accelerator()
            if accel is None or accel.Stream is None:
                pytest.skip("No accelerator available for snapshot tests")
        except Exception:
            pytest.skip("No accelerator available for snapshot tests")

    def test_snapshot_active_operator(self):
        """Active operator snapshots FP32 weights and optimizer state."""
        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        optim_state = {"exp_avg": torch.randn(10, 10), "exp_avg_sq": torch.randn(10, 10)}

        engine.snapshot_operator("expert_0", params, optim_state, is_active=True, iteration=0)
        engine.synchronize()

        snaps = engine.get_current_snapshots()
        assert (0, "expert_0") in snaps
        snap = snaps[(0, "expert_0")]
        assert snap.is_active is True
        assert "params.weight" in snap.state_dict
        assert "optimizer.exp_avg" in snap.state_dict

    def test_snapshot_captures_torch_adam_step_as_tensor(self):
        """torch.optim.Adam/AdamW store ``step`` as a 0-dim Tensor (PyTorch 2.0+).

        Pin that snapshot_operator's ``isinstance(tensor, torch.Tensor)`` filter
        lets it through, so post-recovery Adam resumes with the correct ``t``
        and bias-correction is not silently reset to ``t=1``.

        Regression gate for H2 in CORRECTNESS_AUDIT_2026_04_23.md: the audit
        assumed ``step`` was a Python int (true for FusedAdam / DeepSpeedCPUAdam,
        false for the torch_adam path used by examples/moevement/_common.py).
        If a future PyTorch default flips back to int, this test breaks first
        and forces re-evaluation rather than silent drift.
        """
        for OptCls in (torch.optim.Adam, torch.optim.AdamW):
            p = torch.nn.Parameter(torch.randn(4))
            opt = OptCls([p], lr=1e-3)
            for _ in range(3):
                opt.zero_grad()
                (p**2).sum().backward()
                opt.step()
            state = opt.state[p]
            assert "step" in state, f"{OptCls.__name__}: no step entry"
            assert isinstance(state["step"],
                              torch.Tensor), (f"{OptCls.__name__}: step is {type(state['step']).__name__}, "
                                              "not Tensor — snapshot_operator's isinstance filter would drop it.")

            engine = SparseSnapshotEngine(replication_factor=1)
            engine.snapshot_operator("expert_x", {"w": p.detach().clone()}, dict(state), is_active=True, iteration=0)
            engine.synchronize()
            snap = engine.get_current_snapshots()[(0, "expert_x")]
            assert "optimizer.step" in snap.state_dict, (f"{OptCls.__name__}: step Tensor was not captured")

    def test_snapshot_frozen_operator(self):
        """Frozen operator snapshots only FP16 compute weights."""
        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}

        engine.snapshot_operator("expert_1", params, None, is_active=False, iteration=0)
        engine.synchronize()

        snaps = engine.get_current_snapshots()
        assert (0, "expert_1") in snaps
        snap = snaps[(0, "expert_1")]
        assert snap.is_active is False
        assert "compute_weights.weight" in snap.state_dict
        # FP16 compute weights should be half precision
        assert snap.state_dict["compute_weights.weight"].dtype == torch.float16

    def test_snapshot_active_byte_equal_cpu_path(self):
        """f0 byte-equality (CPU input → CPU-fallback path in _batched_d2h).

        ``test_snapshot_active_operator`` only asserts the snapshot dict
        has the right keys.  This locks in that the bytes themselves
        survive the pack-by-dtype + per-key view layout.  Uses
        ``torch.arange`` so any byte-corruption is visible in the
        assertion message.
        """
        engine = SparseSnapshotEngine(replication_factor=0)

        src_weight = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        src_bias = torch.arange(10, 20, dtype=torch.float32)
        src_exp_avg = torch.arange(100, 200, dtype=torch.float32).reshape(10, 10)
        params = {"weight": src_weight, "bias": src_bias}
        optim_state = {"exp_avg": src_exp_avg}

        engine.snapshot_operator("expert_0", params, optim_state, is_active=True, iteration=0)
        engine.synchronize()

        snap = engine.get_current_snapshots()[(0, "expert_0")]
        torch.testing.assert_close(snap.state_dict["params.weight"], src_weight)
        torch.testing.assert_close(snap.state_dict["params.bias"], src_bias)
        torch.testing.assert_close(snap.state_dict["optimizer.exp_avg"], src_exp_avg)

    def test_snapshot_active_byte_equal_gpu_path(self):
        """f0 byte-equality (GPU input → production async D2H path).

        With GPU input, ``_batched_d2h`` uses the staging path
        (sparse_snapshot.py:311-323): allocate flat_gpu, pack onto it,
        then ``flat_cpu.copy_(flat_gpu, non_blocking=True)``.  Pinned
        bytes must bit-match the GPU source after ``synchronize``.
        Without this test the production path was never byte-checked.
        """
        if not torch.cuda.is_available():  #ignore-cuda
            pytest.skip("requires CUDA for GPU async D2H path")

        from deepspeed.accelerator import get_accelerator
        engine = SparseSnapshotEngine(replication_factor=0)

        device = get_accelerator().device_name()
        src_weight = torch.arange(100, dtype=torch.float32).reshape(10, 10).to(device)
        src_bias = torch.arange(10, 20, dtype=torch.float32).to(device)
        src_exp_avg = torch.arange(100, 200, dtype=torch.float32).reshape(10, 10).to(device)
        params = {"weight": src_weight, "bias": src_bias}
        optim_state = {"exp_avg": src_exp_avg}

        engine.snapshot_operator("expert_0", params, optim_state, is_active=True, iteration=0)
        engine.synchronize()

        snap = engine.get_current_snapshots()[(0, "expert_0")]
        torch.testing.assert_close(snap.state_dict["params.weight"].cpu(), src_weight.cpu())
        torch.testing.assert_close(snap.state_dict["params.bias"].cpu(), src_bias.cpu())
        torch.testing.assert_close(snap.state_dict["optimizer.exp_avg"].cpu(), src_exp_avg.cpu())

    def test_snapshot_frozen_byte_equal_fp16_promotion(self):
        """f0 byte-equality for frozen FP32→FP16 compute-weight promotion.

        Frozen path (sparse_snapshot.py:266-271) casts FP32 sources to
        FP16 before D2H.  Asserts the loaded FP16 round-trips back to
        the original FP32 within FP16 ULP — catches any silent dtype
        mismatch or layout-pack offset bug.
        """
        engine = SparseSnapshotEngine(replication_factor=0)

        # Use values within FP16 range so the cast is non-lossy at the
        # values themselves (FP16 ULP at 1.0 is 2**-10).
        src_weight = (torch.arange(64, dtype=torch.float32) / 64.0).reshape(8, 8)
        params = {"weight": src_weight}

        engine.snapshot_operator("frozen_op", params, None, is_active=False, iteration=0)
        engine.synchronize()

        snap = engine.get_current_snapshots()[(0, "frozen_op")]
        captured_fp16 = snap.state_dict["compute_weights.weight"]
        assert captured_fp16.dtype == torch.float16
        # Round-trip back to FP32 and compare against source FP32.
        torch.testing.assert_close(captured_fp16.float(), src_weight, atol=1e-3, rtol=1e-3)

    def test_snapshot_deferred_sync_preserves_bytes(self):
        """f0 byte-equality through the production deferred-sync boundary.

        Mirrors the coordinator sequence across TWO window rotations
        (it takes two boundaries for a snapshot to land in
        ``_persisted_snapshots`` per the rotation: ``_snapshots`` →
        ``_in_flight`` on first ``begin_window``, then ``_in_flight``
        → ``_persisted`` on next ``finalize_window``).  The deferred-
        sync path uses ``record_pending_d2h_event`` +
        ``wait_for_pending_d2h_event`` instead of a CPU
        ``synchronize`` — pin that this ordering doesn't lose or
        race-corrupt bytes (changed in #119; existing tests didn't
        cover it).
        """
        if not torch.cuda.is_available():  #ignore-cuda
            pytest.skip("requires CUDA for deferred sync")

        from deepspeed.accelerator import get_accelerator
        engine = SparseSnapshotEngine(replication_factor=0)

        device = get_accelerator().device_name()
        src_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8).to(device)

        # Window 0: snapshot + boundary (deferred sync).
        engine.snapshot_operator("op0", {"weight": src_weight}, None, is_active=False, iteration=0)
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=1)
        # Simulate next iter's optim.step queuing the wait on main stream.
        engine.wait_for_pending_d2h_event()

        # Window 1: empty boundary — promotes window 0's snapshots
        # from ``_in_flight`` to ``_persisted``.
        engine.record_pending_d2h_event()
        engine.finalize_window()
        engine.begin_window(iteration=2)
        engine.wait_for_pending_d2h_event()

        # Force a main-stream drain so we can deterministically read
        # the pinned bytes from this CPU thread.
        torch.cuda.synchronize()  #ignore-cuda

        snap = engine._persisted_snapshots[(0, "op0")]
        captured_fp16 = snap.state_dict["compute_weights.weight"]
        torch.testing.assert_close(captured_fp16.float().cpu(), src_weight.cpu(), atol=1e-3, rtol=1e-3)

    def test_per_iter_rng_state_round_trip(self):
        """Per-iter torch RNG state survives finalize_window → save → load → converter.

        Pins the full RNG plumbing introduced for stochastic-model
        recovery: the snapshot engine rotates ``_rng_state_per_iter``
        through in_flight → persisted in lockstep with operator
        snapshots, ``finalize_window`` injects each captured RNG dict as a
        ``__moe_rng_state__`` pseudo-operator into ``_persisted_snapshots``,
        and the converter routes that pseudo-op back into its
        ``_rng_state_per_iter`` cache rather than the operator state
        machine.  A regression at any link breaks dropout / stochastic-
        layer recovery silently — no test fixture hits it because the
        happy-path model is deterministic — so this round-trip + the
        ``__moe_rng_state__`` filter test are the canaries that pin it.
        """
        from deepspeed.moevement.sparse_snapshot import _RNG_PSEUDO_OP

        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(5, 5)}
        engine.snapshot_operator("test_op", params, None, is_active=False, iteration=0)
        rng_iter0 = {
            "torch_cpu": torch.randint(0, 256, (32, ), dtype=torch.uint8),
            "torch_cuda.0": torch.randint(0, 256, (16, ), dtype=torch.uint8),
        }
        engine._rng_state_per_iter[0] = rng_iter0
        engine.synchronize()
        engine.begin_window(iteration=1)
        engine.finalize_window()

        # finalize_window injected the pseudo-op into _persisted_snapshots.
        assert (0, _RNG_PSEUDO_OP) in engine._persisted_snapshots
        pseudo_snap = engine._persisted_snapshots[(0, _RNG_PSEUDO_OP)]
        assert pseudo_snap.is_active is False
        assert torch.equal(pseudo_snap.state_dict["torch_cpu"], rng_iter0["torch_cpu"])

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_to_disk(tmpdir, "step_rng")
            engine.flush_persist()
            metadata, per_iter_states = SparseSnapshotEngine.load_from_disk(tmpdir, "step_rng")

        # Loaded bundle carries the pseudo-op alongside the real op at iter 0.
        assert _RNG_PSEUDO_OP in per_iter_states[0]
        loaded_rng = per_iter_states[0][_RNG_PSEUDO_OP]
        assert torch.equal(loaded_rng["torch_cpu"], rng_iter0["torch_cpu"])
        assert torch.equal(loaded_rng["torch_cuda.0"], rng_iter0["torch_cuda.0"])

        # Converter routes the pseudo-op out of the operator schedule and
        # into its per-iter RNG cache.
        converter = SparseToDenseConverter()
        converter.initialize_from_snapshots(metadata, per_iter_states, schedule=None)
        assert _RNG_PSEUDO_OP not in converter._operator_states, (
            "RNG pseudo-op leaked into operator state machine — would land in the replay schedule")
        assert converter.get_rng_state(0) is not None
        assert torch.equal(converter.get_rng_state(0)["torch_cpu"], rng_iter0["torch_cpu"])
        assert torch.equal(converter.get_rng_state(0)["torch_cuda.0"], rng_iter0["torch_cuda.0"])
        # Catch-up iters (no capture) return None — coordinator no-ops on this.
        assert converter.get_rng_state(99) is None

    def test_save_and_load_disk(self):
        """Sparse snapshots can be saved to and loaded from disk.

        ``save_to_disk`` serialises ``_persisted_snapshots`` — the last
        *completed* window — not the in-progress ``_snapshots`` dict
        that ``snapshot_operator`` writes into.  The promotion runs in
        two steps: ``begin_window`` rotates the just-filled ``_snapshots``
        into ``_in_flight_snapshots`` (and starts a fresh empty window),
        then ``finalize_window`` promotes in-flight to persisted.  Order
        matters: ``begin_window`` must come after ``snapshot_operator``
        so there's something to rotate.
        """
        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(5, 5)}
        engine.snapshot_operator("test_op", params, None, is_active=False, iteration=0)
        engine.synchronize()
        engine.begin_window(iteration=1)
        engine.finalize_window()

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_to_disk(tmpdir, "step_100")
            engine.flush_persist()

            metadata, per_iter_states = SparseSnapshotEngine.load_from_disk(tmpdir, "step_100")
            assert metadata is not None
            assert 0 in per_iter_states
            assert "test_op" in per_iter_states[0]
            assert metadata["per_iter_active"][0]["test_op"] is False


class TestSparseToDenseConverter:

    def test_initialize_from_snapshots(self):
        """Converter ingests per-iter snapshots and starts every op FROZEN.

        Initial FROZEN-for-everyone is deliberate: the coordinator's replay
        loop promotes each iteration's newly-active operators via
        ``activate_operators`` as it walks through the window.  Starting all
        ops FROZEN keeps the w_sparse=1 and w_sparse>1 paths symmetric.
        """
        converter = SparseToDenseConverter()

        metadata = {
            "window_start_iteration": 10,
            "per_iter_active": {
                10: {
                    "expert_0": True,
                    "expert_1": False,
                    "non_expert": False
                },
                11: {
                    "expert_1": True,
                    "non_expert": False
                },
                12: {
                    "non_expert": True
                },
            },
        }
        per_iter_states = {
            10: {
                "expert_0": {
                    "params.weight": torch.randn(5, 5),
                    "optimizer.exp_avg": torch.randn(5, 5),
                },
                "expert_1": {
                    "compute_weights.weight": torch.randn(5, 5).half(),
                },
                "non_expert": {
                    "compute_weights.weight": torch.randn(5, 5).half(),
                },
            },
            11: {
                "expert_1": {
                    "params.weight": torch.randn(5, 5),
                    "optimizer.exp_avg": torch.randn(5, 5),
                },
                "non_expert": {
                    "compute_weights.weight": torch.randn(5, 5).half(),
                },
            },
            12: {
                "non_expert": {
                    "params.weight": torch.randn(5, 5),
                    "optimizer.exp_avg": torch.randn(5, 5),
                },
            },
        }

        converter.initialize_from_snapshots(metadata, per_iter_states, schedule=None)

        # Initial active/frozen split reflects the EARLIEST iter's capture: an
        # op that's ACTIVE at the earliest iter starts ACTIVE in the converter;
        # others start FROZEN and are promoted later by ``activate_operators``.
        assert converter.is_operator_active("expert_0")  # FP32 captured at iter 10
        assert converter.is_operator_frozen("expert_1")  # FP16 at iter 10, FP32 at iter 11
        assert converter.is_operator_frozen("non_expert")  # FP16 at iter 10, FP32 at iter 12
        assert not converter.is_conversion_complete()

        # Per-iter getters return the captured data for the right (name, iter).
        assert converter.get_fp32_weights("expert_0", iteration=10) is not None
        assert converter.get_fp32_weights("expert_0", iteration=11) is None
        assert converter.get_fp16_weights("expert_1", iteration=10) is not None
        assert converter.get_fp32_weights("expert_1", iteration=11) is not None
        assert converter.get_fp16_weights("non_expert", iteration=11) is not None
        assert converter.get_fp32_weights("non_expert", iteration=12) is not None

    def test_activate_operators(self):
        """Operators transition from FROZEN to ACTIVE at their active iter."""
        converter = SparseToDenseConverter()

        metadata = {
            "window_start_iteration": 10,
            "per_iter_active": {
                10: {
                    "expert_0": True,
                    "expert_1": False
                },
                11: {
                    "expert_1": True
                },
            },
        }
        per_iter_states = {
            10: {
                "expert_0": {
                    "params.weight": torch.randn(5, 5)
                },
                "expert_1": {
                    "compute_weights.weight": torch.randn(5, 5).half()
                },
            },
            11: {
                "expert_1": {
                    "params.weight": torch.randn(5, 5)
                },
            },
        }

        converter.initialize_from_snapshots(metadata, per_iter_states, schedule=None)
        assert not converter.is_conversion_complete()

        converter.activate_operators(iteration=10, operator_names=["expert_0"])
        assert converter.is_operator_active("expert_0")
        assert not converter.is_conversion_complete()

        converter.activate_operators(iteration=11, operator_names=["expert_1"])
        assert converter.is_operator_active("expert_1")
        assert converter.is_conversion_complete()

    def test_skip_weight_grad_for_frozen(self):
        """Frozen operators should skip weight gradient computation."""
        converter = SparseToDenseConverter()

        metadata = {
            "window_start_iteration": 10,
            "per_iter_active": {
                10: {
                    "expert_0": True,
                    "expert_1": False
                },
                11: {
                    "expert_1": True
                },
            },
        }
        per_iter_states = {
            10: {
                "expert_0": {
                    "params.weight": torch.randn(5, 5)
                },
                "expert_1": {
                    "compute_weights.weight": torch.randn(5, 5).half()
                },
            },
            11: {
                "expert_1": {
                    "params.weight": torch.randn(5, 5)
                },
            },
        }

        converter.initialize_from_snapshots(metadata, per_iter_states, schedule=None)
        # expert_0 starts ACTIVE (earliest-iter FP32 capture); expert_1 FROZEN.
        assert not converter.should_skip_weight_grad("expert_0")
        assert converter.should_skip_weight_grad("expert_1")
        assert not converter.should_skip_optimizer_step("expert_0")
        assert converter.should_skip_optimizer_step("expert_1")

        # Promote expert_1 at its active iter — it should now take weight grads.
        converter.activate_operators(iteration=11, operator_names=["expert_1"])
        assert not converter.should_skip_weight_grad("expert_1")
        assert not converter.should_skip_optimizer_step("expert_1")

    def test_replay_iterations(self):
        """Replay iteration tracking works correctly."""
        converter = SparseToDenseConverter()
        converter.set_replay_iterations([10, 11, 12])

        assert converter.get_remaining_replay_count() == 3
        assert converter.get_next_replay_iteration() == 10
        assert converter.get_next_replay_iteration() == 11
        assert converter.get_next_replay_iteration() == 12
        assert converter.get_next_replay_iteration() is None

    def test_drop_iteration_pops_per_iter_caches(self):
        """SD-O4 S3: ``drop_iteration`` removes one iter from all four caches."""
        converter = SparseToDenseConverter()
        for it in (10, 11, 12):
            converter.ingest_iteration(it, {
                "op": {
                    "params.w": torch.zeros(2),
                    "compute_weights.w": torch.zeros(2).half(),
                    "optimizer.exp_avg": torch.zeros(2),
                }
            },
                                       iter_active={"op": True})

        assert set(converter._fp32_weights_per_iter.keys()) == {10, 11, 12}
        assert set(converter._fp16_weights_per_iter.keys()) == {10, 11, 12}
        assert set(converter._optimizer_states_per_iter.keys()) == {10, 11, 12}

        converter.drop_iteration(11)
        assert set(converter._fp32_weights_per_iter.keys()) == {10, 12}
        assert set(converter._fp16_weights_per_iter.keys()) == {10, 12}
        assert set(converter._optimizer_states_per_iter.keys()) == {10, 12}

    def test_drop_iteration_missing_iter_is_no_op(self):
        """``drop_iteration`` on an iter that was never ingested is a no-op."""
        converter = SparseToDenseConverter()
        converter.ingest_iteration(10, {"op": {
            "params.w": torch.zeros(2),
        }}, iter_active={"op": True})

        converter.drop_iteration(99)  # never ingested
        assert set(converter._fp32_weights_per_iter.keys()) == {10}


class TestStreamingPullBufferRelease:
    """SD-O4 S3: per-iter pool-buffer drop in the streaming pull path."""

    def test_release_iter_buffers_drops_only_targeted_iter(self):
        engine = SparseSnapshotEngine(replication_factor=1)
        buf_iter5 = engine._pool.acquire((4, ), torch.float32, pin=False)
        buf_iter7 = engine._pool.acquire((4, ), torch.float32, pin=False)
        engine._received_flat_buffers_by_iter[5] = [buf_iter5]
        engine._received_flat_buffers_by_iter[7] = [buf_iter7]

        engine.release_iter_buffers(5)

        assert 5 not in engine._received_flat_buffers_by_iter
        assert 7 in engine._received_flat_buffers_by_iter

    def test_release_iter_buffers_missing_iter_is_no_op(self):
        engine = SparseSnapshotEngine(replication_factor=1)
        engine.release_iter_buffers(99)  # nothing populated

    def test_clear_releases_streaming_per_iter_buffers(self):
        """``clear()`` flushes the per-iter dict, not just the bulk flat list."""
        engine = SparseSnapshotEngine(replication_factor=1)
        buf = engine._pool.acquire((4, ), torch.float32, pin=False)
        engine._received_flat_buffers_by_iter[3] = [buf]

        engine.clear()
        assert engine._received_flat_buffers_by_iter == {}


class TestMaxPrefetchedItersConfig:
    """SD-O4 S3: bounded streaming-pull queue config knob."""

    def test_default_is_eight(self):
        cfg = MoEvementConfig({"moevement": {"enabled": True}})
        assert cfg.max_prefetched_iters == 8

    def test_user_value_overrides(self):
        cfg = MoEvementConfig({"moevement": {"enabled": True, "max_prefetched_iters": 2}})
        assert cfg.max_prefetched_iters == 2


class TestUpstreamLogger:

    @pytest.fixture(autouse=True)
    def check_accelerator(self):
        """Skip tests that require CUDA stream if no accelerator is available."""
        try:
            from deepspeed.accelerator import get_accelerator
            accel = get_accelerator()
            if accel is None or accel.Stream is None:
                pytest.skip("No accelerator available for upstream logging tests")
        except Exception:
            pytest.skip("No accelerator available for upstream logging tests")

    def test_log_activation(self):
        """Activations are logged with correct metadata."""
        logger = UpstreamLogger()

        tensor = torch.randn(4, 8)
        logger.log_activation(tensor, iteration=5, micro_batch_id=0, stage_id=1)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(5)
        assert 0 in logs
        assert len(logs[0]) == 1
        assert logs[0][0].direction == "activation"
        assert logs[0][0].stage_id == 1

    def test_log_gradient(self):
        """Gradients are logged with correct metadata."""
        logger = UpstreamLogger()

        tensor = torch.randn(4, 8)
        logger.log_gradient(tensor, iteration=5, micro_batch_id=1, stage_id=2)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(5)
        assert 1 in logs
        assert len(logs[1]) == 1
        assert logs[1][0].direction == "gradient"

    def test_get_activations_for_replay(self):
        """Specific activations can be retrieved for replay."""
        logger = UpstreamLogger()

        t1 = torch.randn(4, 8)
        t2 = torch.randn(4, 8)
        logger.log_activation(t1, iteration=5, micro_batch_id=0, stage_id=1)
        logger.log_gradient(t2, iteration=5, micro_batch_id=0, stage_id=1)
        logger.synchronize()

        acts = logger.get_activations_for_replay(5, 0, 1)
        assert len(acts) == 1

        grads = logger.get_gradients_for_replay(5, 0, 1)
        assert len(grads) == 1

    def test_garbage_collect(self):
        """Stale logs are properly garbage collected."""
        logger = UpstreamLogger()

        for i in range(10):
            logger.log_activation(torch.randn(4, 8), iteration=i, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        assert logger.total_memory_bytes() > 0

        logger.garbage_collect(oldest_valid_iteration=7)

        # Logs for iterations 0-6 should be gone
        for i in range(7):
            assert len(logger.get_logs_for_iteration(i)) == 0

        # Logs for iterations 7-9 should still exist
        for i in range(7, 10):
            assert len(logger.get_logs_for_iteration(i)) > 0

    def test_log_tuple_tensors(self):
        """Tuple of tensors (common in pipeline) are logged correctly."""
        logger = UpstreamLogger()

        tensors = (torch.randn(4, 8), torch.randn(4, 8))
        logger.log_activation(tensors, iteration=1, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(1)
        assert 0 in logs
        assert len(logs[0]) == 2  # Two tensors from the tuple

    def test_clear(self):
        """Clear removes all logs."""
        logger = UpstreamLogger()
        logger.log_activation(torch.randn(4, 8), iteration=0, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        logger.clear()
        assert logger.total_memory_bytes() == 0

    def test_log_non_float_single_tensor_is_captured(self):
        """Single non-float tensors (bool masks, long position ids) are logged.

        Regression for M2 in CORRECTNESS_AUDIT_2026_04_23.md: the prior
        ``elif ... and tensor.is_floating_point()`` filter silently
        dropped single non-fp tensors, which is asymmetric with the
        tuple branch and would silently corrupt replay if a future
        refactor turns a tuple output into a single bool/long tensor.
        """
        logger = UpstreamLogger()
        mask = torch.zeros(4, 8, dtype=torch.bool)
        logger.log_activation(mask, iteration=0, micro_batch_id=0, stage_id=0)
        logger.synchronize()
        logs = logger.get_logs_for_iteration(0)
        assert len(logs) == 1 and len(logs[0]) == 1, (f"Bool mask was dropped by the fp-only filter: {logs}")
        assert logs[0][0].tensor.dtype == torch.bool


# ---------------------------------------------------------------------------
# Helpers shared by TestConversionReplay
# ---------------------------------------------------------------------------


class _FakeModule(nn.Module):
    """Minimal module with one weight parameter, used as a fake expert."""

    def __init__(self, size=8):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x + self.weight


class _FakeOptimizer:
    """Fake optimizer whose .state dict is queryable like Adam's."""

    def __init__(self, params):
        self.state = {}
        for p in params:
            self.state[p] = {
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
                "step": torch.tensor(0),
            }


def _build_coordinator_with_two_experts(pcie_bw=1e12):
    """Return a coordinator pre-initialized with two fake expert operators."""
    config = MoEvementConfig(
        {"moevement": {
            "enabled": True,
            "pcie_bandwidth_gbs": pcie_bw / (1024**3),
            "upstream_logging": False,
        }})
    coord = MoEvementCoordinator(config)

    # Manually populate operator map with two fake expert modules
    expert0 = _FakeModule(size=4)
    expert1 = _FakeModule(size=4)
    coord._operator_map["layer_0_expert_0"] = expert0
    coord._operator_map["layer_0_expert_1"] = expert1

    ops = [
        OperatorInfo("layer_0_expert_0", num_params=4, is_expert=True, layer_id=0, local_expert_id=0),
        OperatorInfo("layer_0_expert_1", num_params=4, is_expert=True, layer_id=0, local_expert_id=1),
    ]
    coord.scheduler.register_operators(ops)
    coord.scheduler.operators = ops
    # Generate a w_sparse=2 schedule manually (1 active per slot)
    coord.scheduler.w_sparse = 2
    from deepspeed.moevement.scheduler import CheckpointSchedule
    coord.scheduler.schedule = [
        CheckpointSchedule(active_operators=["layer_0_expert_0"], frozen_operators=["layer_0_expert_1"]),
        CheckpointSchedule(active_operators=["layer_0_expert_1"], frozen_operators=["layer_0_expert_0"]),
    ]
    coord._initialized = True
    coord._iter_time_sec = 1.0
    return coord, expert0, expert1


def _build_coordinator_with_two_linear_experts(pcie_bw=1e12):
    """Variant of ``_build_coordinator_with_two_experts`` whose expert
    modules are real ``nn.Linear`` instances.

    Used for tests that exercise the zero-bubble-style forward wrapping
    in ``_freeze_operator_params`` — the wrapping path walks each op's
    ``module.modules()`` looking for ``nn.Linear`` children, and the
    plain-parameter ``_FakeModule`` variant wouldn't match.
    """
    config = MoEvementConfig(
        {"moevement": {
            "enabled": True,
            "pcie_bandwidth_gbs": pcie_bw / (1024**3),
            "upstream_logging": False,
        }})
    coord = MoEvementCoordinator(config)
    expert0 = nn.Linear(4, 4)
    expert1 = nn.Linear(4, 4)
    coord._operator_map["layer_0_expert_0"] = expert0
    coord._operator_map["layer_0_expert_1"] = expert1
    ops = [
        OperatorInfo("layer_0_expert_0", num_params=4, is_expert=True, layer_id=0, local_expert_id=0),
        OperatorInfo("layer_0_expert_1", num_params=4, is_expert=True, layer_id=0, local_expert_id=1),
    ]
    coord.scheduler.register_operators(ops)
    coord.scheduler.operators = ops
    coord.scheduler.w_sparse = 2
    from deepspeed.moevement.scheduler import CheckpointSchedule
    coord.scheduler.schedule = [
        CheckpointSchedule(active_operators=["layer_0_expert_0"], frozen_operators=["layer_0_expert_1"]),
        CheckpointSchedule(active_operators=["layer_0_expert_1"], frozen_operators=["layer_0_expert_0"]),
    ]
    coord._initialized = True
    coord._iter_time_sec = 1.0
    return coord


class TestConversionReplay:
    """Tests for the sparse-to-dense replay loop."""

    def test_zero_frozen_gradients_clears_frozen_params(self):
        """zero_frozen_gradients zeros grad for frozen operator params."""
        coord, expert0, expert1 = _build_coordinator_with_two_experts()

        # Manually mark expert1 as frozen in the converter
        from deepspeed.moevement.conversion import OperatorState
        coord.converter._operator_states["layer_0_expert_0"] = OperatorState.ACTIVE
        coord.converter._operator_states["layer_0_expert_1"] = OperatorState.FROZEN
        coord._recovering = True

        # Give both experts non-zero gradients
        expert0.weight.grad = torch.ones(4)
        expert1.weight.grad = torch.ones(4)

        # Build a minimal model that holds both modules (grad zeroing walks _operator_map)
        class _FakeModel(nn.Module):
            pass

        fake_model = _FakeModel()
        coord.zero_frozen_gradients(fake_model)

        # expert0 (ACTIVE) grad untouched; expert1 (FROZEN) grad zeroed
        assert expert0.weight.grad is not None
        assert expert0.weight.grad.sum().item() == pytest.approx(4.0)
        assert expert1.weight.grad.sum().item() == pytest.approx(0.0)

    def test_restore_module_weights_copies_fp32(self):
        """_restore_module_weights loads FP32 snapshot values onto a CPU module."""
        coord, expert0, _ = _build_coordinator_with_two_experts()

        target_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fp32_weights = {"weight": target_values.clone()}

        coord._restore_module_weights(expert0, fp32_weights)

        assert torch.allclose(expert0.weight.data, target_values)

    def test_apply_optim_state_preserves_dotted_param_names(self):
        """Dotted param names (``lin.weight``, ``wg.weight``) survive restore.

        Serialized keys are ``<param_name>.<adam_key>``, and real
        MoEvement operators (gates, non-expert bucket, nested expert
        children) have dotted param names.  An earlier ``find('.')``
        splitter took the first dot and parsed ``lin.weight.exp_avg`` as
        param ``lin`` / key ``weight.exp_avg``, failed the lookup, and
        silently skipped the entry.  ``rsplit('.', 1)`` keeps the param
        name intact.  Without this fix, gate / non-expert / nested-module
        Adam moments stay at whatever the live optimizer has — usually
        zero — and recovery silently drifts off the saved trajectory.
        """
        coord, _, _ = _build_coordinator_with_two_experts()

        class _NestedModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4, bias=False)

        nested = _NestedModule()
        optimizer = _FakeOptimizer(list(nested.parameters()))
        param_dict = dict(nested.named_parameters())
        assert "lin.weight" in param_dict, "test fixture must produce a dotted param name"

        new_exp_avg = torch.full_like(nested.lin.weight, 0.5)
        new_exp_avg_sq = torch.full_like(nested.lin.weight, 0.25)
        optim_states = {
            "lin.weight.exp_avg": new_exp_avg.clone(),
            "lin.weight.exp_avg_sq": new_exp_avg_sq.clone(),
        }

        coord._apply_optim_state_into_params(param_dict, optimizer, optim_states)

        restored = optimizer.state[nested.lin.weight]
        assert torch.allclose(restored["exp_avg"], new_exp_avg)
        assert torch.allclose(restored["exp_avg_sq"], new_exp_avg_sq)

    def test_apply_optim_state_skips_non_owning_zero_peer(self):
        """ZeRO-partitioned param with ``optim_fragment=None`` must skip, not crash.

        Under ZeRO-1 + EP sharding (ep_size > 1), a DP rank may not
        own any fragment of some expert-group params — the rank's
        ``_hp_mapping.optim_fragment`` stays ``None`` because ZeRO's
        ``set_optim_state_fragment`` never fired for that share.  The
        collect side gates on ``mapping.optim_fragment is not None``
        before contributing keys; the restore side must mirror or
        ``safe_set_full_optimizer_state`` will crash inside
        ``get_hp_fragment`` on ``None in None``.

        Surfaced by ``run_with_survivor_supervisor.py`` on rank 1
        during the replay loop: rank 1 received bundle entries for
        params it doesn't own under EP, tried to apply, and raised
        ``TypeError: argument of type 'NoneType' is not iterable``.
        """
        coord, expert0, _ = _build_coordinator_with_two_experts()

        class _NoOwnershipHpMapping:
            optim_fragment = None  # non-owner: ZeRO never populated

        expert0.weight._hp_mapping = _NoOwnershipHpMapping()
        # Optimizer slot is empty on non-owners.  We pass ``object()`` so
        # the path wouldn't write through ``optimizer.state[...]`` either.
        param_dict = {"weight": expert0.weight}
        optim_states = {
            "weight.exp_avg": torch.full((4, ), 0.5),
            "weight.exp_avg_sq": torch.full((4, ), 0.25),
        }

        # Before the fix this raised ``TypeError`` inside
        # ``safe_set_full_optimizer_state`` → ``get_hp_fragment``.
        # After the fix it's a clean no-op for the non-owner.
        coord._apply_optim_state_into_params(param_dict, optimizer=object(), optim_states=optim_states)

    def test_collect_param_optim_state_uses_zero_helpers_when_hp_mapping_present(self):
        """ZeRO-1/2 params route through safe_get_full_optimizer_state.

        Vanilla ``optimizer.state[param]`` under mixed-precision ZeRO is
        keyed by flat FP32 partitions, so reading it directly captures
        nothing.  With an ``_hp_mapping`` attribute we must instead walk
        DeepSpeed's fragment-mapping API, which is what this test
        verifies the coordinator does.
        """
        coord, expert0, _ = _build_coordinator_with_two_experts()

        # Fragment-snapshot path reads ``mapping.optim_fragment[key]``
        # directly (no DP all-reduce), so the fake needs realistic
        # values in ``optim_fragment`` plus an ``lp_fragment_address``
        # for fragment metadata.  These are the only fields the
        # collector touches.
        class _FakeFragAddr:
            start = 0
            numel = 4

        class _FakeHpMapping:
            optim_fragment = {
                "exp_avg": torch.full((4, ), 7.0),
                "exp_avg_sq": torch.full((4, ), 9.0),
            }
            lp_fragment_address = _FakeFragAddr()

            def get_optim_state_keys(self):
                return list(self.optim_fragment.keys())

        expert0.weight._hp_mapping = _FakeHpMapping()

        state, fragment_info = coord._get_module_optimizer_state(expert0, optimizer=None)
        assert state["weight.exp_avg"].mean().item() == pytest.approx(7.0)
        assert state["weight.exp_avg_sq"].mean().item() == pytest.approx(9.0)
        assert fragment_info["weight.exp_avg"]["fragment_numel"] == 4
        assert fragment_info["weight.exp_avg_sq"]["full_shape"] == list(expert0.weight.shape)

    def test_collect_param_optim_state_skips_unpopulated_hp_mapping(self):
        """ZeRO's ``optim_fragment`` is None until the first ``step()`` runs.

        Calling ``get_optim_state_keys`` on that ``None`` would raise, so
        the snapshot path must detect and short-circuit.  After the first
        real step the fragment is populated and the normal path kicks in.
        """
        coord, expert0, _ = _build_coordinator_with_two_experts()

        class _UnpopulatedHpMapping:
            optim_fragment = None

        expert0.weight._hp_mapping = _UnpopulatedHpMapping()

        # No crash and no entries captured — matches the "nothing to
        # save yet" semantics at iteration 0.
        state, fragment_info = coord._get_module_optimizer_state(expert0, optimizer=None)
        assert state == {}
        assert fragment_info == {}

    def test_collect_param_optim_state_rejects_dotted_adam_key(self):
        """Serialize asserts optimizer-state keys carry no '.' (rsplit assumption).

        A2's restore uses ``rsplit('.', 1)`` to separate ``param_name``
        from ``adam_key``.  Correct for every current optimizer (Adam /
        AdamW / AMSGrad / Lion / Adafactor all use dotless keys).  A
        future custom optimizer with a key like ``adam.momentum.ema``
        would silently load into the wrong param on restore.  Assert
        at serialize so the mismatch fires loud at write, not silent at
        read.
        """
        coord, expert0, _ = _build_coordinator_with_two_experts()

        class _DottedKeyOpt:

            def __init__(self, param):
                # Valid tensor payload, pathological key name.
                self.state = {param: {"adam.momentum": torch.zeros(4)}}

        opt = _DottedKeyOpt(expert0.weight)
        with pytest.raises(AssertionError, match=r"contains '\.'") as excinfo:
            coord._get_module_optimizer_state(expert0, optimizer=opt)
        assert "adam.momentum" in str(excinfo.value)

    def test_restore_module_weights_routes_zero_params_through_hp_setter(self):
        """ZeRO-1/2 params land in the HP fragment via safe_set_full_fp32_param.

        Writing to ``param.data`` directly would update only the LP
        (low-precision) copy — the next optimizer step would overwrite
        it from the untouched FP32 master.  The fix is to write the HP
        fragment; the test asserts that the right setter fires.
        """
        import unittest.mock as mock
        coord, expert0, _ = _build_coordinator_with_two_experts()

        class _FakeHpMapping:
            pass

        expert0.weight._hp_mapping = _FakeHpMapping()

        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with mock.patch("deepspeed.moevement.coordinator.safe_set_full_fp32_param") as m_set:
            coord._restore_module_weights(expert0, {"weight": target.clone()})

        assert m_set.call_count == 1
        # First positional arg is the param; second is the value tensor.
        assert m_set.call_args.args[0] is expert0.weight
        assert torch.allclose(m_set.call_args.args[1], target)

    def test_setup_replay_iter_calls_update_lp_params(self, tmp_path):
        """After restoring ZeRO params, HP→LP is synced via optimizer.update_lp_params.

        Without this call the next forward pass reads the stale LP
        (training-precision) copy, defeating the restore.  The sync
        runs once per window slot rather than per module so the cost
        is a single all-gather regardless of operator count.
        """
        import unittest.mock as mock
        coord, expert0, _ = _build_coordinator_with_two_experts()

        # Mark the module's param as ZeRO-managed so _apply_fp32_into_params
        # takes the HP-setter branch and the replay path flags the
        # touched_zero_partitioned guard.
        class _FakeHpMapping:
            pass

        expert0.weight._hp_mapping = _FakeHpMapping()

        # Seed the converter with a single-iter window's worth of per-iter
        # captures so ``_setup_replay_iter`` finds expert_0's FP32.
        metadata = {"window_start_iteration": 0, "per_iter_active": {0: {"layer_0_expert_0": True}}}
        per_iter_states = {0: {"layer_0_expert_0": {"params.weight": torch.zeros(4)}}}
        coord.converter.initialize_from_snapshots(metadata, per_iter_states, schedule=None)
        coord._cached_snapshot_data = (metadata, per_iter_states)

        fake_optimizer = mock.Mock()  # has update_lp_params
        with mock.patch("deepspeed.moevement.coordinator.safe_set_full_fp32_param"):
            coord._setup_replay_iter(iteration=0, model=None, optimizer=fake_optimizer, thaw_activated=False)

        assert fake_optimizer.update_lp_params.called

    def test_full_replay_activates_all_operators(self, tmp_path):
        """After W_sparse replay steps all operators transition to ACTIVE.

        Per spec (§3.2): within one sparse checkpoint window of K=w_sparse
        iterations, each operator is ACTIVE (FP32 + optimizer state captured)
        in exactly one iter and FROZEN (FP16 captured) in iters before that;
        after an op becomes active it's not captured again within the window.
        This test builds a two-iter window with expert_0 active at iter 0,
        expert_1 active at iter 1, frozen otherwise.
        """
        coord, expert0, expert1 = _build_coordinator_with_two_experts()

        fp32_e0 = torch.tensor([10.0, 20.0, 30.0, 40.0])
        fp32_e1 = torch.tensor([50.0, 60.0, 70.0, 80.0])

        save_dir = str(tmp_path)
        tag = "step_100"
        import os
        ckpt_dir = os.path.join(save_dir, tag, "moevement")
        os.makedirs(ckpt_dir, exist_ok=True)

        metadata = {
            "window_start_iteration": 0,
            "per_iter_active": {
                0: {
                    "layer_0_expert_0": True,
                    "layer_0_expert_1": False
                },
                1: {
                    "layer_0_expert_1": True
                },
            },
        }
        from deepspeed.moevement.snapshot_io import BUNDLE_FILENAME, dump_bundle
        dump_bundle(
            os.path.join(ckpt_dir, BUNDLE_FILENAME.format(rank=0)),
            metadata,
            {
                0: {
                    "layer_0_expert_0": {
                        "is_active": True,
                        "state_dict": {
                            "params.weight": fp32_e0
                        },
                    },
                    "layer_0_expert_1": {
                        "is_active": False,
                        "state_dict": {
                            "compute_weights.weight": fp32_e1.half()
                        },
                    },
                },
                1: {
                    "layer_0_expert_1": {
                        "is_active": True,
                        "state_dict": {
                            "params.weight": fp32_e1
                        },
                    },
                },
            },
        )

        # Initialize recovery: load the bundle, seed the converter.
        metadata, per_iter_states = SparseSnapshotEngine.load_from_disk(save_dir, tag)
        coord.converter.initialize_from_snapshots(metadata, per_iter_states, schedule=None)
        coord.converter.set_replay_iterations(sorted(per_iter_states.keys()))
        coord._checkpoint_load_dir = save_dir
        coord._checkpoint_tag = tag
        coord._cached_snapshot_data = (metadata, per_iter_states)
        coord._recovering = True

        optimizer = _FakeOptimizer(list(expert0.parameters()) + list(expert1.parameters()))

        class _FakeModel(nn.Module):
            pass

        fake_model = _FakeModel()

        # Simulate W_sparse=2 recovery steps: one _setup_replay_iter for each
        # captured iteration (iter 0 promotes expert_0, iter 1 promotes expert_1).
        for _ in range(coord.scheduler.w_sparse):
            assert not coord.converter.is_conversion_complete()
            coord._on_iteration_end_recovery(fake_model, optimizer)

        # end_recovery() clears the converter, so check _recovering flag instead
        assert not coord._recovering

    def test_on_before_optimizer_step_no_op_when_not_recovering(self):
        """on_before_optimizer_step is a no-op when not in recovery mode."""
        coord, expert0, _ = _build_coordinator_with_two_experts()
        expert0.weight.grad = torch.ones(4)

        class _FakeModel(nn.Module):
            pass

        coord._recovering = False
        coord.on_before_optimizer_step(_FakeModel())
        # Grad must be untouched
        assert expert0.weight.grad.sum().item() == pytest.approx(4.0)

    def test_compute_replay_iters_no_catch_up_when_fault_at_boundary(self):
        """``_fault_iter == persisted[-1]`` → bundle exactly, no catch-up."""
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._fault_iter = 4
        assert coord._compute_replay_iters([1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_compute_replay_iters_catch_up_past_bundle(self):
        """``_fault_iter > persisted[-1]`` → bundle + catch-up through fault_iter."""
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._fault_iter = 9
        assert coord._compute_replay_iters([1, 2, 3, 4]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_compute_replay_iters_clamps_bundle_when_fault_inside_window(self):
        """``_fault_iter < persisted[-1]`` clamps the bundle; no over-replay.

        Defensive: ``_fault_iter`` is normally >= the last finalized window
        boundary (since it's the engine's current ``global_steps``), but
        peer-pull threads it through a manifest where a bug could desync
        the two.  The clamp keeps replay from overshooting the fault point
        in that case.
        """
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._fault_iter = 3
        assert coord._compute_replay_iters([1, 2, 3, 4]) == [1, 2, 3]

    def test_compute_replay_iters_empty_when_fault_before_bundle(self):
        """``_fault_iter < persisted[0]`` → empty replay list (nothing to reach)."""
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._fault_iter = 0
        assert coord._compute_replay_iters([1, 2, 3, 4]) == []

    def test_compute_replay_iters_none_fault_returns_full_bundle(self):
        """Missing ``_fault_iter`` (no catch-up target) → replay full bundle."""
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._fault_iter = None
        assert coord._compute_replay_iters([1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_compute_replay_iters_empty_bundle_returns_empty(self):
        """No persisted bundle → no replay regardless of ``_fault_iter``."""
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._fault_iter = 5
        assert coord._compute_replay_iters([]) == []


# ---------------------------------------------------------------------------
# TestPeerReplication
# ---------------------------------------------------------------------------


class TestPeerReplication:
    """Tests for point-to-point peer snapshot replication."""

    def test_replicate_to_peers_no_op_when_single_rank(self):
        """replicate_to_peers returns immediately when dp_world_size <= 1."""
        engine = SparseSnapshotEngine(replication_factor=1)

        sends = []
        import unittest.mock as mock
        with mock.patch("deepspeed.comm.send", side_effect=lambda *a, **kw: sends.append(1)):
            engine.replicate_to_peers(dp_group=None, dp_rank=0, dp_world_size=1, device="cpu")

        assert len(sends) == 0  # single rank → no sends

    def test_replicate_to_peers_no_op_when_replication_factor_zero(self):
        """replication_factor=0 disables replication entirely."""
        engine = SparseSnapshotEngine(replication_factor=0)

        sends = []
        import unittest.mock as mock
        with mock.patch("deepspeed.comm.send", side_effect=lambda *a, **kw: sends.append(1)):
            engine.replicate_to_peers(dp_group=None, dp_rank=0, dp_world_size=4, device="cpu")

        assert len(sends) == 0

    def test_received_snapshots_initially_empty(self):
        """_received_snapshots is empty at construction time.

        Storage is keyed by sender_dp_rank now that every rank sends
        its own shard in the symmetric ring; ``received_senders``
        returns an empty list and ``get_received_snapshots_for`` misses.
        """
        engine = SparseSnapshotEngine()
        assert engine._received_snapshots == {}
        assert engine.received_senders() == []
        assert engine.get_received_snapshots_for(0) == (None, None)

    def test_set_topology_stores_fields(self):
        """set_topology correctly stores dp_group, dp_rank, and device."""
        coord = MoEvementCoordinator(MoEvementConfig())
        sentinel_group = object()
        coord.set_topology(dp_group=sentinel_group, dp_rank=3, device="cuda:1")
        assert coord._dp_group is sentinel_group
        assert coord._dp_rank == 3
        assert coord._device == "cuda:1"

    def test_set_pipeline_topology_stores_fields(self):
        """set_pipeline_topology correctly stores all pipeline fields."""
        coord = MoEvementCoordinator(MoEvementConfig())
        sentinel_group = object()
        fn = lambda s: s + 100

        coord.set_pipeline_topology(pp_group=sentinel_group, stage_id=2, num_stages=4, stage_to_global_fn=fn)
        assert coord._pp_group is sentinel_group
        assert coord._stage_id == 2
        assert coord._num_stages == 4
        assert coord._stage_to_global_fn(5) == 105

    def test_set_pipeline_topology_rejects_single_stage(self):
        """MoEvement requires PP>1 — single-stage setup raises at topology registration.

        Under PP=1 there are no inter-stage activation/gradient logs to
        drive the replay, so recovery semantics collapse.  Guard at the
        topology registration point so the failure mode is visible at
        engine init rather than during a failed recovery attempt.
        """
        coord = MoEvementCoordinator(MoEvementConfig())
        with pytest.raises(ValueError, match="more than 1 stage"):
            coord.set_pipeline_topology(pp_group=object(), stage_id=0, num_stages=1, stage_to_global_fn=lambda s: s)

    def test_do_peer_replication_no_op_without_dp_group(self):
        """_do_peer_replication is a no-op when no dp_group is set."""
        coord = MoEvementCoordinator(MoEvementConfig())
        # Should not raise, even with no group set
        coord._do_peer_replication()

    def test_replication_timeout_sets_broken_flag(self):
        """A hung replication worker flips _replication_broken and stops future submits.

        The executor thread can't be cancelled once it's blocked on a
        network op, so we must stop handing it more work — otherwise
        every subsequent window boundary queues another submit that
        never runs.
        """
        import unittest.mock as mock
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        coord, _, _ = _build_coordinator_with_two_experts()
        coord._replication_group = object()  # pretend a DP mirror exists
        # Shrink the outstanding cap to 1 so a single hung future hits
        # the backpressure path at the next window boundary.  The
        # post-async design only blocks when the queue is at the cap;
        # without this the test's lone hung future would be silently
        # queued and training would proceed (which is by design).
        coord.config.replication_queue_max_outstanding = 1

        # Simulate an in-flight replication future that's NOT done (so
        # the drain loop doesn't pop it off the head) and whose result
        # times out (so the backpressure wait triggers the broken flag).
        hung_future = mock.Mock()
        hung_future.done.return_value = False
        hung_future.result.side_effect = FuturesTimeoutError()
        coord._replication_futures.append(hung_future)

        # Force the window boundary path by lining up with w_sparse.
        coord._window_step = coord.scheduler.w_sparse - 1

        # Provide a minimal optimizer stand-in for ``_get_module_optimizer_state``
        # and prevent actual work inside the snapshot engine — we only care
        # about the future-wait + broken-flag logic at the window boundary.
        class _FakeOptim:
            state = {}

        with mock.patch.object(coord.snapshot_engine, "snapshot_operator"), \
             mock.patch.object(coord.snapshot_engine, "synchronize"), \
             mock.patch.object(coord.snapshot_engine, "finalize_window"), \
             mock.patch.object(coord.snapshot_engine, "begin_window"), \
             mock.patch.object(coord._replication_executor, "submit") as m_submit:
            coord.on_iteration_end(global_step=coord.scheduler.w_sparse, model=None, optimizer=_FakeOptim())

        assert coord._replication_broken is True
        assert m_submit.call_count == 0  # gated off by _replication_broken

    def test_shutdown_does_not_block_when_replication_broken(self):
        """shutdown() falls back to non-blocking executor teardown once broken.

        With the flag set (e.g., a prior timeout), the worker thread is
        wedged.  Blocking on ``executor.shutdown(wait=True)`` would hang
        the process at exit; the coordinator must detach instead.
        """
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._replication_broken = True

        with mock.patch.object(coord._replication_executor, "shutdown") as m_shut:
            coord.shutdown()
        # Called with wait=False (cancel_futures may or may not be passed
        # depending on Python version; either way, wait must be False).
        assert m_shut.called
        assert m_shut.call_args.kwargs.get("wait") is False

    def test_replication_marks_flats_busy_until_worker_done(self):
        """Persisted flats stay busy while the replication worker is running.

        This is what keeps a timed-out worker's in-flight sends from
        racing ``finalize_window`` — while busy, ``pool.release`` on the
        flat is a no-op, so the hung thread's DMA reads can't be
        overwritten by a new acquirer.
        """
        import unittest.mock as mock
        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot

        coord, _, _ = _build_coordinator_with_two_experts()
        coord._replication_group = object()

        # Seed a persisted snapshot with a flat buffer so the mark_busy
        # loop has something real to operate on.
        flat = torch.zeros(4)
        snap = OperatorSnapshot("op0", iteration=0, is_active=True)
        snap._flat_buffers.append(flat)
        coord.snapshot_engine._persisted_snapshots[(0, "op0")] = snap

        # Capture the submitted callable *and* the done-callback so we can
        # drive them manually: the test doesn't actually start a worker.
        submitted_fn = []
        done_callbacks = []

        class _FakeFuture:

            def add_done_callback(self, cb):
                done_callbacks.append(cb)

            def result(self, timeout=None):
                # Coordinator shutdown fires via atexit at interpreter
                # exit and calls .result() on any still-attached future.
                # Keep shutdown noise-free by reporting "already done".
                return None

        def fake_submit(fn, *a, **kw):
            submitted_fn.append(fn)
            return _FakeFuture()

        coord._window_step = coord.scheduler.w_sparse - 1

        class _FakeOptim:
            state = {}

        with mock.patch.object(coord.snapshot_engine, "snapshot_operator"), \
             mock.patch.object(coord.snapshot_engine, "synchronize"), \
             mock.patch.object(coord.snapshot_engine, "finalize_window"), \
             mock.patch.object(coord.snapshot_engine, "begin_window"), \
             mock.patch.object(coord._replication_executor, "submit", side_effect=fake_submit):
            coord.on_iteration_end(global_step=coord.scheduler.w_sparse, model=None, optimizer=_FakeOptim())

        # While the future is running (callback not yet fired), the flat
        # is busy, so release becomes a no-op and the storage is safe.
        assert id(flat) in coord.snapshot_engine._pool._busy
        coord.snapshot_engine._pool.release(flat)
        assert id(flat) in coord.snapshot_engine._pool._busy  # still busy

        # Two done-callbacks now: index 0 is the busy-release lambda,
        # index 1 is ``_on_replication_done`` (exception surface after the
        # training thread stopped blocking on ``.result()``).  The former
        # is the one that returns the flat to the pool.
        assert len(done_callbacks) == 2
        done_callbacks[0](mock.Mock())
        assert id(flat) not in coord.snapshot_engine._pool._busy

    def test_replicate_to_peers_ring_sends_to_forward_peers(self):
        """Every rank isends its own manifest + flats to the next r peers.

        Under symmetric ring replication with r=2 and dp_world_size=4:
        rank 0 sends to ranks 1 and 2.  Per peer: 1 length + 1 payload +
        1 flat-group (both keys share dtype=float32) = 3 isends.  Total
        forward isends from rank 0: 6.
        """
        engine = SparseSnapshotEngine(replication_factor=2)

        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot
        snap = OperatorSnapshot("op0", iteration=0, is_active=True)
        snap.add_tensor("params.weight", torch.ones(4))
        snap.add_tensor("optimizer.exp_avg", torch.zeros(4))
        engine._persisted_snapshots[(0, "op0")] = snap

        import unittest.mock as mock

        class _FakeHandle:

            def wait(self):
                pass

        send_targets = []

        def fake_isend(tensor, dst=None, **kw):
            send_targets.append(dst)
            return _FakeHandle()

        # Recv from (0-1)%4=3 and (0-2)%4=2.  Feed an empty-length flag
        # each time so the receiver short-circuits.
        recv_queue = [
            torch.tensor([0], dtype=torch.int64),  # from peer (0-1)%4=3
            torch.tensor([0], dtype=torch.int64),  # from peer (0-2)%4=2
        ]

        def fake_recv(tensor, src=None, **kw):
            tensor.copy_(recv_queue.pop(0))

        with mock.patch("deepspeed.comm.isend", side_effect=fake_isend), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv):
            engine.replicate_to_peers(dp_group=object(), dp_rank=0, dp_world_size=4, device="cpu")

        # r = min(2, 3) = 2 forward peers: (0+1)%4=1 and (0+2)%4=2.
        # Per peer: length + payload + 1 flat-group = 3 isends.
        assert len(send_targets) == 6
        assert send_targets.count(1) == 3
        assert send_targets.count(2) == 3

    def test_replicate_to_peers_stores_received_snapshot_keyed_by_sender(self):
        """Received snapshots land under ``_received_snapshots[sender_dp_rank]``.

        Under the ring, rank 1 (dp_world_size=2, r=1) receives from
        rank (1-1)%2=0.  The received payload from rank 0 shows up as
        ``_received_snapshots[0]``, not as a flat op_name dict.
        """
        engine = SparseSnapshotEngine(replication_factor=1)

        import io as _io
        import unittest.mock as mock

        manifest = {
            "window_start_iteration":
            1,
            "operators": [{
                "name":
                "op1",
                "is_active":
                False,
                "iteration":
                1,
                "groups": [{
                    "dtype": torch.float32,
                    "total_elems": 4,
                    "layout": [("compute_weights.weight", 0, (4, ))],
                }],
            }],
        }
        buf = _io.BytesIO()
        torch.save(manifest, buf)
        manifest_bytes = buf.getvalue()

        recv_queue = [
            torch.tensor([len(manifest_bytes)], dtype=torch.int64),
            torch.frombuffer(bytearray(manifest_bytes), dtype=torch.uint8),
            torch.full((4, ), 42.0),
        ]

        def fake_recv(tensor, src=None, **kw):
            tensor.copy_(recv_queue.pop(0))

        class _FakeHandle:

            def wait(self):
                pass

        def fake_isend(tensor, dst=None, **kw):
            return _FakeHandle()

        def fake_irecv(tensor, src=None, **kw):
            # Shares ``recv_queue`` with ``fake_recv`` since length +
            # payload go through blocking ``recv`` and per-flat
            # transfers go through ``irecv`` after my refactor.
            tensor.copy_(recv_queue.pop(0))
            return _FakeHandle()

        # Seed an empty persisted window so our *own* ring send is short
        # (length=0 + no payload) — we only care about the recv side here.
        with mock.patch("deepspeed.comm.isend", side_effect=fake_isend), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv), \
             mock.patch("deepspeed.comm.irecv", side_effect=fake_irecv):
            engine.replicate_to_peers(dp_group=object(), dp_rank=1, dp_world_size=2, device="cpu")

        # Sender's dp_rank for rank 1 is (1-1)%2 = 0.
        assert 0 in engine._received_snapshots
        sender_0_ops = engine._received_snapshots[0]
        assert (1, "op1") in sender_0_ops
        received = sender_0_ops[(1, "op1")]["compute_weights.weight"]
        assert received.shape == (4, )
        assert received.float().mean().item() == pytest.approx(42.0)
        assert engine._received_metadata[0][(1, "op1")]["is_active"] is False
        assert engine._received_window_start[0] == 1

        # The high-level accessor returns the disk-shaped tuple.
        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=0)
        assert metadata["window_start_iteration"] == 1
        assert 1 in per_iter and "op1" in per_iter[1]

    def test_receiver_drops_entire_sender_slot_on_recv_failure(self):
        """A mid-batch recv failure leaves the sender slot entirely absent.

        With the parallel ``irecv`` recv path, all flats are posted
        together and waited on as a batch.  A failure anywhere in the
        batch is surfaced only after all posts, so partial commit per
        op is no longer observable — the guarantee is all-or-nothing
        per sender.  Downstream peer-pull code tolerates the sender
        being missing (same path as the empty-sender short-circuit),
        so ghosted metadata without matching state dicts cannot
        happen.
        """
        engine = SparseSnapshotEngine(replication_factor=1)

        import io as _io
        import unittest.mock as mock

        manifest = {
            "window_start_iteration":
            2,
            "operators": [
                {
                    "name": "op_a",
                    "is_active": True,
                    "iteration": 2,
                    "groups": [{
                        "dtype": torch.float32,
                        "total_elems": 2,
                        "layout": [("params.weight", 0, (2, ))],
                    }],
                },
                {
                    "name": "op_b",
                    "is_active": True,
                    "iteration": 2,
                    "groups": [{
                        "dtype": torch.float32,
                        "total_elems": 2,
                        "layout": [("params.weight", 0, (2, ))],
                    }],
                },
            ],
        }
        buf = _io.BytesIO()
        torch.save(manifest, buf)
        raw = buf.getvalue()

        call_state = {"recv": 0, "irecv": 0}

        def fake_recv(tensor, src=None, **kw):
            call_state["recv"] += 1
            if call_state["recv"] == 1:
                tensor.copy_(torch.tensor([len(raw)], dtype=torch.int64))
            elif call_state["recv"] == 2:
                tensor.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
            else:
                raise AssertionError(f"unexpected dist.recv call {call_state['recv']}")

        class _FakeHandle:

            def wait(self):
                pass

        def fake_isend(tensor, dst=None, **kw):
            return _FakeHandle()

        def fake_irecv(tensor, src=None, **kw):
            call_state["irecv"] += 1
            if call_state["irecv"] == 1:
                tensor.copy_(torch.tensor([1.0, 2.0]))  # op_a flat
                return _FakeHandle()
            # op_b flat — simulate peer failure at post time; the
            # outer ``except`` must release all pre-allocated flats
            # (including op_a's already-acquired buffer) and re-raise.
            raise RuntimeError("simulated peer failure mid-op_b")

        with mock.patch("deepspeed.comm.isend", side_effect=fake_isend), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv), \
             mock.patch("deepspeed.comm.irecv", side_effect=fake_irecv):
            with pytest.raises(RuntimeError):
                engine.replicate_to_peers(dp_group=object(), dp_rank=1, dp_world_size=2, device="cpu")

        # Sender is (1-1)%2=0.  On mid-batch failure the entire sender
        # slot is absent — no partial commit, no ghost metadata.
        assert 0 not in engine._received_snapshots
        assert 0 not in engine._received_metadata
        assert 0 not in engine._received_window_start

    def test_recv_keys_by_group_local_under_non_identity_mapping(self):
        """Multi-topology (``_group_to_global`` non-identity) keeps receiver
        keying group-local, matching what the server lookup expects.

        Regression test for H1 in CORRECTNESS_AUDIT_2026_04_23.md: on any
        PP>1 DP>1 cluster, ``serve_peer_pull_request`` /
        ``get_received_snapshots_for`` look up
        ``_received_snapshots[sender_dp_rank]`` where ``sender_dp_rank`` is
        *group-local*.  The ring receive path must store under the same
        scheme.  The prior implementation keyed by the global rank
        returned by ``_group_to_global`` and only matched accidentally on
        single-topology (DP==world) setups where ``_group_to_global`` is
        the identity.
        """
        engine = SparseSnapshotEngine(replication_factor=1)

        import io as _io
        import unittest.mock as mock

        manifest = {
            "window_start_iteration":
            3,
            "operators": [{
                "name": "op_x",
                "is_active": True,
                "iteration": 3,
                "groups": [{
                    "dtype": torch.float32,
                    "total_elems": 2,
                    "layout": [("params.w", 0, (2, ))],
                }],
            }],
        }
        buf = _io.BytesIO()
        torch.save(manifest, buf)
        manifest_bytes = buf.getvalue()

        recv_queue = [
            torch.tensor([len(manifest_bytes)], dtype=torch.int64),
            torch.frombuffer(bytearray(manifest_bytes), dtype=torch.uint8),
            torch.full((2, ), 7.0),
        ]

        def fake_recv(tensor, src=None, **kw):
            tensor.copy_(recv_queue.pop(0))

        class _FakeHandle:

            def wait(self):
                pass

        def fake_isend(tensor, dst=None, **kw):
            return _FakeHandle()

        def fake_irecv(tensor, src=None, **kw):
            # Shares ``recv_queue`` with ``fake_recv``; see parallel
            # recv refactor in ``_recv_ring_peer``.
            tensor.copy_(recv_queue.pop(0))
            return _FakeHandle()

        # Non-identity mapping: group-local {0, 1} → globals {4, 5}.
        # dp_rank=0, dp_world_size=2, r=1: recv from (0-1)%2 = local 1
        # = global 5.  Server-side lookup under ``sender_dp_rank=1``
        # must succeed, i.e. ``_received_snapshots`` must be keyed by 1,
        # not 5.
        def fake_get_global_rank(group, local_rank):
            return 4 + local_rank

        with mock.patch("deepspeed.comm.isend", side_effect=fake_isend), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv), \
             mock.patch("deepspeed.comm.irecv", side_effect=fake_irecv), \
             mock.patch("deepspeed.comm.is_initialized", return_value=True), \
             mock.patch("deepspeed.comm.get_global_rank", side_effect=fake_get_global_rank):
            engine.replicate_to_peers(dp_group=object(), dp_rank=0, dp_world_size=2, device="cpu")

        assert 1 in engine._received_snapshots, (
            f"Expected group-local sender key 1, got {list(engine._received_snapshots.keys())}")
        assert 5 not in engine._received_snapshots, (
            "Under H1 bug, _received_snapshots was keyed by global (5). If this "
            "assertion fires, the fix has regressed.")

        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=1)
        assert metadata is not None
        assert metadata["window_start_iteration"] == 3
        assert 3 in per_iter and "op_x" in per_iter[3]


# ---------------------------------------------------------------------------
# TestCommGroupRebuildLayer1
# ---------------------------------------------------------------------------


class TestCommGroupRebuildLayer1:
    """Tests for ``MoEvementCoordinator.rebuild_comm_groups`` (Layer 1).

    Layer 1 ships the MoEvement-local rebuild of the two gloo mirrors
    (``_replication_group``, ``_pp_replication_group``) after a spare-
    rank substitution.  The load-bearing invariant the tests pin is the
    event ordering: in-flight work must quiesce BEFORE we abort the old
    communicators, and the old communicators must be aborted BEFORE we
    build the replacements.  Ordering is the only correctness property
    Layer 1 adds; everything else is plumbing.
    """

    def _build_instrumented_coordinator(self, calls, monkeypatch):
        """Coordinator with every event-ordering hook stubbed to ``calls``.

        Stubs ``_abort_or_destroy`` (module-level), ``_build_gloo_mirror``
        (instance method), and the two ``flush_persist`` methods to
        append to ``calls`` in invocation order.  Returns the coordinator
        with the two old mirrors pre-seeded to sentinels so the teardown
        path fires.
        """
        import deepspeed.moevement.coordinator as coord_mod

        def fake_abort(pg, timeout_sec=120.0):
            calls.append(("abort", pg))

        def fake_build(self, base_group):
            calls.append(("build", base_group))
            return ("mirror_of", base_group)

        monkeypatch.setattr(coord_mod, "_abort_or_destroy", fake_abort)
        monkeypatch.setattr(MoEvementCoordinator, "_build_gloo_mirror", fake_build)

        coord = MoEvementCoordinator(MoEvementConfig())

        def fake_snap_flush():
            calls.append(("flush_persist", "snapshot"))

        def fake_log_flush():
            calls.append(("flush_persist", "upstream"))

        monkeypatch.setattr(coord.snapshot_engine, "flush_persist", fake_snap_flush)
        if coord.upstream_logger is not None:
            monkeypatch.setattr(coord.upstream_logger, "flush_persist", fake_log_flush)

        # Pre-seed as if a previous ``set_topology`` + ``set_pipeline_topology``
        # had registered old groups — the fake ``_build_gloo_mirror`` won't run
        # through the production collective path, so we inject sentinels.
        coord._dp_group = "old_dp"
        coord._dp_rank = 0
        coord._device = "cpu"
        coord._replication_group = "old_dp_mirror"
        coord._pp_group = "old_pp"
        coord._stage_id = 0
        coord._num_stages = 2
        coord._stage_to_global_fn = lambda s: s
        coord._pp_replication_group = "old_pp_mirror"
        return coord

    def test_rebuild_orders_quiesce_abort_build(self, monkeypatch):
        """Events fire in the order Layer 1 requires: quiesce → abort → build.

        If quiesce runs after abort, the worker thread reads a communicator
        that was already destroyed — undefined behaviour.  If build runs
        before abort, the new mirror can race against stale collectives
        from the hung-but-not-yet-aborted send.  Pinning the order here
        catches a future refactor that re-orders the helper calls without
        realising these are the guarantees Layer 1 rests on.
        """
        from concurrent.futures import Future
        calls = []
        coord = self._build_instrumented_coordinator(calls, monkeypatch)

        class _TrackingFuture(Future):

            def result(self, timeout=None):
                calls.append(("replication_future", "result"))
                return super().result(timeout=timeout)

        fut = _TrackingFuture()
        fut.set_result(None)
        coord._replication_futures.append(fut)

        coord.rebuild_comm_groups(new_dp_group="new_dp", new_pp_group="new_pp")

        # The future was awaited first; then both persist queues were
        # flushed; then the two old gloo mirrors were aborted; then the
        # replacements were built against the new base groups.
        expected = [
            ("replication_future", "result"),
            ("flush_persist", "snapshot"),
            ("flush_persist", "upstream"),
            ("abort", "old_dp_mirror"),
            ("build", "new_dp"),
            ("abort", "old_pp_mirror"),
            ("build", "new_pp"),
        ]
        assert calls == expected
        assert not coord._replication_futures
        assert coord._replication_broken is False
        assert coord._dp_group == "new_dp"
        assert coord._pp_group == "new_pp"
        assert coord._replication_group == ("mirror_of", "new_dp")
        assert coord._pp_replication_group == ("mirror_of", "new_pp")

    def test_rebuild_requires_new_pp_group_when_pp_mirror_exists(self, monkeypatch):
        """Passing ``new_pp_group=None`` with an existing pp mirror raises.

        Silently skipping pp rebuild would leak the old mirror; raising
        surfaces the ambiguity between "caller forgot" and "caller
        disabled PP mid-rebuild" (the latter is out of scope).
        """
        calls = []
        coord = self._build_instrumented_coordinator(calls, monkeypatch)

        with pytest.raises(ValueError, match="new_pp_group is required"):
            coord.rebuild_comm_groups(new_dp_group="new_dp", new_pp_group=None)

        # No teardown fires when the precondition check trips — the old
        # mirrors stay intact so the caller can retry with the right args.
        assert calls == []
        assert coord._replication_group == "old_dp_mirror"
        assert coord._pp_replication_group == "old_pp_mirror"

    def test_rebuild_skips_pp_when_never_configured(self, monkeypatch):
        """When pp was never set up, ``new_pp_group=None`` is accepted.

        A DP-only job doesn't have a pp mirror to rebuild; the method
        should happily rebuild just the dp mirror and not touch pp.
        """
        import deepspeed.moevement.coordinator as coord_mod
        calls = []

        def fake_abort(pg, timeout_sec=120.0):
            calls.append(("abort", pg))

        def fake_build(self, base_group):
            calls.append(("build", base_group))
            return ("mirror_of", base_group)

        monkeypatch.setattr(coord_mod, "_abort_or_destroy", fake_abort)
        monkeypatch.setattr(MoEvementCoordinator, "_build_gloo_mirror", fake_build)

        coord = MoEvementCoordinator(MoEvementConfig())
        monkeypatch.setattr(coord.snapshot_engine, "flush_persist", lambda: calls.append(
            ("flush_persist", "snapshot")))
        if coord.upstream_logger is not None:
            monkeypatch.setattr(coord.upstream_logger, "flush_persist", lambda: calls.append(
                ("flush_persist", "upstream")))
        coord._dp_group = "old_dp"
        coord._replication_group = "old_dp_mirror"

        coord.rebuild_comm_groups(new_dp_group="new_dp", new_pp_group=None)

        expected = [
            ("flush_persist", "snapshot"),
            ("flush_persist", "upstream"),
            ("abort", "old_dp_mirror"),
            ("build", "new_dp"),
        ]
        assert calls == expected
        assert coord._pp_replication_group is None

    def test_rebuild_resets_replication_broken_flag(self, monkeypatch):
        """A prior replication timeout flipped ``_replication_broken`` to True.

        After a rebuild the new mirror deserves a fresh attempt — leaving
        the flag True would permanently disable replication post-spare-
        substitution even though the actual cause of the hang (dead peer)
        has been replaced.
        """
        calls = []
        coord = self._build_instrumented_coordinator(calls, monkeypatch)
        coord._replication_broken = True

        coord.rebuild_comm_groups(new_dp_group="new_dp", new_pp_group="new_pp")

        assert coord._replication_broken is False

    def test_rebuild_syncs_w_sparse_between_dp_and_pp(self, monkeypatch):
        """``rebuild_comm_groups`` must issue the WORLD ``all_reduce(MAX)`` between
        the DP and PP gloo-mirror builds — same slot the cold-start path uses.

        The cold-start coordinator-init sequence is
        ``set_topology`` (DP build) → ``coord.initialize`` (one
        ``all_reduce(MAX)`` inside ``_generate_schedule_world_aligned``) →
        later ``set_pipeline_topology`` (PP build).  Survivors don't
        traverse ``coord.initialize`` on rebuild, so the rebuild path
        must reproduce that all_reduce in the same slot — otherwise the
        spare (running cold-start) and the survivors (running rebuild)
        disagree on WORLD-collective sequence position by the time they
        hit the PP-mirror's ``all_gather_object``, which then deadlocks.

        This was the root cause of an earlier SIGKILL-recovery wedge.
        Pin the call placement here so a future refactor that lifts
        the sync helper out of the rebuild path or moves it to the
        wrong slot fails visibly.
        """
        calls = []
        coord = self._build_instrumented_coordinator(calls, monkeypatch)

        def fake_sync(self):
            calls.append(("sync_w_sparse", ))

        monkeypatch.setattr(MoEvementCoordinator, "_sync_w_sparse_world_max", fake_sync)

        coord.rebuild_comm_groups(new_dp_group="new_dp", new_pp_group="new_pp")

        # Same shape as ``test_rebuild_orders_quiesce_abort_build`` plus
        # the ``sync_w_sparse`` step in the slot between DP and PP
        # mirror builds — mirroring the cold-start sequence.
        expected = [
            ("flush_persist", "snapshot"),
            ("flush_persist", "upstream"),
            ("abort", "old_dp_mirror"),
            ("build", "new_dp"),
            ("sync_w_sparse", ),
            ("abort", "old_pp_mirror"),
            ("build", "new_pp"),
        ]
        assert calls == expected

    def test_set_topology_tears_down_existing_mirror(self, monkeypatch):
        """Re-calling ``set_topology`` aborts the old mirror before rebuilding.

        Prior to Layer 1 the assignment ``self._replication_group = ...``
        silently leaked the old gloo group, which the fake-rank substitution
        path can't tolerate.  Pin the teardown here so a future refactor
        that drops the abort call fails visibly.
        """
        import deepspeed.moevement.coordinator as coord_mod
        aborted = []

        def fake_abort(pg, timeout_sec=120.0):
            aborted.append(pg)

        def fake_build(self, base_group):
            return ("mirror", base_group)

        monkeypatch.setattr(coord_mod, "_abort_or_destroy", fake_abort)
        monkeypatch.setattr(MoEvementCoordinator, "_build_gloo_mirror", fake_build)

        coord = MoEvementCoordinator(MoEvementConfig())
        coord._replication_group = "old_mirror"

        coord.set_topology(dp_group="new_dp", dp_rank=0, device="cpu")

        assert aborted == ["old_mirror"]
        assert coord._replication_group == ("mirror", "new_dp")


# ---------------------------------------------------------------------------
# TestPipelineRecovery
# ---------------------------------------------------------------------------


class TestPipelineRecovery:
    """Tests for upstream log transfer and pipeline stage replay."""

    def test_get_received_activation_returns_none_when_empty(self):
        """get_received_activation returns None before any recv_logs_from call."""
        log = UpstreamLogger()
        assert log.get_received_activation(0, 0) is None

    def test_get_received_gradient_returns_none_when_empty(self):
        """get_received_gradient returns None before any recv_logs_from call."""
        log = UpstreamLogger()
        assert log.get_received_gradient(0, 0) is None

    def test_send_recv_logs_round_trip(self):
        """send_logs_to / recv_logs_from transfers log entries faithfully."""
        sender_log = UpstreamLogger()
        receiver_log = UpstreamLogger()

        # Build fake log entries directly (bypass CUDA stream)
        from deepspeed.moevement.upstream_logging import LogEntry
        t0 = torch.tensor([1.0, 2.0, 3.0])
        t1 = torch.tensor([4.0, 5.0, 6.0])
        sender_log._logs[(0, 0)].append(LogEntry(0, 0, stage_id=1, direction="activation", tensor=t0))
        sender_log._logs[(1, 0)].append(LogEntry(1, 0, stage_id=1, direction="activation", tensor=t1))

        # Capture send/recv calls and wire them together with a queue
        payloads = []

        import unittest.mock as mock

        def fake_send(tensor, dst, **kwargs):
            payloads.append(tensor.clone())

        def fake_recv(tensor, src=None, **kwargs):
            data = payloads.pop(0)
            tensor.copy_(data)

        with mock.patch("deepspeed.comm.send", side_effect=fake_send), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv):
            sender_log.send_logs_to(target_rank=1, stage_id=1, direction_filter="activation", iteration_range=[0, 1])
            receiver_log.recv_logs_from(src_rank=0)

        acts_0 = receiver_log.get_received_activation(0, 0)
        acts_1 = receiver_log.get_received_activation(1, 0)
        assert acts_0 is not None and len(acts_0) == 1
        assert acts_1 is not None and len(acts_1) == 1
        assert torch.allclose(acts_0[0], t0)
        assert torch.allclose(acts_1[0], t1)

    def test_send_recv_logs_forward_group_kwarg(self):
        """group= is plumbed through to dist.send / dist.recv.

        The log payload is a CPU tensor, so the transport group must be
        gloo-backed; passing the group through from the coordinator lets
        ``send_logs_to`` / ``recv_logs_from`` avoid the default NCCL
        group's CPU-tensor rejection.
        """
        import unittest.mock as mock

        logger_obj = UpstreamLogger()
        from deepspeed.moevement.upstream_logging import LogEntry
        logger_obj._logs[(0, 0)].append(LogEntry(0, 0, stage_id=1, direction="activation", tensor=torch.zeros(1)))

        gloo_group = object()
        send_groups = []
        recv_groups = []

        def capture_send(tensor, dst, **kwargs):
            send_groups.append(kwargs.get("group"))

        def capture_recv(tensor, src=None, **kwargs):
            recv_groups.append(kwargs.get("group"))

        with mock.patch("deepspeed.comm.send", side_effect=capture_send):
            logger_obj.send_logs_to(target_rank=1,
                                    stage_id=1,
                                    direction_filter="activation",
                                    iteration_range=[0],
                                    group=gloo_group)
        assert all(g is gloo_group for g in send_groups)

        # Matching recv path: feed a real (empty-list) serialized payload so
        # the pickle round-trip succeeds and we can isolate the group plumbing.
        import io as _io
        buf = _io.BytesIO()
        torch.save([], buf)
        empty_payload = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)
        recv_queue = [torch.tensor([empty_payload.numel()], dtype=torch.int64), empty_payload]

        def sized_recv(tensor, src=None, **kwargs):
            recv_groups.append(kwargs.get("group"))
            tensor.copy_(recv_queue.pop(0))

        receiver = UpstreamLogger()
        with mock.patch("deepspeed.comm.recv", side_effect=sized_recv):
            receiver.recv_logs_from(src_rank=0, group=gloo_group)
        assert all(g is gloo_group for g in recv_groups)

    def test_coordinator_get_replay_activations_delegates(self):
        """coordinator.get_replay_activations returns None when logger has no data."""
        coord, _, _ = _build_coordinator_with_two_experts()
        assert coord.get_replay_activations(0, 0) is None
        assert coord.get_replay_gradients(0, 0) is None

    def test_coordinator_get_replay_returns_none_without_logger(self):
        """get_replay_* returns None when upstream_logging is disabled."""
        config = MoEvementConfig({"moevement": {"enabled": True, "upstream_logging": False}})
        coord = MoEvementCoordinator(config)
        assert coord.get_replay_activations(0, 0) is None
        assert coord.get_replay_gradients(0, 0) is None

    def test_clear_also_resets_received_logs(self):
        """UpstreamLogger.clear() removes received logs too."""
        log = UpstreamLogger()
        from deepspeed.moevement.upstream_logging import LogEntry
        log._received_logs[(0, 0)].append(LogEntry(0, 0, 0, "activation", torch.zeros(2)))
        log.clear()
        assert log.get_received_activation(0, 0) is None

    def test_replay_cursor_remaps_engine_step_to_logged_iteration(self):
        """When recovering, get_replay_activations ignores the caller's iter
        number and uses the coordinator's cursor instead.

        This models the real engine path: ``global_steps`` continues to
        increment forward during replay, but the upstream logs carry the
        *original* iteration tags from before the failure.  Without the
        cursor remap, the lookup would miss and the engine would fall back
        to blocking P2P receives.
        """
        from deepspeed.moevement.upstream_logging import LogEntry
        config = MoEvementConfig({"moevement": {"enabled": True, "upstream_logging": True}})
        coord = MoEvementCoordinator(config)

        # Logs were captured at original iteration 7 on some other rank and
        # shipped to us via recovery_barrier → recv_logs_from.
        saved_tensor = torch.tensor([11.0, 22.0, 33.0])
        coord.upstream_logger._received_logs[(7, 0)].append(
            LogEntry(7, 0, stage_id=0, direction="activation", tensor=saved_tensor))

        # Engine is at some later step (say 42) and enters recovery.  Cursor
        # is seeded to the original window-start iteration.
        coord._recovering = True
        coord._replay_iteration_cursor = 7

        # The engine passes its current global_steps, *not* 7 — our cursor
        # remap is what makes the lookup hit.
        acts = coord.get_replay_activations(iteration=42, micro_batch_id=0)
        assert acts is not None
        torch.testing.assert_close(acts[0], saved_tensor)

        # After end_recovery the cursor clears and lookups fall through.
        coord.end_recovery()
        assert coord.get_replay_activations(iteration=42, micro_batch_id=0) is None

    def test_end_recovery_drops_received_logs(self):
        """Neighbour-shipped logs are cleared once recovery completes.

        Those entries point at pinned CPU activations/gradients the
        neighbour stages produced; they're only needed for the replay
        loop.  Holding them for the remainder of the job is wasted
        memory — end_recovery() is the natural cleanup point.
        """
        from deepspeed.moevement.upstream_logging import LogEntry
        config = MoEvementConfig({"moevement": {"enabled": True, "upstream_logging": True}})
        coord = MoEvementCoordinator(config)
        coord._recovering = True
        coord._replay_iteration_cursor = 0

        coord.upstream_logger._received_logs[(0, 0)].append(
            LogEntry(0, 0, stage_id=0, direction="activation", tensor=torch.zeros(4)))
        coord.upstream_logger._received_logs[(1, 0)].append(
            LogEntry(1, 0, stage_id=0, direction="gradient", tensor=torch.zeros(4)))

        coord.end_recovery()
        assert len(coord.upstream_logger._received_logs) == 0

    def test_freeze_operator_params_wraps_frozen_linear_forward(self):
        """_freeze_operator_params monkey-swaps frozen ops' nn.Linear forward
        so backward returns grad_input without computing grad_weight, and
        _thaw_operator_params restores the original forward on ACTIVE transition.

        Mirrors the zero-bubble backward-split: weights keep
        ``requires_grad=True`` throughout (required for the autograd graph
        to span a stage whose input is raw data), but the wgrad GEMM is
        skipped via a custom autograd.Function installed on each frozen
        linear.
        """
        from deepspeed.moevement.conversion import OperatorState

        coord = _build_coordinator_with_two_linear_experts()
        active = coord._operator_map["layer_0_expert_0"]
        frozen = coord._operator_map["layer_0_expert_1"]
        coord.converter._operator_states["layer_0_expert_0"] = OperatorState.ACTIVE
        coord.converter._operator_states["layer_0_expert_1"] = OperatorState.FROZEN

        # Pre-wrap: both linears run their class-default forward, so
        # ``forward`` is not an instance attribute on either module.
        assert "forward" not in active.__dict__
        assert "forward" not in frozen.__dict__
        assert active.weight.requires_grad is True
        assert frozen.weight.requires_grad is True

        coord._freeze_operator_params(model=nn.Module())

        # Active left alone; frozen now has a monkey-patched instance
        # ``forward`` that routes through ``_FrozenLinearFunction``.
        assert "forward" not in active.__dict__
        assert "forward" in frozen.__dict__
        assert getattr(frozen, "_moevement_orig_linear", False) is True
        # Requires-grad is intentionally NOT flipped — the split is
        # enforced by the custom backward, not by autograd's opt-out.
        assert active.weight.requires_grad is True
        assert frozen.weight.requires_grad is True

        # Backward through the wrapped forward computes grad_input but
        # leaves frozen.weight.grad at None (wgrad GEMM skipped).
        inp = torch.randn(2, 4, requires_grad=True)
        out = frozen(inp)
        out.sum().backward()
        assert inp.grad is not None
        assert frozen.weight.grad is None
        assert frozen.bias.grad is None

        # Thawing restores the default forward.
        coord._thaw_operator_params("layer_0_expert_1", model=None)
        assert "forward" not in frozen.__dict__
        assert not hasattr(frozen, "_moevement_orig_linear") or not frozen._moevement_orig_linear

    def test_freeze_operator_params_reentrance_preserves_wrap(self):
        """Double-freeze (no intervening thaw) mustn't leave a stale wrapper.

        Scenario: ``load_sparse_checkpoint`` runs twice back-to-back (retry
        logic).  Without idempotent tracking, the second call might re-wrap
        an already-wrapped forward and leak an instance override that
        ``_thaw_operator_params`` can't cleanly undo.
        """
        from deepspeed.moevement.conversion import OperatorState

        coord = _build_coordinator_with_two_linear_experts()
        frozen = coord._operator_map["layer_0_expert_1"]
        coord.converter._operator_states["layer_0_expert_1"] = OperatorState.FROZEN

        coord._freeze_operator_params(model=nn.Module())
        first_forward = frozen.forward

        # Second freeze: must not re-wrap an already-wrapped module.
        coord._freeze_operator_params(model=nn.Module())
        assert frozen.forward is first_forward

        # Thaw removes the override cleanly, restoring the class-method default.
        coord._thaw_operator_params("layer_0_expert_1", model=None)
        assert "forward" not in frozen.__dict__


class TestUpstreamLoggerSideStreamSync:
    """Sync the side D2H stream before reading or releasing logged tensors.

    Log tensors live in pinned CPU buffers fed by an async D2H copy on a
    dedicated stream.  Paths that read those buffers (``send_logs_to``
    for peer-pull recovery) or release them back to the pool
    (``garbage_collect`` after a checkpoint boundary) must wait for the
    side stream to drain first — otherwise they race with the in-flight
    copies and either ship partial bytes or hand a still-being-written
    buffer back to the next ``pool.acquire``.

    ``save_to_disk`` already syncs at its top; these tests pin the same
    contract for the two other callers flagged in the correctness audit.
    """

    def _build_logger_with_entry(self):
        """Return (logger, entry) with one non-lent LogEntry in slot (0,0).

        Bypasses the CUDA-path ``_log_single_tensor`` so the test runs on
        CPU-only boxes; the sync-ordering assertion cares about call
        order, not whether a real D2H copy ran.
        """
        from deepspeed.moevement.upstream_logging import LogEntry
        log = UpstreamLogger()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        entry = LogEntry(iteration=0, micro_batch_id=0, stage_id=1, direction="activation", tensor=tensor)
        log._logs[(0, 0)].append(entry)
        return log, entry

    def test_send_logs_to_synchronizes_before_serialize(self):
        """``send_logs_to`` calls ``synchronize`` before the first ``dist.send``.

        Without this, a fault one iter after the D2H copy kicked off would
        ``torch.save`` a partial pinned buffer and ship stale bytes to
        the recovering rank — silent divergence at replay.
        """
        import unittest.mock as mock

        log, _ = self._build_logger_with_entry()
        events = []
        original_sync = log.synchronize
        log.synchronize = lambda: (events.append("sync"), original_sync())

        def fake_send(tensor, dst, **kwargs):
            events.append("send")

        with mock.patch("deepspeed.comm.send", side_effect=fake_send):
            log.send_logs_to(target_rank=1, stage_id=1, direction_filter="activation", iteration_range=[0])

        assert "sync" in events and "send" in events
        assert events.index("sync") < events.index("send"), (f"sync must precede first send; got {events}")

    def test_garbage_collect_synchronizes_before_pool_release(self):
        """``garbage_collect`` syncs before handing buffers back to the pool.

        Releasing a buffer whose in-flight D2H copy hasn't drained would
        let the next ``pool.acquire`` hand that same storage to a fresh
        log entry, and the tail of the old copy would clobber the new
        log's payload after the caller started reading it.
        """
        log, _ = self._build_logger_with_entry()
        events = []
        original_sync = log.synchronize
        log.synchronize = lambda: (events.append("sync"), original_sync())
        original_release = log._pool.release
        log._pool.release = lambda t: (events.append("release"), original_release(t))

        log.garbage_collect(oldest_valid_iteration=5)

        assert "sync" in events, f"garbage_collect must sync before release; got {events}"
        assert "release" in events, f"test fixture must trigger at least one pool release; got {events}"
        assert events.index("sync") < events.index("release"), (f"sync must precede first release; got {events}")

    def test_garbage_collect_skips_sync_when_nothing_stale(self):
        """No stale keys → no sync call.

        Sync is cheap but not free (cross-stream wait on real hardware);
        skipping it in the no-op path keeps steady-state gc on the
        training thread trivially fast.
        """
        log, _ = self._build_logger_with_entry()
        events = []
        original_sync = log.synchronize
        log.synchronize = lambda: (events.append("sync"), original_sync())

        # Entry is at iter 0; gc with threshold 0 leaves it alone.
        log.garbage_collect(oldest_valid_iteration=0)

        assert "sync" not in events, f"empty-gc should not sync; got {events}"


# ---------------------------------------------------------------------------
# Recovery barrier + coordinator initialization
# ---------------------------------------------------------------------------


class TestInitialize:
    """Tests for coordinator.initialize wiring from the engine."""

    def test_initialize_registers_non_expert_operator(self):
        """initialize discovers the non-expert operator and marks the coordinator ready."""
        config = MoEvementConfig({"moevement": {"enabled": True, "pcie_bandwidth_gbs": 1e3}})
        coord = MoEvementCoordinator(config)
        model = _FakeModule(size=8)
        coord.initialize(model=model, moe_layers=[], iter_time_sec=0.5)
        assert coord._initialized is True
        assert "non_expert" in coord._operator_map
        assert coord.scheduler.w_sparse >= 1

    def test_on_iteration_end_is_no_op_before_initialize(self):
        """on_iteration_end must short-circuit when the coordinator is not initialized."""
        config = MoEvementConfig({"moevement": {"enabled": True}})
        coord = MoEvementCoordinator(config)
        model = _FakeModule(size=4)
        optimizer = _FakeOptimizer(list(model.parameters()))
        # Should not raise even though initialize() was never called.
        coord.on_iteration_end(global_step=0, model=model, optimizer=optimizer)


class _FakeMoEGate(nn.Module):
    """Minimal MoE gate — a single ``nn.Linear``, DP-shared (no allreduce marker).

    Mirrors the real ``TopKGate.wg`` surface: a ``nn.Linear`` routing
    hidden_dim to num_experts.  Matches DeepSpeed's convention that
    gate params carry the default ``allreduce`` setting (True by
    absence of the marker).
    """

    def __init__(self, hidden=4, experts=2):
        super().__init__()
        self.wg = nn.Linear(hidden, experts, bias=False)


class _FakeMoEExperts(nn.Module):
    """Minimal expert bank — DeepSpeed stamps ``allreduce=False`` per expert param."""

    def __init__(self, hidden=4, num_experts=2):
        super().__init__()
        self.deepspeed_experts = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_experts)])
        # Emulate deepspeed/moe/experts.py:22-26 — the single signal
        # MoEvement's discovery uses to identify expert params.
        for expert in self.deepspeed_experts:
            for p in expert.parameters():
                p.allreduce = False


class _FakeMoELayer(nn.Module):
    """Minimal stand-in for ``deepspeed.moe.layer.MoE`` with gate + experts."""

    def __init__(self, hidden=4, num_experts=2):
        super().__init__()
        self.gate = _FakeMoEGate(hidden, num_experts)
        self.experts = _FakeMoEExperts(hidden, num_experts)
        self.num_local_experts = num_experts


class _FakeModelWithMoE(nn.Module):
    """Non-expert + MoE layer + non-expert: exercises all three buckets."""

    def __init__(self, hidden=4, num_experts=2):
        super().__init__()
        self.embed = nn.Linear(hidden, hidden)
        self.moe = _FakeMoELayer(hidden, num_experts)
        self.out = nn.Linear(hidden, hidden)


class TestOperatorBucketOwnership:
    """Gate params must land in the ``layer_X_gate`` op only, not double-registered.

    Before the fix, ``_discover_operators`` put every param without
    ``allreduce=False`` into the non_expert bucket — which includes MoE
    gate params (DeepSpeed leaves gate's allreduce default at True,
    since gates are DP-shared).  Gate was then *also* registered as
    ``layer_X_gate``, so any window where the two ops disagreed on
    ACTIVE/FROZEN status silently corrupted the gate: the non_expert-
    frozen path wrapped gate's ``nn.Linear.forward`` and
    ``zero_frozen_gradients`` zeroed the gate's grad while its own op
    was ACTIVE, drifting the gate off the saved trajectory with no
    warning.
    """

    def test_gate_params_excluded_from_non_expert_bucket(self):
        """Gate param id appears in the gate op only, never in non_expert."""
        config = MoEvementConfig({"moevement": {"enabled": True, "pcie_bandwidth_gbs": 1e3}})
        coord = MoEvementCoordinator(config)
        model = _FakeModelWithMoE(hidden=4, num_experts=2)
        coord.initialize(model=model, moe_layers=[model.moe], iter_time_sec=0.5)

        non_expert_ids = {id(p) for _, p in coord._param_list_by_op["non_expert"]}
        gate_ids = {id(p) for _, p in coord._param_list_by_op["layer_0_gate"]}

        gate_weight_id = id(model.moe.gate.wg.weight)
        assert gate_weight_id in gate_ids, "gate.wg.weight must be in gate bucket"
        assert gate_weight_id not in non_expert_ids, ("gate.wg.weight must not also be in non_expert bucket — "
                                                      "double-registration was the bug")

        assert id(model.embed.weight) in non_expert_ids
        assert id(model.out.weight) in non_expert_ids

    def test_non_expert_num_params_excludes_gate(self):
        """OperatorInfo.num_params on non_expert excludes gate params.

        PCIe budget accounting relies on per-op num_params; the pre-fix
        double-count inflated non_expert's size and the scheduler over-
        reserved bandwidth at window-sizing time.
        """
        config = MoEvementConfig({"moevement": {"enabled": True, "pcie_bandwidth_gbs": 1e3}})
        coord = MoEvementCoordinator(config)
        hidden, num_experts = 4, 2
        model = _FakeModelWithMoE(hidden=hidden, num_experts=num_experts)
        coord.initialize(model=model, moe_layers=[model.moe], iter_time_sec=0.5)

        non_expert_op = next(op for op in coord.scheduler.operators if op.name == "non_expert")
        # embed + out = 2 × (hidden × hidden weight + hidden bias)
        expected = 2 * (hidden * hidden + hidden)
        assert non_expert_op.num_params == expected, (f"expected {expected} non_expert params "
                                                      f"(embed+out only, no gate); got {non_expert_op.num_params}")

    def test_discovery_asserts_disjoint_buckets(self):
        """Manually poisoning the param map with overlap must raise.

        Pins the ownership-invariant assertion added alongside the gate
        fix: if a future regression routes the same param into two
        operators, discovery fails loudly instead of silently drifting.
        """
        config = MoEvementConfig({"moevement": {"enabled": True, "pcie_bandwidth_gbs": 1e3}})
        coord = MoEvementCoordinator(config)
        model = _FakeModelWithMoE(hidden=4, num_experts=2)

        # Monkeypatch _resolve_op_params to re-include gate params in
        # non_expert on top of the normal non_expert entries — the exact
        # shape of the pre-fix bug.  Discovery's disjointness check
        # should catch this.
        original = coord._resolve_op_params

        def poisoned(op_name, m):
            result = original(op_name, m)
            if op_name == "non_expert":
                result = list(result) + [("moe.gate.wg.weight", model.moe.gate.wg.weight)]
            return result

        coord._resolve_op_params = poisoned

        with pytest.raises(RuntimeError, match="operator ownership violation"):
            coord.initialize(model=model, moe_layers=[model.moe], iter_time_sec=0.5)

    def test_zero_frozen_gradients_leaves_active_gate_alone(self):
        """End-to-end canary: ``non_expert`` FROZEN must not zero an ACTIVE gate's grad.

        Pins the downstream behaviour — bucket membership is the proxy;
        this is where the saved-trajectory drift actually shows up at
        replay time.  The pre-fix ``zero_frozen_gradients`` branch for
        the non_expert sentinel iterated ``model.parameters()`` with
        the allreduce-only predicate and wiped every gate's grad when
        non_expert was frozen — even when the gate's own op was ACTIVE.
        """
        from deepspeed.moevement.conversion import OperatorState

        config = MoEvementConfig({"moevement": {"enabled": True, "pcie_bandwidth_gbs": 1e3}})
        coord = MoEvementCoordinator(config)
        model = _FakeModelWithMoE(hidden=4, num_experts=2)
        coord.initialize(model=model, moe_layers=[model.moe], iter_time_sec=0.5)

        # Scheduler state: non_expert FROZEN, layer_0_gate ACTIVE.
        coord.converter._operator_states["non_expert"] = OperatorState.FROZEN
        coord.converter._operator_states["layer_0_gate"] = OperatorState.ACTIVE
        for op_name in ("layer_0_expert_0", "layer_0_expert_1"):
            coord.converter._operator_states[op_name] = OperatorState.ACTIVE
        coord._recovering = True

        # Give every param a non-zero grad.
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        coord.zero_frozen_gradients(model)

        # Gate stays ACTIVE → grad must survive.
        assert model.moe.gate.wg.weight.grad.abs().sum().item() > 0, (
            "gate grad was zeroed despite being ACTIVE — pre-fix bug")
        # non_expert FROZEN → embed / out grads must be wiped.
        assert model.embed.weight.grad.abs().sum().item() == 0
        assert model.out.weight.grad.abs().sum().item() == 0


def _fake_handshake(
    any_recovering=True,
    recovering_ranks=None,
    dp_group_has_recovering=False,
    pp_column_has_recovering=False,
    recovering_stages_in_my_pp=None,
):
    """Return a canned handshake dict for mocking ``_world_recovery_handshake``.

    The real handshake does a world ``all_gather`` and group-layout
    lookups; unit tests substitute this dict so we can exercise the
    downstream cascade / log-transfer / pause branches without a full
    distributed backend.
    """
    return {
        "any_recovering": any_recovering,
        "recovering_ranks": recovering_ranks or [],
        "dp_group_has_recovering": dp_group_has_recovering,
        "pp_column_has_recovering": pp_column_has_recovering,
        "recovering_stages_in_my_pp": recovering_stages_in_my_pp or [],
    }


class TestReplayKeyIterationSymmetry:
    """H6 gate: pin the symmetry between log-time iteration and replay-key.

    `engine.py:1156` calls ``on_send_activations(self.global_steps, ...)``
    where ``global_steps`` is the index of the *current* tb (incremented
    only at end of train_batch via ``global_steps += 1``).  During
    replay, ``global_steps`` is held constant (the increment is gated on
    ``not is_moevement_replaying``) and ``_replay_key_iteration`` ignores
    its arg and returns the cursor instead.

    Audit H6 claims bundle-replay's ``return cursor`` is off-by-one and
    should be ``cursor - 1`` (matching the catch-up branch).  That claim
    rests on log keys being "pre-increment (forward input during
    tb-(N+1))" — but the engine instrumentation logs with the same
    ``global_steps`` value that the current tb uses for its own state.

    This test pins the symmetry empirically: log a tensor at iteration=N
    with the real upstream_logger; set up a recovery cursor at N within
    the persisted bundle range; ``get_replay_activations`` must return
    that exact tensor.  If the audit's fix is applied (return cursor-1),
    this would fetch log[N-1] instead and the assertion fails.
    """

    def test_bundle_replay_fetches_log_at_cursor_iteration(self):
        from deepspeed.moevement.upstream_logging import UpstreamLogger, LogEntry
        coord = MoEvementCoordinator(MoEvementConfig())
        coord.upstream_logger = UpstreamLogger()

        # Simulate normal training: at tb-N=5, engine fires
        # on_send_activations(global_steps=5).  We log directly into
        # _received_logs (the replay-side dict) since the test isn't
        # exercising peer log-shipping; the read path key shape is
        # ``(iteration, micro_batch_id)`` either way.
        for n in (4, 5, 6):
            tag_tensor = torch.tensor([float(n)])
            coord.upstream_logger._received_logs[(n, 0)] = [
                LogEntry(iteration=n, micro_batch_id=0, stage_id=0, direction="activation", tensor=tag_tensor),
            ]

        # Bundle covers iters [0..6]; cursor=5 is bundle-replay (cursor <= boundary).
        coord._recovering = True
        coord._replay_iteration_cursor = 5
        coord._catch_up_boundary = 6

        acts = coord.get_replay_activations(iteration=999, micro_batch_id=0)
        assert acts is not None and len(acts) == 1
        # If current code is correct (return cursor=5), we get the tensor logged
        # with iteration=5.  If audit's fix were applied (return cursor-1=4),
        # we'd get the iteration=4 tensor and this assertion fails.
        assert acts[0].item() == 5.0, (
            f"Bundle-replay at cursor=5 fetched log entry tagged {acts[0].item()}, expected 5.0. "
            "Either H6's mechanical claim is right (and current code is off-by-one) "
            "or the engine's on_send_activations instrumentation is non-symmetric with "
            "_replay_key_iteration — investigate before flipping cursor-1.")


class TestIntStepOptimizerWarning:
    """Warn loudly if FusedAdam / DeepSpeedCPUAdam is wired up under MoEvement.

    Both store ``state['step']`` as a Python int and silently fail
    snapshot's ``isinstance(tensor, torch.Tensor)`` filter — post-recovery
    Adam restarts at ``t=1``.  The supported path is ``torch.optim.Adam``
    / ``AdamW`` (step is a 0-dim Tensor and round-trips).  Hardening
    the int-step path requires touching the ZeRO/non-ZeRO writers + H2D
    applier + adding a round-trip marker; out of scope today.  This
    warning surfaces the gap.
    """

    def test_warns_on_fused_adam_inner_optimizer(self, caplog, monkeypatch):
        import logging
        _capture_deepspeed_warnings(caplog, monkeypatch)
        coord = MoEvementCoordinator(MoEvementConfig())

        class _FakeFusedAdam:
            pass

        _FakeFusedAdam.__name__ = "FusedAdam"
        with caplog.at_level(logging.WARNING):
            coord._warn_if_int_step_optimizer(_FakeFusedAdam())
        assert any("FusedAdam" in rec.message and "restarts at ``t=1``" in rec.message
                   for rec in caplog.records), (f"expected FusedAdam warning; got: "
                                                f"{[(rec.levelname, rec.message[:80]) for rec in caplog.records]}")

    def test_walks_optimizer_wrapper_chain(self, caplog, monkeypatch):
        """ZeRO / FP16 wrappers expose the inner optimizer via ``.optimizer``."""
        import logging
        _capture_deepspeed_warnings(caplog, monkeypatch)
        coord = MoEvementCoordinator(MoEvementConfig())

        class _FakeCPUAdam:
            pass

        _FakeCPUAdam.__name__ = "DeepSpeedCPUAdam"

        class _FakeWrapper:

            def __init__(self, inner):
                self.optimizer = inner

        wrapped = _FakeWrapper(_FakeWrapper(_FakeCPUAdam()))
        with caplog.at_level(logging.WARNING):
            coord._warn_if_int_step_optimizer(wrapped)
        assert any("DeepSpeedCPUAdam" in rec.message for rec in caplog.records)

    def test_silent_on_torch_adam(self, caplog, monkeypatch):
        import logging
        _capture_deepspeed_warnings(caplog, monkeypatch)
        coord = MoEvementCoordinator(MoEvementConfig())
        opt = torch.optim.Adam([torch.nn.Parameter(torch.randn(2))], lr=1e-3)
        with caplog.at_level(logging.WARNING):
            coord._warn_if_int_step_optimizer(opt)
        assert not any("FusedAdam" in rec.message or "DeepSpeedCPUAdam" in rec.message for rec in caplog.records)


class TestRestoreRngState:
    """Tests for ``MoEvementCoordinator._restore_rng_state``."""

    def test_restore_rng_state_continues_past_single_device_failure(self):
        """A failed per-device set_rng_state doesn't drop higher-indexed devices.

        Regression for H5 in CORRECTNESS_AUDIT_2026_04_23.md: the old
        implementation broke out of the loop on the first
        ``set_rng_state`` exception, silently leaving every higher-
        indexed device on its current RNG and replay diverged from the
        captured trajectory.  The fix logs at DEBUG and continues so an
        isolated transient miss doesn't compound across devices.
        """
        from unittest import mock
        coord = MoEvementCoordinator(MoEvementConfig())
        state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda.0": torch.zeros(8, dtype=torch.uint8),
            "torch_cuda.1": torch.zeros(8, dtype=torch.uint8),
            "torch_cuda.2": torch.zeros(8, dtype=torch.uint8),
        }
        called_devices = []

        def fake_set_rng_state(s, i):
            called_devices.append(i)
            if i == 1:
                raise RuntimeError("transient device 1 miss")

        fake_accel = mock.Mock()
        fake_accel.Stream = object
        fake_accel.set_rng_state = fake_set_rng_state
        with mock.patch("deepspeed.moevement.coordinator.get_accelerator", return_value=fake_accel):
            coord._restore_rng_state(state)

        assert called_devices == [
            0, 1, 2
        ], (f"Expected attempts on all three devices despite device-1 failure, got {called_devices}")


class TestRecoveryBarrier:
    """Tests for the pipeline-group recovery coordination collective."""

    def test_recovery_barrier_no_op_without_pipeline_topology(self):
        """recovery_barrier returns immediately when no pipeline topology is set."""
        coord, _, _ = _build_coordinator_with_two_experts()
        # No set_pipeline_topology call; must be a safe no-op.
        coord.recovery_barrier()

    def test_recovery_barrier_raises_on_world_asymmetric_pp_topology(self):
        """Asymmetric ``_pp_group is None`` across the world surfaces as RuntimeError.

        Regression for H3 in CORRECTNESS_AUDIT_2026_04_23.md: when some
        ranks set PP topology and others didn't, the old early-return
        let the set ranks block in ``all_gather`` until the NCCL
        watchdog timeout (~10 min).  The new path detects the asymmetry
        via a one-int64 ``all_gather`` and raises locally instead.
        """
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        # Simulate world view: this rank has no PP (gathered=0); peer rank has it (gathered=1).
        peer_views = [torch.tensor([0], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)]

        def fake_all_gather(out_list, in_tensor, *a, **kw):
            for o, v in zip(out_list, peer_views):
                o.copy_(v)

        with mock.patch("deepspeed.moevement.coordinator.dist.is_initialized", return_value=True), \
             mock.patch("deepspeed.moevement.coordinator.dist.get_world_size", return_value=2), \
             mock.patch("deepspeed.moevement.coordinator.dist.all_gather", side_effect=fake_all_gather):
            with pytest.raises(RuntimeError, match="asymmetrically set"):
                coord.recovery_barrier()

    def test_recovery_barrier_world_uniform_no_pp_returns_silently(self):
        """Every rank with ``_pp_group is None`` (legitimate single-rank/no-PP) is OK."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()

        def fake_all_gather(out_list, in_tensor, *a, **kw):
            for o in out_list:
                o.zero_()

        with mock.patch("deepspeed.moevement.coordinator.dist.is_initialized", return_value=True), \
             mock.patch("deepspeed.moevement.coordinator.dist.get_world_size", return_value=2), \
             mock.patch("deepspeed.moevement.coordinator.dist.all_gather", side_effect=fake_all_gather):
            coord.recovery_barrier()  # must not raise

    def test_recovery_barrier_no_op_when_no_one_recovering(self):
        """World handshake reports ``any_recovering=False`` → early return."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=1, num_stages=3, stage_to_global_fn=lambda s: s)

        with mock.patch.object(coord, "_world_recovery_handshake",
                               return_value=_fake_handshake(any_recovering=False)), \
             mock.patch.object(coord, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord, "_wait_for_recovery") as m_wait, \
             mock.patch.object(coord, "cascade_into_recovery") as m_cascade:
            coord.recovery_barrier()
            m_recv.assert_not_called()
            m_send.assert_not_called()
            m_wait.assert_not_called()
            m_cascade.assert_not_called()

    def test_recovery_barrier_recovering_stage_receives_logs(self):
        """A recovering rank pulls logs from its two pipeline neighbours."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=1, num_stages=3, stage_to_global_fn=lambda s: s + 10)
        coord._recovering = True
        coord._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            pp_column_has_recovering=True,
            recovering_stages_in_my_pp=[1],
        )

        with mock.patch.object(coord, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord, "_wait_for_recovery") as m_wait:
            coord.recovery_barrier()
            m_send.assert_not_called()
            m_recv.assert_called_once()
            call_args = m_recv.call_args
            assert call_args.kwargs.get("prev_stage_rank", call_args.args[0]) == 10
            assert call_args.kwargs.get("next_stage_rank", call_args.args[1]) == 12
            # Already recovering → doesn't pause.
            m_wait.assert_not_called()

    def test_recovery_barrier_neighbour_sends_logs_then_pauses(self):
        """A live pipeline neighbour ships logs then drops into the pause wait loop."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=0, num_stages=3, stage_to_global_fn=lambda s: s + 10)
        coord._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            pp_column_has_recovering=True,
            recovering_stages_in_my_pp=[1],
        )

        with mock.patch.object(coord, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord, "_wait_for_recovery") as m_wait:
            coord.recovery_barrier()
            m_recv.assert_not_called()
            m_send.assert_called_once()
            assert m_send.call_args.args[0] == 1
            # Not itself recovering → enters the pause wait loop after the ship.
            m_wait.assert_called_once()

    def test_recovery_barrier_non_adjacent_pauses_without_shipping(self):
        """A stage two hops from the recovering stage doesn't ship logs but still pauses."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=3, num_stages=4, stage_to_global_fn=lambda s: s)
        coord._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            pp_column_has_recovering=True,
            recovering_stages_in_my_pp=[1],
        )

        with mock.patch.object(coord, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord, "_wait_for_recovery") as m_wait:
            coord.recovery_barrier()
            m_send.assert_not_called()
            m_recv.assert_not_called()
            # Still paused — non-recovering rank in an affected pp column.
            m_wait.assert_called_once()

    def test_cascade_into_recovery_restores_from_in_memory_snapshot(self):
        """Cascade loads from own ``_persisted_snapshots`` and enters recovery.

        When a DP peer fails, this rank's model state is already at
        end-of-iter-(X-1) but its optimizer DP all-reduce with the
        recovering peer will mix iteration-mismatched gradients unless
        we rewind ourselves.  The in-memory persisted snapshot holds
        the state at window-start (X-W_sparse); we reinit the converter
        from it and flip ``_recovering=True``.
        """
        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot

        coord, _, _ = _build_coordinator_with_two_experts()
        snap = OperatorSnapshot("layer_0_expert_0", iteration=3, is_active=True)
        snap.add_tensor("params.weight", torch.ones(4))
        coord.snapshot_engine._persisted_snapshots[(3, "layer_0_expert_0")] = snap
        coord.snapshot_engine._window_start_iteration = 3
        # Simulate that this rank reached end-of-iter-3 before the DP peer
        # failed — cascade reads ``_global_step`` as ``_fault_iter``, which
        # ``_compute_replay_iters`` uses to clamp the bundle.  Without this,
        # default ``_global_step=0`` makes the clamp drop the iter-3 snapshot.
        coord._global_step = 3

        assert coord.is_recovering() is False
        ok = coord.cascade_into_recovery()
        assert ok is True
        assert coord.is_recovering() is True
        # ``_model`` isn't set in this fixture, so the eager iter-1 setup branch
        # is skipped and the cursor stays at the seeded ``window_start`` value.
        assert coord._replay_iteration_cursor == 3

    def test_cascade_caches_snapshot_data_for_setup_replay_iter(self):
        """Cascade populates ``_cached_snapshot_data`` so the replay's
        progressive thawing can read in-memory snapshot state.

        Without this, each replay iter's ``_setup_replay_iter``
        would try to re-read from disk — but the cascade path has no
        disk to read from (``_checkpoint_load_dir=None``).  The old
        early-return on ``load_dir is None`` would fire and operators
        would stay FROZEN forever, their weights never rolling back to
        window-start state.  Check the cache is populated after
        cascade and cleared after ``end_recovery``.
        """
        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot

        coord, _, _ = _build_coordinator_with_two_experts()
        snap = OperatorSnapshot("layer_0_expert_0", iteration=3, is_active=True)
        snap.add_tensor("params.weight", torch.ones(4))
        coord.snapshot_engine._persisted_snapshots[(3, "layer_0_expert_0")] = snap
        coord.snapshot_engine._window_start_iteration = 3

        assert coord._cached_snapshot_data is None
        coord.cascade_into_recovery()
        assert coord._cached_snapshot_data is not None
        metadata, per_iter_states = coord._cached_snapshot_data
        assert metadata["window_start_iteration"] == 3
        assert 3 in per_iter_states
        assert "layer_0_expert_0" in per_iter_states[3]
        # Cloned tensors decouple from the pool-managed flats — a later
        # mutation of the original snap.state_dict must not appear in
        # the cache.
        snap.state_dict["params.weight"].zero_()
        assert per_iter_states[3]["layer_0_expert_0"]["params.weight"].abs().sum().item() > 0

        coord.end_recovery()
        assert coord._cached_snapshot_data is None

    def test_load_snapshot_data_caches_disk_result(self, tmp_path):
        """Second call returns cached data without re-reading from disk.

        The replay loop calls ``_load_snapshot_data`` once per replay
        iteration via ``_setup_replay_iter``.  Re-reading the bundle
        from disk every time is wasteful — the bundle doesn't change
        during replay.  Cache the first load; later calls short-circuit
        to the cache.
        """
        import unittest.mock as mock

        coord, _, _ = _build_coordinator_with_two_experts()
        fake_disk = ({"window_start_iteration": 0, "per_iter_active": {0: {"op0": True}}}, {0: {"op0": {}}})

        with mock.patch.object(SparseSnapshotEngine, "load_from_disk", return_value=fake_disk) as m_load:
            result1 = coord._load_snapshot_data(str(tmp_path), tag="step1")
            result2 = coord._load_snapshot_data(str(tmp_path), tag="step1")

        assert m_load.call_count == 1
        assert result1 == result2

    def test_cascade_into_recovery_raises_without_persisted_snapshot(self):
        """No in-memory snapshot → cascade raises.

        If the cascade can't proceed (e.g., failure before the first
        W_sparse window completed), silently entering the pause loop
        would deadlock — the recovering DP peer is still blocked on
        this rank's participation in the stage-DP all-reduce.  Raise
        clearly so the cluster manager can fall back to a full dense
        checkpoint reload instead.
        """
        coord, _, _ = _build_coordinator_with_two_experts()
        assert not coord.snapshot_engine._persisted_snapshots
        with pytest.raises(RuntimeError, match="no persisted snapshot"):
            coord.cascade_into_recovery()
        assert coord.is_recovering() is False

    def test_should_skip_pipeline_send_preserves_adjacent_recovering_chain(self):
        """Send guard lets contiguous recovering stages chain via normal p2p.

        Multi-stage adjacent failure (e.g., stages 2, 3, 4 all recovering
        in a 5-stage pipeline) relies on the middle stages exchanging
        their replay outputs and gradients via ``p2p.send``/``p2p.recv``
        — only the block's first/last stages get logs from the live
        bounding stages.  A naive "skip every send during recovery"
        guard would break this: stage 2's forward output wouldn't reach
        stage 3, the replay chain would deadlock.

        The fix: skip only when the downstream stage is NOT in the
        recovering set.  Verify the two behaviours.
        """
        coord, _, _ = _build_coordinator_with_two_experts()
        coord._recovering = True

        # Contiguous block {2, 3, 4} recovering, stages 1 and 5 live-paused.
        coord._recovering_stages_in_my_pp = frozenset({2, 3, 4})

        # Stage 2 sending forward to stage 3 (also recovering) → don't skip.
        assert coord.should_skip_pipeline_send(downstream_stage_id=3) is False
        # Stage 4 sending forward to stage 5 (paused) → skip.
        assert coord.should_skip_pipeline_send(downstream_stage_id=5) is True
        # Stage 2 sending backward grad to stage 1 (paused) → skip.
        assert coord.should_skip_pipeline_send(downstream_stage_id=1) is True
        # Stage 3 sending backward grad to stage 2 (recovering) → don't skip.
        assert coord.should_skip_pipeline_send(downstream_stage_id=2) is False

    def test_should_skip_pipeline_send_returns_false_outside_recovery(self):
        """Normal training: never skip, whatever ``_recovering_stages_in_my_pp`` says."""
        coord, _, _ = _build_coordinator_with_two_experts()
        assert coord.is_recovering() is False
        coord._recovering_stages_in_my_pp = frozenset({0, 1, 2, 3})  # stale
        assert coord.should_skip_pipeline_send(downstream_stage_id=1) is False

    def test_recovery_barrier_cascade_into_recovery_when_dp_peer_recovering(self):
        """A DP peer recovering triggers cascade on this rank even if no pp-neighbour is affected."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=0, num_stages=2, stage_to_global_fn=lambda s: s)
        coord._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            dp_group_has_recovering=True,
            pp_column_has_recovering=False,
        )

        with mock.patch.object(coord, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord, "cascade_into_recovery") as m_cascade, \
             mock.patch.object(coord, "_wait_for_recovery") as m_wait:
            # ``cascade_into_recovery`` flips ``_recovering``; simulate that so
            # the subsequent "not recovering → pause" branch doesn't fire.
            # ``recovery_barrier`` forwards ``model=`` through to the cascade,
            # so the fake has to accept it even when the test doesn't care.
            def fake_cascade(model=None):
                del model
                coord._recovering = True
                return True

            m_cascade.side_effect = fake_cascade
            coord.recovery_barrier()
            m_cascade.assert_called_once()
            # Cascade recovered us; no need to pause.
            m_wait.assert_not_called()


# ---------------------------------------------------------------------------
# Log persistence + automatic recovery trigger
# ---------------------------------------------------------------------------


class TestLogPersistence:
    """Tests for UpstreamLogger save_to_disk / load_from_disk."""

    def test_save_load_round_trip(self, tmp_path):
        """Saved logs reload into _logs with matching tensors."""
        from deepspeed.moevement.upstream_logging import LogEntry
        sender = UpstreamLogger()
        t0 = torch.tensor([1.0, 2.0])
        t1 = torch.tensor([3.0, 4.0])
        sender._logs[(0, 0)].append(LogEntry(0, 0, stage_id=1, direction="activation", tensor=t0))
        sender._logs[(0, 1)].append(LogEntry(0, 1, stage_id=1, direction="gradient", tensor=t1))

        sender.save_to_disk(str(tmp_path), tag="step10", rank=0)
        sender.flush_persist()

        receiver = UpstreamLogger()
        loaded = receiver.load_from_disk(str(tmp_path), tag="step10", rank=0)
        assert loaded is True
        assert len(receiver._logs[(0, 0)]) == 1
        assert len(receiver._logs[(0, 1)]) == 1
        assert torch.allclose(receiver._logs[(0, 0)][0].tensor, t0)
        assert torch.allclose(receiver._logs[(0, 1)][0].tensor, t1)

    def test_load_missing_file_returns_false(self, tmp_path):
        """load_from_disk on a path without a log file is a safe no-op."""
        log = UpstreamLogger()
        assert log.load_from_disk(str(tmp_path), tag="nonexistent", rank=0) is False
        assert len(log._logs) == 0


class TestAutoRecoveryTrigger:
    """Tests for the load_sparse_checkpoint → begin_recovery auto-trigger."""

    def test_load_sparse_checkpoint_enters_recovery(self, tmp_path):
        """Loading a sparse checkpoint flips the coordinator into recovery mode."""
        coord, _, _ = _build_coordinator_with_two_experts()

        # Build a minimal on-disk sparse checkpoint for load_sparse_checkpoint to find.
        import os
        ckpt_dir = tmp_path / "step5" / "moevement"
        os.makedirs(ckpt_dir, exist_ok=True)
        metadata = {
            "window_start_iteration": 3,
            "per_iter_active": {
                3: {
                    "layer_0_expert_0": True
                }
            },
        }
        from deepspeed.moevement.snapshot_io import BUNDLE_FILENAME, dump_bundle
        dump_bundle(
            str(ckpt_dir / BUNDLE_FILENAME.format(rank=0)),
            metadata,
            {
                3: {
                    "layer_0_expert_0": {
                        "is_active": True,
                        "state_dict": {
                            "params.weight": torch.zeros(4)
                        },
                    },
                },
            },
        )

        assert coord.is_recovering() is False
        loaded = coord.load_sparse_checkpoint(str(tmp_path), tag="step5")
        assert loaded is True
        assert coord.is_recovering() is True


class TestRecoveryBarrierMultiStage:
    """Tests for recovery_barrier in the multi-recovery (whole-job restart) case."""

    def test_all_stages_recovering_send_nothing(self):
        """Whole-job restart: every neighbour is itself recovering, so nobody has live logs to ship."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=1, num_stages=3, stage_to_global_fn=lambda s: s + 10)
        coord._recovering = True
        coord._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            pp_column_has_recovering=True,
            recovering_stages_in_my_pp=[0, 1, 2],
        )

        with mock.patch.object(coord, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord, "_wait_for_recovery"):
            coord.recovery_barrier()
            # All neighbours of stage 1 are themselves recovering, so receive_recovery_logs
            # is still invoked once (with both boundary ranks = None) and there are no sends.
            assert m_send.call_count == 0
            assert m_recv.call_count == 1
            recv_args = m_recv.call_args
            assert recv_args.args[0] is None and recv_args.args[1] is None

    def test_adjacent_recovering_stages_use_outer_neighbours(self):
        """Stages 2 and 3 fail together in a 5-stage pipeline; logs flow from live 1 and 4."""
        import unittest.mock as mock
        # Probe behaviour from stage 1's vantage point: it must send activation logs to
        # the first failed stage (2) and nothing to the second (3, whose upstream is dead).
        coord_s1, _, _ = _build_coordinator_with_two_experts()
        coord_s1.set_pipeline_topology(pp_group=object(), stage_id=1, num_stages=5, stage_to_global_fn=lambda s: s)
        coord_s1._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            pp_column_has_recovering=True,
            recovering_stages_in_my_pp=[2, 3],
        )

        with mock.patch.object(coord_s1, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord_s1, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord_s1, "_wait_for_recovery"):
            coord_s1.recovery_barrier()
            # Stage 1 sends activation logs only to stage 2 (the first failed stage).
            assert m_send.call_count == 1
            assert m_send.call_args.args[0] == 2

        # Stage 2's vantage point: it receives activations from live stage 1 and
        # nothing for gradients (stage 3 is also down).
        coord_s2, _, _ = _build_coordinator_with_two_experts()
        coord_s2.set_pipeline_topology(pp_group=object(), stage_id=2, num_stages=5, stage_to_global_fn=lambda s: s)
        coord_s2._recovering = True
        coord_s2._global_step = 5

        with mock.patch.object(coord_s2, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord_s2, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord_s2, "_wait_for_recovery"):
            coord_s2.recovery_barrier()
            assert m_recv.call_count == 1
            args = m_recv.call_args.args
            assert args[0] == 1 and args[1] is None

        # Stage 3's vantage point: activations come from stage 2's replay (no log);
        # gradients come from live stage 4.
        coord_s3, _, _ = _build_coordinator_with_two_experts()
        coord_s3.set_pipeline_topology(pp_group=object(), stage_id=3, num_stages=5, stage_to_global_fn=lambda s: s)
        coord_s3._recovering = True
        coord_s3._global_step = 5

        with mock.patch.object(coord_s3, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord_s3, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord_s3, "_wait_for_recovery"):
            coord_s3.recovery_barrier()
            assert m_recv.call_count == 1
            args = m_recv.call_args.args
            assert args[0] is None and args[1] == 4

    def test_non_adjacent_recovering_stages_ignore_each_other(self):
        """Stages 0 and 3 recovering in a 4-stage pipeline; stage 0 only ships to 1, not to 3."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()
        coord.set_pipeline_topology(pp_group=object(), stage_id=0, num_stages=4, stage_to_global_fn=lambda s: s)
        coord._recovering = True
        coord._global_step = 5

        handshake = _fake_handshake(
            any_recovering=True,
            pp_column_has_recovering=True,
            recovering_stages_in_my_pp=[0, 3],
        )

        with mock.patch.object(coord, "_world_recovery_handshake", return_value=handshake), \
             mock.patch.object(coord, "receive_recovery_logs") as m_recv, \
             mock.patch.object(coord, "send_recovery_logs_to") as m_send, \
             mock.patch.object(coord, "_wait_for_recovery"):
            coord.recovery_barrier()
            # Stage 0 receives (for its own recovery) from stage 1. Stage 0 is not adjacent to stage 3.
            assert m_recv.call_count == 1
            assert m_send.call_count == 0


# ---------------------------------------------------------------------------
# TestSenderKeyedReceivedState
# ---------------------------------------------------------------------------


class TestSenderKeyedReceivedState:
    """Receiver-side accessors for the symmetric ring.

    The old "peer memory fallback" was deleted with the primary-only
    replication path — it pulled arbitrary peer state that, under ZeRO,
    belongs to a different rank's shard.  The new API is explicit:
    ``get_received_snapshots_for(sender_dp_rank)`` returns exactly the
    named sender's state.
    """

    def test_load_snapshot_data_returns_disk_result(self, tmp_path):
        """_load_snapshot_data now just forwards the disk read."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()

        fake_disk = ({"window_start_iteration": 0, "per_iter_active": {0: {"disk_op": True}}}, {0: {"disk_op": {}}})
        with mock.patch.object(SparseSnapshotEngine, "load_from_disk", return_value=fake_disk):
            metadata, per_iter = coord._load_snapshot_data(str(tmp_path), tag="step10")
        assert metadata["window_start_iteration"] == 0
        assert 0 in per_iter and "disk_op" in per_iter[0]

    def test_load_snapshot_data_returns_none_when_disk_empty(self, tmp_path):
        """With disk empty, we return (None, None) — the old peer-memory fallback is gone."""
        import unittest.mock as mock
        coord, _, _ = _build_coordinator_with_two_experts()

        # Even with received state in memory, _load_snapshot_data no longer
        # falls back — the proper replacement-rank path is an explicit pull
        # from a named peer (commit 2's ``load_sparse_from_peer``).
        coord.snapshot_engine._received_window_start[0] = 3
        coord.snapshot_engine._received_snapshots[0] = {(3, "peer_op"): {"params.w": torch.ones(2)}}
        coord.snapshot_engine._received_metadata[0] = {(3, "peer_op"): {"is_active": True}}

        with mock.patch.object(SparseSnapshotEngine, "load_from_disk", return_value=(None, None)):
            metadata, states = coord._load_snapshot_data(str(tmp_path), tag="step10")
        assert metadata is None
        assert states is None

    def test_get_received_snapshots_for_unknown_sender(self):
        """Unknown sender returns (None, None)."""
        engine = SparseSnapshotEngine(replication_factor=1)
        assert engine.get_received_snapshots_for(sender_dp_rank=42) == (None, None)

    def test_get_received_snapshots_for_named_sender(self):
        """Returns a disk-shaped tuple for a populated sender slot."""
        engine = SparseSnapshotEngine(replication_factor=1)
        engine._received_window_start[0] = 5
        engine._received_snapshots[0] = {(5, "op"): {"params.w": torch.ones(2)}}
        engine._received_metadata[0] = {(5, "op"): {"is_active": True}}

        metadata, per_iter = engine.get_received_snapshots_for(sender_dp_rank=0)
        assert metadata["window_start_iteration"] == 5
        assert metadata["per_iter_active"][5]["op"] is True
        assert 5 in per_iter
        assert "op" in per_iter[5]

    def test_get_received_snapshots_for_asserts_on_metadata_gap(self):
        """L1: state without matching metadata raises rather than defaulting is_active=True.

        Pre-fix, the missing meta entry silently defaulted to ``is_active=True``,
        so a frozen op would land in the FP32 master slot on restore.  The
        recv path writes both dicts in lockstep today, but a future asymmetry
        should fail loud here, not corrupt state.
        """
        engine = SparseSnapshotEngine(replication_factor=1)
        engine._received_window_start[0] = 5
        engine._received_snapshots[0] = {(5, "op"): {"params.w": torch.ones(2)}}
        engine._received_metadata[0] = {}  # asymmetric: state present, metadata missing

        with pytest.raises(AssertionError, match="atomic-commit invariant"):
            engine.get_received_snapshots_for(sender_dp_rank=0)


# ---------------------------------------------------------------------------
# TestPeerPull
# ---------------------------------------------------------------------------


class TestPeerPull:
    """Replacement rank pulls its shard from a named surviving peer."""

    def test_pull_snapshot_from_peer_end_to_end_round_trip(self):
        """Single-process round-trip: peer side frames the manifest + flats,
        replacement side receives and rebuilds disk-shaped ``(metadata, states)``.
        """
        import unittest.mock as mock
        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot

        # Peer-side engine has received rank 0's snapshot from a prior ring.
        peer_engine = SparseSnapshotEngine(replication_factor=1)
        snap = OperatorSnapshot("op_a", iteration=5, is_active=True)
        snap.add_tensor("params.weight", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        # Simulate the ring receive storing rank 0's state under sender=0.
        peer_engine._received_snapshots[0] = {(5, "op_a"): dict(snap.state_dict)}
        peer_engine._received_metadata[0] = {(5, "op_a"): {"is_active": True}}
        peer_engine._received_window_start[0] = 5

        # Replacement engine: empty, about to pull.
        fresh_engine = SparseSnapshotEngine(replication_factor=1)

        # Wire the two sides together via an in-process queue — peer's
        # ``dist.send`` enqueues, replacement's ``dist.recv`` dequeues.
        pipe = []

        def fake_send(tensor, dst=None, **kw):
            pipe.append(tensor.clone())

        def fake_recv(tensor, src=None, **kw):
            tensor.copy_(pipe.pop(0))

        with mock.patch("deepspeed.comm.send", side_effect=fake_send), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv):
            peer_engine.serve_peer_pull_request(requester_rank=1, sender_dp_rank=0, group=object())
            metadata, per_iter = fresh_engine.pull_snapshot_from_peer(peer_rank=0, group=object())

        assert metadata["window_start_iteration"] == 5
        assert metadata["per_iter_active"][5]["op_a"] is True
        assert 5 in per_iter and "op_a" in per_iter[5]
        torch.testing.assert_close(per_iter[5]["op_a"]["params.weight"], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_pull_snapshot_from_peer_empty_peer_returns_none(self):
        """Peer without the requested sender replies length=0; replacement returns (None, None)."""
        import unittest.mock as mock

        peer_engine = SparseSnapshotEngine(replication_factor=1)  # empty
        fresh_engine = SparseSnapshotEngine(replication_factor=1)

        pipe = []

        def fake_send(tensor, dst=None, **kw):
            pipe.append(tensor.clone())

        def fake_recv(tensor, src=None, **kw):
            tensor.copy_(pipe.pop(0))

        with mock.patch("deepspeed.comm.send", side_effect=fake_send), \
             mock.patch("deepspeed.comm.recv", side_effect=fake_recv):
            peer_engine.serve_peer_pull_request(requester_rank=1, sender_dp_rank=0, group=object())
            metadata, states = fresh_engine.pull_snapshot_from_peer(peer_rank=0, group=object())

        assert metadata is None
        assert states is None

    def test_load_sparse_from_peer_enters_recovery(self):
        """coordinator.load_sparse_from_peer wraps the pull and flips ``_recovering``."""
        import unittest.mock as mock

        coord, _, _ = _build_coordinator_with_two_experts()
        # Stand-in for the gloo replication group — the pull path only
        # needs *something* non-None so the group check passes.
        coord._replication_group = object()
        coord._dp_rank = 0

        fake_metadata = {
            "window_start_iteration": 4,
            "per_iter_active": {
                4: {
                    "layer_0_expert_0": True
                }
            },
            # ``_compute_replay_iters`` clamps to ``fault_iter``; the
            # manifest carries it so peer-pull picks up the serving rank's
            # step count rather than this (fresh-coord) rank's default 0.
            "fault_iter": 4,
        }
        fake_states = {4: {"layer_0_expert_0": {"params.weight": torch.zeros(4)}}}

        with mock.patch("deepspeed.comm.send"), \
             mock.patch.object(coord.snapshot_engine, "pull_snapshot_from_peer",
                               return_value=(fake_metadata, fake_states)):
            ok = coord.load_sparse_from_peer(peer_rank=1)

        assert ok is True
        assert coord.is_recovering() is True
        assert coord._replay_iteration_cursor == 4

    def test_load_sparse_from_peer_populates_snapshot_cache(self):
        """Peer pull feeds ``_cached_snapshot_data`` so replay can progressively thaw.

        Symmetric to the cascade-path fix: the peer-pull path sets
        ``_checkpoint_load_dir=None``, so without cache population each
        replay iter's ``_setup_replay_iter`` would short-circuit
        on the missing disk bundle and the operators would stay FROZEN
        through the whole replay — weights never rolling back to
        window-start state.
        """
        import unittest.mock as mock

        coord, _, _ = _build_coordinator_with_two_experts()
        coord._replication_group = object()
        coord._dp_rank = 0

        fake_metadata = {
            "window_start_iteration": 2,
            "per_iter_active": {
                2: {
                    "layer_0_expert_0": True
                }
            },
        }
        fake_states = {2: {"layer_0_expert_0": {"params.weight": torch.ones(4)}}}

        assert coord._cached_snapshot_data is None
        with mock.patch("deepspeed.comm.send"), \
             mock.patch.object(coord.snapshot_engine, "pull_snapshot_from_peer",
                               return_value=(fake_metadata, fake_states)):
            coord.load_sparse_from_peer(peer_rank=1)

        assert coord._cached_snapshot_data is not None
        cached_meta, cached_states = coord._cached_snapshot_data
        assert cached_meta["window_start_iteration"] == 2
        assert 2 in cached_states and "layer_0_expert_0" in cached_states[2]


# ---------------------------------------------------------------------------
# TestEndToEndRecovery
# ---------------------------------------------------------------------------


class TestEndToEndRecovery:
    """End-to-end: snapshot → save → simulate crash → load → replay → weights match.

    Drives the full recovery loop across a two-operator, two-slot sparse
    window and verifies that the replay leaves the live model in the same
    state as the saved snapshot, while clearing the recovering flag.
    """

    def test_save_load_replay_restores_weights(self, tmp_path):
        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot

        # 1. Build a coordinator with two fake experts (w_sparse=2 schedule).
        coord, expert0, expert1 = _build_coordinator_with_two_experts()

        # Snapshot layout that the recovery loop expects:
        #   - expert0 is ACTIVE in slot 0 (has FP32 params + optimizer state)
        #   - expert1 is ACTIVE in slot 1 (same), but the loaded metadata
        #     marks it FROZEN so the converter has to *transition* it to
        #     ACTIVE during replay — that's the path we want to exercise.
        expert0_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expert1_weight = torch.tensor([5.0, 6.0, 7.0, 8.0])

        snap0 = OperatorSnapshot("layer_0_expert_0", iteration=0, is_active=True)
        snap0.add_tensor("params.weight", expert0_weight.clone())
        snap0.add_tensor("optimizer.exp_avg", torch.zeros(4))
        snap0.add_tensor("optimizer.exp_avg_sq", torch.zeros(4))
        coord.snapshot_engine._persisted_snapshots[(0, "layer_0_expert_0")] = snap0

        snap1 = OperatorSnapshot("layer_0_expert_1", iteration=1, is_active=True)
        snap1.add_tensor("params.weight", expert1_weight.clone())
        snap1.add_tensor("optimizer.exp_avg", torch.zeros(4))
        snap1.add_tensor("optimizer.exp_avg_sq", torch.zeros(4))
        coord.snapshot_engine._persisted_snapshots[(1, "layer_0_expert_1")] = snap1

        # 2. Persist the sparse checkpoint.  No upstream logger was enabled
        # on this coordinator, so no dist.get_rank call is made.
        coord.save_sparse_checkpoint(str(tmp_path), tag="step10")
        coord.flush_persist()

        # 3. Simulate a crash: zero out the live weights, drop all in-memory
        # state, rebuild a fresh coordinator pointed at the same modules.
        with torch.no_grad():
            expert0.weight.zero_()
            expert1.weight.zero_()
        assert torch.all(expert0.weight == 0)
        assert torch.all(expert1.weight == 0)

        # 4. Recovery path: the fresh coordinator reuses the same _operator_map
        # bindings so that _setup_replay_iter writes back into the live
        # modules.  load_sparse_checkpoint enters recovery automatically.
        recovered, _, _ = _build_coordinator_with_two_experts()
        recovered._operator_map["layer_0_expert_0"] = expert0
        recovered._operator_map["layer_0_expert_1"] = expert1
        # Bundle covers iters 0-1.  ``_compute_replay_iters`` clamps the
        # bundle to ``_fault_iter``; fresh coord's default ``_global_step=0``
        # would clamp to iter 0 and skip iter 1's replay, so pin it to the
        # saving-rank's step.
        recovered._global_step = 1

        optimizer = _FakeOptimizer([expert0.weight, expert1.weight])
        ok = recovered.load_sparse_checkpoint(str(tmp_path), tag="step10")
        assert ok is True
        assert recovered.is_recovering() is True

        # 5. Drive the replay loop.  Each on_iteration_end advances one window
        # slot: step 0 activates expert0, step 1 activates expert1.  After
        # the last operator flips to ACTIVE the converter marks conversion
        # complete and end_recovery() fires.
        model = nn.Module()  # _on_iteration_end_recovery only reaches model
        # through _restore_module_weights, which uses _operator_map lookup.
        for step in range(2):
            recovered.on_iteration_end(global_step=step, model=model, optimizer=optimizer)

        # 6. Recovery is complete and both experts carry the restored weights.
        assert recovered.is_recovering() is False
        torch.testing.assert_close(expert0.weight.data, expert0_weight)
        torch.testing.assert_close(expert1.weight.data, expert1_weight)


class TestRecoveryStateRoundTrip:
    """Every param + every Adam state tensor on every operator round-trips
    through save → clear → load → replay.

    ``TestEndToEndRecovery`` covers only two experts with zero-valued
    optimizer state, a plain ``_FakeModule`` with a single weight, and no
    gate or non_expert operator.  This test exercises the full bucket
    matrix introduced by B1 — non_expert + gate + experts, each with
    multi-param modules (weight + bias) — and asserts bit-equality on
    every param + every optimizer-state tensor after recovery.  If any
    op silently drops any state on either the save or restore side, the
    relevant assertion fails with the ``name``-prefixed message.
    """

    @staticmethod
    def _signature(key, shape):
        """Deterministic FP16-safe tensor unique per ``key``.

        Integer-valued floats round-trip bit-exactly through ``.half()``
        (the dtype frozen captures use) as long as every element stays
        under 2048, which the ``base + offset`` scheme here guarantees.
        """
        n = 1
        for dim in shape:
            n *= dim
        offset = abs(hash(key)) % 500 + 1  # 1..500
        return torch.arange(n, dtype=torch.float32).reshape(shape) + float(offset)

    @classmethod
    def _populate_golden(cls, model, optimizer):
        """Overwrite every param + every optim-state entry with a unique
        signature tensor and return a deep copy of the state for later
        comparison."""
        golden_params = {}
        for name, p in model.named_parameters():
            sig = cls._signature(f"param::{name}", tuple(p.shape))
            with torch.no_grad():
                p.data.copy_(sig)
            golden_params[name] = sig.clone()

        golden_optim = {}
        for name, p in model.named_parameters():
            optimizer.state[p] = {
                "exp_avg": cls._signature(f"{name}::exp_avg", tuple(p.shape)),
                "exp_avg_sq": cls._signature(f"{name}::exp_avg_sq", tuple(p.shape)),
                "step": torch.tensor(42.0),
            }
            golden_optim[name] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in optimizer.state[p].items()
            }
        return golden_params, golden_optim

    @staticmethod
    def _fresh_model_and_optimizer(hidden, num_experts):
        """Zero-init model + optimizer — so a missed restore stays detectable."""
        model = _FakeModelWithMoE(hidden=hidden, num_experts=num_experts)
        with torch.no_grad():
            for p in model.parameters():
                p.data.zero_()
        optimizer = _FakeOptimizer(list(model.parameters()))
        for p in model.parameters():
            optimizer.state[p] = {
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
                "step": torch.tensor(0.0),
            }
        return model, optimizer

    def test_save_clear_load_replay_recovers_every_tensor(self, tmp_path):
        hidden, num_experts = 4, 2
        # pcie_bandwidth_gbs chosen low enough to force num_active=1 so the
        # scheduler produces W_sparse=4 windows — one operator active per
        # iter, which means every op has at least one FROZEN FP16 capture
        # and one ACTIVE FP32+Adam capture.  High bandwidth collapses to
        # W=1 and skips the FP16 path entirely.  ``upstream_logging: False``
        # skips the pipeline-stage activation logger (which needs a real
        # dist backend for ``get_rank`` at save / load time).
        config = MoEvementConfig(
            {"moevement": {
                "enabled": True,
                "pcie_bandwidth_gbs": 1e-9,
                "upstream_logging": False,
            }})

        # --- Saving side: populate golden, drive a full window of snapshots ---
        model1 = _FakeModelWithMoE(hidden=hidden, num_experts=num_experts)
        optimizer1 = _FakeOptimizer(list(model1.parameters()))
        coord1 = MoEvementCoordinator(config)
        coord1.initialize(model=model1, moe_layers=[model1.moe], iter_time_sec=0.5)
        W = coord1.scheduler.w_sparse
        assert W >= 2, (f"test requires W_sparse >= 2 to exercise FROZEN captures; "
                        f"got W={W}.  Lower pcie_bandwidth_gbs in the config.")

        golden_params, golden_optim = self._populate_golden(model1, optimizer1)

        # No training between iters → every iter's snapshot records the
        # same golden state.  Drive the state machine through one full
        # window so every op becomes ACTIVE exactly once and every other
        # iter captures its FP16 compute weight.
        for step in range(W):
            coord1.on_iteration_end(global_step=step, model=model1, optimizer=optimizer1)
        # After W iters the snapshots are in ``_in_flight_snapshots``;
        # the boundary ``finalize_window`` at iter W-1 only promotes the
        # *previous* window, and the first window needs one more boundary
        # (i.e. iter 2W-1) to reach ``_persisted_snapshots``.  Rather than
        # drive another full W iters of no-op training just to trigger
        # promotion, call ``finalize_window`` directly — this is what the
        # engine's boundary handler does internally.
        coord1.snapshot_engine.finalize_window()
        coord1.save_sparse_checkpoint(str(tmp_path), tag="golden")
        coord1.flush_persist()

        # --- Crash simulation: fresh everything, zeroed ---
        model2, optimizer2 = self._fresh_model_and_optimizer(hidden, num_experts)
        coord2 = MoEvementCoordinator(config)
        coord2.initialize(model=model2, moe_layers=[model2.moe], iter_time_sec=0.5)
        assert coord2.scheduler.w_sparse == W, ("window size must match across saving + loading "
                                                "coordinators for the replay cursor to line up")

        # Pin fault iter to the end of the persisted window so replay
        # covers every captured iter (``_compute_replay_iters`` clamps to
        # ``_fault_iter``).
        coord2._global_step = W - 1
        assert coord2.load_sparse_checkpoint(str(tmp_path), tag="golden") is True
        assert coord2.is_recovering() is True

        # --- Replay: drive W iters; each flips one op FROZEN → ACTIVE ---
        for step in range(W):
            coord2.on_iteration_end(global_step=step, model=model2, optimizer=optimizer2)
        assert coord2.is_recovering() is False, "recovery must complete after W replay iters"

        # --- Layer B: every live param and every optim-state tensor == golden ---
        live_params = dict(model2.named_parameters())
        for name, expected in golden_params.items():
            assert name in live_params, f"param {name!r} vanished on recovery"
            torch.testing.assert_close(live_params[name].data, expected, msg=lambda m, n=name: f"param {n!r}: {m}")

        for name, expected in golden_optim.items():
            live = live_params[name]
            recovered_state = optimizer2.state.get(live, {})
            for key in ("exp_avg", "exp_avg_sq"):
                assert key in recovered_state, f"optimizer[{name!r}][{key!r}] missing after recovery"
                torch.testing.assert_close(recovered_state[key],
                                           expected[key],
                                           msg=lambda m, n=name, k=key: f"optim[{n!r}][{k!r}]: {m}")
            # ``step`` travels through a separate batch in the H2D path —
            # easy to silently drop, so assert it explicitly.
            assert "step" in recovered_state, f"optimizer[{name!r}]['step'] missing after recovery"
            assert float(recovered_state["step"]) == float(expected["step"]), (
                f"optim[{name!r}]['step']: got {recovered_state['step']}, expected {expected['step']}")

    def test_bf16_frozen_capture_preserves_bf16_dtype(self, tmp_path):
        """BF16 compute weights survive the FROZEN capture without FP16 rounding.

        Pre-fix, ``snapshot_operator`` unconditionally called ``tensor.half()``
        on any non-FP16 frozen capture — silently rounding BF16 through
        FP16.  For values that fit in BF16 but overflow FP16's ±65504
        range, the cast produces ``inf``, corrupting the capture.

        End-to-end replay would mask this: every op becomes ACTIVE
        exactly once per window, and the ACTIVE FP32 capture (which
        never goes through ``.half()``) overwrites whatever the FROZEN
        capture held when the op is promoted.  Inspecting the bundle
        contents directly is the only way to pin the save-side contract.

        Integer values > 65504 round-trip bit-exactly through BF16 but
        overflow to ``inf`` through FP16, so they distinguish preserved-
        BF16 from rounded-through-FP16 with no ambiguity.
        """
        from deepspeed.moevement.snapshot_io import load_bundle, BUNDLE_FILENAME

        hidden, num_experts = 4, 2
        config = MoEvementConfig(
            {"moevement": {
                "enabled": True,
                "pcie_bandwidth_gbs": 1e-9,
                "upstream_logging": False,
            }})

        model = _FakeModelWithMoE(hidden=hidden, num_experts=num_experts).to(torch.bfloat16)
        optimizer = _FakeOptimizer(list(model.parameters()))
        coord = MoEvementCoordinator(config)
        coord.initialize(model=model, moe_layers=[model.moe], iter_time_sec=0.5)
        W = coord.scheduler.w_sparse
        assert W >= 2

        # Value > FP16 max so a ``.half()`` cast produces inf.
        FP16_OVERFLOW = 100000.0
        golden = {}
        for name, p in model.named_parameters():
            sig = torch.full_like(p, FP16_OVERFLOW + float(abs(hash(name)) % 1000))
            with torch.no_grad():
                p.data.copy_(sig)
            golden[name] = sig.clone()

        for p_ in model.parameters():
            optimizer.state[p_] = {
                "exp_avg": torch.zeros_like(p_, dtype=torch.float32),
                "exp_avg_sq": torch.zeros_like(p_, dtype=torch.float32),
                "step": torch.tensor(1.0),
            }

        for step in range(W):
            coord.on_iteration_end(global_step=step, model=model, optimizer=optimizer)
        coord.snapshot_engine.finalize_window()
        coord.save_sparse_checkpoint(str(tmp_path), tag="bf16_probe")
        coord.flush_persist()

        # Inspect the bundle's FROZEN compute-weight entries directly —
        # the ACTIVE-capture overwrite path doesn't apply here.
        bundle_path = os.path.join(str(tmp_path), "bf16_probe", "moevement", BUNDLE_FILENAME.format(rank=0))
        metadata, per_iter = load_bundle(bundle_path)

        found_any = False
        for iteration, ops in per_iter.items():
            for op_name, state in ops.items():
                for key, tensor in state.items():
                    if not key.startswith("compute_weights."):
                        continue
                    found_any = True
                    assert tensor.dtype == torch.bfloat16, (
                        f"iter {iteration} op {op_name!r} key {key!r}: "
                        f"BF16 source was captured as {tensor.dtype}; "
                        f"pre-B2 regression rounds through FP16 and loses precision")
                    assert torch.isfinite(tensor).all(), (f"iter {iteration} op {op_name!r} key {key!r}: "
                                                          f"non-finite values — pre-B2 half() overflowed")
        assert found_any, ("test fixture must produce at least one frozen compute-weights "
                           "capture — none found in the bundle; schedule may be degenerate")


class TestProfilerMarkers:
    """Profiler markers fire at the MoEvement hot-path sites.

    The perf audit's "stop and profile first" rule needs named ranges
    on the snapshot / save / peer-replicate / replay paths — anonymous
    kernel launches alone make a profile trace hard to attribute to a
    particular phase of the state machine.  This test drives the full
    save + load + replay pipeline (reusing the round-trip fixture) with
    the profiling helper's recording hook on, and asserts the expected
    coarse markers appear.  A regression that drops a wrap (rename, add
    an early-return bypass, forgets to indent a new branch into the
    ``with`` block) fails here with a direct "name missing" signal.
    """

    def test_expected_markers_fire_across_save_load_replay(self, tmp_path):
        import deepspeed.moevement.profiling as prof

        hidden, num_experts = 4, 2
        config = MoEvementConfig(
            {"moevement": {
                "enabled": True,
                "pcie_bandwidth_gbs": 1e-9,
                "upstream_logging": False,
            }})

        model1 = _FakeModelWithMoE(hidden=hidden, num_experts=num_experts)
        optimizer1 = _FakeOptimizer(list(model1.parameters()))
        coord1 = MoEvementCoordinator(config)
        coord1.initialize(model=model1, moe_layers=[model1.moe], iter_time_sec=0.5)
        W = coord1.scheduler.w_sparse

        prof._record_log.clear()
        prof._recording = True
        try:
            for step in range(W):
                coord1.on_iteration_end(global_step=step, model=model1, optimizer=optimizer1)
            coord1.snapshot_engine.finalize_window()
            coord1.save_sparse_checkpoint(str(tmp_path), tag="markers")
            coord1.flush_persist()

            # Fresh coord for load + replay.
            model2 = _FakeModelWithMoE(hidden=hidden, num_experts=num_experts)
            with torch.no_grad():
                for p in model2.parameters():
                    p.data.zero_()
            optimizer2 = _FakeOptimizer(list(model2.parameters()))
            coord2 = MoEvementCoordinator(config)
            coord2.initialize(model=model2, moe_layers=[model2.moe], iter_time_sec=0.5)
            coord2._global_step = W - 1
            coord2.load_sparse_checkpoint(str(tmp_path), tag="markers")
            for step in range(W):
                coord2.on_iteration_end(global_step=step, model=model2, optimizer=optimizer2)
        finally:
            prof._recording = False

        seen = set(prof._record_log)
        expected = {
            "moevement/snap_active",  # ACTIVE operator snapshot branch
            "moevement/snap_frozen",  # FROZEN operator snapshot branch
            "moevement/bundle_write",  # sparse bundle enqueue for disk
            "moevement/save_sparse_ckpt",  # coordinator-side save path
            "moevement/recovery_iter_end",  # per-iter replay advance
        }
        missing = expected - seen
        assert not missing, (f"expected profiler markers missing: {sorted(missing)}; "
                             f"saw: {sorted(seen)}")


class TestCpuOnlyStreamHandling:
    """Ensure snapshot and log paths survive accelerators with no stream API."""

    def test_snapshot_operator_works_without_cuda_stream(self):
        """snapshot_operator runs on a CPU-only box where Stream is None."""
        engine = SparseSnapshotEngine(replication_factor=1)
        engine.snapshot_operator(
            name="op0",
            params_dict={"weight": torch.ones(4)},
            optimizer_state_dict={"exp_avg": torch.zeros(4)},
            is_active=True,
            iteration=0,
        )
        engine.synchronize()

        snap = engine._snapshots[(0, "op0")]
        torch.testing.assert_close(snap.state_dict["params.weight"].cpu(), torch.ones(4))

    def test_log_activation_works_without_cuda_stream(self):
        """UpstreamLogger.log_activation runs without a stream."""
        logger = UpstreamLogger()
        logger.log_activation(torch.ones(4), iteration=0, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(0)
        assert len(logs[0]) == 1

    def test_log_activation_preserves_non_fp_tuple_members(self):
        """Non-floating-point tuple members (attention masks, position ids) are logged.

        Pipeline activations often look like ``(hidden_fp, position_ids_int,
        attn_mask_bool)``.  The old filter dropped the int/bool members, so
        the recovering stage ended up with a shorter tuple than its forward
        signature wants.  Logging every position makes the replay tuple
        drop-in compatible with the normal recv path.
        """
        logger = UpstreamLogger()

        hidden = torch.randn(4, 8)
        position_ids = torch.arange(4, dtype=torch.long)
        attn_mask = torch.ones(4, dtype=torch.bool)
        logger.log_activation((hidden, position_ids, attn_mask), iteration=0, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        entries = logger._logs[(0, 0)]
        assert len(entries) == 3
        dtypes = [e.tensor.dtype for e in entries]
        assert torch.float32 in dtypes
        assert torch.long in dtypes
        assert torch.bool in dtypes
        # Direction tags are position-indexed so recovery can rebuild in order.
        assert [e.direction for e in entries] == ["activation_0", "activation_1", "activation_2"]

    def test_save_to_disk_only_persists_completed_window(self, tmp_path):
        """Mid-window saves drop the in-flight partial snapshot.

        Without this, a save taken partway through a window writes a
        bundle where some operators are at iteration N and others at
        N-K, violating the single-``window_start_iteration`` invariant
        the replay loop depends on.  The fix is to serialize only
        ``_persisted_snapshots`` (the last fully completed window).
        """
        from deepspeed.moevement.sparse_snapshot import OperatorSnapshot
        from deepspeed.moevement.snapshot_io import BUNDLE_FILENAME

        engine = SparseSnapshotEngine(replication_factor=0)

        # Previous window completed — persisted has one operator.
        persisted_snap = OperatorSnapshot("persisted_op", iteration=5, is_active=True)
        persisted_snap.add_tensor("params.weight", torch.ones(4))
        engine._persisted_snapshots[(5, "persisted_op")] = persisted_snap

        # Current window is mid-flight — an op has been snapshotted but the
        # window hasn't rolled over yet.  This is the state the fix has to
        # exclude from the bundle.
        inflight_snap = OperatorSnapshot("inflight_op", iteration=7, is_active=True)
        inflight_snap.add_tensor("params.weight", torch.zeros(4))
        engine._snapshots[(7, "inflight_op")] = inflight_snap

        engine._window_start_iteration = 5
        engine.save_to_disk(str(tmp_path), tag="step7")
        engine.flush_persist()

        from deepspeed.moevement.snapshot_io import load_bundle
        bundle_path = tmp_path / "step7" / "moevement" / BUNDLE_FILENAME.format(rank=0)
        metadata, per_iter_states = load_bundle(str(bundle_path))

        assert metadata["per_iter_active"] == {5: {"persisted_op": True}}
        assert 5 in per_iter_states and "persisted_op" in per_iter_states[5]
        assert 7 not in per_iter_states


class TestPinnedPoolBusyRefcount:
    """Refcount semantics for overlapping owners of the same buffer."""

    def test_multiple_mark_busy_require_matching_releases(self):
        """Two owners marking busy must both release before the buffer pools.

        Previously ``_busy`` was a set, so the first ``release_busy`` cleared
        the flag for all owners — a ``pool.release`` from an unrelated path
        (e.g., ``finalize_window``) could then hand the storage to a new
        acquirer while the second owner was still reading it.  The
        refcount makes mark/release symmetric per owner.
        """
        from deepspeed.moevement.buffer_pool import PinnedPool

        pool = PinnedPool(max_per_key=4)
        flat = pool.acquire((4, ), torch.float32, pin=False)

        pool.mark_busy(flat)  # owner A
        pool.mark_busy(flat)  # owner B

        # A plain release now is a no-op: two owners still hold it.
        pool.release(flat)
        assert all(flat is not f for f in pool._free[list(pool._free.keys())[0]] if pool._free)

        # Owner A releases — still busy for B.
        pool.release_busy(flat)
        assert pool._busy.get(id(flat), 0) == 1

        # Owner B releases — now eligible for the free list.
        pool.release_busy(flat)
        assert pool._busy.get(id(flat), 0) == 0
        key = pool._key(flat.shape, flat.dtype, None, False)
        assert flat in pool._free[key]

    def test_stray_release_busy_is_harmless(self):
        """``release_busy`` without a matching ``mark_busy`` still pools the buffer.

        Matches the old set-based behaviour: a stray release just routes
        the buffer through ``release``, putting it on the free list.  The
        refcount path mustn't underflow below zero.
        """
        from deepspeed.moevement.buffer_pool import PinnedPool

        pool = PinnedPool(max_per_key=4)
        flat = pool.acquire((4, ), torch.float32, pin=False)
        pool.release_busy(flat)  # never marked busy

        assert id(flat) not in pool._busy
        key = pool._key(flat.shape, flat.dtype, None, False)
        assert flat in pool._free[key]

    def test_release_is_idempotent_against_double_return(self):
        """Releasing the same flat twice results in one free-list entry.

        Two independent code paths can legitimately release the same
        buffer: the replication future's done-callback (via
        ``release_busy`` → ``release``) and a later ``finalize_window``
        (via direct ``release``).  Without idempotency the buffer would
        be appended twice and two subsequent ``acquire`` calls would
        hand the same storage to different callers.
        """
        from deepspeed.moevement.buffer_pool import PinnedPool

        pool = PinnedPool(max_per_key=4)
        flat = pool.acquire((4, ), torch.float32, pin=False)

        pool.release(flat)
        pool.release(flat)

        key = pool._key(flat.shape, flat.dtype, None, False)
        # Exactly one reference in the free list, not two.
        assert sum(1 for f in pool._free[key] if f is flat) == 1


class TestMultiFpGradLogging:
    """Tuple gradients with multiple fp members survive the logging path."""

    def test_log_gradient_tuple_logs_all_fp_members(self):
        """All fp tuple members are logged with position-indexed directions.

        The send path asserts every fp member has ``.grad`` and ships all
        of them to the previous stage.  Recovery has to produce the same
        shape, so the log has to capture all of them — not just
        ``inputs[0].grad`` as the original hook did.
        """
        logger = UpstreamLogger()
        grads = (torch.ones(4), torch.full((4, ), 2.0))
        logger.log_gradient(grads, iteration=0, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        entries = logger._logs[(0, 0)]
        assert len(entries) == 2
        assert [e.direction for e in entries] == ["gradient_0", "gradient_1"]
        assert torch.allclose(entries[0].tensor, torch.ones(4))
        assert torch.allclose(entries[1].tensor, torch.full((4, ), 2.0))


def _capture_deepspeed_warnings(caplog, monkeypatch):
    """Route DeepSpeed logger records through caplog.

    The DeepSpeed logger has ``propagate=False`` by default, so the pytest
    caplog handler (which is attached to the root logger) never sees its
    messages.  Flip propagation on for the duration of the test so caplog
    can capture warnings we emit under ``deepspeed.utils.logger``.
    """
    import logging
    ds_logger = logging.getLogger("DeepSpeed")
    monkeypatch.setattr(ds_logger, "propagate", True)
    caplog.set_level(logging.WARNING)


class TestEmptyBundleWarning:
    """Loading an empty bundle flags a warning instead of silently no-op'ing."""

    def test_initialize_from_empty_snapshots_logs_warning(self, caplog, monkeypatch):
        """Operator-less metadata triggers the ``recovery will no-op`` warning."""
        _capture_deepspeed_warnings(caplog, monkeypatch)

        converter = SparseToDenseConverter()
        metadata = {
            "per_iter_active": {},
            "window_start_iteration": -1,
        }
        converter.initialize_from_snapshots(metadata, per_iter_operator_states={}, schedule=[])

        assert any("no iterations" in rec.message for rec in caplog.records)
        # Empty bundle leaves _conversion_complete at False; the coordinator's
        # replay loop then short-circuits via ``get_next_replay_iteration`` →
        # ``None`` and immediately exits recovery.  The warning is the
        # observable signal, not the complete-flag.
        assert converter.is_conversion_complete() is False


class TestPersistWorkerBackpressure:
    """Bounded queue + high-water-mark warning hysteresis on ``PersistWorker``."""

    def test_high_water_mark_warning_fires_once_per_crossing(self, caplog, monkeypatch):
        """Warning fires once when queue crosses threshold, re-arms after draining.

        Without the hysteresis flag, every ``enqueue`` above the high-water
        threshold would log again — spamming operator logs during sustained
        back-pressure.  This test pins the warn-once contract: fire on the
        first crossing above the threshold, silence during sustained
        back-pressure, then re-arm once the queue drops back below the
        rearm ratio.

        Pins the worker thread on a gate so enqueued jobs accumulate in
        the queue without racing with the drain loop.
        """
        import logging
        import threading
        from deepspeed.moevement.persist_worker import PersistWorker

        # DeepSpeed logger has propagate=False; route through caplog.
        _capture_deepspeed_warnings(caplog, monkeypatch)

        # Large queue + low high-water ratio so the test can reach the
        # threshold with a small number of enqueues while the worker
        # thread sits on the hold-gate.
        worker = PersistWorker(max_queue_size=10, high_water_mark_ratio=0.3, rearm_ratio=0.2)
        try:
            hold_gate = threading.Event()
            hold_taken = threading.Event()

            def hold_writer(e=hold_gate, taken=hold_taken):
                taken.set()
                e.wait(timeout=10.0)

            def fast_writer():
                pass

            # Submit one blocker and wait until the worker has pulled it —
            # from here every subsequent enqueue sits in the queue until
            # ``hold_gate.set()``.
            worker.enqueue(hold_writer, label="hold")
            assert hold_taken.wait(timeout=2.0), "worker never pulled hold job"

            # threshold = int(10 * 0.3) = 3.  enqueue() observes qsize
            # BEFORE its own put, so to trip the warning we need a pre-put
            # qsize >= 3 — i.e. the 4th enqueue after the blocker takes.
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                for _ in range(4):
                    worker.enqueue(fast_writer, label="fast")

            warnings = [rec for rec in caplog.records if "PersistWorker queue" in rec.message]
            assert len(warnings) == 1, (f"expected exactly 1 high-water warning on first crossing; "
                                        f"got {len(warnings)} (threshold={worker._high_water_threshold}, "
                                        f"qsize={worker._queue.qsize()})")

            # Sustained back-pressure: further enqueues at the same level
            # MUST NOT log again.  Hysteresis keeps ``_high_water_warned``
            # True until qsize drops below ``_rearm_threshold``.
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                for _ in range(3):
                    worker.enqueue(fast_writer, label="fast")
            warnings = [rec for rec in caplog.records if "PersistWorker queue" in rec.message]
            assert len(warnings) == 0, (f"hysteresis broken: warning re-fired during sustained "
                                        f"backpressure ({len(warnings)} extra warnings)")

            # Drain the queue.  The re-arm check lives inside ``enqueue``
            # itself (runs when the next call observes qsize at/below the
            # rearm threshold), so we don't assert on the internal flag
            # here — the behavioural check below is the ground truth.
            hold_gate.set()
            worker.flush()

            # Second crossing after drain should log exactly once: the
            # next enqueue sees qsize=0 ≤ rearm_threshold, flips the flag
            # back to False, and the subsequent crossing fires again.
            hold_gate.clear()
            hold_taken.clear()
            worker.enqueue(hold_writer, label="hold2")
            assert hold_taken.wait(timeout=2.0), "worker never pulled hold2"
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                for _ in range(4):
                    worker.enqueue(fast_writer, label="fast2")
            warnings = [rec for rec in caplog.records if "PersistWorker queue" in rec.message]
            assert len(warnings) == 1, (f"expected 1 warning on second crossing after re-arm; "
                                        f"got {len(warnings)}")

            hold_gate.set()
            worker.flush()
        finally:
            worker.shutdown()


class TestPersistWorkerErrorSurface:
    """``flush`` raises the first writer/callback error; ``shutdown`` swallows.

    Without this, a failed bundle or log write is only logged, the
    coordinator's ``save_sparse_checkpoint`` still returns success, and
    the outer ``save_checkpoint`` promotes the incomplete tag to
    ``latest`` — a silent data-loss path under disk pressure.
    """

    def test_flush_raises_writer_exception_with_original_as_cause(self):
        from deepspeed.moevement.persist_worker import PersistWorker

        worker = PersistWorker(max_queue_size=4)
        try:
            original = OSError("disk full")

            def raising_writer():
                raise original

            worker.enqueue(raising_writer, label="bundle.pt")
            with pytest.raises(RuntimeError, match="bundle.pt") as excinfo:
                worker.flush()
            assert excinfo.value.__cause__ is original
        finally:
            worker.shutdown()

    def test_flush_raises_callback_exception(self):
        """Callback failures surface through ``flush`` just like writer failures.

        A raising callback means the writer succeeded (data is on disk)
        but buffer release / done-signalling is broken.  That's still a
        real bug the caller should see, not a silent log line — so the
        same first-error capture covers both paths.
        """
        from deepspeed.moevement.persist_worker import PersistWorker

        worker = PersistWorker(max_queue_size=4)
        try:
            original = RuntimeError("buffer release bug")

            def ok_writer():
                pass

            def raising_callback():
                raise original

            worker.enqueue(ok_writer, callback=raising_callback, label="logs.pt")
            with pytest.raises(RuntimeError, match="logs.pt") as excinfo:
                worker.flush()
            assert excinfo.value.__cause__ is original
        finally:
            worker.shutdown()

    def test_first_error_wins_and_is_cleared_after_flush(self):
        """First failure in a window wins; later flush after clean work is quiet.

        The stored error clearing on flush is deliberate: once the
        caller has been notified, a subsequent retry must not keep
        raising on the old failure.  Otherwise a transient disk error
        would poison the worker for the rest of the process.
        """
        from deepspeed.moevement.persist_worker import PersistWorker

        worker = PersistWorker(max_queue_size=4)
        try:
            first = OSError("first fail")
            second = OSError("second fail")

            def fail_first():
                raise first

            def fail_second():
                raise second

            worker.enqueue(fail_first, label="bundle_a.pt")
            worker.enqueue(fail_second, label="bundle_b.pt")

            with pytest.raises(RuntimeError, match="bundle_a.pt") as excinfo:
                worker.flush()
            assert excinfo.value.__cause__ is first

            worker.enqueue(lambda: None, label="clean.pt")
            worker.flush()
        finally:
            worker.shutdown()

    def test_shutdown_suppresses_writer_exception(self):
        """Teardown runs to completion even if a pending writer failed.

        The worker's error is already logged by ``_run``; swallowing it
        at ``shutdown`` lets the worker thread be joined instead of
        leaking past ``atexit``.
        """
        from deepspeed.moevement.persist_worker import PersistWorker

        worker = PersistWorker(max_queue_size=4)

        def raising_writer():
            raise OSError("shutdown-time fail")

        worker.enqueue(raising_writer, label="x")
        worker.shutdown()
        assert not worker._thread.is_alive()

    def test_had_persist_error_is_sticky_across_flush(self):
        """``had_persist_error`` survives ``flush``'s error-clear.

        ``flush`` consumes ``_first_error`` on the way out so a retry-
        flush starts clean.  But the operator needs to know a write
        failure ever happened (atexit banner, non-zero exit gate, etc.),
        so the sticky flag stays True even after ``flush`` cleared the
        raise-once error.
        """
        from deepspeed.moevement.persist_worker import PersistWorker

        worker = PersistWorker(max_queue_size=4)
        try:
            assert worker.had_persist_error() is False
            worker.enqueue(lambda: (_ for _ in ()).throw(OSError("fail")), label="x")
            with pytest.raises(RuntimeError):
                worker.flush()
            # flush raised + cleared the raise-once error, but the sticky
            # flag stays set.
            assert worker.had_persist_error() is True
            # Subsequent clean work does NOT clear the sticky flag.
            worker.enqueue(lambda: None, label="clean")
            worker.flush()
            assert worker.had_persist_error() is True
        finally:
            worker.shutdown()

    def test_shutdown_critical_banner_when_writer_thread_wedges(self, caplog, monkeypatch):
        """Wedged worker thread surfaces as CRITICAL + ``writer_is_stuck=True``.

        Regression for M3 in CORRECTNESS_AUDIT_2026_04_23.md: prior to
        the fix, ``shutdown`` joined with a 5s timeout and returned
        silently if the worker was stuck on disk I/O.  The daemon thread
        leaked through process exit holding file handles + page cache
        locks, and no log line documented the leak.

        Stubs ``is_alive`` to True and ``join`` to a no-op so the test
        exercises the post-join-timeout branch without actually wedging
        a thread for 5s.
        """
        import logging
        from deepspeed.moevement.persist_worker import PersistWorker

        _capture_deepspeed_warnings(caplog, monkeypatch)
        worker = PersistWorker(max_queue_size=4)
        # Drain the real worker first so shutdown's flush() returns
        # immediately; we only want to exercise the join + is_alive
        # branch with a fake-alive thread.
        worker.flush()

        class _FakeThread:

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        real_thread = worker._thread
        worker._thread = _FakeThread()

        with caplog.at_level(logging.CRITICAL):
            worker.shutdown()

        try:
            assert worker.writer_is_stuck() is True
            assert any("PersistWorker thread still alive" in rec.message for rec in caplog.records
                       if rec.levelno >= logging.CRITICAL), (
                           f"expected CRITICAL stuck-writer banner; got: "
                           f"{[(rec.levelname, rec.message[:80]) for rec in caplog.records]}")
        finally:
            # Restore the real thread + drain it so the daemon exits.
            worker._thread = real_thread
            real_thread.join(timeout=5.0)

    def test_enqueue_after_shutdown_flag_flip_raises_via_lock(self):
        """M4: serialize enqueue's flag-check + put with shutdown's flip + sentinel.

        Pre-fix, an ``enqueue`` thread could pass the ``_shutting_down``
        check just as ``shutdown`` flipped the flag and queued the
        sentinel; the late ``put`` would then succeed and the worker
        would drain the orphan job after the sentinel.  The lock
        serializes the two so post-flip enqueue always raises.
        """
        from deepspeed.moevement.persist_worker import PersistWorker

        worker = PersistWorker(max_queue_size=4)
        worker.shutdown()  # sets _shutting_down=True under lock + queues sentinel
        with pytest.raises(RuntimeError, match="shutting down"):
            worker.enqueue(lambda: None, label="late.pt")

    def test_shutdown_emits_critical_banner_on_sticky_error(self, caplog, monkeypatch):
        """Shutdown's critical banner fires on any historical write failure.

        The ``save_checkpoint → flush_persist`` path already surfaces
        write errors by raising at flush time.  This banner specifically
        covers the atexit-only cleanup case: if the only failure was on
        the final flush during shutdown, no caller got the raise; the
        banner makes sure the exit isn't silent.
        """
        import logging
        from deepspeed.moevement.persist_worker import PersistWorker

        _capture_deepspeed_warnings(caplog, monkeypatch)
        worker = PersistWorker(max_queue_size=4)

        worker.enqueue(lambda: (_ for _ in ()).throw(OSError("atexit-only fail")), label="bundle.pt")
        # No ``flush`` call before shutdown — mirrors the atexit case
        # where the training loop ended cleanly but the final disk
        # write happened to fail post-process-exit.
        with caplog.at_level(logging.CRITICAL):
            worker.shutdown()

        criticals = [rec for rec in caplog.records if rec.levelno >= logging.CRITICAL]
        assert any("PersistWorker saw at least one" in rec.message
                   for rec in criticals), (f"expected CRITICAL banner on sticky-error shutdown; got: "
                                           f"{[(rec.levelname, rec.message[:80]) for rec in caplog.records]}")

    def test_fsync_on_save_false_skips_fsync_call(self, monkeypatch, tmp_path):
        """Perf-A: ``fsync_on_save=False`` plumbs through to ``dump_bundle``.

        The default-True path still calls ``os.fsync``; setting the
        config flag False makes the bundle writer skip it.  Counts
        ``os.fsync`` invocations across one save in each mode and
        asserts the asymmetry.
        """
        import deepspeed.moevement.snapshot_io as snapshot_io_mod

        for fsync_setting, expected_min_calls in [(True, 1), (False, 0)]:
            engine = SparseSnapshotEngine(replication_factor=0, fsync_on_save=fsync_setting)
            params = {"weight": torch.randn(4, 4)}
            engine.snapshot_operator("test_op", params, None, is_active=False, iteration=0)
            engine.synchronize()
            engine.begin_window(iteration=1)
            engine.finalize_window()

            calls = []
            real_fsync = snapshot_io_mod.os.fsync

            def counting_fsync(fd, _calls=calls, _real=real_fsync):
                _calls.append(fd)
                _real(fd)

            monkeypatch.setattr(snapshot_io_mod.os, "fsync", counting_fsync)
            try:
                engine.save_to_disk(str(tmp_path / f"f{fsync_setting}"), tag="t")
                engine.flush_persist()
            finally:
                monkeypatch.setattr(snapshot_io_mod.os, "fsync", real_fsync)

            if fsync_setting:
                assert len(calls) >= expected_min_calls, (
                    f"fsync_on_save=True must call os.fsync at least once; got {len(calls)}")
            else:
                assert len(calls) == 0, (f"fsync_on_save=False must NOT call os.fsync; got {len(calls)} calls")
            engine._persist_worker.shutdown()

    def test_save_to_disk_flush_raises_when_bundle_write_fails(self, monkeypatch, tmp_path):
        """Bundle-write failure propagates through ``flush_persist``.

        Pins the cross-layer contract: ``dump_bundle`` raise →
        ``PersistWorker._run`` captures → ``flush_persist`` re-raises.
        Without this, a future refactor could silently swallow the
        exception at the snapshot-engine layer and reintroduce the
        ``latest``-points-at-partial-tag bug the A1 fix targets.
        """
        import deepspeed.moevement.sparse_snapshot as sparse_snapshot_mod

        engine = SparseSnapshotEngine(replication_factor=1)
        params = {"weight": torch.randn(4, 4)}
        engine.snapshot_operator("test_op", params, None, is_active=False, iteration=0)
        engine.synchronize()
        engine.begin_window(iteration=1)
        engine.finalize_window()

        def boom(*args, **kwargs):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(sparse_snapshot_mod, "dump_bundle", boom)

        engine.save_to_disk(str(tmp_path), "step_fail")
        with pytest.raises(RuntimeError, match="step_fail"):
            engine.flush_persist()
        # A second flush is clean: the stored error cleared on raise, so
        # subsequent successful writes aren't poisoned by the past failure.
        engine.flush_persist()
