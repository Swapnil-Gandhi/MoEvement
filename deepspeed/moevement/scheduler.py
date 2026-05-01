"""Sparse checkpoint scheduling policy (Algorithm 1 from MoEvement paper).

Determines the sparse checkpoint window size (W_sparse) and generates
a per-iteration schedule of which operators are active (full FP32 snapshot)
vs frozen (FP16 compute weights only).
"""

import collections
import math

from deepspeed.utils import logger


class OperatorInfo:
    """Metadata for a single operator (expert, non-expert, or gating).

    Attributes:
        name: Unique identifier for this operator.
        num_params: Number of parameters in this operator.
        activation_count: Cumulative token activation count (for experts).
        is_expert: Whether this operator is an expert.
        layer_id: The layer index this operator belongs to.
        local_expert_id: Local expert index (None for non-expert/gating operators).
    """

    def __init__(self, name, num_params, is_expert=True, layer_id=0, local_expert_id=None):
        self.name = name
        self.num_params = num_params
        self.activation_count = 0
        self.is_expert = is_expert
        self.layer_id = layer_id
        self.local_expert_id = local_expert_id

    def __repr__(self):
        return f"OperatorInfo(name={self.name}, params={self.num_params}, act_count={self.activation_count})"


class CheckpointSchedule:
    """A single iteration's checkpoint assignment.

    Attributes:
        active_operators: List of operator names whose FP32 state is checkpointed.
        frozen_operators: List of operator names that only store FP16 compute weights.
    """

    def __init__(self, active_operators, frozen_operators):
        self.active_operators = active_operators
        self.frozen_operators = frozen_operators


class SparseCheckpointScheduler:
    """Implements Algorithm 1 from the MoEvement paper.

    Computes the sparse checkpoint window size and generates per-iteration
    schedules that assign operators to active or frozen states.
    """

    # Bytes per parameter for each state component (mixed-precision FP16-FP32 with Adam)
    FP32_MASTER_WEIGHT_BYTES = 4
    FP32_OPTIMIZER_STATE_BYTES = 8  # Adam: momentum (4) + variance (4)
    FP16_COMPUTE_WEIGHT_BYTES = 2

    def __init__(self,
                 pcie_bandwidth_bytes_per_sec,
                 reorder_threshold=0.10,
                 reorder_fraction=0.25,
                 activation_count_window_iters=100):
        self.pcie_bandwidth = pcie_bandwidth_bytes_per_sec
        self.reorder_threshold = reorder_threshold
        self.reorder_fraction = reorder_fraction
        self.activation_count_window_iters = activation_count_window_iters
        self.operators = []
        self.w_sparse = 1
        self.schedule = []
        self._prev_popularity_order = None
        # Per-expert token counts accumulated for the *current* iteration,
        # written to by ``update_activation_counts`` as each MoE layer's
        # forward completes.  Flushed into ``_interval_history`` by
        # ``tick_interval`` at every on_iteration_end boundary.
        self._pending_interval = collections.defaultdict(int)
        # Rolling window of per-iteration {op_name: token_count} dicts.
        # Oldest at the left; capped at ``activation_count_window_iters``.
        # This is the substrate the paper's §3.5 "frequency change" metric
        # operates on — rates, not cumulative totals, so late-training
        # shifts remain detectable.
        self._interval_history = collections.deque(maxlen=activation_count_window_iters)
        # Running window-sum kept in lockstep with ``_interval_history``:
        # add the new interval's counts when it's pushed, subtract the
        # evicted interval's counts when the deque rolls.  Lets
        # ``tick_interval`` refresh per-expert ``activation_count`` in
        # ``O(active experts in this iter)`` instead of the
        # ``O(window_iters * num_experts)`` full rebuild that fired every
        # iter on the snapshot hot path.
        self._window_totals = collections.defaultdict(int)
        # Per-layer expert-op index, built once in ``register_operators``.
        # Avoids scanning the full operator list on every
        # ``update_activation_counts`` call (one call per MoE layer per
        # iter on the snapshot hot path).
        self._expert_ops_by_layer = collections.defaultdict(list)
        # Dedupe set for CC6's one-shot warning on oversized
        # ``local_expert_id`` — keys are ``(layer_id, op.name)``.  First
        # miss logs; subsequent mismatches for the same pair stay silent.
        self._oversized_expert_warned = set()

    def register_operators(self, operators):
        """Register the list of operators to schedule.

        Args:
            operators: List of OperatorInfo objects.
        """
        self.operators = operators
        # Rebuild the per-layer expert-ops index.  Cheap (runs once) and
        # lets ``update_activation_counts`` iterate only that layer's
        # experts instead of filtering the full operator list each call.
        self._expert_ops_by_layer.clear()
        for op in operators:
            if op.is_expert and op.local_expert_id is not None:
                self._expert_ops_by_layer[op.layer_id].append(op)

    def find_window_size(self, iter_time_sec, overlap_target=1.0):
        """Pick (w_sparse, num_active) for the given per-iter PCIe budget.

        Args:
            iter_time_sec: Duration of a single training iteration in seconds.
            overlap_target: Fraction of one iter's PCIe budget the per-iter
                snapshot may consume.  ``1.0`` (default) preserves the
                historical recovery-optimal behavior — pick the SMALLEST
                ``w_sparse`` such that the snapshot fits in one iter (paper
                §3.5 Algorithm 1).  ``< 1.0`` widens ``w_sparse`` to leave
                compute headroom for the side-stream D2H to drain without
                blocking next iter's ``optimizer.step`` — the perf-optimal
                regime at production scale.  E.g. ``0.5`` picks the
                LARGEST ``w_sparse`` such that per-iter D2H ≤ half of
                ``iter_time``, leaving 2× compute headroom.

        Returns:
            Tuple of ``(w_sparse, num_active_per_iter)``.

        Two regimes from a single algorithm:

        - ``overlap_target == 1.0``: scan ``num_active`` from ``total_ops``
          DOWN.  First fit wins → smallest ``w_sparse`` (paper algo).
        - ``overlap_target < 1.0``: scan ``num_active`` from 1 UP.  Last
          fit before exceeding the tighter budget wins → largest
          ``w_sparse`` whose per-iter snapshot honors the headroom target.

        Both regimes use the same per-slot byte formula: pricing the
        WORST slot (the one whose active set carries the most params)
        rather than slot 0 — earlier versions priced ``operators[:num_active]``
        only, which always sat near the head of the ordered list (mostly-
        expert ops).  Tail slots carry ``non_expert`` (transformer
        backbone, often 10x+ bigger than any single expert) and silently
        blew the budget.  Cost is monotonic in active-params-in-slot, so
        the worst slot is the one with the most active params.
        """
        total_ops = len(self.operators)
        if total_ops == 0:
            return 1, 0

        s_compute = self.FP16_COMPUTE_WEIGHT_BYTES
        s_master = self.FP32_MASTER_WEIGHT_BYTES
        s_optim = self.FP32_OPTIMIZER_STATE_BYTES
        total_params = sum(op.num_params for op in self.operators)
        budget_bytes = iter_time_sec * self.pcie_bandwidth * max(0.01, overlap_target)

        def _slot_bytes(num_active):
            max_active_params = max(
                sum(op.num_params for op in self.operators[start:start + num_active])
                for start in range(0, total_ops, num_active))
            return (s_master + s_optim) * max_active_params + s_compute * (total_params - max_active_params)

        if overlap_target >= 1.0:
            # Recovery-optimal: smallest w_sparse that fits in budget.
            # Scan num_active high → low; first fit (= largest num_active
            # = smallest w_sparse) wins.  Paper Algorithm 1 line 10 uses
            # ``while O_Active > 2``; we permit ``num_active == 1`` because
            # that yields a larger ``w_sparse`` rather than a hard failure.
            num_active = total_ops
            while num_active > 1:
                if _slot_bytes(num_active) <= budget_bytes:
                    break
                num_active -= 1
        else:
            # Perf-optimal: largest w_sparse whose snapshot still honors
            # the tighter ``overlap_target × iter_time`` budget.  Scan
            # num_active low → high; last fit (= smallest num_active
            # = largest w_sparse) wins.  If even ``num_active = 1``
            # overruns the budget, fall back to it (matching the
            # ``num_active = 1`` floor of the recovery-optimal path).
            num_active = 1
            for candidate in range(1, total_ops + 1):
                if _slot_bytes(candidate) > budget_bytes:
                    break
                num_active = candidate

        w_sparse = math.ceil(total_ops / num_active)
        # Loop above terminates at ``num_active == 1`` even if that slot
        # still overruns the per-iter PCIe budget (paper's deviation,
        # kept here so a slow-PCIe config yields a larger window rather
        # than a hard failure).  But the snapshot stream then falls
        # behind iter 1 and downstream assumptions ("one window's D2H
        # fits in one iter" — pipelined replication, the ``begin_recovery``
        # 2*w_sparse check) silently absorb the overrun as wall-clock
        # stretch, which is ~impossible to attribute from a profile
        # alone.  Surface the overrun here as a single warning so the
        # operator can re-tune ``pcie_bandwidth_gbs`` (if measured
        # conservative) or shrink the largest operator (typically
        # ``non_expert``).  Use the *unscaled* iter budget for this
        # check so a tight ``overlap_target`` doesn't spuriously fire it.
        worst_bytes = _slot_bytes(num_active)
        unscaled_budget = iter_time_sec * self.pcie_bandwidth
        if worst_bytes > unscaled_budget:
            logger.warning(
                "[MoEvement] Schedule PCIe budget cannot be met: worst-slot snapshot = %.1f MB, "
                "iter budget = %.1f MB (pcie_bandwidth_gbs * iter_time); w_sparse chosen = %d, "
                "num_active = %d.  Snapshot stream will fall behind training; either raise "
                "pcie_bandwidth_gbs if measured conservative, or shrink the largest operator "
                "(typically non_expert).", worst_bytes / 1e6, unscaled_budget / 1e6, w_sparse, num_active)
        return w_sparse, num_active

    def order_operators(self):
        """Sort operators by ascending activation frequency (popularity).

        Experts come first, sorted ascending by activation count (least
        popular at the head of the expert section).  Non-expert / gating
        operators are placed at the **tail** of the schedule: they fire on
        every token, so their effective popularity is maximal — paper Fig. 6
        matches this layout and defers their snapshot D2H to the window's
        final iter for overlap with the backward pass.

        Tail placement is sensitive to a known FP16 limitation in the
        ``w_sparse > 1`` recovery path: stage-1 catch-up replay can drift
        ~10 FP16 ULP outside the fault-free envelope on the first-activated
        expert, more than head placement does on the same param.  The
        ``TestMoEvementRecoveryEquivalenceMultiWindow`` and
        ``TestMoEvementMiddleStageFailurePP3`` tolerances accommodate this
        (atol = 3e-3 ≈ 30 ULP near 1.0); a structural fix would need
        bit-exact catch-up replay (activation logs persisted across
        whole-job restart so replay can re-feed exact stage-0→1 inputs).

        Returns:
            List of OperatorInfo with experts first (ascending by
            activation count), non-expert / gating operators at the tail.
        """
        experts = [op for op in self.operators if op.is_expert]
        non_experts = [op for op in self.operators if not op.is_expert]
        experts.sort(key=lambda op: op.activation_count)
        return experts + non_experts

    def generate_schedule(self, iter_time_sec, w_sparse_override=None):
        """Generate the full sparse checkpoint schedule.

        Args:
            iter_time_sec: Duration of a single training iteration in seconds.
            w_sparse_override: When supplied, skip ``find_window_size`` and
                use this value verbatim.  The coordinator pins ``w_sparse``
                across the world via ``all_reduce(MAX)`` so DP peers (whose
                bundles must be exchangeable for peer-pull / replication)
                and PP peers (whose ``recovery_barrier`` is a global op)
                share one cadence.  Passing ``None`` falls back to the
                stage-local sizing used by single-rank tests.

        Returns:
            Tuple of (w_sparse, list of CheckpointSchedule for each iteration in the window).
        """
        ordered = self.order_operators()
        self.operators = ordered

        if w_sparse_override is not None:
            # ``num_active`` derives from the global w_sparse.  When the
            # global max exceeds this rank's op count, the trailing slots
            # carry empty active sets — a "rest" iter with no D2H — which
            # keeps the schedule index-aligned with peers that have more
            # ops in the same window.
            w_sparse = w_sparse_override
            num_active = max(1, math.ceil(len(ordered) / max(1, w_sparse)))
        else:
            w_sparse, num_active = self.find_window_size(iter_time_sec)
        self.w_sparse = w_sparse

        schedule = []
        for i in range(w_sparse):
            start = i * num_active
            end = min(start + num_active, len(ordered))
            active_names = [op.name for op in ordered[start:end]]
            # Per paper Algorithm 1, slot ``i``'s frozen set is strictly
            # the not-yet-captured tail (``ordered[end:]``) — operators
            # captured ACTIVE in earlier slots of the same window are
            # already done and shouldn't appear here.  An earlier version
            # emitted ``ordered`` minus the active set (which included
            # earlier-slot ACTIVEs) and relied on a coordinator-side
            # ``_already_active_in_window`` filter to drop them at
            # capture time — two invariants for the same fact, with no
            # cross-check.  Restricting at the source removes the cache.
            frozen_names = [op.name for op in ordered[end:]]
            schedule.append(CheckpointSchedule(active_operators=active_names, frozen_operators=frozen_names))

        self.schedule = schedule
        self._prev_popularity_order = [op.name for op in ordered if op.is_expert]

        logger.info(f"[MoEvement] Sparse checkpoint schedule: W_sparse={w_sparse}, "
                    f"active_per_iter={num_active}, total_operators={len(ordered)}")

        return w_sparse, schedule

    def should_reorder(self):
        """Check if expert activation frequencies have changed enough to trigger reordering.

        Paper §3.5: "MoEvement reorders operators when activation frequencies
        change by over 10% for at least 25% of experts."  We implement this
        as a rate comparison between the older half and the newer half of
        the rolling activation-count window: an expert "counts as changed"
        when ``|newer_rate - older_rate| / max(older_rate, eps)`` exceeds
        ``reorder_threshold``.  The reorder fires when the fraction of
        changed experts meets ``reorder_fraction``.

        Returns:
            True if reordering should be triggered.
        """
        if self._prev_popularity_order is None:
            return True

        experts = [op for op in self.operators if op.is_expert]
        if len(experts) == 0:
            return False

        # Require a full window of history before making a rate-based
        # decision — comparing a half-populated window to itself would give
        # misleading rates (the "older half" may be empty iterations at
        # startup).  This is a deliberate quiet period; we'd rather miss a
        # few iterations of detection than reorder on noise.
        if len(self._interval_history) < self.activation_count_window_iters:
            return False

        half = len(self._interval_history) // 2
        older_counts = self._window_sums(0, half)
        newer_counts = self._window_sums(half, len(self._interval_history))
        older_iters = max(half, 1)
        newer_iters = max(len(self._interval_history) - half, 1)

        changed_count = 0
        for op in experts:
            older_rate = older_counts.get(op.name, 0) / older_iters
            newer_rate = newer_counts.get(op.name, 0) / newer_iters
            # ``max(older_rate, eps)`` lets a dormant→active transition
            # register as a strictly-greater-than-threshold change rather
            # than a divide-by-zero.
            denom = max(older_rate, 1e-9)
            relative_delta = abs(newer_rate - older_rate) / denom
            if relative_delta > self.reorder_threshold:
                changed_count += 1

        fraction_changed = changed_count / len(experts)
        return fraction_changed >= self.reorder_fraction

    def update_activation_counts(self, layer_id, exp_counts):
        """Add one MoE layer's per-expert token counts to the pending iteration.

        Coordinator invokes this once per MoE layer on every training
        iteration, passing the layer's ``exp_counts`` tensor produced by
        the gate.  Preferred callers pass a CPU tensor — the coordinator
        batches the per-layer D2H copies into a single stream sync before
        calling (see ``_schedule_exp_counts_copies`` /
        ``_fence_exp_counts_copies``).  Legacy callers / tests that pass
        a GPU tensor still work via a synchronous ``.cpu()`` fallback.

        Convert the CPU tensor to a Python list once via ``tolist()`` and
        index that list per expert; the prior per-expert ``.item()``
        emitted one ``aten::item`` event per call (cheap individually but
        adds up across MoE layers × microbatches, ~16 events / iter at
        HIDDEN=4096 NUM_EXPERTS=16).  ``tolist()`` is one event total.

        Args:
            layer_id: The layer index.
            exp_counts: Tensor of shape [num_experts] with token counts per expert.
        """
        if exp_counts is None:
            return

        if exp_counts.device.type != 'cpu':
            exp_counts = exp_counts.detach().cpu()
        counts_list = exp_counts.tolist()
        n_counts = len(counts_list)
        for op in self._expert_ops_by_layer.get(layer_id, ()):
            if op.local_expert_id < n_counts:
                self._pending_interval[op.name] += counts_list[op.local_expert_id]
            else:
                # Operator discovery saw more experts on this layer than
                # the gate tensor reports.  The rolling-window reorder
                # decision would otherwise paper over this with zero
                # counts, biasing toward undercounted ops.  One warning
                # per ``(layer_id, op.name)`` pair — further mismatches
                # stay silent so the log doesn't flood.
                key = (layer_id, op.name)
                if key not in self._oversized_expert_warned:
                    self._oversized_expert_warned.add(key)
                    logger.warning(f"[MoEvement] layer {layer_id} op {op.name}: "
                                   f"local_expert_id {op.local_expert_id} >= gate exp_counts "
                                   f"length {n_counts}; dropping count.  Operator/gate mismatch?")

    def tick_interval(self):
        """Finalize the current iteration's counts into the rolling window.

        Called once per training iteration from the coordinator's
        ``on_iteration_end`` — after all MoE layers have reported via
        ``update_activation_counts``.  Also refreshes each expert's
        ``activation_count`` so ``order_operators`` uses window-aggregated
        popularity (rather than full-history cumulative, which would freeze
        the ordering after a few thousand iterations).

        Maintains ``_window_totals`` incrementally: subtract the evicted
        interval's counts when the deque rolls, then add the new
        interval's counts.  Per-expert refresh becomes
        ``O(active experts in this iter)`` rather than the
        ``O(window_iters * num_experts)`` full rebuild it used to be.
        """
        new_interval = dict(self._pending_interval)
        self._pending_interval.clear()

        # Subtract the about-to-be-evicted oldest interval before the
        # ``append`` rolls it off — only fires once the deque is full.
        # The ``in`` guard makes the push/evict invariant explicit: every
        # name in the evicted dict should already be in ``_window_totals``
        # because a prior push added it.  A defaultdict's silent
        # zero-create would leave a phantom negative entry if the
        # invariant ever broke via an upstream regression.
        if len(self._interval_history) == self._interval_history.maxlen:
            evicted = self._interval_history[0]
            for name, count in evicted.items():
                if name in self._window_totals:
                    self._window_totals[name] -= count
                    if self._window_totals[name] == 0:
                        del self._window_totals[name]

        self._interval_history.append(new_interval)

        for name, count in new_interval.items():
            self._window_totals[name] += count

        for op in self.operators:
            if op.is_expert:
                op.activation_count = self._window_totals.get(op.name, 0)

    def _window_sums(self, start, stop):
        """Sum per-expert counts across ``_interval_history[start:stop]``.

        Returns a dict mapping op_name → integer total, omitting experts
        that saw no activations in the slice.
        """
        totals = collections.defaultdict(int)
        for i in range(start, stop):
            if i < 0 or i >= len(self._interval_history):
                continue
            for name, count in self._interval_history[i].items():
                totals[name] += count
        return totals

    def get_schedule_for_iteration(self, global_step, window_start=0):
        """Get the checkpoint schedule for a specific iteration.

        Args:
            global_step: The current training iteration number.
            window_start: First iteration of the current sparse window.
                Used to align slot 0 with the window's first iter even
                after a mid-training schedule regen — without this, a
                regen that changes ``w_sparse`` mid-window leaves slot
                indexing rotated by ``global_step % w_sparse_new``,
                disrupting the paper's tail-slot-overlap-with-next-window
                story for one full window per regen.

        Returns:
            CheckpointSchedule for this iteration, or None if no schedule exists.
        """
        if not self.schedule:
            return None
        idx = (global_step - window_start) % self.w_sparse
        return self.schedule[idx]
