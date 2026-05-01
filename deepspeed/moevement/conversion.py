"""Sparse-to-dense checkpoint conversion for MoEvement.

Incrementally reconstructs a logically consistent dense checkpoint from
a sparse checkpoint's per-iteration snapshots.  One sparse checkpoint
covers ``w_sparse`` consecutive iterations; each iteration's snapshot
carries:

- For the operator that becomes ACTIVE at that iteration: FP32 master
  weights and optimizer state.
- For every operator still FROZEN at that iteration: FP16 compute
  weights (each iteration preserves its own FP16 snapshot — needed for
  exact equivalence because a frozen op's weights evolve iter-by-iter
  in the pre-fault run).
- Nothing for operators that already transitioned to ACTIVE earlier in
  the window (spec §3.2: "nothing is captured after active").

During recovery the converter feeds the coordinator's replay loop: at
replay iteration I it exposes that iteration's new-active FP32/optimizer
state, that iteration's still-frozen FP16 weights, and tracks which
operators have already been promoted in the current replay.
"""

from collections import OrderedDict
from enum import Enum

from deepspeed.utils import logger
from deepspeed.moevement.sparse_snapshot import _RNG_PSEUDO_OP


class OperatorState(Enum):
    """Execution state of an operator during sparse-to-dense conversion."""
    FROZEN = "frozen"
    ACTIVE = "active"


class SparseToDenseConverter:
    """Manages the incremental reconstruction of a dense checkpoint.

    During recovery, operators start FROZEN and transition to ACTIVE as
    the replay loop promotes them at each snapshot iteration.  The
    converter caches the entire sparse checkpoint's per-iter state at
    ``initialize_from_snapshots`` time, so per-iter lookups during replay
    are plain dict accesses.

    The conversion is complete when all operators are in the ACTIVE state.
    """

    def __init__(self):
        self._operator_states = OrderedDict()  # name -> OperatorState (current replay state)
        # Per-iter caches: ``{iteration: {op_name: {param_key: tensor}}}``.
        # Populated once at ``initialize_from_snapshots`` by splitting each
        # snapshot's state_dict on its key prefix (``compute_weights.`` /
        # ``params.`` / ``optimizer.``).  Not every op has an entry in every
        # iter — an op absent from ``_fp16_weights[I]`` was either ACTIVE
        # at some iter ≤ I (captured as FP32 there, nothing further) or
        # simply wasn't in the schedule.
        self._fp16_weights_per_iter = {}
        self._fp32_weights_per_iter = {}
        self._optimizer_states_per_iter = {}
        # Per-iter torch RNG state captured at on_iteration_end and shipped
        # in the bundle as a pseudo-operator (see ``_RNG_PSEUDO_OP``).  Each
        # entry is the post-iter-K state — applied at ``_setup_replay_iter(K)``
        # so the next replay tb's forward consumes the same RNG stream as
        # fault-free iter K+1's forward.  Empty dict on bundles that
        # predate per-iter RNG capture (no entry → no restore, the RNG
        # simply advances naturally — same behavior as before this feature).
        self._rng_state_per_iter = {}
        self._conversion_complete = False
        self._replay_iterations = []  # original-iteration numbers in replay order
        self._current_replay_idx = 0

    def initialize_from_snapshots(self, metadata, per_iter_operator_states, schedule):
        """Ingest a sparse checkpoint's per-iteration snapshots.

        Args:
            metadata: Bundle metadata dict.  Must carry ``per_iter_active``
                ``{iteration: {op_name: bool}}`` so the converter can
                initialize ``_operator_states`` to the split at the
                earliest captured iteration (ops ACTIVE at the first
                iter start ACTIVE; every other op starts FROZEN and is
                promoted later by ``activate_operators`` as the replay
                loop reaches that op's active iteration).
            per_iter_operator_states: ``{iteration: {op_name: state_dict}}``
                where ``state_dict`` may contain ``params.*`` (FP32),
                ``optimizer.*`` (optimizer state), and/or
                ``compute_weights.*`` (FP16) keys — as emitted by
                ``SparseSnapshotEngine.save_to_disk`` / ring replication.
            schedule: List of ``CheckpointSchedule`` objects describing
                the window (used by the coordinator for active-operator
                lookup; stored here for reference).
        """
        del schedule  # retained in coordinator; converter uses per-iter data directly
        self._operator_states.clear()
        self._fp16_weights_per_iter.clear()
        self._fp32_weights_per_iter.clear()
        self._optimizer_states_per_iter.clear()
        self._rng_state_per_iter = {}

        if not per_iter_operator_states:
            self._conversion_complete = False
            logger.warning("[MoEvement] Loaded sparse checkpoint contains no iterations; "
                           "recovery will complete immediately as a no-op")
            return

        per_iter_active = metadata.get("per_iter_active", {}) if isinstance(metadata, dict) else {}
        # Feed the shared ``ingest_iteration`` path in sorted order so the
        # first-iter operator-state seed (ACTIVE/FROZEN split from the
        # earliest iter) lands exactly as the pre-incremental version did.
        for iteration in sorted(per_iter_operator_states.keys()):
            self.ingest_iteration(iteration,
                                  per_iter_operator_states[iteration],
                                  iter_active=per_iter_active.get(iteration))

        active_count = sum(1 for s in self._operator_states.values() if s == OperatorState.ACTIVE)
        frozen_count = len(self._operator_states) - active_count
        logger.info(f"[MoEvement] Initialized conversion across {len(per_iter_operator_states)} iterations: "
                    f"{active_count} initially-active, {frozen_count} initially-frozen operators")

    def ingest_iteration(self, iteration, iter_op_states, iter_active=None):
        """Add one iteration's state to the per-iter caches incrementally.

        The streaming peer-pull path calls this per-iter as each
        mini-manifest lands, so the replacement can start replaying
        iter N while iter N+1's flats are still arriving over the
        wire.  ``initialize_from_snapshots`` also routes through here
        (iterating in sorted iter order) so the two ingest paths
        share code.

        The **first** call seeds ``_operator_states`` under the
        assumption that ``iteration`` is the earliest iter — true by
        the streaming sender's sorted-order guarantee and by
        ``initialize_from_snapshots``'s explicit sort.  An op seen in
        that first iter starts ACTIVE iff the bundle flagged it active
        AND the iter carried its FP32 payload; every other op starts
        FROZEN.  Subsequent calls register any newly-seen ops as
        FROZEN; they stay FROZEN until ``activate_operators`` promotes
        them at their active replay iter.

        Args:
            iteration: Iter number this ingest belongs to.
            iter_op_states: ``{op_name: state_dict}`` for this iter —
                values are the raw per-op state with ``compute_weights.``
                / ``params.`` / ``optimizer.`` prefixes.  The RNG pseudo-
                op is routed to ``_rng_state_per_iter`` and dropped from
                the operator space.
            iter_active: ``{op_name: bool}`` active flag for this iter,
                or ``None`` when unavailable.  Consulted only on the
                first ingest (to set the earliest-iter split); later
                calls ignore it since an op's ACTIVE/FROZEN decision
                at replay start is locked at the earliest iter.
        """
        is_first = not self._operator_states
        self._ingest_single_iter_splits(iteration, iter_op_states)

        op_names_this_iter = [n for n in iter_op_states if n != _RNG_PSEUDO_OP]
        if is_first:
            earliest_fp32 = self._fp32_weights_per_iter.get(iteration, {})
            active_flags = iter_active or {}
            for name in op_names_this_iter:
                starts_active = active_flags.get(name, False) and name in earliest_fp32
                self._operator_states[name] = OperatorState.ACTIVE if starts_active else OperatorState.FROZEN
        else:
            # Any op appearing after the earliest iter cannot have been
            # active at the earliest iter, so it starts FROZEN and is
            # promoted by ``activate_operators`` when replay reaches
            # its active iter.  Skip ops already registered — this
            # call may see ops that were handled by the first ingest.
            for name in op_names_this_iter:
                if name not in self._operator_states:
                    self._operator_states[name] = OperatorState.FROZEN

        self._conversion_complete = all(s == OperatorState.ACTIVE for s in self._operator_states.values())

    def _ingest_single_iter_splits(self, iteration, iter_states):
        """Split one iter's op state_dicts into the three per-iter caches.

        Shared between bulk ``initialize_from_snapshots`` (via
        ``ingest_iteration``) and the streaming path.  Does not touch
        ``_operator_states``; ``ingest_iteration`` owns that seeding.
        """
        # The RNG pseudo-operator rides the per-iter byte stream to reuse
        # the bundle serializer; route it to the per-iter RNG cache and
        # skip it in the operator iteration below so the replay
        # schedule never sees it.
        rng_entry = iter_states.get(_RNG_PSEUDO_OP)
        if rng_entry:
            self._rng_state_per_iter[iteration] = dict(rng_entry)
        fp16 = {}
        fp32 = {}
        opt = {}
        for op_name, state_dict in iter_states.items():
            if op_name == _RNG_PSEUDO_OP:
                continue
            op_fp16 = {
                k[len("compute_weights."):]: v
                for k, v in state_dict.items() if k.startswith("compute_weights.")
            }
            op_fp32 = {k[len("params."):]: v for k, v in state_dict.items() if k.startswith("params.")}
            op_opt = {k[len("optimizer."):]: v for k, v in state_dict.items() if k.startswith("optimizer.")}
            # Surface unknown prefixes so a future extension that adds a
            # new bundle prefix via ``snapshot_operator`` doesn't silently
            # fail to restore.
            for k in state_dict:
                if not (k.startswith("compute_weights.") or k.startswith("params.") or k.startswith("optimizer.")):
                    logger.warning(f"[MoEvement] ingest_iteration: dropping unknown key "
                                   f"prefix {k!r} from operator {op_name} — add a handler for "
                                   "this prefix in conversion.py")
            if op_fp16:
                fp16[op_name] = op_fp16
            if op_fp32:
                fp32[op_name] = op_fp32
            if op_opt:
                opt[op_name] = op_opt
        if fp16:
            self._fp16_weights_per_iter[iteration] = fp16
        if fp32:
            self._fp32_weights_per_iter[iteration] = fp32
        if opt:
            self._optimizer_states_per_iter[iteration] = opt

    def activate_operators(self, iteration, operator_names):
        """Transition operators from FROZEN to ACTIVE at the given replay iteration.

        Called by the coordinator's replay loop as each replay iter promotes
        its newly-active operators.  The FP32 weights and optimizer state for
        these operators live at ``self._fp32_weights_per_iter[iteration]`` /
        ``self._optimizer_states_per_iter[iteration]`` — this call only flips
        the state; the coordinator separately copies the payload into the
        live model via ``_restore_module_weights`` / ``_apply_optim_state_into_params``.
        """
        del iteration  # kept in the signature so future bookkeeping can use it
        for name in operator_names:
            if name in self._operator_states:
                self._operator_states[name] = OperatorState.ACTIVE

        self._conversion_complete = all(s == OperatorState.ACTIVE for s in self._operator_states.values())

        if self._conversion_complete:
            logger.info("[MoEvement] Sparse-to-dense conversion complete: all operators active")

    def is_operator_active(self, name):
        """Check if an operator is in the ACTIVE state."""
        return self._operator_states.get(name, OperatorState.FROZEN) == OperatorState.ACTIVE

    def is_operator_frozen(self, name):
        """Check if an operator is in the FROZEN state."""
        return self._operator_states.get(name, OperatorState.FROZEN) == OperatorState.FROZEN

    def is_conversion_complete(self):
        """Check if all operators have been transitioned to ACTIVE state."""
        return self._conversion_complete

    def get_operator_state(self, name):
        """Get the current state of an operator."""
        return self._operator_states.get(name, OperatorState.FROZEN)

    def get_active_operators(self):
        """Get names of all currently-active operators."""
        return [name for name, state in self._operator_states.items() if state == OperatorState.ACTIVE]

    def get_frozen_operators(self):
        """Get names of all currently-frozen operators."""
        return [name for name, state in self._operator_states.items() if state == OperatorState.FROZEN]

    def get_fp32_weights(self, name, iteration):
        """Get FP32 master weights captured for ``name`` at ``iteration``.

        Returns ``None`` if the operator had no FP32 capture at that
        iteration (either because it was FROZEN there, or because it had
        already become ACTIVE earlier in the window).
        """
        return self._fp32_weights_per_iter.get(iteration, {}).get(name)

    def get_optimizer_state(self, name, iteration):
        """Get optimizer state captured for ``name`` at ``iteration``.

        Returns ``None`` under the same conditions as :func:`get_fp32_weights`.
        """
        return self._optimizer_states_per_iter.get(iteration, {}).get(name)

    def get_fp16_weights(self, name, iteration):
        """Get FP16 compute weights captured for ``name`` at ``iteration``.

        Returns ``None`` if the operator had no FP16 capture at that
        iteration (the op was ACTIVE there, or had already been active
        earlier in the window and was no longer being captured).
        """
        return self._fp16_weights_per_iter.get(iteration, {}).get(name)

    def get_rng_state(self, iteration):
        """Get torch RNG state captured at ``iteration`` end.

        Returns ``None`` when the bundle has no RNG capture at that iter
        (older bundles that predate per-iter RNG, or catch-up iters beyond
        the last persisted window).  The coordinator's
        ``_setup_replay_iter`` no-ops on ``None`` so RNG advances naturally
        from whatever state the prior tb left.
        """
        return self._rng_state_per_iter.get(iteration)

    def set_replay_iterations(self, iterations):
        """Set the list of original-iteration numbers to replay, in order."""
        self._replay_iterations = list(iterations)
        self._current_replay_idx = 0

    def get_next_replay_iteration(self):
        """Get the next original-iteration number to replay.

        Returns ``None`` once the replay sequence is exhausted.
        """
        if self._current_replay_idx >= len(self._replay_iterations):
            return None
        iteration = self._replay_iterations[self._current_replay_idx]
        self._current_replay_idx += 1
        return iteration

    def get_remaining_replay_count(self):
        """Get the number of iterations still to replay."""
        return len(self._replay_iterations) - self._current_replay_idx

    def should_skip_weight_grad(self, operator_name):
        """Check if weight-gradient computation should be skipped for an operator.

        Frozen operators skip weight-gradient computation and optimizer updates,
        performing only forward and input-gradient computations.
        """
        return self.is_operator_frozen(operator_name)

    def should_skip_optimizer_step(self, operator_name):
        """Check if optimizer step should be skipped for an operator."""
        return self.is_operator_frozen(operator_name)

    def clear(self):
        """Reset all conversion state."""
        self._operator_states.clear()
        self._fp16_weights_per_iter.clear()
        self._fp32_weights_per_iter.clear()
        self._optimizer_states_per_iter.clear()
        self._rng_state_per_iter = {}
        self._conversion_complete = False
        self._replay_iterations = []
        self._current_replay_idx = 0

    def drop_iteration(self, iteration):
        """Free one iteration's per-iter caches after replay has consumed it.

        SD-O4 S3.  Each per-iter dict entry is a view into pool-managed
        CPU storage that the snapshot stream's ``non_blocking=True``
        H2D copies read from; the coordinator's ``_drop_replayed_iter``
        synchronises that iter's H2D-completion event before calling
        this so the views can be discarded without racing the copy.
        Releases of the underlying pool buffers happen on the engine
        side via ``release_iter_buffers``.

        Idempotent — dropping an iter that was never ingested (or was
        already dropped) is a no-op.
        """
        self._fp16_weights_per_iter.pop(iteration, None)
        self._fp32_weights_per_iter.pop(iteration, None)
        self._optimizer_states_per_iter.pop(iteration, None)
        self._rng_state_per_iter.pop(iteration, None)
