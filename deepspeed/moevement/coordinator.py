"""MoEvement coordinator: orchestrates sparse checkpointing, conversion, and logging.

The coordinator runs on each worker alongside DeepSpeed, managing the lifecycle
of sparse checkpoints, triggering snapshots each iteration, and coordinating
localized recovery on failure using upstream logs.
"""

import atexit
import contextlib
import statistics
import time
import types
import weakref
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
import torch.nn as nn

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger
from deepspeed.utils.tensor_fragment import (
    safe_get_full_fp32_param,
    safe_set_full_fp32_param,
    safe_set_full_optimizer_state,
)

from deepspeed.moevement.scheduler import SparseCheckpointScheduler, OperatorInfo
from deepspeed.moevement.sparse_snapshot import (
    PEER_PULL_PROTOCOL_BULK,
    PEER_PULL_PROTOCOL_STREAMING,
    PEER_PULL_PROTOCOL_SUPPORTED,
    SparseSnapshotEngine,
)
from deepspeed.moevement.conversion import SparseToDenseConverter
from deepspeed.moevement.upstream_logging import UpstreamLogger
from deepspeed.moevement.comm_rebuild import _abort_or_destroy
from deepspeed.moevement.profiling import trace_range
from deepspeed.runtime.utils import noop_decorator
from deepspeed.runtime.zero.linear import autocast_custom_fwd, autocast_custom_bwd


def _probe_autocast_decorators():
    """Fall back to no-op decorators on accelerators that don't support AMP.

    ``deepspeed.runtime.zero.linear`` binds ``autocast_custom_fwd`` /
    ``autocast_custom_bwd`` via ``get_accelerator().device_name()``.  On
    real training hardware (CUDA, HCCL) that produces working decorators
    that preserve AMP autocast state across the custom forward/backward
    — which is what MoEvement wants, so AMP-enabled runs don't hit a
    dtype mismatch in the saved-tensor matmul.

    On MPS (local Mac testing) and CPU-only nodes, the decorator's
    per-call ``torch.get_autocast_dtype(device_name)`` raises
    ``"unsupported scalarType"`` because there's no autocast dtype
    registered for the device.  Rather than teaching every non-CUDA
    test to mock AMP, we probe the decorator once at import time and
    fall back to ``noop_decorator`` if it fails — AMP behavior is
    already off on those devices, so a no-op is semantically correct.
    """

    class _Probe(torch.autograd.Function):

        @staticmethod
        @autocast_custom_fwd
        def forward(ctx, x):
            return x

        @staticmethod
        @autocast_custom_bwd
        def backward(ctx, g):
            return g

    try:
        _Probe.apply(torch.zeros(1, requires_grad=True))
        return autocast_custom_fwd, autocast_custom_bwd
    except RuntimeError:
        return noop_decorator, noop_decorator


_moevement_custom_fwd, _moevement_custom_bwd = _probe_autocast_decorators()


class _FrozenLinearFunction(torch.autograd.Function):
    """Linear forward whose backward returns grad_input but NOT grad_weight.

    Mirrors the zero-bubble pipeline-parallelism split (sail-sg) in shape:
    weight keeps ``requires_grad=True`` so the autograd graph is built and
    the loss tensor has a ``grad_fn`` (required for ``.backward()`` to run
    through the stage at all — stage 0's input is raw data with
    ``requires_grad=False``, so without the weight's grad entry the graph
    would be empty and backward would raise ``"element 0 of tensors does
    not require grad"``).  What we skip is the weight-grad GEMM itself:
    returning ``None`` for the weight's grad slot tells autograd not to
    accumulate into ``weight.grad``, so no wgrad compute happens and
    ``weight.grad`` stays whatever it was coming in (typically ``None``).
    Pipeline p2p still receives the returned ``grad_input`` so the
    upstream stage can step through its own backward.

    ``@autocast_custom_fwd`` / ``@autocast_custom_bwd`` (imported from
    ``deepspeed.runtime.zero.linear``, which handles the torch < 2.4
    legacy path) preserve AMP autocast state across the custom forward
    and backward so the ``grad_output.matmul(weight)`` in backward sees
    the same dtype as the saved weight.  Without them, an AMP autocast
    wrapping the forward would cast inputs to float16 while the saved
    tensor stays float32, and the backward matmul would raise on the
    dtype mismatch.
    """

    @staticmethod
    @_moevement_custom_fwd
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(weight)
        output = input.matmul(weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @_moevement_custom_bwd
    def backward(ctx, grad_output):
        (weight, ) = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        # ``None`` for weight / bias tells autograd to skip the wgrad GEMM
        # + accumulation.  ``zero_frozen_gradients`` in
        # ``on_before_optimizer_step`` still runs as a safety net in case
        # a ZeRO pre-partitioned grad buffer was pre-populated via
        # another path.
        return grad_input, None, None


def _wrap_linear_for_recovery(linear_module):
    """Swap ``linear_module.forward`` to use ``_FrozenLinearFunction``.

    Idempotent: a second call on an already-wrapped module is a no-op.
    ``_moevement_orig_linear`` is the sentinel — its presence means the
    instance has a monkey-patched ``forward`` attribute shadowing the
    class method.
    """
    if getattr(linear_module, "_moevement_orig_linear", False):
        return

    def _frozen_forward(self, input):
        return _FrozenLinearFunction.apply(input, self.weight, self.bias)

    linear_module._moevement_orig_linear = True
    linear_module.forward = types.MethodType(_frozen_forward, linear_module)


def _unwrap_linear(linear_module):
    """Restore the class-level ``forward`` by removing the instance override."""
    if not getattr(linear_module, "_moevement_orig_linear", False):
        return
    del linear_module.forward
    del linear_module._moevement_orig_linear


def _is_zero_partitioned(param):
    """True when ``param`` carries a ZeRO-1/2 or bf16_optimizer fragment mapping.

    Those mappings are the hand-off point from DeepSpeed's flat FP32
    master partition back to an individual ``nn.Parameter``; when
    present, we must go through ``safe_get_full_fp32_param`` /
    ``safe_get_full_optimizer_state`` rather than reading the module's
    low-precision ``param.data`` directly.
    """
    return getattr(param, '_hp_mapping', None) is not None


# Relative change in observed median iter time (vs. the value baked into
# the current schedule) that qualifies as drift worth recomputing
# ``w_sparse`` for.  Cheaper than re-running ``find_window_size`` every
# boundary and still tight enough to catch throughput-warmup or
# routing-shift slowdowns.  10% is the same knob paper §3.5 uses for
# the frequency-change threshold; using it here keeps one tuning knob.
_ITER_TIME_DRIFT_TO_REGEN = 0.10


def _supports_pinned_memory():
    """True iff the current accelerator services ``pin_memory=True`` allocations.

    ``torch.empty(..., pin_memory=True)`` is a CUDA-specific allocator —
    HPU / XPU / MPS all expose ``get_accelerator().Stream`` but raise
    ``RuntimeError("Need to provide pin_memory allocator to use pin
    memory.")`` on the allocation call.  The ``non_blocking=True`` H2D
    hint is silently ignored for unpinned destinations on every
    backend, so the two decisions move together: we pin only when the
    backend supports it, and we rely on ``non_blocking`` only when the
    source is pinned.
    """
    accelerator = get_accelerator()
    if accelerator is None:
        return False
    return accelerator.device_name() in ("cuda", "rocm")


class MoEvementCoordinator:
    """Orchestrates MoEvement's sparse checkpointing system.

    Manages the interaction between:
    - SparseCheckpointScheduler: determines what to checkpoint each iteration
    - SparseSnapshotEngine: handles GPU-to-CPU transfers and persistence
    - SparseToDenseConverter: reconstructs dense checkpoints during recovery
    - UpstreamLogger: logs activations/gradients for localized recovery
    """

    def __init__(self, config):
        """Initialize the MoEvement coordinator.

        Args:
            config: MoEvementConfig instance.
        """
        self.config = config
        self.scheduler = SparseCheckpointScheduler(
            pcie_bandwidth_bytes_per_sec=config.pcie_bandwidth_bytes_per_sec,
            reorder_threshold=config.reorder_threshold,
            reorder_fraction=config.reorder_fraction,
            activation_count_window_iters=config.activation_count_window_iters,
        )
        self.snapshot_engine = SparseSnapshotEngine(replication_factor=config.replication_factor,
                                                    fsync_on_save=config.fsync_on_save)
        self.converter = SparseToDenseConverter()
        self.upstream_logger = UpstreamLogger(fsync_on_save=config.fsync_on_save) if config.upstream_logging else None

        self._initialized = False
        self._iter_time_sec = None
        # Rolling window of observed per-iteration wall-clock durations.
        # Driven from ``on_iteration_end`` via ``time.perf_counter`` deltas;
        # feeds back into ``scheduler.find_window_size`` at window
        # boundaries so ``w_sparse`` tracks the real iter time rather
        # than staying pinned to ``config.initial_iter_time_sec`` for the
        # life of the job (CheckFreq-style runtime recalibration).
        self._iter_time_window = deque(maxlen=config.iter_time_window_iters)
        self._last_iter_end_time = None
        self._global_step = 0
        self._window_step = 0  # step within current sparse window
        self._recovering = False
        # Tracks whether this recovery session's one-time pp-log-transfer has
        # already run.  ``_pp_log_transfer`` needs to execute exactly once per
        # session: the paused live peers ship their logs and then stay in the
        # wait loop, so re-entering log transfer on subsequent replay iters
        # would deadlock the recovering side on a recv that nobody sends.
        self._pp_log_transfer_done = False
        # Set for exactly one iteration's worth of ``recovery_barrier`` →
        # ``_aggregate_total_loss`` traversal: the recovering rank just
        # finished its last replay iter, called ``end_recovery`` (flipping
        # ``_recovering=False``), and issued ``_release_paused_peers``'s
        # final handshake — but ``_aggregate_total_loss`` in the same
        # train_batch would still try to broadcast across ``pp_group``,
        # which the paused peers never join.  The flag lets that broadcast
        # short-circuit symmetrically with paused ranks (who ``"abandon"``
        # this iter via ``recovery_barrier``'s return status).  Cleared at
        # the top of the next ``recovery_barrier``.
        self._post_recovery_exit = False
        # Original training iteration at which recovery was triggered (i.e.
        # the global_step every paused rank had most recently completed
        # before the fault was observed).  Captured once at recovery entry
        # so catch-up replay knows how many iters past the last persisted
        # window it needs to reconstruct.  ``None`` outside recovery.
        self._fault_iter = None
        # Latest engine-scalar state handed in by ``on_iteration_end``
        # (global_steps, global_samples, lr_scheduler.state_dict(),
        # compression_scheduler.state_dict()).  Stashed per-iter and
        # latched into the persisted bundle at ``finalize_window`` time so
        # peer-pull / disk-reload / cascade all restore the spare's engine
        # counters + scheduler step to the paused peers' state.  Engine
        # threads this in as an optional kwarg so the coordinator stays
        # engine-agnostic (no back-reference to ``DeepSpeedEngine``).
        self._pending_engine_scalars = None
        # Cursor into the replay window: the original iteration number whose
        # logs should feed the engine on the *current* training step.  The
        # engine's ``global_steps`` counter continues to increment forward
        # during replay, so we translate it via this cursor before looking up
        # upstream-logged tensors.  None = cursor not yet set.
        self._replay_iteration_cursor = None
        # Last persisted bundle iter — the boundary between bundle-replay tbs
        # (whose forward output is overwritten by ``_setup_replay_iter`` at tb
        # end, so log-key off-by-one doesn't matter) and catch-up tbs (whose
        # forward output is the real state advancement, so log-key must line
        # up exactly).  Bundle keys are post-increment ``global_steps=N``
        # (state after tb-N's optimizer step) while log keys are pre-increment
        # (forward input during tb-(N+1)); for replay tb with cursor ``c`` in
        # the catch-up range, the forward input we want is log key ``c-1``.
        # Set to ``None`` outside recovery.
        self._catch_up_boundary = None
        # Backup of ``requires_grad`` flags flipped to False for frozen-op
        # params during recovery.  Restored in end_recovery() / when the op
        # transitions to ACTIVE.
        self._frozen_param_backup = {}
        self._moe_layers = []
        # Pinned CPU destinations for per-layer ``exp_counts`` D2H copies.
        # Keyed by ``(layer_idx, slot_idx)`` where ``slot_idx`` cycles
        # through ``[0, w_sparse)``: each iter within a sparse window
        # writes to its own slot so all ``w_sparse`` iters' counts
        # coexist until the boundary fences once and flushes them to
        # the scheduler.  Replaces a per-iter fence that fired multiple
        # stream syncs per iter.
        self._exp_counts_pinned = {}
        # Dedicated side stream for ``exp_counts`` D2H copies.  Lazy-init
        # on first use (CPU accelerators expose ``Stream`` as None).
        # Issuing the per-layer copies on this stream lets
        # ``_fence_exp_counts_copies`` sync only the small exp_counts
        # work rather than ``current_stream().synchronize()``, which
        # waited for all in-flight training kernels too.
        self._exp_counts_stream = None
        # Buffered ``[(layer_idx, cpu_tensor), ...]`` per iter in the
        # current window; flushed to the scheduler (one
        # ``update_activation_counts`` + ``tick_interval`` pair per
        # buffered iter) after the window-boundary fence.  Preserves
        # per-iter granularity in ``_interval_history`` without paying
        # a stream sync every iteration.
        self._pending_activation_counts = []
        # Cache of canonical optimizer-state keys per param, keyed by
        # ``id(param)``.  ``_collect_param_optim_state`` normally fires
        # a ``dist.all_gather_object`` once per (param, iter) to agree
        # on the canonical keyset across the DP group (non-owner ranks
        # contribute empty keys, owner contributes
        # ``[exp_avg, exp_avg_sq]`` etc.).  For a stable optimizer
        # (AdamW family) the keyset is fixed after the first ``step()``
        # populates ``_hp_mapping.optim_fragment``, so the collective
        # is redundant on subsequent iters.  Only non-empty results are
        # cached — pre-first-step iters return ``[]`` and we re-run
        # the collective next call until the owner's ``local_keys``
        # populate.  All ranks in the DP group transition cache-miss
        # → cache-hit at the same iter (ZeRO lazy-init fires
        # symmetrically), so cache hits don't desynchronise the
        # collective.  Invalidated at recovery entry.
        self._param_optim_keys_cache = {}
        self._operator_map = OrderedDict()  # name -> (module, param_group_idx)
        # Pre-resolved ``nn.Linear`` submodules per operator, populated
        # once at ``_discover_operators`` time.  ``_freeze_operator_params``
        # used to walk ``model.modules()`` for the ``non_expert`` sentinel
        # on every recovery — a non-trivial Python iteration over the full
        # model tree.  Caching the resolution at discovery (operators
        # don't change during a run) collapses that to a list lookup.
        self._frozen_linears_by_op = {}
        # Pre-resolved ``[(param_name, param), ...]`` per operator, also
        # populated once at ``_discover_operators`` time.  Mirrors the P8
        # pattern: the frozen-op snapshot loop used to call
        # ``module.named_parameters()`` every iter and build a fresh
        # dict — a full module-subtree walk per op per iter.  Operators'
        # parameter layouts are static for the run (dynamic-router
        # models would need to invalidate this cache — not supported).
        # The non-expert sentinel maps to the full "non-MoE params"
        # list, matching the filter predicate in ``_get_non_expert_params``.
        self._param_list_by_op = {}
        # ``id(p)`` for every MoE gate parameter on this rank.  Populated
        # at ``_discover_operators`` time and consumed by
        # ``_is_non_expert_param`` so gate params land in the
        # ``layer_X_gate`` operator only — *not* also in ``non_expert``.
        # Without the exclusion, a window where the two ops disagreed on
        # ACTIVE/FROZEN status would silently corrupt the gate: the
        # non_expert-frozen path wrapped the gate's ``nn.Linear.forward``
        # and ``zero_frozen_gradients`` zeroed the gate's grad even
        # though its own op was ACTIVE.
        self._gate_param_ids = set()
        self._model = None  # set in initialize(); needed by cascade recovery
        self._optimizer = None  # set by engine via set_optimizer(); used in cascade replay setup
        self._checkpoint_load_dir = None
        self._checkpoint_tag = None
        # Snapshot data cached across replay iterations.  Populated once
        # at recovery entry — either by ``load_sparse_checkpoint`` (disk
        # read) or by ``cascade_into_recovery`` (own in-memory
        # ``_persisted_snapshots``) — and consumed by every replay
        # iteration's ``_setup_replay_iter``.  Caching avoids a
        # per-iteration disk re-read on the disk-load path and gives
        # the cascade path a way to feed in-memory snapshot data to
        # ``_setup_replay_iter`` without routing through disk.
        # Cleared at ``end_recovery``.
        self._cached_snapshot_data = None

        # Streaming peer-pull state (SD-O4 S2).  When
        # ``config.streaming_recovery`` is on, ``load_sparse_from_peer``
        # starts a background pull thread that feeds iters into a
        # ``queue.Queue`` as their mini-manifests land.  The main
        # thread returns after ingesting iter 0; subsequent iters are
        # drained on demand from ``_setup_replay_iter`` (or en masse at
        # ``end_recovery``).  All three are ``None`` outside streaming
        # recovery so bulk / cascade / disk paths are unaffected.
        self._streaming_pull_thread = None
        self._streaming_pull_queue = None
        self._streaming_pull_exc = None
        self._streaming_model_ref = None

        # Topology — set by engine after initialisation
        self._dp_group = None
        self._dp_rank = 0
        self._device = 'cpu'
        # Dedicated NCCL group for background peer replication.  We can't
        # reuse ``_dp_group`` because the training thread issues ZeRO
        # collectives on it while the replication worker is sending on
        # the same communicator — concurrent ops on one NCCL comm are
        # undefined behaviour.  Built once in ``set_topology`` via
        # ``all_gather_object`` + ``new_group`` so every rank collectively
        # materialises the same mirror group.
        self._replication_group = None
        self._pp_group = None
        self._stage_id = None
        self._num_stages = None
        self._stage_to_global_fn = None
        # Gloo mirror of ``_pp_group`` for upstream-log transfer.  The
        # logs are pinned CPU activations/grads, so the default NCCL
        # group can't ship them; we build a gloo mirror at topology
        # registration time the same way replication does for the DP
        # group.
        self._pp_replication_group = None

        # Peer replication runs on a dedicated worker so the training
        # thread never waits on gloo TCP sends at the window boundary.
        # Jobs queue in ``_replication_futures`` FIFO; the window boundary
        # drains completed entries, warns past ``replication_queue_warn_threshold``,
        # and only blocks once ``replication_queue_max_outstanding`` windows
        # are in flight (pinned-memory safety rail).  ``max_workers=1``
        # keeps sends serialized so a slow peer applies natural
        # backpressure by growing the queue rather than fighting for the
        # socket.
        self._replication_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MoEvement-Replicate")
        self._replication_futures = deque()
        # Debounce flag for the warn-threshold crossing log; re-armed once
        # depth drops to half the warn threshold so a single slow boundary
        # doesn't produce a warn per subsequent window.
        self._replication_warn_active = False
        # Upper bound on how long a backpressure-block wait for replication
        # can block training.  Sized to be much larger than a reasonable
        # DP-group broadcast but finite, so a hung gloo send surfaces as
        # a loud error instead of a permanent training hang.
        self._replication_timeout_sec = 300.0
        # Set once a replication job times out: a hung gloo send cannot be
        # cancelled, so the worker thread stays wedged forever.  Once
        # flagged we stop submitting new jobs (they'd queue behind the
        # hung one and never run) and fall back to a non-blocking
        # executor teardown at shutdown.
        self._replication_broken = False
        self._shutdown_done = False

        # Pause state for the recovery-cascade handshake.  A rank that
        # isn't recovering itself and isn't in an affected DP group
        # drops into a blocking wait while the recovering DP group
        # replays its W_sparse window, then resumes.
        self._paused_for_recovery = False
        # Set of pipeline stage ids in THIS rank's pp column that are
        # currently (post-cascade) in recovery.  Consumed by
        # ``should_skip_pipeline_send`` so the pipe engine's send-side
        # recovery guard can distinguish "downstream is paused, skip"
        # from "downstream is recovering, send via p2p" — the latter
        # needed so adjacent failed stages can chain replay through
        # normal pipeline channels.
        self._recovering_stages_in_my_pp = frozenset()
        # How long to sleep between handshake polls while paused — the
        # wait loop participates in a periodic world all_gather so the
        # recovering ranks can signal completion.  Short enough that
        # short recoveries don't pay much wall clock, long enough that
        # we don't burn CPU for longer ones.
        self._pause_poll_interval_sec = 0.05

        # Register a process-exit flush+shutdown via a weakref so the
        # coordinator can still be GC'd before exit.
        self_ref = weakref.ref(self)

        def _atexit_shutdown():
            inst = self_ref()
            if inst is not None:
                inst.shutdown()

        atexit.register(_atexit_shutdown)

    def initialize(self,
                   model,
                   moe_layers,
                   iter_time_sec,
                   stage_id=None,
                   optimizer=None,
                   gradient_accumulation_steps=1):
        """Initialize the coordinator with model information.

        Discovers all operators (experts, non-experts, gating) and builds
        the initial checkpoint schedule.

        Args:
            model: The DeepSpeed model (nn.Module).
            moe_layers: List of MOELayer instances found in the model.
            iter_time_sec: Measured iteration time in seconds.
            stage_id: This rank's pipeline stage id.  Used to namespace
                operator names so stage 0's ``layer_0_expert_0`` doesn't
                collide with stage 1's under any cross-stage lookup.
                Optional because unit tests build coordinators without a
                pipeline topology; the engine path always passes it.
            optimizer: The optimizer.  Stored for cascade-triggered
                recovery (auto-entered from the world handshake with no
                caller available to pass it); used to restore Adam state
                during eager iter-1 setup.
            gradient_accumulation_steps: Engine-side ``gas`` value (number
                of micro-batches per ``train_batch``).  Used to auto-size
                the upstream-logger pinned pool: every micro-batch logs
                one activation per MoE layer, so per-iter buffer demand
                scales linearly with ``gas``.  Defaults to 1 for legacy
                callers / unit tests.
        """
        self._gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self._moe_layers = moe_layers
        self._iter_time_sec = iter_time_sec
        # Iter-time value baked into the current schedule.  Compared against
        # the rolling-window median at every window boundary; drift beyond
        # ``_ITER_TIME_DRIFT_TO_REGEN`` triggers a recompute of ``w_sparse``
        # via ``find_window_size``.  Reset after every successful regen so
        # subsequent comparisons are against the freshest baseline.
        self._last_scheduled_iter_time = iter_time_sec
        # Keep live references to the model and optimizer so cascade-
        # triggered recovery (auto-entered from the world handshake with no
        # caller available to pass them) can still reach the non-expert
        # params and restore optimizer state.
        self._model = model
        self._optimizer = optimizer
        # ``set_pipeline_topology`` is called later from
        # ``PipelineEngine.__init__`` and would set the same value;
        # preempting it here lets ``_discover_operators`` stage-qualify
        # the names it emits without depending on call order.
        if stage_id is not None:
            self._stage_id = stage_id

        self._warn_if_tensor_parallel_linears(model)
        self._warn_if_int_step_optimizer(optimizer)

        operators = self._discover_operators(model, moe_layers)
        self.scheduler.register_operators(operators)

        w_sparse, schedule = self._generate_schedule_world_aligned(iter_time_sec)

        # Auto-size the upstream-log retention window to the scheduler's
        # ``w_sparse``.  ``UpstreamLogger._max_window`` is purely advisory
        # (nothing inside the logger enforces it as a cap — actual
        # retention is driven by the coordinator's per-iter gc policy);
        # the field only feeds the assertion in ``begin_recovery``, which
        # needs ``retained >= 2 * w_sparse`` for catch-up replay to have
        # log sources.  The default 10 is fine for small w_sparse; bump
        # it here once w_sparse is known so configs with starved PCIe
        # bandwidth (disk-resume demos, low-bandwidth boxes) don't trip
        # ``begin_recovery`` at load-checkpoint time.
        if self.upstream_logger is not None:
            required = 2 * max(1, w_sparse)
            if self.upstream_logger._max_window < required:
                self.upstream_logger._max_window = required

        # Configure the pinned-buffer pool to grow in bulk on the first
        # miss for each unique ``(shape, dtype, device, pin)`` key.
        # Without this, every acquire past the first for a given shape
        # pays ``cudaMallocHost`` (tens of ms per 64+ MB pinned buffer)
        # synchronously inside ``on_iteration_end`` — projecting shapes
        # at init was fragile because runtime acquire sizes depend on
        # per-rank ZeRO fragments, MoE-expert optim state availability
        # (not populated on the owner under ZeRO-1 + EP sharding), and
        # the frozen-FP16 capture path's own sizing.  Grow-on-miss lets
        # the first capture of each shape allocate enough buffers to
        # cover the full in-flight depth, and every subsequent acquire
        # hits the pool.
        #
        # ``_batched_d2h`` holds a buffer in a snapshot across two
        # generations (``_in_flight_snapshots``, ``_persisted_snapshots``)
        # before ``finalize_window`` returns it to the free list.  Async
        # peer replication can hold up to ``max_outstanding`` more
        # generations in flight simultaneously (each outstanding window
        # retains its per-op flat until the worker's done-callback
        # fires), so ``grow_on_miss`` covers the 2-gen rotation plus the
        # replication backlog.
        #
        # Empirically at factor=0, over-sizing the pinned pool to the
        # max_outstanding ceiling inflates iter time substantially —
        # large pinned allocations appear to inflate CUDA-driver
        # bookkeeping on every cudaStreamSynchronize / cudaMemcpyAsync;
        # mechanism unexplained but reproducible.  The factor-based cap
        # below is the same workaround the prior init-time prewarm
        # carried.
        if self.config.replication_factor > 0:
            grow_on_miss = 2 + self.config.replication_queue_max_outstanding
        else:
            # Each window holds ~``w_sparse`` × (active + frozen) buffers
            # in flight before ``finalize_window`` releases the previous
            # window's flats back to the pool — buffer lifetime is
            # 2 windows.  The pool needs enough headroom per shape to
            # span that without missing on every acquire, so size
            # ``grow_on_miss`` to fill in 1-2 misses rather than many.
            #
            # The dominant cost on a pool miss is the synchronous
            # ``cudaHostAlloc`` itself, not pinned-residency-driven
            # inflation of cudaStreamSynchronize; eliminating misses
            # dwarfs any residency effect.
            grow_on_miss = 16
        self.snapshot_engine._pool.set_grow_on_miss(grow_on_miss)
        self.snapshot_engine._pool.set_max_per_key(self.config.pool_max_per_key)
        # Upstream logger pool sizing.  At gas>1 with low w_sparse, every
        # iter creates a persisted snapshot that protects its activations
        # from GC, so the pool's working set grows to ``gas × num_moe_layers
        # × w_sparse × in_flight_windows`` buffers per shape and never
        # returns to the pool.  Sizing too small forces mid-iter
        # cudaHostAlloc bursts on cold iters.  We err generous: host
        # pinned memory is cheap, perf loss from undersizing is not.
        # Auto-default ``8 × gas × num_moe_layers × w_sparse`` gives
        # ample headroom over the protection horizon.  User-overridable
        # via ``MoEvementConfig.pool_grow_on_miss_activation``; 0 means
        # "auto-size."
        if self.upstream_logger is not None:
            grow_activation = self.config.pool_grow_on_miss_activation
            if grow_activation == 0:
                num_moe_layers = max(1, len(self._moe_layers))
                grow_activation = max(512, 8 * self._gradient_accumulation_steps * num_moe_layers * w_sparse)
                logger.info(f"[MoEvement] auto-sized upstream-logger pool grow_on_miss="
                            f"{grow_activation} (gas={self._gradient_accumulation_steps}, "
                            f"moe_layers={num_moe_layers}, w_sparse={w_sparse}); "
                            f"override via MoEvementConfig.pool_grow_on_miss_activation")
            self.upstream_logger._pool.set_grow_on_miss(grow_activation)
            self.upstream_logger._pool.set_max_per_key(self.config.pool_max_per_key)

        self._initialized = True
        logger.info(f"[MoEvement] Coordinator initialized: {len(operators)} operators, "
                    f"W_sparse={w_sparse}, iter_time={iter_time_sec:.3f}s")

    def _generate_schedule_world_aligned(self, iter_time_sec):
        """Pick ``w_sparse`` as the world max, then generate the schedule.

        Each PP stage has a different operator set (stage 0 owns the
        embed, stage 2 the head, middle stages neither) so the
        stage-local ``find_window_size`` would pick a different cadence
        per stage.
        A per-rank cadence corrupts every cross-rank protocol that is
        iter-keyed:

        - DP peer-pull would index the donor's bundle by the donor's slot
          map but replay it through the recipient's slot map (silent
          state corruption).
        - The per-iter ``_world_recovery_handshake`` would fire at iters
          when not all ranks participate (deadlock).
        - Cross-stage ``pp_log_transfer`` would key activations by an
          iter that the neighbour stage doesn't have logged.

        Taking the max keeps every rank's per-slot PCIe budget at most as
        tight as its local choice would have been; smaller-op ranks just
        get extra empty slots ("rest" iters with no D2H).  Used both at
        ``initialize`` and at the mid-training regen path so the
        invariant is preserved across schedule rotations triggered by
        popularity or iter-time drift.
        """
        # User override takes precedence over scheduler's find_window_size.
        # find_window_size MINIMIZES w_sparse subject to "snapshot fits in
        # one iter PCIe budget" — the right choice for paper §3.5's
        # recovery-frequency goal but suboptimal for steady-state perf
        # when compute headroom is plenty (boundary cost amortizes
        # inversely with w_sparse, and per-iter D2H drain has more
        # compute time to overlap).  Production users tuning for
        # throughput set ``MoEvementConfig.w_sparse_override`` to a
        # larger value (e.g. 16 or 32); leaving 0 keeps the recovery-
        # optimal default.
        if self.config.w_sparse_override > 0:
            local_w_sparse = int(self.config.w_sparse_override)
        else:
            local_w_sparse, _ = self.scheduler.find_window_size(iter_time_sec,
                                                                overlap_target=self.config.snapshot_overlap_target)
        if dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor(local_w_sparse, dtype=torch.long, device=get_accelerator().current_device_name())
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            global_w_sparse = int(t.item())
        else:
            global_w_sparse = local_w_sparse
        if global_w_sparse != local_w_sparse:
            logger.info(
                "[MoEvement] Aligning w_sparse to world max: local=%d, global=%d "
                "(stage operator-count asymmetry; trailing slots will be empty on this rank)", local_w_sparse,
                global_w_sparse)
        return self.scheduler.generate_schedule(iter_time_sec, w_sparse_override=global_w_sparse)

    def _sync_w_sparse_world_max(self):
        """Re-issue the WORLD ``all_reduce(MAX)`` after a topology rebuild.

        Mirrors the WORLD all_reduce in ``_generate_schedule_world_aligned``
        which the spare runs once during cold-start
        (``coordinator.initialize``).  Survivors don't traverse
        ``initialize`` on rebuild, so without this call the spare's
        WORLD-collective sequence sits one ahead of the survivors and the
        very next WORLD collective on the rebuilt default group deadlocks
        (observed: ``_build_gloo_mirror``'s ``all_gather_object`` hangs
        with all 4 ranks at the same line but mismatched seq positions).

        Pure sync — does not touch ``self.scheduler.schedule``.  Survivors
        keep their pre-fault schedule (still correct after a 1:1 spare
        substitution); the spare already aligned its w_sparse during
        ``initialize``.  The all_reduce just verifies agreement and
        consumes the matching WORLD seq slot.
        """
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return
        t = torch.tensor(self.scheduler.w_sparse, dtype=torch.long, device=get_accelerator().current_device_name())
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        world_max = int(t.item())
        if world_max != self.scheduler.w_sparse:
            # Rebuild-time disagreement implies a topology / workload
            # state change between cold-start and rebuild that the
            # coordinator hasn't reconciled yet.  Log and let the next
            # ``on_iteration_end`` regen path resolve via the standard
            # symmetric flow rather than racing a fix into the rebuild.
            logger.warning(
                "[MoEvement] Post-rebuild w_sparse asymmetry: this rank=%d, world max=%d. "
                "Likely the spare and survivors initialised with different iter_time estimates; "
                "next regen will resync.", self.scheduler.w_sparse, world_max)

    def _discover_operators(self, model, moe_layers):
        """Discover all checkpointable operators in the model.

        Operators include:
        - Each expert in each MoE layer
        - Non-expert (shared) layers
        - Gating networks

        Args:
            model: The model module.
            moe_layers: List of MOELayer instances.

        Returns:
            List of OperatorInfo objects.
        """
        operators = []

        # Stage-qualify every operator name.  Pipeline stages enumerate
        # ``moe_layers`` locally, so both stage 0 and stage 1 would
        # otherwise emit the same ``layer_0_*`` labels; that worked for
        # purely local lookups but any cross-stage lookup (serialisation,
        # peer shipping) would silently mis-address.  Falls back to no
        # prefix when ``_stage_id`` hasn't been set (direct-construct
        # unit tests), matching the pre-change behaviour.
        prefix = f"stage{self._stage_id}_" if self._stage_id is not None else ""

        # Snapshot every MoE gate's param id set *before* walking the
        # model for non_expert — ``_is_non_expert_param`` reads this to
        # exclude gate params from the non_expert bucket.  See that
        # helper's docstring for the full ownership rationale.
        self._gate_param_ids = set()
        for moe_layer in moe_layers:
            self._gate_param_ids.update(id(p) for p in moe_layer.gate.parameters())

        # Discover non-MoE parameters as a single non-expert operator
        non_expert_params = 0
        for name, param in model.named_parameters():
            if self._is_non_expert_param(param):
                non_expert_params += param.numel()

        if non_expert_params > 0:
            non_expert_name = f"{prefix}non_expert"
            op = OperatorInfo(name=non_expert_name,
                              num_params=non_expert_params,
                              is_expert=False,
                              layer_id=-1,
                              local_expert_id=None)
            operators.append(op)
            self._operator_map[non_expert_name] = None
            self._frozen_linears_by_op[non_expert_name] = self._resolve_frozen_linears(non_expert_name, model)
            self._param_list_by_op[non_expert_name] = self._resolve_op_params(non_expert_name, model)

        # Discover experts and gating from each MoE layer
        for layer_idx, moe_layer in enumerate(moe_layers):
            # Gating operator
            gate_params = sum(p.numel() for p in moe_layer.gate.parameters())
            if gate_params > 0:
                gate_name = f"{prefix}layer_{layer_idx}_gate"
                op = OperatorInfo(name=gate_name,
                                  num_params=gate_params,
                                  is_expert=False,
                                  layer_id=layer_idx,
                                  local_expert_id=None)
                operators.append(op)
                self._operator_map[gate_name] = moe_layer.gate
                self._frozen_linears_by_op[gate_name] = self._resolve_frozen_linears(gate_name, model)
                self._param_list_by_op[gate_name] = self._resolve_op_params(gate_name, model)

            # Expert operators
            num_local = moe_layer.num_local_experts
            for expert_idx in range(num_local):
                expert_module = moe_layer.experts.deepspeed_experts[expert_idx]
                expert_params = sum(p.numel() for p in expert_module.parameters())
                expert_name = f"{prefix}layer_{layer_idx}_expert_{expert_idx}"
                op = OperatorInfo(name=expert_name,
                                  num_params=expert_params,
                                  is_expert=True,
                                  layer_id=layer_idx,
                                  local_expert_id=expert_idx)
                operators.append(op)
                self._operator_map[expert_name] = expert_module
                self._frozen_linears_by_op[expert_name] = self._resolve_frozen_linears(expert_name, model)
                self._param_list_by_op[expert_name] = self._resolve_op_params(expert_name, model)

        # Ownership invariant: every param appears in at most one operator's
        # bucket.  Without this, a window where two overlapping ops disagree
        # on ACTIVE/FROZEN status silently corrupts the shared param — the
        # exact bug the ``_is_non_expert_param`` gate-id exclusion fixes.
        # Asserting here makes future double-registration regressions loud.
        seen_ids = {}
        for op_name, entries in self._param_list_by_op.items():
            for pname, p in entries:
                if id(p) in seen_ids:
                    raise RuntimeError(f"[MoEvement] operator ownership violation: param '{pname}' "
                                       f"registered in both '{seen_ids[id(p)]}' and '{op_name}'")
                seen_ids[id(p)] = op_name

        num_experts = sum(1 for o in operators if o.is_expert)
        num_gates = sum(1 for o in operators if (not o.is_expert) and o.name.endswith("_gate"))
        num_non_expert = sum(1 for o in operators if (not o.is_expert) and o.name.endswith("non_expert"))
        logger.info(f"[MoEvement] Operator ownership: "
                    f"{num_experts} expert, {num_gates} gate, {num_non_expert} non_expert; "
                    f"total params tracked: {len(seen_ids)}")

        return operators

    def on_iteration_end(self, global_step, model, optimizer, engine_scalars=None):
        """Called after each training iteration to perform sparse checkpointing.

        During normal training: pulls expert activation counts, snapshots scheduled
        operators, and manages window progression.

        During recovery: advances the replay window by loading the next sparse snapshot
        and transitioning newly-active operators from FROZEN to ACTIVE.

        Args:
            global_step: Current global training step.
            model: The model module.
            optimizer: The optimizer.
            engine_scalars: Optional dict shaped
                ``{"global_steps": int, "global_samples": int,
                   "lr_scheduler": dict | None,
                   "compression_scheduler": dict | None}``.  Engine hands
                these in so ``finalize_window`` can latch them onto the
                persisted bundle for peer-pull / cascade recovery.
                ``None`` on callers that predate engine-scalar sync
                (older bundles, unit tests) — the peer-pull path tolerates
                missing entries at restore time.
        """
        if not self._initialized:
            return

        # Promote pending log_d2h tuples (collected during fwd / bwd send
        # hooks) into the ``_logs`` dict in one tight loop.  Hot per-
        # microbatch path stays a tuple append; the LogEntry alloc + dict
        # insert + max() amortize across the iter.
        if self.upstream_logger is not None:
            self.upstream_logger.flush_pending()

        # Record the engine's step counter so collectives that run *before*
        # on_iteration_end (e.g. recovery_barrier at the top of train_batch)
        # can compute iteration ranges relative to where training is.
        self._global_step = global_step
        # Stash engine scalars for finalize_window; keep last-seen value so
        # a mid-window fault cascading off another DP peer still has a
        # plausible state to restore from.
        if engine_scalars is not None:
            self._pending_engine_scalars = engine_scalars

        if self._recovering:
            self._on_iteration_end_recovery(model, optimizer)
            return

        # Record one iteration's wall-clock duration from the last
        # ``on_iteration_end`` to now.  The first call has no prior
        # timestamp and is skipped — iter 0 picks up on the second call.
        self._record_iter_duration()

        # Issue the per-layer ``exp_counts`` D2H copies as non_blocking
        # ops into pre-allocated pinned destinations slotted by
        # ``_window_step``.  Defer both the stream fence and the
        # ``scheduler.update_activation_counts`` + ``tick_interval``
        # feed to the window boundary: one sync covers all ``w_sparse``
        # iters' copies instead of firing every iteration.
        # Per-iter granularity in the scheduler's rolling window is
        # preserved because each buffered slot still gets its own
        # ``tick_interval`` at flush time.  The paper's reorder trigger
        # only evaluates at window boundaries, so delivering counts in
        # a batch rather than live is semantically equivalent.
        self._pending_activation_counts.append(self._schedule_exp_counts_copies(self._window_step))

        # Use ``_window_step`` (slot index within the current window,
        # incremented at end of every iter, reset at window boundary) to
        # derive ``window_start`` rather than ``snapshot_engine._window_start_iteration``
        # — the latter is set only at ``begin_window`` (window boundary)
        # and stays at -1 during the first window before any boundary
        # has fired.  ``_window_step`` is correct from iter 0.  This
        # also keeps slot indexing aligned with the paper after a
        # mid-training schedule regen that changes ``w_sparse``: a bare
        # ``global_step % w_sparse`` would rotate by whatever residue
        # ``global_step`` carries against the new modulus.
        window_start = global_step - self._window_step
        schedule_entry = self.scheduler.get_schedule_for_iteration(global_step, window_start=window_start)
        if schedule_entry is None:
            return

        # Snapshot active operators (FP32 master weights + optimizer state).
        # Under ZeRO-1/2 the master weights live as partitioned fragments
        # inside the flat FP32 buffer; ``_active_params_dict`` walks the
        # ``_hp_mapping`` handles to reassemble full tensors via a DP
        # all-reduce.  Every DP rank iterates the same ops in the same
        # order, so the collectives line up.
        for op_name in schedule_entry.active_operators:
            if op_name not in self._operator_map:
                continue
            module = self._operator_map[op_name]
            # ``module is None`` is the non-expert sentinel (see
            # ``_discover_operators``); any real module is stored non-
            # None.  Using the map's value directly keeps this branch
            # oblivious to the operator name format.
            if module is None:
                params_dict, params_frag = self._active_non_expert_params(model, op_name)
                optim_state, optim_frag = self._get_non_expert_optimizer_state(model, optimizer)
            else:
                params_dict, params_frag = self._active_module_params(module)
                optim_state, optim_frag = self._get_module_optimizer_state(module, optimizer)

            # Tag fragment_info with the same ``params.``/``optimizer.``
            # prefixes snapshot_operator applies to entry keys, so the
            # metadata stays aligned with the state_dict.
            fragment_info = {}
            for k, v in params_frag.items():
                fragment_info[f"params.{k}"] = v
            for k, v in optim_frag.items():
                fragment_info[f"optimizer.{k}"] = v

            self.snapshot_engine.snapshot_operator(name=op_name,
                                                   params_dict=params_dict,
                                                   optimizer_state_dict=optim_state,
                                                   is_active=True,
                                                   iteration=global_step,
                                                   fragment_info=fragment_info)

        # Snapshot frozen operators (FP16 compute weights only).  The
        # scheduler restricts ``frozen_operators`` to slot ``i``'s
        # not-yet-captured tail (paper Algorithm 1: ``ordered[end:]``),
        # so operators captured ACTIVE in earlier slots of this window
        # are already absent from this list.  No coordinator-side
        # filter required.
        for op_name in schedule_entry.frozen_operators:
            if op_name not in self._operator_map:
                continue
            # Unified via the ``_param_list_by_op`` cache populated at
            # ``_discover_operators`` — same shape for module-backed ops
            # (walks ``module.named_parameters()``) and the non-expert
            # sentinel (filters the full model by ``allreduce``).
            # Eliminates the per-iter module-subtree walk for every
            # frozen op and the full-model walk for the non-expert path.
            params_dict = {name: p.data for name, p in self._iter_op_params(op_name, model)}

            self.snapshot_engine.snapshot_operator(name=op_name,
                                                   params_dict=params_dict,
                                                   optimizer_state_dict=None,
                                                   is_active=False,
                                                   iteration=global_step)

        # Capture this iter's post-optim-step torch RNG state so replay can
        # restore it before the next tb's forward — without this, dropout /
        # stochastic-layer models silently diverge from fault-free starting
        # at iteration zero of replay.  Lives in the snapshot engine's
        # per-iter dict so it rotates through in_flight → persisted in
        # lockstep with the operator snapshots at every window boundary.
        self.snapshot_engine._rng_state_per_iter[global_step] = self._capture_rng_state()

        # Manage window progression
        self._window_step += 1
        if self._window_step >= self.scheduler.w_sparse:
            self._window_step = 0
            # Record the side-stream D2H event up-front so the replication
            # worker can ``event.synchronize()`` itself before reading
            # pinned CPU.  Replaces the training-thread CPU stall that
            # used to gate the worker's safe-to-read condition: the
            # boundary critical path no longer blocks on D2H drain, so
            # forward + backward of iter N+1 overlap with whatever's
            # left of this window's snapshot drain.  Captured here (not
            # later) so the same handle threads through to the worker.
            self.snapshot_engine.record_pending_d2h_event()
            boundary_d2h_event = self.snapshot_engine._pending_d2h_event

            # Drain the per-iter ``exp_counts`` D2Hs — these were issued
            # on the dedicated exp_counts side stream, so this sync only
            # waits for those small copies (not training kernels).
            # Snapshot side-stream D2H stays in flight; the boundary
            # block records its event via ``record_pending_d2h_event``
            # and the next iter's ``on_before_optimizer_step`` queues
            # the matching ``wait_event`` so forward + backward overlap
            # with side-stream drain.
            if self._pending_activation_counts:
                self._fence_exp_counts_copies()
                for per_iter_pending in self._pending_activation_counts:
                    for layer_idx, counts_cpu in per_iter_pending:
                        self.scheduler.update_activation_counts(layer_idx, counts_cpu)
                    self.scheduler.tick_interval()
                self._pending_activation_counts.clear()

            # Drain already-completed replications off the head of the
            # queue so the depth check below reflects only in-flight
            # work.  ``done()`` on a FIFO-ordered queue is monotonic
            # (oldest finishes first under ``max_workers=1``) so we
            # never leave a completed entry behind.
            while self._replication_futures and self._replication_futures[0].done():
                self._replication_futures.popleft()

            # Backpressure: once the outstanding queue hits the configured
            # cap, block the window boundary on the oldest future.  This
            # is the only path that can block training on replication —
            # the cap is the pinned-memory safety rail, so we must stop
            # growing past it even if it means waiting.  Timeout keeps a
            # wedged gloo peer from hanging training silently.
            max_outstanding = self.config.replication_queue_max_outstanding
            if len(self._replication_futures) >= max_outstanding:
                logger.error(f"[MoEvement] replication queue hit max_outstanding={max_outstanding}; "
                             f"blocking window boundary on oldest outstanding replication")
                oldest = self._replication_futures.popleft()
                try:
                    oldest.result(timeout=self._replication_timeout_sec)
                except FuturesTimeoutError:
                    logger.error(f"[MoEvement] peer replication timed out after "
                                 f"{self._replication_timeout_sec}s; disabling further replication")
                    self._replication_broken = True
                except Exception as exc:
                    logger.error(f"[MoEvement] peer replication failed: {exc}")

            # Warn once when queue depth first crosses the warn threshold.
            # Re-arm only after the depth drops to half of it, so a single
            # slow boundary doesn't produce a warn on every subsequent
            # window while the depth decays.
            warn_threshold = self.config.replication_queue_warn_threshold
            depth = len(self._replication_futures)
            if not self._replication_warn_active and depth >= warn_threshold:
                logger.warning(f"[MoEvement] peer replication queue depth {depth} "
                               f"has reached warn_threshold={warn_threshold}; "
                               f"replication is falling behind window cadence "
                               f"(max_outstanding={max_outstanding})")
                self._replication_warn_active = True
            elif self._replication_warn_active and depth <= warn_threshold // 2:
                self._replication_warn_active = False

            # Capture loss-scaler state before finalize promotes this window:
            # the scaler has advanced to post-tb-``global_step-1`` state since
            # the optimizer step for that iter has fired.  Store on the
            # snapshot engine so save_to_disk, peer-pull, and in-memory
            # cascade all see the same scaler snapshot tied to this bundle.
            self.snapshot_engine._persisted_loss_scaler_state = self._capture_loss_scaler_state()
            self.snapshot_engine.finalize_window()
            self.snapshot_engine.begin_window(global_step)

            # Replicate this window's snapshot to DP peers in the background.
            # Training continues into the next iteration while the sends
            # overlap with compute on the main stream.
            if self._replication_group is not None and not self._replication_broken:
                # Mark the persisted flats busy before dispatching so the
                # next window's ``finalize_window`` can't race the worker's
                # sends.  On success a done-callback returns them to the
                # pool; on a worker hang they stay busy (bounded leak but
                # no data race over storage the hung thread is still reading).
                busy_flats = self._mark_persisted_flats_busy()
                # Pin the persisted-snapshots dict reference here (on the
                # training thread, between ``finalize_window`` and the
                # next iter's boundary) so the worker threads receive a
                # stable handle to THIS window's snapshots.  Pinning
                # inside the worker would race: a subsequent window
                # boundary could rotate ``_persisted_snapshots`` before
                # the worker picks up the job.
                persisted_snapshot_ref = self.snapshot_engine._persisted_snapshots
                # Closure over ``busy_flats`` so the worker can release
                # pinned D2H flats back to the training pool as soon as
                # its clone phase completes — training's next D2H
                # acquire doesn't wait on the gloo ship anymore.
                # ``release_busy`` is idempotent, so the done-callback
                # below stays as a safety net for the legacy (no-clone)
                # path and for hung-worker scenarios.
                pinned_release_fn = (lambda flats=busy_flats: self._release_replication_busy(flats))
                future = self._replication_executor.submit(self._do_peer_replication, persisted_snapshot_ref,
                                                           pinned_release_fn, boundary_d2h_event)
                future.add_done_callback(lambda _fut, flats=busy_flats: self._release_replication_busy(flats))
                # Separate callback for exception surface: the training
                # thread no longer awaits ``.result()`` at the boundary,
                # so a worker-raised exception would be swallowed silently
                # without this hook.  Flagging ``_replication_broken`` is
                # the same state the timeout path uses, so the submit
                # guard above will stop accepting new jobs.
                future.add_done_callback(self._on_replication_done)
                self._replication_futures.append(future)

            # Garbage collect stale upstream logs.  Protect iters covered by the
            # last persisted snapshot window: that window can be ``w_sparse``
            # iters older than the default rolling retention because ``_persisted``
            # lags one ``finalize_window`` cycle behind the live snapshots.
            # Under peer-pull recovery the live neighbour must retain logs for
            # exactly those persisted iters so the recovering rank can replay
            # through real pipeline p2p instead of falling through to a blocking
            # recv with no matching send.
            if self.upstream_logger is not None:
                oldest_valid = global_step - self.scheduler.w_sparse
                persisted_iters = {it for (it, _) in self.snapshot_engine._persisted_snapshots.keys()}
                self.upstream_logger.garbage_collect(oldest_valid, protected_iterations=persisted_iters)

            # Refresh the iter-time estimate from the rolling window of
            # observed durations, then decide whether to regenerate the
            # schedule.  Either a popularity shift (§3.5) or an iter-time
            # drift past the configured threshold warrants a fresh
            # ``find_window_size`` so ``w_sparse`` tracks reality.
            self._maybe_update_iter_time()
            regen_reason = None
            if self.scheduler.should_reorder():
                regen_reason = "popularity"
            elif self._iter_time_drift_exceeds_threshold():
                regen_reason = "iter-time drift"
            # ``_generate_schedule_world_aligned`` issues a world-group
            # ``all_reduce(MAX)``; every rank must agree on whether to
            # call it.  ``should_reorder`` and ``_iter_time_drift_exceeds_threshold``
            # both look at per-rank state and can diverge, so OR-reduce a
            # single bit before the call to pull every rank into the same
            # branch.  Without this, a rank that decides to regen waits on
            # the world all_reduce while peers advance to the next iter's
            # ``recovery_barrier`` — different world collectives queue out
            # of order and NCCL deadlocks.
            if dist.is_initialized() and dist.get_world_size() > 1:
                decision = torch.tensor([1 if regen_reason is not None else 0],
                                        dtype=torch.long,
                                        device=get_accelerator().current_device_name())
                dist.all_reduce(decision, op=dist.ReduceOp.MAX)
                any_rank_regen = bool(decision.item())
            else:
                any_rank_regen = regen_reason is not None
            if any_rank_regen:
                old_w = self.scheduler.w_sparse
                self._generate_schedule_world_aligned(self._iter_time_sec)
                self._last_scheduled_iter_time = self._iter_time_sec
                logger.info(f"[MoEvement] Regenerated schedule at step {global_step} "
                            f"({regen_reason or 'peer-triggered'}): "
                            f"w_sparse {old_w} → {self.scheduler.w_sparse}, "
                            f"iter_time={self._iter_time_sec:.4f}s")

            # The D2H event was already recorded at the top of the
            # boundary block (and threaded into the replication worker
            # so it can event.synchronize() before reading pinned).
            # ``on_before_optimizer_step`` of the next iter consumes
            # the same handle from ``snapshot_engine._pending_d2h_event``
            # to queue the main-stream ``wait_event`` that orders
            # optim.step after the snapshot drain.

    def _on_iteration_end_recovery(self, model, optimizer):
        """Set up state for the NEXT replay iteration.

        Called at the end of each replay iter's ``train_batch``.  Advances
        the converter's replay cursor, promotes that iter's newly-active
        operators (load FP32 + optimizer state, thaw autograd), and refreshes
        frozen ops' FP16 weights to that iter's per-snapshot capture.

        Recovery ends when the replay-iter list is exhausted — we just set
        up the last replay iter and no more catch-up tbs remain, so the
        recovering rank has reached ``_fault_iter`` state and the next
        ``train_batch`` would run post-recovery.  The previous
        ``is_conversion_complete`` gate fired as soon as every operator was
        promoted (i.e., at ``bundle[-1]`` when ``num_ops <= w_sparse``),
        which short-circuited the catch-up iters that bridge
        ``bundle[-1]`` to ``_fault_iter`` — ending recovery ``w_sparse``
        iters short of the pre-fault state.  Using the replay-list
        exhaustion signal instead keeps catch-up iters in the loop when
        ``_fault_iter > bundle[-1]``.
        """
        with trace_range("recovery_iter_end"):
            next_iter = self.converter.get_next_replay_iteration()
            if next_iter is None:
                self.end_recovery()
                self._release_paused_peers()
                return

            self._setup_replay_iter(next_iter, model, optimizer, thaw_activated=True)

            # Advance the replay cursor so the next iteration's log lookups pull
            # the next logged timestamp.  The engine's global_steps is unrelated
            # to the original iteration numbers stored in the logs.
            if self._replay_iteration_cursor is not None:
                self._replay_iteration_cursor += 1

            # Last replay iter just set up: next train_batch would step past
            # ``_fault_iter``, so release paused peers and exit recovery now.
            if self.converter.get_remaining_replay_count() == 0:
                self.end_recovery()
                self._release_paused_peers()

    def on_before_optimizer_step(self, model):
        """Called by the engine before the optimizer step.

        Two responsibilities:

        1. Queue the cross-stream wait_event for the previous boundary's
           snapshot D2H.  ``optimizer.step`` is the only consumer of the
           snapshot D2H source tensors (post-step master + Adam state),
           so the wait belongs HERE — not at the end of the prior iter's
           ``on_iteration_end``, where it would block iter N+1's first
           forward kernel that has no read-after-write dependency on the
           D2H.  Moving the wait to this hook lets iter N+1's forward +
           backward overlap with side-stream drain; only ``optimizer.step``
           pays a wait, and at gas>1 with adequate compute headroom it's
           typically a no-op (event already fired by the time it's
           enqueued).
        2. During recovery, zero weight gradients for all frozen
           operators so that ZeRO's gradient partition contains zeros
           for those parameters and the optimizer step leaves them
           unchanged.

        Args:
            model: The model module (DeepSpeedEngine.module).
        """
        self.snapshot_engine.wait_for_pending_d2h_event()
        if self._recovering:
            self.zero_frozen_gradients(model)

    def _freeze_operator_params(self, model):
        """Intercept frozen operators' ``nn.Linear`` forward so backward
        returns ``grad_input`` but skips the wgrad GEMM.

        Paper §3.3 specifies that frozen operators skip weight-gradient
        computation.  The naive realization — flipping
        ``requires_grad=False`` on every frozen param — does NOT work under
        pipeline parallelism: once every ``nn.Linear`` on a stage is
        marked non-trainable, the forward produces a tensor with no
        ``grad_fn`` (stage 0's input is raw data with
        ``requires_grad=False``; stages 1+'s recv buffer's
        ``requires_grad=True`` flag flows through weighted ops but not
        past them when weights are also non-trainable), and the
        subsequent ``.backward(grad=...)`` call raises ``"element 0 of
        tensors does not require grad"``.

        Instead we take the zero-bubble split's shape: keep
        ``requires_grad=True`` on the weight so autograd builds the
        graph, and swap ``nn.Linear.forward`` for a
        ``_FrozenLinearFunction`` invocation whose backward returns
        ``None`` for the weight / bias grad slots.  The wgrad GEMM never
        runs, ``weight.grad`` stays untouched, and pipeline p2p still
        receives ``grad_input`` so upstream stages step through their
        own backward.

        Idempotent: an op already in ``_frozen_param_backup`` is skipped,
        so a re-entry path (``load_sparse_checkpoint`` twice without an
        intervening ``end_recovery``) doesn't double-wrap.
        """
        for op_name in self.converter.get_frozen_operators():
            if op_name in self._frozen_param_backup:
                continue
            wrapped = []
            for linear in self._iter_frozen_linears(op_name, model):
                _wrap_linear_for_recovery(linear)
                wrapped.append(linear)
            if wrapped:
                self._frozen_param_backup[op_name] = wrapped

    def _is_non_expert_param(self, param):
        """Predicate: does ``param`` belong in the ``non_expert`` operator bucket?

        Two exclusions collapse into one membership test:

        - **Expert** params carry ``allreduce=False`` (stamped by
          ``deepspeed.moe.experts.Experts.__init__``).  The non_expert
          bucket is "everything not expert" by the paper's
          categorisation, so these must go to ``layer_X_expert_N`` ops.
        - **Gate** params have *no* expert marker — DeepSpeed leaves
          the gate's allreduce default (``True``) because gates are
          shared across DP — so the ``allreduce`` predicate alone lets
          them leak into non_expert.  MoEvement registers gates as
          their own ``layer_X_gate`` operator, so we exclude them by
          ``id()`` via ``_gate_param_ids`` populated at discovery.

        Pre-discovery (before ``initialize``) ``_gate_param_ids`` is
        empty — matches the old behaviour for test fixtures that build
        coordinators without MoE layers.
        """
        if hasattr(param, 'allreduce') and not param.allreduce:
            return False
        if id(param) in self._gate_param_ids:
            return False
        return True

    def _resolve_frozen_linears(self, op_name, model):
        """Walk ``op_name``'s subtree once and collect its ``nn.Linear`` modules.

        Called from ``_discover_operators`` at coordinator init; the
        resolved list is stored in ``_frozen_linears_by_op`` and reused
        by every subsequent recovery, so this function never runs on the
        hot path.

        For MoE experts / gate ops the operator map holds the owning
        module directly, so we walk its ``.modules()`` and pick every
        linear child.  The ``non_expert`` sentinel has ``module=None``;
        its linears live scattered across the model and are identified
        by ``_is_non_expert_param`` (DeepSpeed's ``allreduce`` marker
        plus the MoEvement gate-id exclusion).
        """
        module = self._operator_map.get(op_name)
        if module is not None:
            return [sub for sub in module.modules() if isinstance(sub, nn.Linear)]
        if model is None:
            return []
        out = []
        for sub in model.modules():
            if not isinstance(sub, nn.Linear):
                continue
            if all(self._is_non_expert_param(p) for p in sub.parameters()):
                out.append(sub)
        return out

    def _iter_frozen_linears(self, op_name, model):
        """Return cached ``nn.Linear`` submodules belonging to ``op_name``.

        Resolution happens once at ``_discover_operators`` time via
        ``_resolve_frozen_linears``; this is just a list lookup.  ``model``
        is retained for the defensive fallback when an op was never seen
        at discovery (shouldn't happen under normal flow but keeps the
        signature compatible if a caller hands in an unmapped name).
        """
        cached = self._frozen_linears_by_op.get(op_name)
        if cached is not None:
            return cached
        return self._resolve_frozen_linears(op_name, model)

    def _resolve_op_params(self, op_name, model):
        """Walk ``op_name``'s parameters once and return ``(name, Parameter)`` pairs.

        Mirrors ``_resolve_frozen_linears`` but for the full parameter
        set, not just linears.  Called once at ``_discover_operators``
        time; results cached on ``_param_list_by_op`` so the per-iter
        snapshot loops skip the module-subtree walk +
        ``model.named_parameters()`` rebuild.

        Module-backed ops return ``list(module.named_parameters())``.
        The ``non_expert`` sentinel filters the full model by the
        MoE-layer ``allreduce`` marker (the same predicate
        ``_get_non_expert_params`` / ``_active_non_expert_params`` apply
        on the fallback path).  Parameter layouts are static for the run;
        dynamic-router models that mutate layouts mid-run would need to
        invalidate this cache — not supported.
        """
        module = self._operator_map.get(op_name)
        if module is not None:
            return list(module.named_parameters())
        if model is None:
            return []
        return [(name, p) for name, p in model.named_parameters() if self._is_non_expert_param(p)]

    def _iter_op_params(self, op_name, model):
        """Return cached ``[(name, Parameter)]`` pairs for ``op_name``.

        List lookup against ``_param_list_by_op``; defensive fallback to
        ``_resolve_op_params`` for unknown names (shouldn't happen under
        normal flow, kept for signature parity with ``_iter_frozen_linears``).
        """
        cached = self._param_list_by_op.get(op_name)
        if cached is not None:
            return cached
        return self._resolve_op_params(op_name, model)

    def _get_exp_counts_stream(self):
        """Lazy-init the dedicated side stream for exp_counts D2Hs.

        Returns ``None`` on CPU-only accelerators (no streams available)
        — callers fall back to the synchronous-copy path.
        """
        if self._exp_counts_stream is not None:
            return self._exp_counts_stream
        ctor = get_accelerator().Stream
        if ctor is None:
            return None
        self._exp_counts_stream = ctor()
        return self._exp_counts_stream

    def _schedule_exp_counts_copies(self, slot_idx):
        """Issue non_blocking D2H for every MoE layer's ``exp_counts``.

        Returns a list of ``(layer_idx, cpu_tensor)`` pairs; the copies
        have been *posted* to the dedicated exp_counts side stream but
        not necessarily completed.  Call ``_fence_exp_counts_copies()``
        before reading the CPU tensors.

        Each layer's destination buffer is a pinned CPU tensor cached on
        ``self._exp_counts_pinned`` keyed by ``(layer_idx, slot_idx)``
        where ``slot_idx = _window_step``.  One buffer per slot per
        layer ensures all ``w_sparse`` iters' counts coexist until the
        window-boundary fence — without the per-slot separation, a
        later iter's D2H would overwrite an earlier iter's counts
        before the scheduler read them at boundary flush.  Shape and
        dtype are stable after the first forward so the buffer
        allocates once per ``(layer_idx, slot_idx)`` pair then
        amortises.  On CPU-only accelerators (no pinned memory), falls
        back to plain ``torch.empty``; the copy is still batched but
        not truly async (PyTorch can't non_blocking to an unpinned
        destination).

        Stream coordination: the gate's ``exp_counts`` is produced on
        the default stream (forward), so the side stream
        ``wait_stream(default)`` before the copies queue.  Using
        ``default_stream()`` rather than ``current_stream()`` makes the
        coordination explicit and robust to any future engine that
        sets a non-default current stream.
        """
        pending = []
        use_pinned = _supports_pinned_memory()
        side_stream = self._get_exp_counts_stream()
        if side_stream is not None:
            side_stream.wait_stream(get_accelerator().default_stream())
        ctx = get_accelerator().stream(side_stream) if side_stream is not None else contextlib.nullcontext()
        with ctx:
            for layer_idx, moe_layer in enumerate(self._moe_layers):
                exp_counts = getattr(moe_layer, 'exp_counts', None)
                if exp_counts is None:
                    continue
                cache_key = (layer_idx, slot_idx)
                cached = self._exp_counts_pinned.get(cache_key)
                if (cached is None or cached.shape != exp_counts.shape or cached.dtype != exp_counts.dtype):
                    # First visit to this ``(layer, slot)`` pair, or the
                    # gate's shape/dtype changed (defensive — shouldn't
                    # happen after the first forward).  Warn loudly on the
                    # mid-run mismatch path so a silent per-iter realloc
                    # is visible to operators: if the gate's shape/dtype
                    # drifts across iters, pinned allocation (hundreds of
                    # ms on CUDA) fires every step instead of amortising.
                    if cached is not None:
                        logger.warning(f"[MoEvement] exp_counts shape/dtype changed on layer {layer_idx} "
                                       f"({cached.shape}/{cached.dtype} → {exp_counts.shape}/{exp_counts.dtype}); "
                                       f"re-allocating pinned buffer.  Upstream regression?")
                    cached = torch.empty(exp_counts.shape, dtype=exp_counts.dtype, pin_memory=use_pinned)
                    self._exp_counts_pinned[cache_key] = cached
                cached.copy_(exp_counts.detach(), non_blocking=use_pinned)
                pending.append((layer_idx, cached))
        return pending

    def _fence_exp_counts_copies(self):
        """Block until all pending ``exp_counts`` D2H copies are readable.

        Syncs only the dedicated ``_exp_counts_stream`` rather than the
        full main/current stream — the main stream's pending training
        kernels (forward/backward leftover, NCCL collectives, optimizer
        ops) are NOT waited on here.  Earlier this fence used
        ``current_stream().synchronize()`` and dragged on those kernels.

        No-op on CPU-only accelerators (the copies ran synchronously
        under ``_schedule_exp_counts_copies``'s fallback).
        """
        if self._exp_counts_stream is not None:
            self._exp_counts_stream.synchronize()

    def _record_iter_duration(self):
        """Append the elapsed time since the last call to the rolling window.

        First call in a run has no prior timestamp and is silently
        skipped — the window starts filling on the second call.  Recovery
        iterations deliberately do not feed the window: their duration
        is dominated by log-transfer / catch-up replay, not steady-state
        training, so they would poison ``find_window_size``'s estimate.
        """
        now = time.perf_counter()
        if self._last_iter_end_time is not None:
            self._iter_time_window.append(now - self._last_iter_end_time)
        self._last_iter_end_time = now

    def _maybe_update_iter_time(self):
        """Refresh ``_iter_time_sec`` to the median of the rolling window.

        Waits until the window has at least half its configured capacity
        — with fewer samples, a single outlier skews the median
        meaningfully.  Median (rather than mean) is robust to the
        occasional GC pause or DP-collective straggler that would
        otherwise inflate the estimate and over-shrink ``w_sparse``.
        """
        if self._iter_time_window.maxlen is None:
            return
        min_samples = max(1, self._iter_time_window.maxlen // 2)
        if len(self._iter_time_window) < min_samples:
            return
        self._iter_time_sec = statistics.median(self._iter_time_window)

    def _iter_time_drift_exceeds_threshold(self):
        """True when the current iter-time estimate has drifted far enough
        from the value used to build the current schedule to justify a
        regen (see ``_ITER_TIME_DRIFT_TO_REGEN``).

        No-op when the window hasn't populated yet (``_iter_time_sec`` and
        ``_last_scheduled_iter_time`` both unchanged from init).
        """
        if self._iter_time_sec is None or self._last_scheduled_iter_time is None:
            return False
        baseline = max(self._last_scheduled_iter_time, 1e-9)
        drift = abs(self._iter_time_sec - self._last_scheduled_iter_time) / baseline
        return drift >= _ITER_TIME_DRIFT_TO_REGEN

    def _warn_if_tensor_parallel_linears(self, model):
        """Emit a one-shot warning if the model contains TP-sharded linears.

        ``_iter_frozen_linears`` filters on ``isinstance(..., nn.Linear)``
        only.  Megatron-style ``ColumnParallelLinear`` / ``RowParallelLinear``
        subclass ``nn.Module`` rather than ``nn.Linear``, so they slip
        past that filter — the frozen-op wgrad-skip wrapper is never
        installed and the GEMM still fires during recovery replay.
        Additionally, those variants carry a dgrad all-reduce (or
        reduce-scatter under sequence parallelism) that a naive swap to
        ``_FrozenLinearFunction`` would drop.

        Current MoEvement targets vanilla DeepSpeed MoE with stock
        ``nn.Linear`` experts.  If a contributor wires it up against a
        TP'd model, this warning surfaces the gap immediately rather
        than silently producing incorrect gradients.
        """
        if model is None:
            return

        offenders = set()
        for sub in model.modules():
            cls_name = type(sub).__name__
            if cls_name in ('ColumnParallelLinear', 'RowParallelLinear'):
                offenders.add(cls_name)
                continue
            # Generic Megatron / custom-TP marker — an attribute explicitly
            # declaring multi-way parameter sharding.
            tp_size = getattr(sub, 'tensor_model_parallel_size', None)
            if isinstance(tp_size, int) and tp_size > 1:
                offenders.add(cls_name)

        if offenders:
            logger.warning(
                "[MoEvement] Detected tensor-parallel linear modules (%s) in the model. "
                "MoEvement's frozen-op wgrad-skip wrapper targets stock ``nn.Linear`` only; "
                "these modules will silently execute full weight-gradient GEMMs during "
                "recovery replay and may drop required dgrad collectives.  Real TP support "
                "is not implemented — treat MoEvement results on this model as incorrect.",
                sorted(offenders),
            )

    def _warn_if_int_step_optimizer(self, optimizer):
        """Emit a one-shot warning if the inner optimizer stores ``step`` as a Python int.

        ``snapshot_operator``'s ``isinstance(tensor, torch.Tensor)``
        filter drops anything that isn't a tensor.  ``torch.optim.Adam``
        / ``AdamW`` (PyTorch 2.0+) store ``state['step']`` as a 0-dim
        Tensor and pass the filter; ``deepspeed.ops.adam.fused_adam.FusedAdam``
        and ``DeepSpeedCPUAdam`` initialize ``state['step'] = 0`` and
        ``+= 1`` it as a Python int.  Under those optimizers, MoEvement
        silently fails to round-trip ``step``: post-recovery Adam restarts
        at ``t=1`` and the first step's bias correction is far too large.

        Promoting int->tensor on capture + demoting back on restore would
        require touching the ZeRO/non-ZeRO write paths, the H2D batched
        applier, and adding a round-trip marker.  Out of scope for the
        examples (``torch.optim.AdamW``); this warning surfaces the gap
        for anyone wiring up FusedAdam/CPUAdam against MoEvement.
        """
        if optimizer is None:
            return
        # Walk the wrapper chain (DeepSpeed wraps the user optimizer in
        # ZeRO / FP16 / BF16 wrappers exposing ``.optimizer``).
        offender = None
        seen = set()
        cur = optimizer
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if type(cur).__name__ in ('FusedAdam', 'DeepSpeedCPUAdam'):
                offender = type(cur).__name__
                break
            cur = getattr(cur, 'optimizer', None)
        if offender is not None:
            logger.warning(
                "[MoEvement] Detected %s as the inner optimizer.  MoEvement's snapshot "
                "path captures only ``torch.Tensor`` optimizer-state entries, but %s "
                "stores Adam's ``step`` as a Python int and silently drops it from the "
                "bundle.  Post-recovery, Adam restarts at ``t=1`` and the first step's "
                "bias correction is ~10x too large.  Use ``torch.optim.Adam`` / ``AdamW`` "
                "(``torch_adam=True`` in DeepSpeed config) for MoEvement-compatible "
                "optimizer-state round-trip.", offender, offender)

    def _thaw_operator_params(self, op_name, model):
        """Restore the original ``forward`` on every linear we wrapped for ``op_name``.

        ``model`` is unused but kept in the signature for call-site symmetry
        with ``_freeze_operator_params``.
        """
        del model
        wrapped = self._frozen_param_backup.pop(op_name, None)
        if wrapped is None:
            return
        for linear in wrapped:
            _unwrap_linear(linear)

    def zero_frozen_gradients(self, model):
        """Zero weight gradients for all frozen operator parameters.

        Must be called after loss.backward() and before the optimizer step.
        Frozen operators have their gradients zeroed so the optimizer update
        is a no-op for those parameters, preserving the FP16 snapshot weights.

        Args:
            model: The model module.
        """
        for op_name in self.converter.get_frozen_operators():
            if op_name not in self._operator_map:
                continue
            module = self._operator_map[op_name]
            if module is None:
                for param in model.parameters():
                    if self._is_non_expert_param(param) and param.grad is not None:
                        param.grad.zero_()
            else:
                for param in module.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

    def _operator_param_dict(self, op_name, model):
        """Return the ``{param_name: tensor}`` dict for an operator.

        Handles both paths uniformly:

        - Module-backed ops: the op's ``_operator_map`` entry is a live
          ``nn.Module``; we return ``dict(module.named_parameters())``.
        - ``non_expert`` sentinel: ``_operator_map[name] is None``; we
          filter ``model.named_parameters()`` down to the non-expert
          subset (params without ``allreduce=False``).

        Returns ``{}`` for unknown ops or when ``model`` is needed for
        the non-expert path but isn't provided (e.g., converter-only
        unit tests).
        """
        if op_name not in self._operator_map:
            return {}
        module = self._operator_map[op_name]
        if module is not None:
            return dict(module.named_parameters())
        if model is None:
            return {}
        return {name: p for name, p in model.named_parameters() if self._is_non_expert_param(p)}

    def _setup_replay_iter(self, iteration, model, optimizer, thaw_activated=False):
        """Prepare model state for the replay train_batch at ``iteration``.

        Two steps:

        1. Promote the operators whose schedule says they become ACTIVE
           at this iteration — load their FP32 + optimizer-state capture
           from the converter's per-iter cache into the live model, thaw
           autograd, and mark them ACTIVE in the converter.
        2. Refresh frozen ops' FP16 weights in the live model to this
           iteration's per-snapshot capture.  Spec §3.2: each iteration's
           snapshot carries the frozen-op FP16 values AT that iteration,
           and replay's forward pass reads them from ``param.data``; if
           we didn't refresh, iter N's forward would see iter N-1's FP16
           values (or whatever was in memory post-fault) and the replay
           trajectory would diverge from fault-free.
        """
        # If streaming peer-pull is active, block until this iter's
        # state has been ingested from the pull thread.  No-op when
        # the stream is already closed or the target iter is already
        # present — the common case in steady-state replay once pull
        # is running ahead of replay.
        if self._streaming_pull_queue is not None:
            self._drain_streaming_iter(iteration)
        # Derive the iter's new-active operators directly from the bundle:
        # anything that has an FP32 capture at this iter became ACTIVE here at
        # save time.  Using the live ``scheduler.schedule`` would disagree when
        # save-time and load-time scheduler states differ (e.g., the post-load
        # scheduler starts with zero expert activation counts and orders ops
        # differently from a trained scheduler at save time).  The bundle's
        # per-iter capture is the canonical source of truth.
        active_names = list(self.converter._fp32_weights_per_iter.get(iteration, {}).keys())
        window_start = self._window_start_for_replay()
        window_idx = iteration - window_start if window_start is not None else iteration

        # (1) Promote new actives.
        touched_zero_partitioned = False
        missing = []
        with trace_range("recovery/replay_apply_active"):
            for name in active_names:
                param_dict = self._operator_param_dict(name, model)
                if not param_dict:
                    missing.append(name)
                    continue
                fp32 = self.converter.get_fp32_weights(name, iteration) or {}
                optim = self.converter.get_optimizer_state(name, iteration) or {}
                if not fp32 and not optim:
                    missing.append(name)
                    continue
                self._apply_fp32_into_params(param_dict, fp32)
                self._apply_optim_state_into_params(param_dict, optimizer, optim)
                # ZeRO-enrollment predicate must be cross-rank uniform.  Using
                # ``_is_zero_partitioned`` here (truthy ``_hp_mapping``) only fires
                # on the DP peer that OWNS a fragment, so non-owner peers skip the
                # ``update_lp_params`` call below and its all-gather deadlocks NCCL
                # on the owner side.  ``hasattr(p, '_hp_mapping')`` is set cross-
                # rank uniformly by ``link_hp_params`` regardless of ownership and
                # is the correct predicate for "should every peer run the
                # HP→LP propagation collective."
                if not touched_zero_partitioned:
                    touched_zero_partitioned = any(hasattr(p, '_hp_mapping') for p in param_dict.values())

        if missing:
            preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            logger.warning(f"[MoEvement] Iter {iteration} (window slot {window_idx}) expected "
                           f"{len(active_names)} active operators, but {len(missing)} have no "
                           f"capture ({preview}); running replay against current model state")

        self.converter.activate_operators(iteration, active_names)

        if thaw_activated:
            for name in active_names:
                self._thaw_operator_params(name, model)

        # Propagate HP (FP32 master) → LP (training-precision) in one
        # batched all-gather after every active-op restore finishes.  Only
        # needed when a ZeRO wrapper is managing a separate LP buffer;
        # calling it on a plain optimizer would AttributeError, so gate on
        # both the flag and the method presence.
        if touched_zero_partitioned and hasattr(optimizer, 'update_lp_params'):
            with trace_range("recovery/replay_update_lp_params"):
                optimizer.update_lp_params()

        # (2) Refresh still-frozen ops' FP16 to this iteration's capture.
        if model is not None:
            with trace_range("recovery/replay_restore_fp16"):
                self._restore_fp16_weights_into_model(model, iteration)

        # (3) Restore the post-iter-K RNG state captured at fault-free
        # ``on_iteration_end`` so the next replay tb's forward consumes
        # the same RNG stream as fault-free iter K+1's forward.  No-op
        # for catch-up iters (no capture beyond the persisted window) and
        # for older bundles without per-iter RNG — RNG simply advances
        # naturally from whatever state the prior tb left, matching the
        # pre-feature behavior.
        rng_state = self.converter.get_rng_state(iteration)
        if rng_state is not None:
            with trace_range("recovery/replay_restore_rng"):
                self._restore_rng_state(rng_state)

        # SD-O4 S3: free this iter's per-iter caches now that
        # ``_batched_h2d_apply`` has copied every view into its own
        # fresh pinned flat buffer.  Per-iter views are read only by
        # the synchronous CPU memcpy inside ``_batched_h2d_apply``;
        # the H2D's source is the fresh buffer, not these views, so
        # there is no in-flight stream dependency on them and no
        # event sync is required.  Bounding the converter dicts is
        # what lets the bounded pull queue actually deliver
        # ``max_prefetched_iters`` worth of peak CPU memory under
        # streaming — without this, the dicts grow O(N) until
        # ``end_recovery`` regardless of queue cap.
        if self._streaming_pull_queue is not None or self._streaming_pull_thread is not None:
            self.converter.drop_iteration(iteration)
            self.snapshot_engine.release_iter_buffers(iteration)

        logger.info(f"[MoEvement] Set up replay iter {iteration} (window slot {window_idx}): "
                    f"{len(active_names) - len(missing)} ops promoted to ACTIVE")

    def _window_start_for_replay(self):
        """Return the window's first captured iteration for replay bookkeeping.

        The schedule indexes by ``window_idx = iteration - window_start``, so
        ``window_start`` must name the FIRST iteration whose snapshot landed
        in the bundle — not the ``_window_start_iteration`` value carried in
        metadata, which ``begin_window(global_step)`` records as the global
        step at which the *next* window begins (i.e. the last iter of the
        completed window, not its first).

        Computes the first iteration from the cached per-iter keys — that's
        the source of truth for which iterations the bundle covers.  Falls
        back to the replay cursor for the rare case where the cache wasn't
        populated (tests driving the converter directly).
        """
        if self._cached_snapshot_data is not None:
            _, per_iter_operator_states = self._cached_snapshot_data
            if per_iter_operator_states:
                return min(per_iter_operator_states.keys())
        return self._replay_iteration_cursor

    @staticmethod
    def _batched_h2d_apply(items, target_device):
        """Batch CPU→device copies by dtype, then run per-item appliers.

        ``items`` is a list of ``(src_cpu, apply_fn)`` pairs — the callers
        in the restore path build these lazily as they walk per-operator
        snapshot dicts.  Tensors sharing a dtype pack into one flat
        **pinned** CPU buffer that crosses PCIe in a single
        ``non_blocking=True`` H2D; each ``apply_fn`` is then invoked
        with the device-side slice of that flat buffer reshaped to the
        source tensor's shape.

        Without this batching the restore path issued one ``.to(device)``
        per parameter — tens to hundreds of H2D launches per replay
        iter, each its own sync point.

        Pinning the packed flat buffer is what lets the H2D actually run
        async: PyTorch silently ignores ``non_blocking=True`` when the
        source is unpinned (which is the case for ``load_bundle``-cloned
        snapshot tensors), so a naive ``torch.cat`` of those sources
        would still stall the host per copy.  Allocating a fresh pinned
        flat, ``copy_``-ing sources into it (memcpy on CPU), then a
        ``non_blocking=True`` ``.to(device)`` kicks the transfer onto
        the stream and lets the caller's subsequent work overlap.

        On backends without a pinned-memory allocator (HPU / XPU / MPS /
        CPU-only), ``_supports_pinned_memory()`` returns False and this
        falls back to unpinned + sync H2D — same end-state as the
        pre-patch behaviour; we just lose the async-overlap win.

        Empty items lists, empty dtype groups, and zero-element tensors
        all short-circuit safely.
        """
        if not items:
            return
        by_dtype = defaultdict(list)
        for src, apply_fn in items:
            by_dtype[src.dtype].append((src, apply_fn))
        use_pinned = _supports_pinned_memory()
        for dtype, group in by_dtype.items():
            total = sum(t.numel() for t, _ in group)
            if total == 0:
                continue
            flat_cpu = torch.empty(total, dtype=dtype, pin_memory=use_pinned)
            offset = 0
            for src, _ in group:
                n = src.numel()
                flat_cpu[offset:offset + n].copy_(src.contiguous().view(-1))
                offset += n
            flat_gpu = flat_cpu.to(target_device, non_blocking=use_pinned)
            offset = 0
            for src, apply_fn in group:
                n = src.numel()
                view = flat_gpu[offset:offset + n].view(src.shape)
                apply_fn(view)
                offset += n

    def _apply_fp32_into_params(self, param_dict, fp32_weights):
        """Copy FP32 snapshot weights into a pre-built named-param dict.

        ZeRO-1/2 path routes through ``safe_set_full_fp32_param`` so the
        partitioned master sees the restored value; plain PyTorch falls
        back to a direct ``param.data.copy_``.  Batches the per-param
        H2D copies via ``_batched_h2d_apply`` so the restore incurs one
        H2D launch per dtype rather than one per parameter.
        """
        items = []
        target_device = None
        for param_name, weight_tensor in fp32_weights.items():
            if param_name not in param_dict:
                continue
            param = param_dict[param_name]
            if target_device is None:
                target_device = param.device

            if _is_zero_partitioned(param):
                hp_mapping = getattr(param, '_hp_mapping', None)
                # FS1 captures ``_hp_mapping.hp_fragment.data`` directly
                # (tensor.numel() == lp_fragment_address.numel).  The legacy
                # full-shape FP32 capture routed through
                # ``safe_set_full_fp32_param`` / ``set_full_hp_param`` which
                # narrows from full-shape — that narrow crashes when the
                # captured tensor is already the per-rank fragment.  Mirror
                # the fragment-vs-full-shape discrimination used for optim
                # state in ``_apply_optim_state_into_params``.
                if (hp_mapping is not None and getattr(hp_mapping, 'hp_fragment', None) is not None
                        and weight_tensor.numel() == hp_mapping.lp_fragment_address.numel
                        and param.numel() != hp_mapping.lp_fragment_address.numel):

                    def _apply(view, hp_mapping=hp_mapping):
                        hp_mapping.hp_fragment.data.copy_(view.data.reshape_as(hp_mapping.hp_fragment.data))
                else:

                    def _apply(view, param=param):
                        safe_set_full_fp32_param(param, view)
            else:

                def _apply(view, param=param):
                    with torch.no_grad():
                        param.data.copy_(view)

            items.append((weight_tensor, _apply))
        if target_device is not None:
            self._batched_h2d_apply(items, target_device)

    def _apply_optim_state_into_params(self, param_dict, optimizer, optim_states):
        """Copy Adam-style optimizer state into the optimizer for each param.

        Mirrors ``_apply_fp32_into_params`` — ``optim_states`` is keyed as
        ``"param_name.adam_key"`` (e.g. ``"weight.exp_avg"``) and splits
        into the per-param, per-Adam-key destination.  Adam's
        ``exp_avg``/``exp_avg_sq`` share dtype, so one H2D carries every
        (param, key) pair for the op; ``step`` (int) goes in its own
        batch.  See ``_batched_h2d_apply`` for the shape of the batching.

        The split uses ``rsplit('.', 1)`` because param names can contain
        dots (``wg.weight``, nested expert children).  An earlier
        ``find('.')`` splitter took the first dot, parsed ``wg.weight.exp_avg``
        as ``param='wg'`` / ``key='weight.exp_avg'``, failed the
        ``param_name in param_dict`` check, and silently skipped the
        entry — so gate / non-expert / nested-module Adam moments never
        got restored on recovery.
        """
        if optimizer is None:
            return
        items = []
        target_device = None
        for composite_key, tensor in optim_states.items():
            parts = composite_key.rsplit('.', 1)
            if len(parts) != 2:
                continue
            param_name, adam_key = parts
            if param_name not in param_dict:
                continue
            param = param_dict[param_name]
            if target_device is None:
                target_device = param.device

            if _is_zero_partitioned(param):
                # ``safe_set_full_optimizer_state`` writes into the rank's own
                # ``_hp_mapping.optim_fragment`` dict.  Under ZeRO-1 + EP
                # sharding (ep_size > 1 with the EP group sized as the DP
                # group) a rank may not own any fragment of an expert param
                # — ``optim_fragment`` is still ``None`` post-first-step for
                # non-owners because ``set_optim_state_fragment`` never ran
                # for params outside this rank's share.  ``get_hp_fragment``
                # then does ``key in self.optim_fragment`` which crashes on
                # ``None``.  The owner of that fragment will apply its own
                # slice separately, so skipping on non-owners is safe; the
                # collect side gates the SAME way at
                # ``_collect_param_optim_state``'s ``mapping.optim_fragment
                # is not None`` check, so restore must mirror to stay
                # cross-rank symmetric.
                hp_mapping = getattr(param, '_hp_mapping', None)
                if hp_mapping is None or getattr(hp_mapping, 'optim_fragment', None) is None:
                    continue

                # FS1 captures per-rank ``optim_fragment[key]`` directly
                # (tensor.numel() == lp_fragment_address.numel).  The legacy
                # full-shape path in ``safe_set_full_optimizer_state`` does
                # ``narrow(value, 0, start, numel)`` which crashes when the
                # captured tensor is already the fragment (start can exceed
                # fragment.numel under uneven partitioning).  Discriminate
                # by numel: fragment-shaped → direct copy into the mapping's
                # hp_fragment; full-shape → narrow-based path.
                frag_numel = hp_mapping.lp_fragment_address.numel
                if tensor.numel() == frag_numel and param.numel() != frag_numel:

                    def _apply(view, hp_mapping=hp_mapping, adam_key=adam_key):
                        hp_frag = hp_mapping.get_hp_fragment(adam_key)
                        hp_frag.data.copy_(view.data.reshape_as(hp_frag.data))
                else:

                    def _apply(view, param=param, adam_key=adam_key):
                        safe_set_full_optimizer_state(param, view, adam_key)
            else:

                def _apply(view, param=param, adam_key=adam_key):
                    optimizer.state[param][adam_key] = view

            items.append((tensor, _apply))
        if target_device is not None:
            self._batched_h2d_apply(items, target_device)

    def _restore_fp16_weights_into_model(self, model, iteration):
        """Copy this iteration's FROZEN-op FP16 weights into the live model.

        Called per replay iteration with ``iteration`` = the original
        iter number being replayed.  The converter's per-iter FP16
        cache (``_fp16_weights_per_iter[iteration][op_name]``) maps
        dotted param names to CPU FP16 tensors matching the op's
        ``named_parameters()`` layout — module-relative for modules
        in ``_operator_map``, top-level for the ``non_expert`` sentinel.

        Writes directly to ``param.data``: FROZEN ops have
        ``requires_grad=False`` and skip the optimizer step, so the LP
        buffer is the single source of truth during replay.  Under ZeRO-1
        the LP params are fully replicated per-rank, so a full-shape copy
        is safe.
        """
        # Collect every (src_fp16, target_param) pair across all frozen ops
        # that have an FP16 capture at this iter, then issue ONE H2D per
        # dtype group via ``_batched_h2d_apply``.  FP16 tensors share dtype,
        # so this collapses to a single H2D regardless of how many frozen
        # ops fire at this iter — the pre-batching path issued one per
        # param per op.  Dtype mismatches (bundle FP16 vs. model BF16 on
        # bf16_optimizer) still re-cast inside the applier, after the
        # single H2D has landed the bulk in device memory.
        items = []
        target_device = None
        for op_name in self.converter.get_frozen_operators():
            fp16_weights = self.converter.get_fp16_weights(op_name, iteration)
            if not fp16_weights:
                continue
            param_dict = self._operator_param_dict(op_name, model)
            if not param_dict:
                continue
            for name, param in param_dict.items():
                if name not in fp16_weights:
                    continue
                if target_device is None:
                    target_device = param.device

                def _apply(view, param=param):
                    value = view if view.dtype == param.dtype else view.to(param.dtype)
                    with torch.no_grad():
                        param.data.copy_(value)

                items.append((fp16_weights[name], _apply))
        if target_device is not None:
            self._batched_h2d_apply(items, target_device)

    def _restore_module_weights(self, module, fp32_weights):
        """Copy FP32 snapshot weights into a live module's parameters.

        Under ZeRO-1/2 we write into the HP (FP32 master) fragment via
        ``safe_set_full_fp32_param`` so the next optimizer step sees
        the restored values.  The LP (training-precision) copy is
        synced separately by the caller via ``optimizer.update_lp_params()``
        after all restores in this window slot finish; batching the
        sync avoids N redundant all-gathers, one per module.

        Args:
            module: The nn.Module whose parameters will be updated.
            fp32_weights: Dict of dotted param name -> CPU FP32 tensor from snapshot.
        """
        param_dict = dict(module.named_parameters())
        for param_name, weight_tensor in fp32_weights.items():
            if param_name in param_dict:
                param = param_dict[param_name]
                value = weight_tensor.to(param.device)
                if _is_zero_partitioned(param):
                    safe_set_full_fp32_param(param, value)
                else:
                    with torch.no_grad():
                        param.data.copy_(value)

    def on_send_activations(self, tensor, iteration, micro_batch_id, stage_id):
        """Hook called when pipeline stage sends activations downstream.

        Args:
            tensor: Activation tensor being sent.
            iteration: Current iteration.
            micro_batch_id: Current microbatch.
            stage_id: Sending stage ID.
        """
        if self.upstream_logger is not None:
            self.upstream_logger.log_activation(tensor, iteration, micro_batch_id, stage_id)

    def on_send_gradients(self, tensor, iteration, micro_batch_id, stage_id):
        """Hook called when pipeline stage sends gradients upstream.

        Args:
            tensor: Gradient tensor being sent.
            iteration: Current iteration.
            micro_batch_id: Current microbatch.
            stage_id: Sending stage ID.
        """
        if self.upstream_logger is not None:
            self.upstream_logger.log_gradient(tensor, iteration, micro_batch_id, stage_id)

    def save_sparse_checkpoint(self, save_dir, tag):
        """Save the current sparse checkpoint state to disk.

        Also persists this rank's upstream logs so that a whole-job restart
        can replay the last sparse window without relying on neighbour memory.

        Direct caller; the gate that suppresses this on ``engine.save_checkpoint``
        when ``MoEvementConfig.persist_to_disk`` is False lives at the
        ``runtime/engine.py`` call site so tests / advanced users that drive
        ``save_sparse_checkpoint`` directly always get the bundle write they
        asked for.

        Args:
            save_dir: Checkpoint directory.
            tag: Checkpoint tag.
        """
        with trace_range("save_sparse_ckpt"):
            self.snapshot_engine.synchronize()
            rank = dist.get_rank() if dist.is_initialized() else 0
            # Loss-scaler state was captured at finalize_window time (see
            # on_iteration_end), matching the bundle's "state at the end of
            # the last finalized window" semantics.  Passing it through
            # ``extra_metadata`` keeps it on the same code path as the bundle
            # itself, so whole-job-restart recovery sees the scaler at the
            # same iter the ops were captured at.
            extra_metadata = {}
            scaler_state = self.snapshot_engine._persisted_loss_scaler_state
            if scaler_state is not None:
                extra_metadata["loss_scaler_state"] = scaler_state
            # Engine scalars (global_steps, global_samples, LR + compression
            # scheduler state) are tied to fault-time, not bundle-finalize
            # time.  Unlike the loss scaler (which replay advances iter-for-
            # iter), these are pinned during replay — so the restoring rank
            # needs the LIVE value at save/serve moment, not the window-
            # boundary snapshot.  ``_pending_engine_scalars`` tracks the
            # last-seen live value from ``on_iteration_end``.
            if self._pending_engine_scalars is not None:
                extra_metadata["engine_scalars"] = self._pending_engine_scalars
            self.snapshot_engine.save_to_disk(save_dir, tag, rank=rank, extra_metadata=extra_metadata or None)
            if self.upstream_logger is not None:
                self.upstream_logger.save_to_disk(save_dir, tag, rank=rank)
            # A caller (launcher, cluster manager) often tars the checkpoint
            # directory the moment save_checkpoint returns.  Flush so the
            # bundle is on disk rather than still in the worker queue.
            self.flush_persist()

    def flush_persist(self):
        """Block until every background disk write for this rank is done.

        Call before ``load_sparse_checkpoint`` in the same process — the
        snapshot and log writes run asynchronously by default, so the files
        may not exist yet when ``save_sparse_checkpoint`` returns.
        """
        self.snapshot_engine.flush_persist()
        if self.upstream_logger is not None:
            self.upstream_logger.flush_persist()

    def shutdown(self):
        """Flush pending writes and tear down background workers.

        Idempotent; registered as an ``atexit`` hook so a process that
        exits mid-training doesn't leave a half-written bundle.  The
        ``_shutdown_done`` flag is set in a ``finally`` so a raise during
        teardown (timeout, worker-thread exception) doesn't short-circuit
        the next caller — ``atexit`` + an explicit user ``shutdown()`` is
        a realistic retry path.
        """
        if self._shutdown_done:
            return
        try:
            # Drain in-flight replication jobs before we tear down the
            # executor — they may be holding pool buffers we also want to
            # free.  Single 30s budget applied to the oldest outstanding
            # future; a wedge there flags the executor broken and we
            # detach the rest rather than waiting on each in turn (atexit
            # must not hang on a dead peer).
            while self._replication_futures:
                fut = self._replication_futures.popleft()
                try:
                    fut.result(timeout=30.0)
                except FuturesTimeoutError:
                    logger.error("[MoEvement] replication job still running after 30s during "
                                 "shutdown; detaching executor instead of waiting")
                    self._replication_broken = True
                    break
                except Exception as exc:
                    logger.error(f"[MoEvement] replication job raised during shutdown: {exc}")
            if self._replication_broken:
                # Non-blocking teardown: the hung worker thread can't be
                # cancelled, but cancel_futures prevents queued submits
                # from firing post-shutdown.  ``cancel_futures`` is 3.9+;
                # on 3.8 we fall back to plain wait=False.
                try:
                    self._replication_executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    self._replication_executor.shutdown(wait=False)
            else:
                self._replication_executor.shutdown(wait=True)
            # ``flush_persist`` now raises on writer/callback failures so
            # ``save_checkpoint`` can refuse to promote a partial tag.
            # During teardown the error has already been logged by the
            # worker and the goal here is to exit cleanly — swallow it
            # so the paired ``_persist_worker.shutdown`` still runs and
            # the worker thread is joined.
            try:
                self.snapshot_engine.flush_persist()
            except Exception as exc:
                logger.error(f"[MoEvement] snapshot flush during shutdown failed: {exc}")
            self.snapshot_engine._persist_worker.shutdown()
            if self.upstream_logger is not None:
                try:
                    self.upstream_logger.flush_persist()
                except Exception as exc:
                    logger.error(f"[MoEvement] upstream-log flush during shutdown failed: {exc}")
                self.upstream_logger._persist_worker.shutdown()
        finally:
            self._shutdown_done = True

    def load_sparse_checkpoint(self, load_dir, tag, schedule=None, model=None, optimizer=None, engine=None):
        """Load sparse checkpoint and initialize conversion.

        Args:
            load_dir: Checkpoint directory.
            tag: Checkpoint tag.
            schedule: Optional checkpoint schedule override (default: use the
                engine's scheduler.schedule).
            model: The live model module.  When provided, this call seeds
                state for the FIRST replay iteration: loads that iter's
                ACTIVE-op FP32 + optimizer state into the live model, loads
                still-frozen ops' FP16 captures, and flips
                ``requires_grad=False`` on frozen params so autograd skips
                their weight-gradient computation during replay.  Safe to
                omit for single-rank unit tests that drive the converter
                directly.
            optimizer: Required alongside ``model`` to restore optimizer
                state for iter-1 actives.

        Returns:
            True if a sparse checkpoint was found and loaded.
        """
        metadata, per_iter_operator_states = self._load_snapshot_data(load_dir, tag)
        if metadata is None:
            return False

        with trace_range("recovery/converter_init"):
            self.converter.initialize_from_snapshots(metadata, per_iter_operator_states, schedule)
        self._checkpoint_load_dir = load_dir
        self._checkpoint_tag = tag

        # ``_global_step`` at load time equals the paused peers' most recently
        # completed iter — the target catch-up iter.  Cascade / peer-pull use
        # the same convention; see ``_compute_replay_iters``.
        #
        # Under whole-cluster-restart (the ``engine.load_checkpoint`` path),
        # the engine has already restored its ``global_steps`` from the
        # checkpoint's client state by the time this runs, but the
        # coordinator's own ``_global_step`` is still at its ``__init__``
        # value (0) because no ``on_iteration_end`` has fired yet — so
        # ``fault_iter`` would clamp the replay to an empty range and the
        # saved window would never actually replay.  Sync from the engine
        # here so disk-resume exercises the real replay path.  Cascade /
        # peer-pull entries come in via ``_on_iteration_end_recovery``
        # where ``_global_step`` is already driven by the live training
        # step; they don't carry an ``engine`` arg so this branch is a
        # no-op for them.
        if engine is not None and hasattr(engine, 'global_steps'):
            self._global_step = engine.global_steps
        self._fault_iter = self._global_step

        # Replay iterations = the bundle's captured iters (sparse-to-dense
        # conversion window) followed by catch-up iters up to ``_fault_iter``.
        # Feeding original iter numbers keeps the replay cursor, converter,
        # and schedule lookup in the same coordinate space.  Under whole-
        # job restart every rank is recovering, catch-up iters find no
        # logs and fall through to real p2p; under single-stage failure
        # paused pp peers ship logs covering the catch-up range.
        replay_iters = self._compute_replay_iters(sorted(per_iter_operator_states.keys()))
        self.converter.set_replay_iterations(replay_iters)

        # Cursor = first captured iteration so upstream-log lookups remap the
        # engine's post-load train_batches onto the original iter numbers the
        # logs are keyed by.
        if replay_iters:
            self._replay_iteration_cursor = replay_iters[0]
        persisted_sorted = sorted(per_iter_operator_states.keys())
        self._catch_up_boundary = persisted_sorted[-1] if persisted_sorted else None

        # Restore the FP16 loss-scaler's dynamic state to what the saving
        # rank had at bundle finalize time.  Without this, the replacement
        # rank starts with a factory-initial scale and makes different
        # overflow decisions than the paused peers made during training,
        # so Adam's ``step_count`` diverges and weights drift within FP16
        # precision for each mis-skipped step.
        self._restore_loss_scaler_state(metadata.get("loss_scaler_state"))
        # Restore engine scalars (global_steps, global_samples, LR +
        # compression scheduler state) alongside the loss scaler.  Both
        # describe the bundle's iteration state; the spare lands on the
        # paused peers' counters + scheduler step in one block.  ``engine``
        # kwarg is optional for backward compat with older callers (tests
        # that construct a coordinator directly without an engine); skip
        # silently when absent.
        self._restore_engine_scalars(metadata.get("engine_scalars"), engine)

        if self.upstream_logger is not None:
            self.upstream_logger.load_from_disk(load_dir, tag, rank=dist.get_rank())

        self.begin_recovery()

        # Replay setup happens per-iter in ``_on_iteration_end_recovery`` at
        # the END of each replay train_batch.  The first train_batch runs on
        # post-fault (zeroed) state and produces garbage, which the end-of-iter
        # setup then clobbers with the correct iter-1 state before iter-2's
        # forward runs — matching the legacy single-iter recovery semantics
        # and keeping the full window's trajectory correct under ``w_sparse > 1``.
        #
        # Intercept frozen ops' ``nn.Linear`` forward to skip wgrad compute
        # on the backward pass — see ``_freeze_operator_params`` for why
        # this runs via a custom autograd Function rather than simply
        # flipping ``requires_grad=False``.
        self._freeze_operator_params(model)
        del optimizer  # replay uses ``self._optimizer`` via on_iteration_end hooks
        del model  # recovery enters via the engine's ``on_iteration_end`` with model refs
        return True

    def _load_snapshot_data(self, load_dir, tag):
        """Return ``(metadata, per_iter_operator_states)`` honoring the cache.

        ``per_iter_operator_states`` is ``{iteration: {op_name: state_dict}}``
        matching the shape ``SparseSnapshotEngine.load_from_disk`` produces.
        The cache path (populated at recovery entry by
        ``load_sparse_checkpoint`` or ``cascade_into_recovery``) feeds the
        same shape so downstream consumers see a uniform input regardless
        of whether data came off disk or in-memory ``_persisted_snapshots``.

        Falls back to a fresh disk read on a first-time call with a
        valid ``load_dir``; that result is then cached.  Returns
        ``(None, None)`` when neither cache nor disk has data.
        """
        if self._cached_snapshot_data is not None:
            return self._cached_snapshot_data
        if load_dir is None:
            return None, None
        rank = dist.get_rank() if dist.is_initialized() else 0
        metadata, operator_states = SparseSnapshotEngine.load_from_disk(load_dir, tag, rank=rank)
        if metadata is not None:
            self._cached_snapshot_data = (metadata, operator_states)
        return metadata, operator_states

    def load_sparse_from_peer(self, peer_rank, my_dp_rank_in_replication_group=None, model=None, engine=None):
        """Pull this rank's sparse snapshot from a surviving DP peer.

        Called on a replacement rank that came up with no local disk
        state.  The cluster manager tells us which peer ranks received
        our replicated shard (they are ranks ``(my_dp_rank + 1)..(+r)``
        on the replication ring, mapped to global ranks); we connect to
        one of them, send a request on the gloo replication group, and
        receive our shard back over the same channel.  Peer serves the
        request from its ``_received_snapshots[my_dp_rank]`` slot.

        After a successful pull, we initialise the converter from the
        pulled state and enter recovery — the same flow as
        ``load_sparse_checkpoint`` from disk.  Returns ``False`` if the
        peer had nothing for us (the caller can retry another peer).

        Args:
            peer_rank: Group-local rank on ``_replication_group`` of a
                surviving peer that holds our shard.  The peer is told
                externally (cluster manager) which replacement rank to
                serve; here we only identify the source.
            my_dp_rank_in_replication_group: The dp_rank we're taking
                over.  Defaults to this process's own ``_dp_rank`` —
                appropriate when the replacement slots into the failed
                rank's place at the same dp_rank index.
            model: The live model module.  Enables the same
                ``_freeze_operator_params`` pass that
                ``load_sparse_checkpoint`` does.
        """
        if self._replication_group is None:
            logger.warning("[MoEvement] load_sparse_from_peer called before set_topology; "
                           "no replication group available")
            return False

        sender_rank = my_dp_rank_in_replication_group
        if sender_rank is None:
            sender_rank = self._dp_rank

        # Handshake: ``(sender_rank, protocol_version)`` as a 2-int64
        # tensor.  ``sender_rank`` tells the peer which slot of
        # ``_received_snapshots`` to serve; ``protocol_version`` selects
        # the wire shape (bulk or streaming per-iter) so both sides
        # agree before any manifest flows.  An unknown version on the
        # server side causes it to send ``length=0`` and the caller
        # retries another peer.
        streaming = bool(getattr(self.config, 'streaming_recovery', False))
        protocol_version = (PEER_PULL_PROTOCOL_STREAMING if streaming else PEER_PULL_PROTOCOL_BULK)
        request = torch.tensor([sender_rank, protocol_version], dtype=torch.int64)
        # ``dist.send`` treats ``dst`` as global regardless of
        # ``group=``; translate from the group-local ``peer_rank`` the
        # API accepts.  Guard on ``dist.is_initialized()`` so mocked-
        # dist unit tests (sentinel group, patched ``send``) keep
        # working with their group-local expectations.
        if dist.is_initialized():
            peer_global = dist.get_global_rank(self._replication_group, peer_rank)
        else:
            peer_global = peer_rank
        dist.send(request, dst=peer_global, group=self._replication_group)

        if streaming:
            return self._load_sparse_from_peer_streaming(peer_rank=peer_rank,
                                                         sender_rank=sender_rank,
                                                         protocol_version=protocol_version,
                                                         model=model,
                                                         engine=engine)

        metadata, per_iter_operator_states = self.snapshot_engine.pull_snapshot_from_peer(
            peer_rank=peer_rank, group=self._replication_group, protocol_version=protocol_version)
        if metadata is None:
            logger.warning(f"[MoEvement] peer rank {peer_rank} had no shard for "
                           f"sender_rank={sender_rank}")
            return False

        # Feed ``_setup_replay_iter`` via the cache so each replay iteration's
        # progressive thawing reads from this pulled state rather than re-trying
        # a disk read that won't succeed — ``load_dir`` isn't set on the
        # peer-pull path.  Same motivation as ``cascade_into_recovery``.
        self._cached_snapshot_data = (metadata, per_iter_operator_states)

        # Fault iter comes from the serving peer's ``global_step`` at
        # serve time, threaded through the manifest.  Falls back to
        # this rank's own ``_global_step`` if the sender didn't attach
        # it (older bundles / tests that construct a manifest by hand).
        self._fault_iter = metadata.get("fault_iter", self._global_step)

        with trace_range("recovery/converter_init"):
            self.converter.initialize_from_snapshots(metadata, per_iter_operator_states, schedule=None)
        replay_iters = self._compute_replay_iters(sorted(per_iter_operator_states.keys()))
        self.converter.set_replay_iterations(replay_iters)
        if replay_iters:
            self._replay_iteration_cursor = replay_iters[0]
        persisted_sorted = sorted(per_iter_operator_states.keys())
        self._catch_up_boundary = persisted_sorted[-1] if persisted_sorted else None

        # Peer-pull manifest threads the serving rank's scaler state (captured
        # at its last finalize_window) through ``loss_scaler_state``.  Restore
        # to match the bundle's iteration so catch-up replay starts from the
        # same scaler state training had at bundle[persisted[-1]].
        self._restore_loss_scaler_state(metadata.get("loss_scaler_state"))
        # Engine scalars (global_steps, global_samples, LR + compression
        # scheduler state) ride the same manifest and restore immediately
        # after the scaler — both describe the bundle's iteration state.
        # ``engine=None`` is tolerated for older callers that only pass
        # ``model``; those callers stay on the manual
        # ``engine.global_steps = coord._fault_iter`` workaround.
        self._restore_engine_scalars(metadata.get("engine_scalars"), engine)

        self.begin_recovery()
        # Wrap frozen ops' nn.Linear forward — see ``load_sparse_checkpoint``
        # / ``_freeze_operator_params`` for why the zero-bubble-style
        # Function is used instead of flipping ``requires_grad``.
        self._freeze_operator_params(model)
        del model  # kept in signature for API parity
        logger.info(f"[MoEvement] Loaded sparse snapshot from peer rank {peer_rank} "
                    f"(sender_dp_rank={sender_rank})")
        return True

    def _load_sparse_from_peer_streaming(self, peer_rank, sender_rank, protocol_version, model, engine):
        """SD-O4 S2: streaming variant of ``load_sparse_from_peer``.

        Pull runs on a background thread; this method returns after
        iter 0 has been ingested and recovery state committed.  Iter
        K's state for K > 0 is drained on demand by
        ``_drain_streaming_iter``, which ``_setup_replay_iter`` calls
        at the top of its body so the replay loop blocks on the pull
        only if the pull hasn't kept up.

        Key deviation from bulk: ``replay_iters`` are synthesised from
        ``[window_start_iteration, _fault_iter]`` rather than from the
        sorted keys of the per-iter dict, because we don't have the
        full key set at first-iter time.  The assumption is that a
        persisted window covers a contiguous iter range — which is
        true under the existing snapshot flow (one snapshot per
        training iter within the window).
        """
        import queue as _queue_mod
        import threading as _threading_mod

        # SD-O4 S3: bound the queue so a fast pull on a slow replay
        # can't balloon CPU memory.  Cap covers iters between pull and
        # replay; converter state for already-drained iters is dropped
        # by ``_drop_replayed_iter`` so the dict size is also bounded.
        max_prefetched = max(1, int(getattr(self.config, 'max_prefetched_iters', 8)))
        q = _queue_mod.Queue(maxsize=max_prefetched)
        self._streaming_pull_exc = None

        def _pull_worker():
            try:
                self.snapshot_engine.pull_snapshot_from_peer(
                    peer_rank=peer_rank,
                    group=self._replication_group,
                    protocol_version=protocol_version,
                    on_iter_ready=lambda it, ops, active, bundle: q.put((it, ops, active, bundle)),
                )
            except BaseException as exc:  # noqa: BLE001 — propagate any failure to the consumer
                self._streaming_pull_exc = exc
            finally:
                q.put(None)  # end-of-stream sentinel

        pull_thread = _threading_mod.Thread(target=_pull_worker, name="moevement-pull-thread", daemon=True)
        pull_thread.start()

        # Block on iter 0.  Bundle-level fields ride iter 0's mini-manifest
        # so everything needed to commit to recovery state is here.
        first = q.get()
        if first is None:
            # Pull thread ended before producing any iter — either the
            # peer had nothing for us (length=0 reply) or the pull
            # raised.  Match bulk-path semantics: caller retries the
            # next peer on False; we re-raise real exceptions.
            pull_thread.join(timeout=5.0)
            if self._streaming_pull_exc is not None:
                raise self._streaming_pull_exc
            logger.warning(f"[MoEvement] peer rank {peer_rank} had no shard for "
                           f"sender_rank={sender_rank}")
            return False

        iteration, iter_op_states, iter_active, bundle_fields = first
        bundle_fields = bundle_fields or {}

        metadata = {
            "window_start_iteration": bundle_fields.get("window_start_iteration", -1),
            "per_iter_active": {
                iteration: iter_active
            },
        }
        if "fault_iter" in bundle_fields:
            metadata["fault_iter"] = bundle_fields["fault_iter"]
        if "loss_scaler_state" in bundle_fields:
            metadata["loss_scaler_state"] = bundle_fields["loss_scaler_state"]
        if "engine_scalars" in bundle_fields:
            metadata["engine_scalars"] = bundle_fields["engine_scalars"]

        # This dict grows on the main thread as later iters drain
        # through ``_drain_streaming_iter``.  Only the main thread
        # mutates it, so there's no race against the pull thread.
        per_iter_operator_states = {iteration: iter_op_states}

        with trace_range("recovery/converter_init"):
            self.converter.ingest_iteration(iteration, iter_op_states, iter_active=iter_active)

        self._cached_snapshot_data = (metadata, per_iter_operator_states)
        self._fault_iter = metadata.get("fault_iter", self._global_step)

        # Use the same ``_compute_replay_iters`` helper the cascade and
        # bulk paths use, fed by the donor's full persisted-iter list
        # (shipped as a bundle field on iter 0).  Without this, the
        # spare and its DP cascade peer derive different replay-iter
        # sets — N differs across recovering ranks — and paused
        # peers' ``2N+1`` handshake count desyncs from the slower
        # recovering rank, deadlocking the world.
        window_start = metadata["window_start_iteration"]
        if window_start < 0:
            window_start = iteration
        persisted_iters = bundle_fields.get("persisted_iters") or sorted(per_iter_operator_states.keys())
        replay_iters = self._compute_replay_iters(persisted_iters)
        self.converter.set_replay_iterations(replay_iters)
        if replay_iters:
            self._replay_iteration_cursor = replay_iters[0]
        # Catch-up boundary is the last persisted iter — that's where
        # bundle-state-driven replay ends and upstream-log-driven
        # catch-up replay starts.
        self._catch_up_boundary = persisted_iters[-1] if persisted_iters else None

        self._restore_loss_scaler_state(metadata.get("loss_scaler_state"))
        self._restore_engine_scalars(metadata.get("engine_scalars"), engine)

        # Stash pull machinery for later drain + join.  Also keep a
        # weak-ish reference to the model so late-arriving ops (ops
        # that first appear in iter > 0) can still be freeze-wrapped
        # by ``_drain_streaming_iter``.  The user's training loop
        # holds the strong reference; we just need to call
        # ``_freeze_operator_params(model)`` again if new FROZEN ops
        # appear.
        self._streaming_pull_queue = q
        self._streaming_pull_thread = pull_thread
        self._streaming_model_ref = model

        self.begin_recovery()
        self._freeze_operator_params(model)
        logger.info(f"[MoEvement] streaming peer-pull from rank {peer_rank} started; "
                    f"first iter {iteration} ingested, pull thread running in background "
                    f"(replay_iters={len(replay_iters)})")
        return True

    def _drain_streaming_iter(self, target_iter):
        """Drain the streaming pull queue until the converter holds ``target_iter``.

        Called at the top of ``_setup_replay_iter`` so the replay loop
        blocks the minimum necessary on the pull thread's progress.
        When target_iter is a catch-up iter past the end of the
        bundle, the pull stream ends without ever producing it —
        that's fine because catch-up replay runs train_batch on live
        weights, not converter state.

        Propagates exceptions from the pull thread on EOS.  Idempotent
        after the stream closes (subsequent calls are no-ops).
        """
        if self._streaming_pull_queue is None:
            return
        if (target_iter in self.converter._fp32_weights_per_iter
                or target_iter in self.converter._fp16_weights_per_iter
                or target_iter in self.converter._optimizer_states_per_iter):
            return

        _, per_iter_operator_states = self._cached_snapshot_data
        per_iter_active = self._cached_snapshot_data[0]["per_iter_active"]

        with trace_range("recovery/streaming_drain"):
            while True:
                item = self._streaming_pull_queue.get()
                if item is None:
                    if self._streaming_pull_thread is not None and self._streaming_pull_thread.is_alive():
                        self._streaming_pull_thread.join(timeout=5.0)
                    exc = self._streaming_pull_exc
                    self._streaming_pull_queue = None
                    self._streaming_pull_thread = None
                    if exc is not None:
                        raise exc
                    return
                iteration, iter_op_states, iter_active, _ = item
                prior_frozen = set(self.converter.get_frozen_operators())
                self.converter.ingest_iteration(iteration, iter_op_states, iter_active=iter_active)
                per_iter_operator_states[iteration] = iter_op_states
                per_iter_active[iteration] = iter_active

                # Any op that first appeared in this later iter landed
                # in ``_operator_states`` as FROZEN.  Freeze-wrap its
                # nn.Linear forward in the model if we have a ref.
                # Ops that were already registered (present in an
                # earlier iter) were wrapped at load time — skipped by
                # ``_freeze_operator_params``'s idempotency guard.
                if self._streaming_model_ref is not None:
                    newly_frozen = set(self.converter.get_frozen_operators()) - prior_frozen
                    if newly_frozen:
                        self._freeze_operator_params(self._streaming_model_ref)

                if iteration == target_iter:
                    return

    def serve_sparse_snapshot_to_peer(self, requester_rank):
        """Serve a replacement rank's pull request for a specific sender's shard.

        Called on a surviving rank after the cluster manager tells a
        replacement that this rank holds the shard it needs.  We block
        on a single int64 recv (the replacement's "give me sender
        dp_rank=S" request), then forward the matching slice of
        ``_received_snapshots[S]`` on the same gloo replication group.
        """
        if self._replication_group is None:
            logger.warning("[MoEvement] serve_sparse_snapshot_to_peer called before set_topology")
            return
        # Handshake: 2-int64 ``(sender_rank, protocol_version)``.  An
        # unsupported version gets a single ``length=0`` reply so the
        # replacement retries another peer rather than hanging on a
        # protocol shape we don't know how to speak.
        request = torch.zeros(2, dtype=torch.int64)
        if dist.is_initialized():
            requester_global = dist.get_global_rank(self._replication_group, requester_rank)
        else:
            requester_global = requester_rank
        dist.recv(request, src=requester_global, group=self._replication_group)
        sender_rank = int(request[0].item())
        protocol_version = int(request[1].item())
        if protocol_version not in PEER_PULL_PROTOCOL_SUPPORTED:
            logger.warning(f"[MoEvement] unsupported peer-pull protocol_version={protocol_version} "
                           f"from requester {requester_rank}; sending empty manifest")
            dist.send(torch.tensor([0], dtype=torch.int64), dst=requester_global, group=self._replication_group)
            return
        self.snapshot_engine.serve_peer_pull_request(
            requester_rank=requester_rank,
            sender_dp_rank=sender_rank,
            group=self._replication_group,
            fault_iter=self._global_step,
            loss_scaler_state=self.snapshot_engine._persisted_loss_scaler_state,
            engine_scalars=self._pending_engine_scalars,
            protocol_version=protocol_version,
        )

    def begin_recovery(self, failed_stage_id=None, dp_group_rank=None):
        """Begin localized recovery after a failure.

        Only the affected data-parallel group rolls back. If upstream logging
        is enabled, recovery is confined to the failed pipeline stage.

        Args:
            failed_stage_id: The pipeline stage that failed (None for non-pipeline).
            dp_group_rank: The data-parallel group that needs recovery.
        """
        self._recovering = True
        self._pp_log_transfer_done = False
        self._disable_mp_collectives_during_replay()

        # Upstream-log retention must span the persisted window (``w_sparse``
        # iters the sparse-to-dense conversion replays) plus every iter past
        # the last finalized window that the catch-up branch will replay.
        # The worst case is a fault right before the next ``finalize_window``:
        # the still-in-flight window has ``w_sparse`` iters of uncommitted
        # logs past the previous persisted window, so retention must cover
        # at minimum ``2 * w_sparse`` iters for catch-up to have sources.
        # Fire here rather than at config-parse so the assertion sees a
        # scheduler that has actually populated ``w_sparse``.
        if self.upstream_logger is not None:
            required_retention = 2 * max(1, self.scheduler.w_sparse)
            retained = self.upstream_logger._max_window
            assert retained >= required_retention, (
                f"[MoEvement] upstream_logger.max_window_iterations={retained} "
                f"< 2 * w_sparse = {required_retention}; catch-up replay needs "
                f"logs covering the persisted window plus the in-flight iters. "
                f"Increase max_window_iterations when constructing UpstreamLogger.")

        if self.upstream_logger is not None and failed_stage_id is not None:
            logger.info(f"[MoEvement] Beginning localized recovery for stage {failed_stage_id}, "
                        f"DP group {dp_group_rank}")
        else:
            logger.info(f"[MoEvement] Beginning recovery (DP group {dp_group_rank})")

    def _disable_mp_collectives_during_replay(self):
        """Replace the optimizer's PP-column ``_model_parallel_all_reduce`` with a no-op.

        ``ds_model_proc_group`` under PP=2 DP=2 spans each per-dp PP column
        ({0,2} for dp=0, {1,3} for dp=1), so the ZeRO optimizer's overflow +
        gradient-norm sync calls out across ranks that don't share a DP group
        with the recovering rank.  Under localized failure those cross-column
        peers are paused in ``_wait_for_recovery`` and don't participate in
        the collective, leaving it queued on the replaying rank's default
        stream — the subsequent ``.item()`` then blocks.  Stripping the MP
        all_reduce during replay is safe: the DP-local all_reduce on
        ``dp_process_group`` still fires one line earlier, so overflow /
        gradient-norm decisions stay consistent within each DP group.  The
        cross-PP-stage sync is only a no-op because the paused stage isn't
        producing gradients anyway.
        """
        opt = getattr(self, "_optimizer", None)
        if opt is None or not hasattr(opt, "_model_parallel_all_reduce"):
            return
        if hasattr(opt, "_moevement_original_mp_all_reduce"):
            # Idempotent: recovery may re-enter via cascade on top of an existing
            # replay (rare but possible); don't double-wrap.
            return
        opt._moevement_original_mp_all_reduce = opt._model_parallel_all_reduce

        def _noop_mp_all_reduce(tensor, op):
            # ``tensor`` and ``op`` are captured for the signature the optimizer
            # calls with; the whole point is to NOT issue any collective here.
            del tensor, op
            return None

        opt._model_parallel_all_reduce = _noop_mp_all_reduce

    def _restore_mp_collectives_after_replay(self):
        """Undo ``_disable_mp_collectives_during_replay`` at ``end_recovery``."""
        opt = getattr(self, "_optimizer", None)
        if opt is None or not hasattr(opt, "_moevement_original_mp_all_reduce"):
            return
        opt._model_parallel_all_reduce = opt._moevement_original_mp_all_reduce
        del opt._moevement_original_mp_all_reduce

    def end_recovery(self):
        """Mark recovery as complete."""
        self._recovering = False
        self._pp_log_transfer_done = False
        self._fault_iter = None
        self._catch_up_boundary = None
        self._restore_mp_collectives_after_replay()
        self._replay_iteration_cursor = None
        # Thaw any params that did not get activated mid-replay (shouldn't
        # happen under normal flow but keeps state clean against partial
        # aborts).
        for op_name in list(self._frozen_param_backup.keys()):
            self._thaw_operator_params(op_name, model=None)
        self.converter.clear()
        # Logs shipped in by neighbour stages were only useful for the
        # just-finished replay.  Drop them so they don't linger for the
        # entire rest of the job (potentially hundreds of MB of pinned
        # activations and gradients per stage boundary).
        if self.upstream_logger is not None:
            self.upstream_logger._received_logs.clear()
        # Drop the cached snapshot-load result — its tensors (either
        # cloned in the cascade path or freshly allocated by
        # ``load_bundle`` on the disk path) are no longer needed; a
        # fresh recovery will repopulate the cache.
        self._cached_snapshot_data = None
        # Streaming pull teardown: drain any iters the pull thread
        # produced past the last replayed one, then join the thread.
        # Normally the stream has already closed by this point because
        # ``_setup_replay_iter`` drained the last bundle iter before
        # the final replay step.
        if self._streaming_pull_queue is not None:
            while True:
                item = self._streaming_pull_queue.get()
                if item is None:
                    break
        if self._streaming_pull_thread is not None and self._streaming_pull_thread.is_alive():
            self._streaming_pull_thread.join(timeout=5.0)
        self._streaming_pull_queue = None
        self._streaming_pull_thread = None
        self._streaming_pull_exc = None
        self._streaming_model_ref = None
        # Forget the per-pp-column recovering-stage set so the next
        # recovery_barrier call starts from a clean slate.
        self._recovering_stages_in_my_pp = frozenset()
        self._reset_window_state_post_recovery()
        logger.info("[MoEvement] Recovery complete")

    def _reset_window_state_post_recovery(self):
        """Restart the sparse-snapshot window cleanly after recovery ends.

        Recovery discards whatever pre-fault iters lived in the in-flight
        ``_snapshots`` / ``_in_flight_snapshots`` dicts: the recovering
        rank never re-captures them, and every other rank must drop
        them too so the next ``finalize_window`` promotes the same
        (fresh) iter range on every peer.  Reset ``_window_step`` to 0
        so the first post-recovery iter starts a new window rather than
        landing mid-stride through a pre-fault one.

        Called from ``end_recovery`` (recovering and cascaded ranks) and
        from the pause branch of ``recovery_barrier`` (paused peers that
        never entered recovery but still saw the fault) so post-recovery
        state is cross-rank symmetric.
        """
        engine = self.snapshot_engine
        # Drain the snapshot stream before returning these buffers to the
        # pool: any in-flight D2H from on_iteration_end may still be writing
        # into them, and the pool will hand them straight back out to the
        # next acquire.  Sibling hot-path (UpstreamLogger.garbage_collect)
        # already follows this discipline.
        engine.synchronize()
        for snap in engine._snapshots.values():
            for flat in snap._flat_buffers:
                engine._pool.release(flat)
        engine._snapshots = OrderedDict()
        for snap in engine._in_flight_snapshots.values():
            for flat in snap._flat_buffers:
                engine._pool.release(flat)
        engine._in_flight_snapshots = OrderedDict()
        for gpu_buf in engine._pending_gpu_staging:
            engine._pool.release(gpu_buf)
        engine._pending_gpu_staging = []
        self._window_step = 0
        # Drop any ``exp_counts`` D2Hs buffered for the aborted window:
        # flushing them into the scheduler would attribute pre-fault
        # counts to post-recovery slots.
        self._pending_activation_counts.clear()
        # Recovery may re-attach the optimizer or change optim state
        # shape (e.g., loading a bundle with a different optimizer
        # class); drop cached optimizer-key canonical lists so a stale
        # entry can't mask a real change.
        self._param_optim_keys_cache.clear()

    def is_recovering(self):
        """True while the coordinator is replaying sparse-to-dense recovery.

        The pipe engine checks this on every ``train_batch`` to decide
        whether ``_exec_recv_activations`` / ``_exec_recv_grads`` should
        pull logged tensors from ``upstream_logger`` (recovery path) or
        block on real pipeline p2p (steady-state training path).  Flips
        True when any recovery entry point — ``load_sparse_checkpoint``,
        ``cascade_into_recovery``, or ``load_sparse_from_peer`` — fires,
        and back to False inside ``end_recovery`` once the replay cursor
        exhausts the bundle + catch-up iterations.
        """
        return self._recovering

    # ------------------------------------------------------------------
    # Topology registration
    # ------------------------------------------------------------------

    def set_topology(self, dp_group, dp_rank, device):
        """Register data-parallel topology for peer replication.

        Called by DeepSpeedEngine after creating the coordinator so that
        window-boundary replication knows which process group and device to use.
        Also re-invoked by ``rebuild_comm_groups`` after a spare-rank
        substitution — the pre-existing gloo mirror is torn down here so
        subsequent replication doesn't issue collectives on a communicator
        that still references the departed rank.

        Args:
            dp_group: The data-parallel process group (or None for single-rank).
            dp_rank: This rank's index within dp_group.
            device: Device string or torch.device for temporary GPU tensors
                    during broadcast (e.g. ``'cuda:0'``).
        """
        if self._replication_group is not None:
            _abort_or_destroy(self._replication_group, timeout_sec=self.config.comm_rebuild_timeout_sec)
            self._replication_group = None
        self._dp_group = dp_group
        self._dp_rank = dp_rank
        self._device = device
        self._replication_group = self._build_gloo_mirror(dp_group)

    def _build_gloo_mirror(self, base_group):
        """Create a gloo group mirroring ``base_group`` across the world.

        CPU-tensor collectives (peer replication, upstream log transfer)
        can't run on NCCL groups, and they also shouldn't race training
        collectives on the same communicator.  The solution is a
        dedicated gloo mirror per base group.

        Every rank in the world gathers its own layout on ``base_group``,
        then collectively calls ``new_group`` once per unique layout so
        the mirrors are materialised in a consistent order on all ranks.

        Returns ``None`` for single-rank jobs or when ``base_group`` is
        None — callers treat that as "skip the gloo-backed path".
        """
        if base_group is None:
            return None
        try:
            world_size = dist.get_world_size()
        except Exception:
            return None
        if world_size <= 1:
            return None

        my_layout = tuple(sorted(dist.get_all_ranks_from_group(base_group)))
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_layout)

        unique_layouts = sorted({tuple(r) for r in gathered if r is not None})
        my_rank = dist.get_rank()
        my_group = None
        # ``new_group`` must be called by every rank in the world in the same
        # order, even if this rank isn't a member of the layout being built.
        for layout in unique_layouts:
            group = dist.new_group(ranks=list(layout), backend='gloo')
            if my_rank in layout:
                my_group = group
        return my_group

    def set_pipeline_topology(self, pp_group, stage_id, num_stages, stage_to_global_fn):
        """Register pipeline-parallel topology for localized recovery.

        Called by PipelineEngine after it has set up ``self.grid``.

        Args:
            pp_group: The pipeline-parallel process group.
            stage_id: This rank's pipeline stage index.
            num_stages: Total number of pipeline stages.  Must be > 1 —
                see ``ValueError`` below.
            stage_to_global_fn: Callable ``(stage_id) -> global_rank`` mapping
                                 a stage index to the global rank in the same
                                 DP group (``grid.stage_to_global``).

        Raises:
            ValueError: when ``num_stages <= 1``.  The engine-level
                ``isinstance(self, PipelineEngine)`` check catches the
                "no pipeline at all" case; this one catches "pipeline
                engine is used but configured with a single stage",
                which is equally incompatible with MoEvement's replay.
        """
        if num_stages is not None and num_stages <= 1:
            raise ValueError(f"MoEvement requires pipeline parallelism with more than 1 stage; "
                             f"got num_stages={num_stages}.  Increase the pipeline depth in "
                             f"your DeepSpeed pipeline config or disable moevement.")
        if self._pp_replication_group is not None:
            _abort_or_destroy(self._pp_replication_group, timeout_sec=self.config.comm_rebuild_timeout_sec)
            self._pp_replication_group = None
        self._pp_group = pp_group
        self._stage_id = stage_id
        self._num_stages = num_stages
        self._stage_to_global_fn = stage_to_global_fn
        self._pp_replication_group = self._build_gloo_mirror(pp_group)

    # ------------------------------------------------------------------
    # Communication-group rebuild (spare-rank substitution)
    # ------------------------------------------------------------------

    def _quiesce_pending_replication_and_persist(self, timeout_sec):
        """Drain in-flight peer replication + disk-persist before comm teardown.

        Any op queued on the existing gloo mirrors must be drained before we
        abort the underlying communicators — otherwise the worker threads
        hold references to mid-flight buffers and the abort races with the
        send loop.  Two moving parts:

        - ``_replication_futures``: peer-send jobs submitted at recent
          window boundaries.  Any still in flight get drained under a
          shared time bound.  Timeout → flag the executor broken (same
          state the steady-state path uses when gloo wedges) and stop
          waiting on the rest, so the post-rebuild resubmit path knows
          to stay away.
        - ``_persist_worker`` queues on both snapshot engine and the
          upstream logger: the enqueued closures hold pinned buffers
          that the pool would otherwise reclaim mid-write on the next
          acquire.  ``flush_persist`` blocks until the queue drains.

        ``mark_busy``'d pinned buffers and ``_pending_gpu_staging`` entries
        are released by the normal done-callback path once the replication
        future completes; we don't force-release them here because doing so
        would racy against a still-running worker.

        Args:
            timeout_sec: Upper bound on the replication-wait step.  The
                persist flushes are unbounded but expected to complete in
                under a second (queue holds tens of entries at most).
        """
        while self._replication_futures:
            fut = self._replication_futures.popleft()
            try:
                fut.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                logger.error(f"[MoEvement] comm rebuild: replication job did not finish within "
                             f"{timeout_sec}s; marking executor broken and proceeding with abort")
                self._replication_broken = True
                break
            except Exception as exc:
                logger.error(f"[MoEvement] comm rebuild: replication job raised during quiesce: {exc}")

        self.snapshot_engine.flush_persist()
        if self.upstream_logger is not None:
            self.upstream_logger.flush_persist()

    def rebuild_comm_groups(self, new_dp_group, new_pp_group=None):
        """Tear down and rebuild MoEvement's gloo mirrors after a world-topology change.

        Called by the cluster/job manager after ``init_process_group`` has
        been re-invoked against a fresh world (e.g. a spare rank has joined
        to replace a failed peer).  The previous ``_replication_group`` and
        ``_pp_replication_group`` were mirrors of the *old* dp/pp groups
        and hold dangling references to the departed rank; any collective
        issued on them after the topology change deadlocks.

        Scope is deliberately local: this method only rebuilds the gloo
        mirrors MoEvement owns.  Rebuilding the underlying DeepSpeed dp/pp
        groups, expert-parallel groups, and ZeRO ``_hp_mapping`` refs is
        caller-driven (Layer 2 of the design — not implemented here).  The
        caller must pass the freshly-built ``new_dp_group`` / ``new_pp_group``
        so this method can build mirrors against them.

        Must be invoked by every rank in lockstep — every rank participates
        in the ``all_gather_object`` / ``new_group`` calls inside
        ``_build_gloo_mirror``.

        Assumes 1:1 spare substitution: the spare takes the departed
        rank's exact number, so ``self._dp_rank`` / ``self._stage_id``
        remain valid.  k-of-n rebuilds would need explicit new
        coordinates and are deferred (design §4).

        Args:
            new_dp_group: The newly-built data-parallel process group
                against the fresh world.  May be ``None`` for single-rank
                jobs (in which case the mirror stays ``None``).
            new_pp_group: The newly-built pipeline-parallel process group,
                if pipeline parallelism was configured.  Required when this
                coordinator previously had a pipeline mirror — passing
                ``None`` in that case raises ``ValueError`` rather than
                silently leaking the old mirror.

        Raises:
            ValueError: when a pipeline mirror exists but ``new_pp_group``
                is ``None`` — ambiguous between "caller forgot" and
                "caller intentionally disabled PP mid-rebuild" (the
                latter is out of scope, see design §4).
        """
        if self._pp_replication_group is not None and new_pp_group is None:
            raise ValueError("rebuild_comm_groups: coordinator has an existing pipeline mirror; "
                             "new_pp_group is required to rebuild it.  Pass the refreshed pp group, "
                             "or explicitly tear down the pp mirror via set_pipeline_topology(None, ...) "
                             "first if PP is being disabled for this rebuild.")
        timeout_sec = self.config.comm_rebuild_timeout_sec
        self._quiesce_pending_replication_and_persist(timeout_sec)
        # ``_replication_broken`` was set by the old mirror's hung worker;
        # the new mirror deserves a fresh chance at replication.  Reset
        # the warn debounce too so a fresh mirror logs its own crossings
        # rather than inheriting the pre-rebuild silence window.
        self._replication_broken = False
        self._replication_warn_active = False
        # ``set_topology`` / ``set_pipeline_topology`` are idempotent:
        # they tear down the existing mirror before building the replacement.
        self.set_topology(new_dp_group, self._dp_rank, self._device)
        # Cold-starting spare's WORLD-collective order in
        # ``_configure_moevement_coordinator`` is:
        #   1. ``coord.set_topology`` → ``_build_gloo_mirror`` (DP)
        #   2. ``coord.initialize`` → ``_generate_schedule_world_aligned``
        #      (one ``all_reduce(MAX)``)
        #   3. later: ``coord.set_pipeline_topology`` → ``_build_gloo_mirror``
        #      (PP)
        # Survivors don't re-run ``initialize`` on rebuild, so step 2 is
        # missing — and without a matching WORLD collective here, the
        # spare and survivors disagree on WORLD-collective sequence
        # position by the time they hit step 3 (``_build_gloo_mirror``'s
        # ``all_gather_object``), which then deadlocks.  Re-issue the
        # all_reduce in the same slot to keep both paths in lockstep.
        self._sync_w_sparse_world_max()
        if new_pp_group is not None:
            self.set_pipeline_topology(new_pp_group, self._stage_id, self._num_stages, self._stage_to_global_fn)

    # ------------------------------------------------------------------
    # Peer replication
    # ------------------------------------------------------------------

    def _do_peer_replication(self, persisted_snapshots=None, pinned_release_fn=None, d2h_event=None):
        """Replicate persisted snapshots to ``r`` DP peers at each window boundary.

        A no-op when running single-rank, when no dp_group has been set, or
        when the configured ``replication_factor`` is zero.  Called by the
        primary (dp_rank == 0) and the first ``r`` peers; other ranks skip it.

        ``persisted_snapshots`` is the dict captured on the training thread
        at submit time.  The hot path always passes it so the worker's
        view stays pinned to the window we're actually replicating, even
        when later window boundaries rotate the live attribute before
        this future runs.  The default (None) falls back to the live
        attribute and exists only for legacy direct-invocation callers
        (unit tests); production paths must supply it.

        ``pinned_release_fn`` is an optional closure (over the
        submit-time ``busy_flats`` list) that releases every pinned
        D2H flat back to the training thread's pool.  The snapshot
        engine invokes it once its clone phase completes so training
        can reuse those flats without waiting on the slow gloo ship.
        Safe to double-invoke: ``release_busy`` is idempotent when the
        done-callback fires later.

        ``d2h_event`` is the side-stream event recorded at submit time.
        The worker calls ``event.synchronize()`` on its own thread before
        reading pinned CPU buffers — replaces the training-thread
        ``snapshot_engine.synchronize()`` that used to gate the boundary,
        so the wire ship stays entirely off training's critical path.
        Honors the design invariant that nothing past D2H-to-CPU may
        block the next iter's optim.step.

        Sends run on ``_replication_group`` — a dedicated mirror of the DP
        group — so they don't race the training thread's ZeRO collectives
        on the primary DP communicator.
        """
        if self._replication_group is None:
            return
        dp_world_size = dist.get_world_size(group=self._replication_group)
        if dp_world_size <= 1:
            return
        if d2h_event is not None:
            d2h_event.synchronize()
        # When replication is disabled (factor=0) ``replicate_to_peers``
        # early-returns without touching the pinned snapshot pages.
        # Iterating those pages on this worker thread closes a single-host
        # PP=2 DP=2 perf gap: the next iter's ``cudaStreamSynchronize``
        # calls tail-block when no CPU thread reads the just-D2H'd pinned
        # pages first.  Mechanism is empirical — the read forces
        # something in the CUDA driver / NCCL proxy / kernel-pinned
        # bookkeeping to settle off the training thread's critical path.
        # f1+ already triggers the same touch through the clone phase in
        # ``replicate_to_peers``.
        if (self.config.replication_factor <= 0 and persisted_snapshots is not None):
            for _snap in persisted_snapshots.values():
                for _flat in _snap._flat_buffers:
                    _flat.sum()
        with trace_range("peer_replicate"):
            self.snapshot_engine.replicate_to_peers(
                self._replication_group,
                dp_rank=self._dp_rank,
                dp_world_size=dp_world_size,
                device=self._device,
                replication_factor=self.config.replication_factor,
                persisted_snapshots=persisted_snapshots,
                pinned_release_fn=pinned_release_fn,
            )

    def _mark_persisted_flats_busy(self):
        """Mark every persisted flat buffer busy for the replication window.

        Called on the training thread before submitting the replication
        future.  While the worker is sending from these buffers, any
        concurrent ``pool.release`` on them (e.g. from ``finalize_window``
        after we give up waiting on a hung worker) becomes a no-op, so
        the storage can't be reassigned mid-DMA.  Receivers have no
        persisted flats of their own, so this is a no-op on non-primary
        ranks.
        """
        flats = []
        for snap in self.snapshot_engine._persisted_snapshots.values():
            for flat in snap._flat_buffers:
                self.snapshot_engine._pool.mark_busy(flat)
                flats.append(flat)
        return flats

    def _release_replication_busy(self, flats):
        """Return replication buffers to the pool once the worker is done.

        Invoked via ``Future.add_done_callback`` — fires on success *or*
        a completed-with-exception future, but not when the worker is
        hung forever.  The hung case is precisely what we want: leaving
        the flats busy prevents a later ``finalize_window`` release from
        racing with sends that never finish.
        """
        for flat in flats:
            self.snapshot_engine._pool.release_busy(flat)

    def _on_replication_done(self, future):
        """Surface worker-thread exceptions now that training no longer awaits.

        The window-boundary path stopped calling ``future.result()`` so
        training can continue without blocking on gloo.  Without this
        callback a worker-raised exception would be swallowed by the
        ``ThreadPoolExecutor``; flagging ``_replication_broken`` mirrors
        the backpressure-timeout path and prevents further submits until
        a comm rebuild clears the flag.
        """
        exc = future.exception()
        if exc is not None:
            logger.error(f"[MoEvement] peer replication raised on worker thread: "
                         f"{type(exc).__name__}: {exc!r}")
            self._replication_broken = True

    # ------------------------------------------------------------------
    # Pipeline-parallel recovery log transfer
    # ------------------------------------------------------------------

    def send_recovery_logs_to(self, recovering_stage_id, iteration_range):
        """Send upstream logs to a recovering pipeline stage.

        Called on neighbour stages (S-1 for activations, S+1 for gradients)
        when they are informed that stage S needs recovery.  Must be paired
        with ``receive_recovery_logs`` on the recovering rank.

        Args:
            recovering_stage_id: Pipeline stage index of the recovering rank.
            iteration_range: Iterable of iteration numbers to transfer.
        """
        if self.upstream_logger is None or self._stage_to_global_fn is None:
            return
        # Stage S-1 holds activations it sent toward S (direction="activation").
        # Stage S+1 holds gradients it sent toward S (direction="gradient").
        direction = "activation" if self._stage_id < recovering_stage_id else "gradient"
        group = self._pp_replication_group
        # ``dist.send(dst=X, group=G)`` requires X to be the GLOBAL rank
        # and X must be part of G — passing a group-local ordinal or a
        # stage id fails with "global rank N not part of group" whenever
        # the stage id doesn't happen to coincide with a global rank that
        # is in the group (e.g. dp_rank != 0 columns where pp_group spans
        # non-0-starting ranks).  Always translate the recovering stage
        # id to the global rank in THIS pp column.
        target_rank = self._stage_to_global_fn(recovering_stage_id)
        self.upstream_logger.send_logs_to(
            target_rank=target_rank,
            stage_id=self._stage_id,
            direction_filter=direction,
            iteration_range=iteration_range,
            group=group,
        )

    def receive_recovery_logs(self, prev_stage_rank, next_stage_rank, iteration_range):
        """Receive activation and gradient logs from neighbouring pipeline stages.

        Blocks until both neighbours have sent their payloads.  The logs are
        stored internally and made available via ``get_replay_activations`` and
        ``get_replay_gradients``.

        Args:
            prev_stage_rank: Global rank of stage S-1 (sends activations).
            next_stage_rank: Global rank of stage S+1 (sends gradients).
            iteration_range: Iterable of iteration numbers being transferred
                             (must match what the neighbours pass to
                             ``send_recovery_logs_to``).
        """
        if self.upstream_logger is None:
            return
        group = self._pp_replication_group
        # ``dist.recv(src=X, group=G)`` requires X to be the GLOBAL rank
        # of the sender and X must be part of G — see the matching comment
        # in ``send_recovery_logs_to``.  The caller already computes the
        # correct global ranks for the adjacent stages, so pass them
        # through unchanged.
        # Activations come from the previous stage (which sent them to us).
        if prev_stage_rank is not None:
            self.upstream_logger.recv_logs_from(src_rank=prev_stage_rank, group=group)
        # Gradients come from the next stage (which sent them back to us).
        if next_stage_rank is not None:
            self.upstream_logger.recv_logs_from(src_rank=next_stage_rank, group=group)

    def recovery_barrier(self, model=None):
        """World-level handshake that drives the recovery cascade.

        Called at the top of every training iteration.  Under steady
        state, does one small ``all_gather`` (two int64 per rank) and
        returns ``"continue"``.  When any rank is recovering, it drives
        three extra responsibilities:

        1. **Cascade**: ranks in the same DP group as a recovering rank
           auto-enter recovery (loading their own in-memory persisted
           snapshot and replaying W_sparse iterations in lockstep).  The
           DP all-reduce at each optimizer step then averages aligned-
           iteration gradients rather than mixing a recovering rank's
           replay gradient with a healthy peer's current-iteration one.

        2. **Log transfer** within each affected pipeline column: the
           recovering stage receives activation/gradient logs from its
           pipeline neighbours for the W_sparse-iteration replay.

        3. **Pause**: ranks that aren't recovering and whose DP group
           isn't affected drop into a blocking wait-loop (staying inside
           this call) until every rank reports ``recovering=0``.  The
           wait loop participates in the same world handshake every
           tick so the recovering ranks' replay iterations pair up with
           a matching collective on the paused ranks.  After the pause
           exits, returns ``"abandon"`` so the caller skips the rest of
           its ``train_batch`` — the paused rank didn't run forward /
           backward and has nothing to aggregate or broadcast.

        No-op (single-rank early return) when distributed isn't set up.
        """
        # Starting a new iteration's barrier means the previous iter's
        # post-recovery exit (if any) is done — recovering ranks flipped
        # ``_recovering=False`` via ``end_recovery`` last time around,
        # paused peers saw the release handshake and left their wait
        # loop.  Clear the flag now so this iter's ``_aggregate_total_loss``
        # takes the normal path.
        self._post_recovery_exit = False

        if self._pp_group is None or self._stage_id is None:
            # Early-return is load-bearing for legitimate single-rank /
            # no-PP runs (every rank's ``_pp_group`` is None).  Catch the
            # asymmetric case — some ranks set, others not — explicitly so
            # the deadlock surfaces as a RuntimeError here instead of an
            # NCCL watchdog timeout 10 minutes into the next handshake.
            self._raise_if_pp_topology_world_asymmetric()
            return "continue"

        # Round 1: discover originally-recovering ranks and decide whether
        # to cascade.  We can't determine pp log-transfer targets from
        # round 1 alone because cascade-triggered ranks weren't yet
        # announcing ``recovering=1`` when this round ran.
        round1 = self._world_recovery_handshake()
        if not round1["any_recovering"]:
            return "continue"

        # Localized cascade: only DP peers of the failed rank rewind.  PP
        # peers pause and contribute via upstream-log shipment instead of
        # rolling back themselves — their stage's weights aren't lost, so
        # forcing them to replay would throw away work.  Transitive cascade
        # (rank 3 is neither a DP peer of rank 0 nor a PP peer, but its DP
        # peer rank 2 has a pp-column recovering rank and may itself cascade
        # once round 2 observes rank 2's post-cascade state) is handled by
        # running the cascade check only on rank-pairs with a *direct* DP
        # link — see the PP-peer pause branch below for indirect cases.
        if not self._recovering and round1["dp_group_has_recovering"]:
            self.cascade_into_recovery(model=model)

        # Round 2: re-collect status now that cascade targets have flipped
        # ``_recovering=True``.  Their pp neighbours need to see them as
        # recovering to ship logs; round 1 wouldn't have reported them.
        # In the no-cascade case round 2 mirrors round 1, so the extra
        # collective is negligible (small int64 all_gather).
        round2 = self._world_recovery_handshake()
        # Cache the effective recovering-stage set for the pipe engine's
        # send-side guard.  Computed from round 2 so cascade-triggered
        # stages are included; used by ``should_skip_pipeline_send``
        # throughout this iteration's pipeline schedule.
        self._recovering_stages_in_my_pp = frozenset(round2.get("recovering_stages_in_my_pp", ()))

        # Replay window — ship logs covering every iter the recovering rank
        # will replay, including catch-up iters past the last persisted window.
        # Conversion iters come from ``_persisted_snapshots`` (each DP peer
        # snapshots the same global window schedule, so its own ``_persisted``
        # yields the same iter set as the failed rank's).  Catch-up iters
        # extend forward from the last persisted iter up to the live peer's
        # current ``_global_step`` — that's the iter the paused DP peers
        # most recently completed, which also equals the recovering rank's
        # ``_fault_iter`` after the peer-pull manifest or cascade captured
        # it.  Shipping the extended range lets the recovering rank replay
        # past the window boundary using logged pp-peer activations/gradients,
        # reconstructing all training past the sparse checkpoint rather than
        # losing up to ``w_sparse - 1`` iters.
        #
        # Fall back to the rolling-window range when no window has persisted
        # yet (early training / no checkpoint) — nothing useful to ship in
        # that case, but an empty ``iter_range`` would let the live side
        # ship zero entries and deadlock the recovering side on a missing
        # log lookup anyway.
        persisted_iters = sorted({it for (it, _) in self.snapshot_engine._persisted_snapshots.keys()})
        if persisted_iters:
            iter_end = max(persisted_iters[-1], self._global_step) + 1
            iter_range = range(persisted_iters[0], iter_end)
        else:
            w_sparse = max(1, self.scheduler.w_sparse)
            iter_start = max(0, self._global_step - w_sparse)
            iter_range = range(iter_start, self._global_step)

        # Local pp-column log transfer.  Only runs when this pp column
        # contains a recovering rank AND the transfer hasn't already fired
        # this recovery session: paused live peers ship logs exactly once
        # and then stay in ``_wait_for_recovery``'s world-handshake loop,
        # so re-entering log transfer on subsequent replay iters would
        # hang the recovering side on a recv that nobody matches.
        if round2["pp_column_has_recovering"] and not self._pp_log_transfer_done:
            self._pp_log_transfer(round2["recovering_stages_in_my_pp"], iter_range)
            self._pp_log_transfer_done = True

        # If we're still not recovering at this point, we're paused —
        # block in the wait loop until the world handshake shows every
        # rank cleared back to ``recovering=0``.  Paused ranks keep
        # participating in the handshake collective on each loop
        # iteration so recovering ranks aren't starved for a peer.
        # Once ``_wait_for_recovery`` returns, signal ``"abandon"`` so
        # the caller skips ``_exec_schedule`` + ``_aggregate_total_loss``:
        # those would fire collectives (pp broadcast, DP all-reduce) in
        # groups the paused rank joins from a different starting point
        # than the ranks that just replayed, guaranteeing a mismatch.
        if not self._recovering:
            self._paused_for_recovery = True
            # Symmetric mp-disable: paused peers patch their optimizer's
            # MP all_reduce too, mirroring the spare + cascaded survivor
            # who get patched via ``begin_recovery``.  Without this, any
            # path that lets a paused peer reach ``optimizer.step``
            # (e.g. an early-release handshake that happens before the
            # spare has called ``end_recovery``) issues a real MP
            # all_reduce that the spare's no-op patch skips, wedging
            # the PP-column comm.  Idempotent: ``_disable_mp_collectives_
            # during_replay`` short-circuits if already patched.
            self._disable_mp_collectives_during_replay()
            self._wait_for_recovery()
            self._paused_for_recovery = False
            # Paused ranks don't pass through ``end_recovery``, so
            # leftover per-session state (``_pp_log_transfer_done`` from
            # the pp-log shipment and ``_recovering_stages_in_my_pp``
            # from round2) plus the in-progress window bookkeeping
            # (pre-fault ``_snapshots`` / ``_in_flight_snapshots`` and
            # ``_window_step``) must be scrubbed here to keep every
            # rank's post-recovery window state symmetric with the
            # recovering ranks.
            self._pp_log_transfer_done = False
            self._recovering_stages_in_my_pp = frozenset()
            self._reset_window_state_post_recovery()
            # Symmetric mp-restore mirrors ``end_recovery``'s call on
            # the recovering ranks — both halves of the patch installed
            # at recovery entry are torn down before returning to
            # normal training.
            self._restore_mp_collectives_after_replay()
            return "abandon"

        return "continue"

    def _raise_if_pp_topology_world_asymmetric(self):
        """Detect ``_pp_group is None`` set on a strict subset of the world.

        World-uniform None (legitimate no-PP run) returns silently;
        every rank skips the handshake together.  Asymmetric None is a
        bug — some ranks would enter the handshake's ``all_gather``
        while others returned early — so raise locally rather than
        wedge the world for a watchdog timeout.
        """
        if not dist.is_initialized():
            return
        has_pp = torch.tensor([0 if self._pp_group is None else 1], dtype=torch.int64, device=self._device)
        gathered = [torch.zeros_like(has_pp) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, has_pp)
        # ``torch.stack(...).cpu().tolist()`` drains all gathered tensors
        # in a single D2H + sync rather than ``world_size`` separate
        # ``g.item()`` calls (each forcing its own cudaStreamSynchronize).
        # Fires every iter via ``recovery_barrier``; the per-rank-loop
        # variant added ~world_size cuda syncs per iter on the hot path.
        gathered_cpu = torch.stack(gathered).cpu().tolist()
        if any(int(g[0]) for g in gathered_cpu):
            raise RuntimeError("recovery_barrier: PP topology asymmetrically set across world; "
                               "every rank must call set_pipeline_topology before any iter.")

    def _world_recovery_handshake(self):
        """One all_gather of (is_recovering, stage_id) across the world.

        Returns a dict describing the recovery state from this rank's
        perspective: whether anyone is recovering, whether our DP group
        contains a recovering peer (cascade trigger), whether our PP
        column contains a recovering rank (log-transfer trigger), and
        the list of recovering stages within our PP column (driving
        the per-stage log shipment).
        """
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()

        # Steady-state fast path: a 1-element ``all_reduce(MAX)`` of
        # ``is_recovering`` is sufficient to decide "anyone recovering?"
        # and the cascade / log-transfer logic below is only needed if
        # the answer is yes.  The full ``all_gather`` of
        # ``(is_recovering, stage_id)`` per rank forces a ``.cpu()`` D2H
        # sync that can serialize against side-stream snapshot work in
        # flight from the previous iter's ``on_iteration_end``.  Defer
        # the gather to the rare branch where we already know somebody
        # is recovering.  ``recovery_barrier`` runs every iter, so the
        # fast path lifts a per-iter D2H out of the steady-state hot
        # path.
        is_recovering_t = torch.tensor([1 if self._recovering else 0], dtype=torch.int64, device=self._device)
        dist.all_reduce(is_recovering_t, op=dist.ReduceOp.MAX)
        if int(is_recovering_t.item()) == 0:
            return {"any_recovering": False}

        # Recovery branch (rare): we now need per-rank announcements to
        # drive cascade / pp-log-transfer / pause decisions.
        announcement = torch.tensor([1 if self._recovering else 0, self._stage_id],
                                    dtype=torch.int64,
                                    device=self._device)
        gathered = [torch.zeros_like(announcement) for _ in range(world_size)]
        dist.all_gather(gathered, announcement)

        # Drain the whole gather to CPU in a single D2H + sync rather
        # than ``world_size`` per-rank ``g[0].item()`` calls (each its
        # own cudaStreamSynchronize).
        gathered_cpu = torch.stack(gathered).cpu().tolist()
        recovering_ranks = [i for i, row in enumerate(gathered_cpu) if int(row[0]) == 1]
        if not recovering_ranks:
            return {"any_recovering": False}

        # Which global ranks share our DP group / PP group?  Used to
        # decide cascade vs log-ship vs pause.
        try:
            dp_ranks = set(dist.get_all_ranks_from_group(self._dp_group)) \
                if self._dp_group is not None else {world_rank}
        except Exception:
            dp_ranks = {world_rank}
        try:
            pp_ranks = set(dist.get_all_ranks_from_group(self._pp_group)) \
                if self._pp_group is not None else {world_rank}
        except Exception:
            pp_ranks = {world_rank}

        dp_group_has_recovering = any(r in dp_ranks for r in recovering_ranks)
        pp_column_has_recovering = any(r in pp_ranks for r in recovering_ranks)

        # Recovering stage ids within OUR pp column — the set that
        # drives the pp-log-transfer logic (prev/next-neighbour ship).
        recovering_stages_in_my_pp = [int(gathered_cpu[r][1]) for r in recovering_ranks if r in pp_ranks]

        return {
            "any_recovering": True,
            "recovering_ranks": recovering_ranks,
            "dp_group_has_recovering": dp_group_has_recovering,
            "pp_column_has_recovering": pp_column_has_recovering,
            "recovering_stages_in_my_pp": sorted(set(recovering_stages_in_my_pp)),
        }

    def _pp_log_transfer(self, recovering_stages, iter_range):
        """Ship activation/gradient logs between recovering stages and their live neighbours.

        Called only when this pp column contains a recovering rank.
        Each recovering stage receives logs from its immediate live
        neighbours; pipeline-adjacent live stages send.  A contiguous
        block of K failed stages is supported as long as the stages
        bounding the block are alive — the middle failed stages get
        their inputs/gradients via normal pipeline p2p from the
        bordering stages' own replay outputs instead of from logs.
        """
        recovering_set = set(recovering_stages)
        for recovering_stage_id in sorted(recovering_stages):
            prev_stage = recovering_stage_id - 1
            if prev_stage < 0 or prev_stage in recovering_set:
                prev_stage = None
            next_stage = recovering_stage_id + 1
            if next_stage >= self._num_stages or next_stage in recovering_set:
                next_stage = None
            prev_rank = self._stage_to_global_fn(prev_stage) if prev_stage is not None else None
            next_rank = self._stage_to_global_fn(next_stage) if next_stage is not None else None

            if self._stage_id == recovering_stage_id:
                self.receive_recovery_logs(prev_rank, next_rank, iter_range)
                if self._replay_iteration_cursor is None and self.upstream_logger is not None:
                    received = self.upstream_logger._received_logs
                    if received:
                        self._replay_iteration_cursor = min(k[0] for k in received.keys())
            elif self._stage_id == prev_stage or self._stage_id == next_stage:
                self.send_recovery_logs_to(recovering_stage_id, iter_range)

    def _wait_for_recovery(self):
        """Block inside the world handshake until no rank reports recovering.

        Paused ranks stay here for the duration of the cascade.  They
        keep participating in the world ``all_gather`` so recovering
        ranks aren't starved for a collective peer; the blocking
        semantics of ``all_gather`` naturally pace this loop to the
        recovering ranks' iteration rate (one handshake per replay
        iter).  No explicit sleep is needed.
        """
        while True:
            info = self._world_recovery_handshake()
            if not info.get("any_recovering"):
                return

    def _release_paused_peers(self):
        """Final post-``end_recovery`` handshake that releases paused peers.

        Called exactly once, immediately after ``end_recovery()`` on the
        recovering rank's last replay iter.  Publishes ``_recovering=False``
        via one world ``all_gather``; each paused peer's next
        ``_wait_for_recovery`` loop iteration reads ``any_recovering=False``
        and returns.

        Handshake accounting (recovering side, N total replay iters,
        N >= 1): 2 from ``recovery_barrier`` round 1 + round 2, then
        2*(N-1) more from subsequent replay iters (each contributing 2),
        plus 1 from this release = 2N + 1 total.  Paused peer performs
        2 initial handshakes in its own ``recovery_barrier`` plus
        (2N - 1) ``_wait_for_recovery`` loop iterations, matching the
        2N + 1 count so the last loop tick pairs exactly with this call.
        For N == 0, the degenerate path fires 3 handshakes total on each
        side and the formula doesn't apply (round 1 + round 2 + this
        release on the recovering side; 2 initial + 1 wait-loop tick on
        the paused side).
        Also sets ``_post_recovery_exit`` so the current train_batch's
        ``_aggregate_total_loss`` takes the paused-peer-free short-circuit
        path — the pp broadcast otherwise hangs on the peers that just
        exited their wait loop and are about to return "abandon" from
        their own barrier.
        """
        self._post_recovery_exit = True
        if self._pp_group is None or self._stage_id is None:
            return
        self._world_recovery_handshake()

    def cascade_into_recovery(self, model=None):
        """Enter recovery using this rank's own in-memory snapshot.

        Triggered from the world handshake when a DP peer is recovering.
        We roll back to the last persisted window by re-initialising the
        converter against ``_persisted_snapshots`` (no disk or peer pull
        needed — this rank's own state is the right shard, and the last
        full window is already in memory from the most recent
        ``finalize_window``).  Replay then drives W_sparse iterations
        forward so the DP all-reduce at each step lines up with the
        recovering peer's replay.

        Raises ``RuntimeError`` when no persisted snapshot is available.
        This happens before the first window completes — MoEvement has
        nothing to rewind to, and continuing would deadlock (the
        recovering DP peer is stuck waiting on our participation in
        the stage-DP all-reduce, and we'd enter the pause loop
        indefinitely).  The cluster manager should observe the raise
        and trigger a full dense-checkpoint reload instead.
        """
        engine = self.snapshot_engine
        if not engine._persisted_snapshots:
            raise RuntimeError("[MoEvement] DP peer entered recovery but this rank has no persisted "
                               "snapshot to cascade from (likely a failure before the first W_sparse "
                               "window completed).  MoEvement recovery cannot proceed; the cluster "
                               "manager should reload the full dense checkpoint across all ranks.")

        # Build per-iter operator_states + per-iter active-flags from the
        # ``(iteration, name)``-keyed persisted dict — matches the shape
        # ``load_from_disk`` produces on the disk path so the converter
        # sees a uniform input regardless of which entry point fed it.
        #
        # Clone every tensor out of the snapshot's state_dict: the
        # replication future's done-callback can release-busy the
        # pool-managed flat buffers while we're still replaying, and a
        # subsequent ``pool.acquire`` would hand the same storage to a
        # new caller.  Clones decouple.
        per_iter_operator_states = {}
        per_iter_active = {}
        for (iteration, op_name), snap in engine._persisted_snapshots.items():
            iter_states = per_iter_operator_states.setdefault(iteration, {})
            iter_states[op_name] = {k: v.clone() for k, v in snap.state_dict.items()}
            per_iter_active.setdefault(iteration, {})[op_name] = bool(snap.is_active)
        metadata = {
            "window_start_iteration": engine._window_start_iteration,
            "per_iter_active": per_iter_active,
        }
        self._cached_snapshot_data = (metadata, per_iter_operator_states)

        # Fault iter = every DP peer's most recently completed global_step.
        # DP peers advance in lockstep pre-fault, so this rank's own
        # ``_global_step`` matches the failed peer's last completed iter
        # and is the target catch-up iter the replay must reconstruct.
        self._fault_iter = self._global_step

        with trace_range("recovery/converter_init"):
            self.converter.initialize_from_snapshots(metadata, per_iter_operator_states, schedule=None)
        replay_iters = self._compute_replay_iters(sorted(per_iter_operator_states.keys()))
        self.converter.set_replay_iterations(replay_iters)
        if replay_iters:
            self._replay_iteration_cursor = replay_iters[0]
        persisted_sorted = sorted(per_iter_operator_states.keys())
        self._catch_up_boundary = persisted_sorted[-1] if persisted_sorted else None

        # Restore loss-scaler to its state at the last finalize_window.
        # Cascade is a DP-group rollback: weights + optim state wind back
        # to bundle[persisted[-1]] so the recovering DP peer's replay
        # aligns iter-for-iter with this rank's.  The scaler needs the
        # same rollback — without it, the live scaler's fault-time state
        # persists through replay and makes different overflow decisions
        # than the ones captured in the bundle's Adam state.
        self._restore_loss_scaler_state(engine._persisted_loss_scaler_state)
        # Engine scalars (global_steps, global_samples, LR + compression
        # scheduler) are NOT rolled back on cascade: this surviving rank
        # already holds the correct fault-time values, and replay is a
        # no-op on those counters by design.  Rolling them back to
        # bundle-time would leave the rank stuck behind real time post-
        # recovery.

        self.begin_recovery()
        # Wrap frozen ops' nn.Linear forward — see ``load_sparse_checkpoint``
        # / ``_freeze_operator_params``.  ``model`` is threaded through
        # ``recovery_barrier`` by the pipe engine; single-rank unit
        # tests that invoke cascade directly pass ``model=None`` which
        # skips only the non-expert-sentinel path (real expert / gate
        # modules are still wrapped from ``self._operator_map``).
        self._freeze_operator_params(model)
        first = replay_iters[0] if replay_iters else None
        logger.info(f"[MoEvement] Cascaded into recovery from own in-memory snapshot "
                    f"spanning iters {first}..{replay_iters[-1] if replay_iters else None}")
        return True

    def _compute_replay_iters(self, persisted_iters):
        """Extend the persisted-iter list with catch-up iters up to ``_fault_iter``.

        The last persisted window's iters drive sparse-to-dense conversion:
        each iter's active-op FP32 snapshot restores that op's weights, and
        by the end of the window every operator is ACTIVE.  Catch-up iters
        cover the gap between the window's last iter and ``_fault_iter`` —
        the pp-peer-logged activations/gradients fill in the stage's inputs
        for these iters so the replay reproduces the pre-fault trajectory
        rather than stopping one window short and losing w_sparse iters of
        training.

        ``persisted_iters`` is assumed sorted; catch-up iters are appended
        in ascending order with no gap.  A fault that lands exactly at a
        window boundary has zero catch-up iters — ``_fault_iter`` equals
        the last persisted iter and the range is empty.

        The bundle is clamped to iters ``<= _fault_iter`` so that a
        ``_fault_iter`` inside an already-persisted window replays only
        to the fault point (not the window's end).  In practice
        ``_fault_iter >= persisted_iters[-1]`` because it's read from the
        engine's current ``global_steps`` which is always ≥ the last
        finalized window boundary, but the clamp is defensive — the
        invariant could silently break if ``_fault_iter`` capture moves
        (peer-pull threads it through a manifest, for example) and the
        over-replay would produce wrong weights past ``_fault_iter``.
        """
        if not persisted_iters:
            return []
        if self._fault_iter is None:
            return list(persisted_iters)
        bundle = [it for it in persisted_iters if it <= self._fault_iter]
        catch_up = list(range(persisted_iters[-1] + 1, self._fault_iter + 1))
        return bundle + catch_up

    def is_paused_for_recovery(self):
        """True while this rank is waiting inside ``recovery_barrier`` for peer recovery."""
        return self._paused_for_recovery

    def should_skip_pipeline_send(self, downstream_stage_id):
        """True iff the pipe engine should short-circuit a p2p send during replay.

        The send-side guard has two branches:

        1. Our downstream is also recovering (originally or via cascade):
           it will either pair with us via ``p2p.recv`` (if it has no log
           for this iter/mb, which is the case when we — its upstream — are
           also recovering and didn't ship logs to it) or drop our send
           on the floor (if it had logs from a live upstream).  Either way,
           sending is safe.  Do NOT skip — this is the mechanism that lets
           a contiguous block of adjacent failed stages chain replay through
           normal pipeline channels.

        2. Our downstream is paused (non-recovering rank in an affected pp
           column): it won't run the pipeline schedule at all, so no
           matching ``p2p.recv`` is ever posted.  Send would sit unmatched
           in the NCCL queue and eventually deadlock the next iteration.
           Skip.

        The ``_recovering_stages_in_my_pp`` frozenset, populated from
        round 2 of the recovery handshake, tells us which stages in our
        pp column are effectively recovering (original + cascade).
        """
        if not self._recovering:
            # Non-recovering ranks shouldn't be in the schedule at all
            # during recovery (they're paused), but if they are, they
            # should send normally.
            return False
        return downstream_stage_id not in self._recovering_stages_in_my_pp

    def get_replay_activations(self, iteration, micro_batch_id):
        """Return logged activation tensors for replay at the given step.

        During recovery, ``iteration`` is remapped through the replay cursor
        so the engine's monotonically-increasing ``global_steps`` indexes
        into the original iteration numbers stored in the logs.  Callers
        outside recovery (tests, direct lookups) keep their explicit key.

        Returns None if upstream logging is disabled or no entry is stored.
        """
        if self.upstream_logger is None:
            return None
        key_iter = self._replay_key_iteration(iteration)
        return self.upstream_logger.get_received_activation(key_iter, micro_batch_id)

    def get_replay_gradients(self, iteration, micro_batch_id):
        """Return logged gradient tensors for replay at the given step.

        Remaps ``iteration`` through the replay cursor during recovery — see
        :meth:`get_replay_activations` for the rationale.

        Returns None if upstream logging is disabled or no entry is stored.
        """
        if self.upstream_logger is None:
            return None
        key_iter = self._replay_key_iteration(iteration)
        return self.upstream_logger.get_received_gradient(key_iter, micro_batch_id)

    def _replay_key_iteration(self, requested_iter):
        """Map the caller's iteration to the original logged iteration.

        Returns ``requested_iter`` unchanged outside recovery or when the
        replay cursor has not been seeded (e.g., tests that pre-populate
        ``_received_logs`` and look up by original key directly).

        For catch-up replay tbs (cursor past the last persisted bundle iter),
        shift the lookup key down by one: bundle keys are post-increment
        ``global_steps=N`` (state at end of tb-N) while log keys are
        pre-increment (forward input during tb-(N+1)), so a catch-up tb
        replaying tb-i with cursor=i needs log key i-1.  Bundle-replay tbs
        don't take this branch — their forward output is overwritten by
        ``_setup_replay_iter`` at tb end, so the mismatched log lookup is
        harmless noise rather than a correctness bug.
        """
        if self._recovering and self._replay_iteration_cursor is not None:
            cursor = self._replay_iteration_cursor
            if self._catch_up_boundary is not None and cursor > self._catch_up_boundary:
                return cursor - 1
            return cursor
        return requested_iter

    def _capture_loss_scaler_state(self):
        """Snapshot the FP16 loss-scaler's dynamic state for the bundle.

        FP16 dynamic loss scaling maintains a scalar ``cur_scale`` that
        halves on gradient overflow and doubles every ``scale_window``
        overflow-free iters.  The scale trajectory depends on the
        gradient history; without preserving it across a fault, the
        recovering rank starts from the factory initial scale (e.g.
        ``2**16``) while the paused peers have advanced the scale
        dynamically during training.  Replay then makes different
        overflow decisions than training — some optimizer steps get
        skipped that weren't skipped originally, or vice versa — and
        Adam's ``step_count`` and weight trajectory diverge.

        The capture is a plain dict of the four dynamic fields
        (``cur_scale``, ``cur_iter``, ``last_overflow_iter``,
        ``cur_hysteresis``) plus a type discriminator so the restore
        can no-op on a static scaler.  Static scalers have a fixed
        ``cur_scale`` and no evolving state, so they don't need
        preservation.

        Returns ``None`` when this rank has no FP16 optimizer attached,
        e.g., BF16 or FP32 training.  Callers store whatever this
        method returns in the bundle metadata and pass it back to
        ``_restore_loss_scaler_state`` on recovery entry.
        """
        opt = getattr(self, "_optimizer", None)
        if opt is None:
            return None
        scaler = getattr(opt, "loss_scaler", None)
        if scaler is None:
            return None
        if not getattr(scaler, "dynamic", False):
            return {"dynamic": False, "cur_scale": float(scaler.cur_scale)}
        return {
            "dynamic": True,
            "cur_scale": float(scaler.cur_scale),
            "cur_iter": int(scaler.cur_iter),
            "last_overflow_iter": int(scaler.last_overflow_iter),
            "cur_hysteresis": int(scaler.cur_hysteresis),
        }

    def _restore_loss_scaler_state(self, state):
        """Apply a captured loss-scaler state from a bundle.

        Inverse of ``_capture_loss_scaler_state``.  A captured ``None``
        (BF16 / FP32 training, or older bundles that don't carry scaler
        state) is a no-op.  A static-scaler capture restores only
        ``cur_scale``; a dynamic capture restores all four fields.
        """
        if not state:
            return
        opt = getattr(self, "_optimizer", None)
        if opt is None:
            return
        scaler = getattr(opt, "loss_scaler", None)
        if scaler is None:
            return
        scaler.cur_scale = state["cur_scale"]
        if state.get("dynamic") and getattr(scaler, "dynamic", False):
            scaler.cur_iter = state["cur_iter"]
            scaler.last_overflow_iter = state["last_overflow_iter"]
            scaler.cur_hysteresis = state["cur_hysteresis"]

    @staticmethod
    def capture_engine_scalars(engine):
        """Build the per-iter engine-scalar dict the coordinator latches.

        Static method so the engine can call it without the coordinator
        holding a back-reference.  ``None`` scheduler fields are preserved
        (disk bundles for bf16/fp32 training or non-pipe engines often
        have no LR / compression schedulers, and the restore path
        tolerates that).
        """
        lr = getattr(engine, "lr_scheduler", None)
        lr_state = lr.state_dict() if lr is not None and hasattr(lr, "state_dict") else None
        comp = getattr(engine, "compression_scheduler", None)
        comp_state = comp.state_dict() if comp is not None and hasattr(comp, "state_dict") else None
        return {
            "global_steps": int(getattr(engine, "global_steps", 0)),
            "global_samples": int(getattr(engine, "global_samples", 0)),
            "lr_scheduler": lr_state,
            "compression_scheduler": comp_state,
        }

    def _restore_engine_scalars(self, state, engine):
        """Apply a captured engine-scalar dict to a live ``DeepSpeedEngine``.

        Called from ``load_sparse_from_peer`` / ``load_sparse_checkpoint`` /
        ``cascade_into_recovery``.  A captured ``None`` (older bundles, or
        callers with no engine handy) is a no-op.  Missing fields inside
        the dict are tolerated — the bundle may have been produced before
        one of the capture keys existed.

        LR scheduler and compression scheduler state is applied via
        ``load_state_dict`` on whatever scheduler the engine currently
        holds; if the engine's scheduler type has changed across restarts
        (reconfigured LR schedule, say), the caller is responsible for
        keeping shapes aligned — we don't validate.
        """
        if state is None or engine is None:
            return
        if "global_steps" in state:
            engine.global_steps = int(state["global_steps"])
        if "global_samples" in state:
            engine.global_samples = int(state["global_samples"])
        lr_state = state.get("lr_scheduler")
        lr = getattr(engine, "lr_scheduler", None)
        if lr_state is not None and lr is not None and hasattr(lr, "load_state_dict"):
            try:
                lr.load_state_dict(lr_state)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"[MoEvement] lr_scheduler.load_state_dict failed "
                               f"({type(lr).__name__}): {exc}")
        elif lr_state is not None and lr is None:
            logger.warning("[MoEvement] bundle carries lr_scheduler state but engine has none; skipping")
        comp_state = state.get("compression_scheduler")
        comp = getattr(engine, "compression_scheduler", None)
        if comp_state is not None and comp is not None and hasattr(comp, "load_state_dict"):
            try:
                comp.load_state_dict(comp_state)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"[MoEvement] compression_scheduler.load_state_dict failed: {exc}")
        elif comp_state is not None and comp is None:
            logger.warning("[MoEvement] bundle carries compression_scheduler state but engine has none; "
                           "skipping")

    def _capture_rng_state(self):
        """Snapshot torch CPU + accelerator RNG state at this moment.

        Returns an ``OrderedDict`` of ByteTensors so the snapshot
        engine's existing per-iter byte stream serializes it without any
        bundle-format change.  Keys are ``torch_cpu`` for the host RNG and
        ``torch_cuda.<dev_index>`` for each accelerator device's RNG.

        Captured at on_iteration_end (post-optim-step), so each iter's
        entry is the **post-iter-K** RNG state — i.e. the state that
        iter K+1's forward will start with.  ``_setup_replay_iter(K)``
        applies it before the next replay tb so dropout / stochastic-layer
        replay matches the fault-free trajectory bit-for-bit.

        Numpy and Python ``random`` are intentionally NOT captured:
        DeepSpeed's accelerator abstraction handles only torch RNG, and
        almost all DL stochasticity (Dropout, randperm, multinomial,
        bernoulli) goes through torch.  Models that consume numpy /
        python random in forward will still drift post-recovery — that's
        a known gap, not a regression.
        """
        state = OrderedDict()
        state["torch_cpu"] = torch.get_rng_state()
        accelerator = get_accelerator()
        if accelerator is None or accelerator.Stream is None:
            return state
        try:
            for i in range(accelerator.device_count()):
                state[f"torch_cuda.{i}"] = accelerator.get_rng_state(i)
        except (AttributeError, RuntimeError) as exc:
            # CPU-only test runners and accelerators without per-device
            # RNG throw ``AttributeError`` (missing ``get_rng_state``) or
            # ``RuntimeError`` (e.g. "Device count not available").  Log
            # at DEBUG so an unexpected miss on a production backend is
            # still discoverable — anything broader (OOM, CUDA driver
            # error, API rename) stays visible as an uncaught raise
            # rather than a silent missing-RNG-on-replay bug.
            logger.debug(f"[MoEvement] per-device RNG capture skipped: {exc}; "
                         f"replay on this bundle will advance CPU RNG only")
        return state

    def _restore_rng_state(self, state):
        """Apply a captured RNG state from the bundle.

        Inverse of ``_capture_rng_state``.  ``None`` / empty is a no-op
        (older bundles without per-iter RNG, or catch-up iters beyond the
        last persisted window).  Per-device sets stop at the first missing
        ``torch_cuda.<i>`` key so a bundle saved on a different device count
        loads cleanly on a smaller world (the missing devices keep their
        current RNG; replay's other guarantees still hold).
        """
        if not state:
            return
        cpu = state.get("torch_cpu")
        if cpu is not None:
            torch.set_rng_state(cpu)
        accelerator = get_accelerator()
        if accelerator is None or accelerator.Stream is None:
            return
        i = 0
        while True:
            key = f"torch_cuda.{i}"
            if key not in state:
                break
            try:
                accelerator.set_rng_state(state[key], i)
            except (AttributeError, RuntimeError) as exc:
                # Mirror ``_capture_rng_state``: tolerate per-device misses
                # (CPU-only runners, accelerators without per-device RNG)
                # at DEBUG so an unexpected production miss stays
                # discoverable.  ``continue`` (not ``break``) so a single
                # bad device doesn't silently drop every higher-indexed
                # device's RNG restore — the original ``break`` left those
                # devices on whatever RNG they happened to hold and replay
                # diverged silently from the captured trajectory.
                logger.debug(f"[MoEvement] per-device RNG restore skipped for device {i}: {exc}")
            i += 1

    def is_conversion_complete(self):
        """Check if sparse-to-dense conversion is complete."""
        return self.converter.is_conversion_complete()

    def should_skip_weight_grad(self, operator_name):
        """Check if weight gradient should be skipped during conversion.

        Args:
            operator_name: Name of the operator.

        Returns:
            True if the operator is frozen and should skip weight gradients.
        """
        if not self._recovering:
            return False
        return self.converter.should_skip_weight_grad(operator_name)

    def should_skip_optimizer_step(self, operator_name):
        """Check if optimizer step should be skipped during conversion.

        Args:
            operator_name: Name of the operator.

        Returns:
            True if the operator is frozen and should skip optimizer step.
        """
        if not self._recovering:
            return False
        return self.converter.should_skip_optimizer_step(operator_name)

    def _active_non_expert_params(self, model, op_name):
        """Return ``(params_dict, fragment_info)`` for non-expert FP32 master weights.

        Under ZeRO-1/2 / bf16_optimizer the FP32 copy lives inside the
        flat partition as a per-rank fragment — return that fragment
        directly and record ``{full_shape, fragment_numel}`` in
        ``fragment_info`` so the bundle writer + restore path can
        route through the fragment-direct copy.  Without a ZeRO wrapper
        (``_hp_mapping`` absent), ``param.data`` is already the full
        replicated tensor; ``fragment_info`` stays empty.
        """
        return self._collect_active_params(self._iter_op_params(op_name, model))

    def _active_module_params(self, module):
        """ZeRO-aware ``(params_dict, fragment_info)`` for one operator module."""
        return self._collect_active_params(module.named_parameters())

    @staticmethod
    def _collect_active_params(name_param_iter):
        """Return ``({name: tensor}, fragment_info)`` for one operator's FP32 master weights.

        ZeRO-1/2 / bf16_optimizer partition the FP32 master (``_hp_mapping.hp_fragment``)
        across the DP group even though the live LP buffer is replicated.
        Pre-FS1 the bundle stored the full-shape FP32 master by reassembling
        it via ``safe_get_full_fp32_param``'s DP all-reduce; that collective
        dominated snap_active CPU and round-tripped redundant data through
        every peer's pinned memory.  FS1 captures only this rank's local
        fragment — ``fragment_info[name]`` carries the full shape + fragment
        numel so the restore path can route directly into
        ``_hp_mapping.hp_fragment`` without the narrow-from-full-shape path.

        Non-ZeRO fallback: no ``_hp_mapping`` → the param's ``.data`` IS
        the full-shape FP32 master (e.g., dense training in tests); emit
        it directly and leave ``fragment_info`` empty so the restore routes
        through the standard ``param.data.copy_`` path.
        """
        params_dict = {}
        fragment_info = {}
        for name, p in name_param_iter:
            mapping = getattr(p, '_hp_mapping', None)
            if mapping is not None and getattr(mapping, 'hp_fragment', None) is not None:
                params_dict[name] = mapping.hp_fragment.data
                fragment_info[name] = {
                    "full_shape": list(p.shape),
                    "fragment_numel": int(mapping.lp_fragment_address.numel),
                }
            else:
                params_dict[name] = p.data
        return params_dict, fragment_info

    @staticmethod
    def _full_fp32_view(param):
        """Return the full-shape FP32 view of ``param``.

        ZeRO-1/2 / bf16_optimizer attach ``_hp_mapping`` (initially
        ``None``) and ``get_full_hp_param`` to every lp_param in the
        optimizer at init time — regardless of whether this rank owns
        a fragment.  The method internally issues an unconditional DP
        all-reduce; peers without a fragment must still call it,
        contributing zeros, so the collective pairs up across the DP
        group.

        Discrimination here uses ``hasattr(param, '_hp_mapping')``
        rather than ``_hp_mapping is not None``.  The truthy check
        tests *this rank's fragment ownership* — gating on it silently
        skips the all_reduce on non-owning peers, and the owning peers
        then wait forever on an NCCL collective that never pairs.
        Presence of the attribute is the right "ZeRO-enrolled"
        predicate because it's set cross-rank uniformly by
        ``link_hp_params`` regardless of ownership.
        """
        if hasattr(param, '_hp_mapping'):
            gathered = safe_get_full_fp32_param(param)
            if gathered is not None:
                return gathered
        return param.data

    def _get_non_expert_optimizer_state(self, model, optimizer):
        """``(state_dict, fragment_info)`` for non-expert optimizer state."""
        state_dict = {}
        fragment_info = {}
        for name, param in model.named_parameters():
            if self._is_non_expert_param(param):
                sub_state, sub_frag = self._collect_param_optim_state(name, param, optimizer)
                state_dict.update(sub_state)
                fragment_info.update(sub_frag)
        return state_dict, fragment_info

    def _get_module_optimizer_state(self, module, optimizer):
        """``(state_dict, fragment_info)`` for a single module's optimizer state."""
        state_dict = {}
        fragment_info = {}
        for name, param in module.named_parameters():
            sub_state, sub_frag = self._collect_param_optim_state(name, param, optimizer)
            state_dict.update(sub_state)
            fragment_info.update(sub_frag)
        return state_dict, fragment_info

    def _collect_param_optim_state(self, name_prefix, param, optimizer):
        """Return ``({name_prefix.key: tensor}, fragment_info)`` for one param's optimizer state.

        ZeRO-1/2 / bf16_optimizer path: emit each rank's LOCAL
        ``mapping.optim_fragment[key]`` slice with fragment metadata
        (``full_shape`` + ``fragment_numel``).  No DP all-reduce — the
        old path reassembled via ``safe_get_full_optimizer_state`` so
        every rank held a redundant full-shape copy.  Fragment-direct
        capture cuts snap_active CPU and shrinks replication payload
        ``dp_world×``.

        Keyset symmetry required a cross-rank ``all_gather_object`` on
        the first call for each param, so every rank iterates the
        canonical key order even when a rank owns no fragment.  The
        result is cached on ``self._param_optim_keys_cache`` keyed by
        ``id(param)`` and reused on subsequent calls (AdamW-family
        keysets are stable across a run), eliminating the per-iter
        collective for already-seen params.  Ranks that don't own a
        fragment contribute empty local_keys and skip the per-key
        emit — the receiver-side code tolerates asymmetric entry sets.

        Without ``_hp_mapping`` (plain PyTorch / ZeRO-0) fall back to
        the vanilla ``optimizer.state[param]`` dict; fragment_info
        stays empty for those entries.

        Returns an empty ``(dict, dict)`` when the optimizer hasn't
        populated state yet (pre-first-step).
        """
        out = {}
        fragment_info = {}
        if hasattr(param, '_hp_mapping'):
            mapping = getattr(param, '_hp_mapping', None)

            cache_key = id(param)
            cached_keys = self._param_optim_keys_cache.get(cache_key)
            if cached_keys is not None:
                # Cache hit — skip the collective.  Only populated
                # canonicals are cached (see the conditional on
                # ``all_keys`` below) so a hit always carries a
                # non-empty Adam keyset.  Cross-rank symmetry is
                # preserved because the cache-hit/miss decision is
                # driven by the post-collective ``all_keys`` value,
                # which is identical on every DP-group rank.
                all_keys = cached_keys
            else:
                local_keys = []
                if mapping is not None and getattr(mapping, 'optim_fragment', None) is not None:
                    local_keys = list(mapping.get_optim_state_keys())

                # Canonical key order via all_gather across the DP
                # group so every rank iterates the same set even when
                # a rank owns no fragment.  Don't remove this — it's
                # the "every-rank symmetric collective" guarantee that
                # keeps replication manifest iteration deadlock-free
                # (see P6 deferral note).  Fires once per param
                # lifetime (post-first-step); cached below.
                dp_group = getattr(param, '_dp_group', None)
                dp_world = dist.get_world_size(group=dp_group) if dp_group is not None else 1
                if dp_world > 1:
                    gathered = [None] * dp_world
                    dist.all_gather_object(gathered, local_keys, group=dp_group)
                    canonical = []
                    seen = set()
                    for peer_keys in gathered:
                        if peer_keys:
                            for k in peer_keys:
                                if k not in seen:
                                    seen.add(k)
                                    canonical.append(k)
                            if canonical:
                                break
                    all_keys = canonical
                else:
                    all_keys = local_keys

                # Cache only once a populated canonical is observed.
                # Gating on ``_global_step >= 1`` was too eager: an
                # iter-0 fp16 overflow skips the optimizer step, leaves
                # ``optim_fragment`` empty, and would lock the cache to
                # ``[]`` for the param's lifetime — every subsequent
                # snapshot would short-circuit at the empty-key check
                # and silently drop optimizer.* entries.  ``all_keys``
                # is symmetric across the DP group (built from the
                # all_gather above), so caching only when it's non-empty
                # is still a lockstep decision — every rank either all
                # cache or all re-run the collective on the next iter.
                # Adam keysets are stable once populated, so re-running
                # the collective until that point is bounded.
                if all_keys:
                    self._param_optim_keys_cache[cache_key] = all_keys

            if not all_keys:
                return out, fragment_info

            for key in all_keys:
                assert '.' not in key, (f"optimizer state key {key!r} contains '.'; "
                                        f"restore uses rsplit('.', 1) assuming dotless keys")
                # Fragment-direct read: no all-reduce.  Ranks that don't
                # own the fragment for this param skip the emit; the
                # receiver-side (restore / peer-pull) handles asymmetric
                # entry sets via the bundle manifest.
                if mapping is None or mapping.optim_fragment is None:
                    continue
                if key not in mapping.optim_fragment:
                    continue
                frag = mapping.optim_fragment[key]
                composite = f"{name_prefix}.{key}"
                out[composite] = frag
                fragment_info[composite] = {
                    "full_shape": list(param.shape),
                    "fragment_numel": int(mapping.lp_fragment_address.numel),
                }
        elif optimizer is not None and hasattr(optimizer, 'state'):
            entries = optimizer.state.get(param, {})
            for key, val in entries.items():
                if isinstance(val, torch.Tensor):
                    assert '.' not in key, (f"optimizer state key {key!r} contains '.'; "
                                            f"restore uses rsplit('.', 1) assuming dotless keys")
                    out[f"{name_prefix}.{key}"] = val
        return out, fragment_info

    def get_memory_usage(self):
        """Get memory usage breakdown for MoEvement components.

        Returns:
            Dict with memory usage in bytes for each component.
        """
        usage = {
            "snapshot_bytes": sum(s.total_bytes() for s in self.snapshot_engine.get_all_snapshots().values()),
        }
        if self.upstream_logger is not None:
            usage["upstream_log_bytes"] = self.upstream_logger.total_memory_bytes()
        return usage

    def cleanup(self):
        """Release all resources held by MoEvement."""
        self.snapshot_engine.clear()
        self.converter.clear()
        if self.upstream_logger is not None:
            self.upstream_logger.clear()
