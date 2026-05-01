"""Sparse snapshot engine for MoEvement.

Handles per-operator granularity GPU-to-CPU snapshots and asynchronous
replication to peer nodes. Active operators snapshot FP32 master weights
and optimizer state; frozen operators snapshot only FP16 compute weights.
"""

import contextlib
import io
import os
import time
from collections import OrderedDict, defaultdict, deque

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

from deepspeed.moevement.buffer_pool import PinnedPool
from deepspeed.moevement.persist_worker import PersistWorker
from deepspeed.moevement.profiling import trace_range
from deepspeed.moevement.snapshot_io import BUNDLE_FILENAME, dump_bundle, load_bundle

# Reserved op_name for the per-iter RNG-state pseudo-operator embedded in
# the bundle by ``save_to_disk``.  The converter filters this name out of
# the operator iteration on load so it never appears in the replay
# schedule; the corresponding ``state_dict`` is routed to the converter's
# ``_rng_state_per_iter`` cache and consumed by ``_setup_replay_iter``.
_RNG_PSEUDO_OP = "__moe_rng_state__"

# Peer-pull handshake protocol versions.  The handshake is always a
# 2-int64 tensor ``(sender_rank, protocol_version)``; the server rejects
# unsupported versions by sending a single ``length=0`` manifest so the
# replacement falls back to the next peer.
#
# - ``PEER_PULL_PROTOCOL_BULK`` (1): original bulk-manifest flow — one
#   length-prefixed manifest covering every iter in the window, then all
#   flats back-to-back.
# - ``PEER_PULL_PROTOCOL_STREAMING`` (2): iter-major flow — one
#   mini-manifest per iteration carrying that iter's ``operators`` +
#   ``is_last_iter`` sentinel, followed by that iter's flats.  Only the
#   last mini-manifest carries ``fault_iter`` / ``loss_scaler_state`` /
#   ``engine_scalars`` since those describe the bundle as a whole.
PEER_PULL_PROTOCOL_BULK = 1
PEER_PULL_PROTOCOL_STREAMING = 2
PEER_PULL_PROTOCOL_SUPPORTED = frozenset({PEER_PULL_PROTOCOL_BULK, PEER_PULL_PROTOCOL_STREAMING})


def _stream_context(stream):
    """Return an accelerator stream context, or a no-op if none is available."""
    if stream is None:
        return contextlib.nullcontext()
    return get_accelerator().stream(stream)


class OperatorSnapshot:
    """Stores a snapshot of a single operator's state.

    For active operators: contains FP32 master weights and optimizer state.
    For frozen operators: contains only FP16 compute weights.
    """

    def __init__(self, name, iteration, is_active):
        self.name = name
        self.iteration = iteration
        self.is_active = is_active
        self.state_dict = {}
        # Optional per-entry fragment metadata populated by the
        # fragment-snapshot path under ZeRO-1/2: each entry in this dict
        # is ``{key: {"full_shape": [...], "fragment_numel": int}}``.
        # Absent (or the specific key absent) means the corresponding
        # ``state_dict`` entry is a full-shape tensor — non-ZeRO path or
        # test-built snapshots via ``add_tensor``.
        self.fragment_info = {}
        # Flat pinned CPU buffers backing the tensors in ``state_dict`` when
        # produced by the batched snapshot path.  Released back to the pool
        # at ``finalize_window`` once the snapshot rotates out.
        self._flat_buffers = []

    def add_tensor(self, key, tensor):
        """Store a tensor (already on CPU) in this snapshot."""
        self.state_dict[key] = tensor

    def total_bytes(self):
        """Return the raw byte footprint of every tensor in this snapshot.

        Used by logging and sanity checks — the pinned-buffer accounting
        is tracked separately on ``_flat_buffers`` and doesn't need to
        round-trip through this method.
        """
        total = 0
        for t in self.state_dict.values():
            total += t.nelement() * t.element_size()
        return total


class SparseSnapshotEngine:
    """Manages per-operator sparse snapshots with async GPU-to-CPU transfers.

    Each iteration, only a scheduled subset of operators have their full FP32
    state snapshotted. The rest store only FP16 compute weights.
    """

    def __init__(self, replication_factor=2, fsync_on_save=True):
        self.replication_factor = replication_factor
        # Driven by ``MoEvementConfig.fsync_on_save``; passed through to
        # ``dump_bundle`` so cloud-VM workloads can opt out of the
        # fsync barrier that dominates ``save_sparse_checkpoint`` cost.
        self._fsync_on_save = fsync_on_save
        self._cuda_stream = None
        # All three dicts are keyed by ``(iteration, name)``; the inner value
        # is an ``OperatorSnapshot`` holding that capture's payload.  Per-iter
        # keying is required so intermediate FROZEN-op FP16 captures within a
        # window are preserved (see spec §3.2 and note at ``snapshot_operator``).
        self._snapshots = OrderedDict()  # (iteration, name) -> OperatorSnapshot
        self._persisted_snapshots = OrderedDict()
        self._in_flight_snapshots = OrderedDict()
        # FP16 loss-scaler state captured at the end of the last finalized
        # window; travels through the same bundle channel (disk save + peer
        # pull) so a replacement rank can restore the scale factor +
        # overflow counters that were live when the last persisted bundle
        # was captured.  Without this, replay's overflow/skip decisions
        # diverge from training's and Adam's ``step_count`` drifts.
        # ``None`` until the first finalize_window fires.
        self._persisted_loss_scaler_state = None
        # Per-iter torch RNG state captured at on_iteration_end, mirroring
        # the operator-snapshot lifecycle (live → in-flight → persisted at
        # window boundaries).  Each entry is the post-iter-K state =
        # the state that iter K+1's forward will consume.  Restoring per-
        # tb during replay makes dropout / stochastic-layer recovery bit-
        # faithful to the fault-free trajectory; without it those models
        # silently diverge from iteration zero.  Numpy / Python random are
        # NOT captured — almost all DL stochasticity goes through torch
        # (Dropout, randperm, etc.); models that consume numpy random in
        # forward will still drift.
        self._rng_state_per_iter = OrderedDict()
        self._in_flight_rng_state_per_iter = OrderedDict()
        self._persisted_rng_state_per_iter = OrderedDict()
        # Peer replication is symmetric under ZeRO-1/2: every rank owns
        # a non-overlapping shard (optimizer state is partitioned across
        # the DP group), so every rank ships its own snapshot to r
        # forward peers on the ring and receives from r backward peers.
        # Receiver-side state is keyed by the SENDER's dp_rank so a
        # replacement rank for rank k can pull "what you received from
        # rank k" from any of k's forward peers.  The inner dicts are
        # keyed per ``(iteration, op_name)`` to match the per-iter local
        # snapshots — a window-spanning capture ships the full window's
        # worth of per-iter payloads, not the legacy per-name aggregate.
        self._received_snapshots = {}  # sender_dp_rank -> {(iter, op_name): state_dict}
        self._received_metadata = {}  # sender_dp_rank -> {(iter, op_name): {"is_active"}}
        self._received_window_start = {}  # sender_dp_rank -> window_start_iteration
        # Pool-acquired flat pinned buffers backing the views in
        # ``_received_snapshots``.  Released back to the pool only after
        # we clear the view dict for the next window so no reader can
        # observe storage that has been handed to a fresh ``acquire``.
        self._received_flat_buffers = []
        # Streaming-recovery (SD-O4 S3) per-iter buffer tracking.  When
        # the streaming pull path is active each iter's flats are kept
        # under their own iter key so ``release_iter_buffers(iter)`` can
        # drop just that iter's storage once replay has consumed it,
        # bounding peak CPU memory regardless of pull/replay rate skew.
        # Bulk path keeps the flat list (above) — bulk has no notion of
        # per-iter drop and frees everything in one shot at recovery
        # end.
        self._received_flat_buffers_by_iter = {}
        self._snapshot_iteration = -1
        self._window_start_iteration = -1
        # Pool of pinned CPU and device-side buffers reused across windows,
        # and a background worker that keeps torch.save off the training
        # thread.  ``_pending_gpu_staging`` holds device-side flat buffers
        # used during D2H; they must outlive any in-flight non_blocking
        # copies and are returned to the pool by ``synchronize``.
        # ``max_per_key`` is pushed in by the coordinator after init via
        # ``MoEvementConfig.pool_max_per_key`` (default 4096).
        self._pool = PinnedPool()
        self._persist_worker = PersistWorker()
        self._pending_gpu_staging = []
        # FIFO of ``(event, [staging_buffers])`` for in-flight D2H batches.
        # ``flush_to_main_stream`` records an event on the side stream and
        # rotates ``_pending_gpu_staging`` here so that the staging buffers
        # are released back to the pool only after the corresponding D2H
        # has actually drained — without paying a CPU sync at the boundary.
        # Sibling-stream ordering (next iter's optimizer.step waits for
        # the D2H to finish reading the source) is provided by
        # ``current_stream.wait_event(event)`` queued at the next iter's
        # ``on_before_optimizer_step`` call (NOT at boundary time, so
        # forward + backward overlap with side-stream drain).
        self._inflight_iter_buffers = deque()
        # Most-recently recorded boundary-D2H event awaiting a
        # main-stream wait_event at the next optimizer step.  ``None``
        # outside boundary windows; see ``record_pending_d2h_event`` /
        # ``wait_for_pending_d2h_event``.
        self._pending_d2h_event = None

    def _get_cuda_stream(self):
        # CPU accelerators expose ``Stream`` as None; fall back to synchronous
        # copies there instead of crashing on ``None()``.
        if self._cuda_stream is None:
            stream_ctor = get_accelerator().Stream
            if stream_ctor is not None:
                self._cuda_stream = stream_ctor()
        return self._cuda_stream

    def snapshot_operator(self, name, params_dict, optimizer_state_dict, is_active, iteration, fragment_info=None):
        """Snapshot a single operator's state from GPU to CPU.

        For active operators, captures FP32 master weights and optimizer state.
        For frozen operators, captures only FP16 compute weights.

        All tensors of the same dtype for a single operator are packed into
        one flat pinned CPU buffer so the D2H is issued as a single copy
        rather than one per key — fewer launch overheads and better PCIe
        utilization on operators with many optimizer state entries.

        Args:
            name: Operator identifier.
            params_dict: Dict of parameter name -> GPU tensor.
            optimizer_state_dict: Dict of optimizer state key -> GPU tensor (or None for frozen).
            is_active: Whether this operator is active (full FP32) or frozen (FP16 only).
            iteration: Current training iteration number.
            fragment_info: Optional ``{prefixed_key: {"full_shape", "fragment_numel"}}``
                mapping identifying which ``state_dict`` entries are per-rank
                fragments of a larger ZeRO-partitioned param.  Keys use the
                same ``params.``/``optimizer.`` prefixes the snapshot applies
                to ``state_dict``.  Entries absent from this dict are stored
                as full-shape tensors and routed through the
                ``safe_set_full_*`` restore path; present entries carry the
                fragment metadata through ``dump_bundle`` + ``load_bundle``
                and copy directly into ``_hp_mapping.hp_fragment`` on
                restore.
        """
        with trace_range("snap_active" if is_active else "snap_frozen"):
            snap = OperatorSnapshot(name, iteration, is_active)
            if fragment_info:
                snap.fragment_info = dict(fragment_info)
            stream = self._get_cuda_stream()
            # Pinned memory + non_blocking copies only make sense when a CUDA-like
            # stream is available; on CPU accelerators we fall back to a plain
            # synchronous copy so the destination tensor is valid on return.
            async_copy = stream is not None

            # Collect (key, gpu_tensor) pairs once; the batched copy groups by
            # dtype and allocates one flat pinned buffer per group.
            entries = []
            if is_active:
                for key, tensor in params_dict.items():
                    entries.append((f"params.{key}", tensor))
                if optimizer_state_dict is not None:
                    for key, tensor in optimizer_state_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            entries.append((f"optimizer.{key}", tensor))
            else:
                # Compute-weight capture is paper §3.2's memory-saving win: FP32
                # masters get compressed to FP16 (2x shrink).  Already-low-
                # precision dtypes (FP16, BF16) are stored as-is — the prior
                # unconditional ``.half()`` silently rounded BF16 through FP16
                # on save, so BF16 compute weights lost precision on every
                # FROZEN capture and the restore path got an FP16-rounded
                # value cast back to BF16 (lossy round-trip).  Per-tensor dtype
                # already lives in the bundle header, so the load path picks
                # up the preserved dtype without any schema change.
                for key, tensor in params_dict.items():
                    if tensor.dtype == torch.float32:
                        captured = tensor.half()
                    else:
                        captured = tensor
                    entries.append((f"compute_weights.{key}", captured))

            # The input tensors live on the default compute stream.  Tell the
            # snapshot stream to wait so the D2H copies read the final values
            # produced by the training step rather than racing with them.
            if async_copy:
                stream.wait_stream(get_accelerator().current_stream())

            with _stream_context(stream):
                self._batched_d2h(snap, entries, async_copy)

            # Keyed per (iteration, name) so a frozen op captured in multiple
            # iterations of one window preserves every per-iter FP16 snapshot —
            # overwriting by name alone would drop iter N's capture as soon as
            # iter N+1 re-snapshotted the same op (spec §3.2: frozen-op FP16 is
            # captured every iteration until the op becomes active).
            self._snapshots[(iteration, name)] = snap

    def _batched_d2h(self, snap, entries, async_copy):
        """Pack ``entries`` by dtype, D2H each group into one pinned buffer, expose views."""
        if not entries:
            return
        if os.environ.get("MOEV_D2H_BYTES_LOG", "0") == "1":
            tot = sum(t.numel() * t.element_size() for _, t in entries)
            iter_id = getattr(snap, "iteration", -1)
            name = getattr(snap, "name", "?")
            active = getattr(snap, "is_active", "?")
            print(f"[d2h-bytes] iter={iter_id} name={name} active={active} "
                  f"bytes={tot} entries={len(entries)}",
                  flush=True)

        by_dtype = defaultdict(list)
        for key, tensor in entries:
            by_dtype[tensor.dtype].append((key, tensor))

        for dtype, items in by_dtype.items():
            total_elems = sum(t.numel() for _, t in items)
            if total_elems == 0:
                continue

            with trace_range("moevement/d2h_acquire_cpu"):
                flat_cpu = self._pool.acquire((total_elems, ), dtype, pin=async_copy)
            device = items[0][1].device

            if async_copy:
                # Stage on the GPU side so the whole operator hits one D2H.
                # The staging buffer is pool-owned; release happens in
                # ``synchronize`` once the non_blocking copy has drained.
                with trace_range("moevement/d2h_acquire_gpu"):
                    flat_gpu = self._pool.acquire((total_elems, ), dtype, device=device)
                self._pending_gpu_staging.append(flat_gpu)
                offset = 0
                layout = []
                with trace_range("moevement/d2h_stage_pack"):
                    for key, tensor in items:
                        n = tensor.numel()
                        flat_gpu[offset:offset + n].copy_(tensor.contiguous().view(-1))
                        layout.append((key, offset, n, tensor.shape))
                        offset += n
                with trace_range("moevement/d2h_async"):
                    flat_cpu.copy_(flat_gpu, non_blocking=True)
            else:
                # CPU accelerator path: copy each tensor directly into its
                # slice of the flat buffer.  No staging buffer needed.
                offset = 0
                layout = []
                for key, tensor in items:
                    n = tensor.numel()
                    flat_cpu[offset:offset + n].copy_(tensor.contiguous().view(-1))
                    layout.append((key, offset, n, tensor.shape))
                    offset += n

            # Views share storage with ``flat_cpu``; releasing the flat buffer
            # invalidates them, so we track it for pool release at window
            # rollover.
            for key, start, n, shape in layout:
                snap.state_dict[key] = flat_cpu[start:start + n].view(shape)
            snap._flat_buffers.append(flat_cpu)

    def synchronize(self):
        """Wait for all async GPU-to-CPU transfers to complete.

        Used by save / recovery-flush paths that need to read the pinned
        CPU buffers (or recycle staging) immediately on the calling CPU
        thread.  Boundary path uses ``record_pending_d2h_event`` +
        ``wait_for_pending_d2h_event`` instead so training doesn't pay a
        CPU stall waiting for D2H to drain.
        """
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
        if self._pending_gpu_staging:
            for gpu_buf in self._pending_gpu_staging:
                self._pool.release(gpu_buf)
            self._pending_gpu_staging = []
        # All in-flight events are now complete (we just CPU-synced); release
        # everything held under per-iter event FIFOs.
        while self._inflight_iter_buffers:
            _evt, bufs = self._inflight_iter_buffers.popleft()
            for b in bufs:
                self._pool.release(b)
        # Any pending boundary event we were holding for the next
        # optim_step has fired by now too — clear it so a stale handle
        # isn't waited on after the source D2H is long gone.
        self._pending_d2h_event = None

    def record_pending_d2h_event(self):
        """Record a side-stream event marking the current D2H drain point.

        Called from the coordinator's ``on_iteration_end`` boundary block.
        Pairs with ``wait_for_pending_d2h_event`` called at the NEXT
        iter's ``on_before_optimizer_step`` — placing the main-stream
        wait there (rather than at boundary time) lets iter N+1's
        forward + backward overlap with side-stream D2H drain.  Only
        ``optimizer.step`` (the actual writer of the source tensors)
        needs to wait, not the entire next iter.

        Bookkeeping:
        - Stashes the just-recorded event so the next
          ``wait_for_pending_d2h_event`` can queue a wait on it.
        - Rotates ``_pending_gpu_staging`` into the per-event FIFO
          ``_inflight_iter_buffers`` so the device staging buffers are
          held until the D2H actually drains.
        - Sweeps the FIFO head, releasing buffers whose events have
          already fired.

        Side-effect-free if no D2H was issued (no stream / no staging) —
        safe to call every boundary unconditionally.
        """
        if self._cuda_stream is None:
            return
        if self._pending_gpu_staging:
            evt = get_accelerator().Event()
            self._cuda_stream.record_event(evt)
            self._pending_d2h_event = evt
            self._inflight_iter_buffers.append((evt, self._pending_gpu_staging))
            self._pending_gpu_staging = []
        # Sweep completed events at the head of the FIFO.  Side-stream is
        # FIFO-ordered, so events fire in insertion order — once we hit one
        # that's not done, no later one can be done either.
        while self._inflight_iter_buffers:
            evt, bufs = self._inflight_iter_buffers[0]
            if not evt.query():
                break
            self._inflight_iter_buffers.popleft()
            for b in bufs:
                self._pool.release(b)

    def wait_for_pending_d2h_event(self):
        """Queue a main-stream wait on the most recent boundary D2H event.

        Called from the coordinator's ``on_before_optimizer_step``.  The
        wait is queued on the current/main stream — the GPU stalls only
        if the side stream still hasn't drained by the time
        ``optimizer.step`` is launched.  At gas>1 with normal
        forward + backward times exceeding per-iter D2H, this is
        typically a no-op (event already fired by the time the wait
        is enqueued).

        No-op when no event is pending (between boundaries) or when
        running on a CPU-only accelerator.
        """
        if self._cuda_stream is None:
            return
        if self._pending_d2h_event is None:
            return
        get_accelerator().current_stream().wait_event(self._pending_d2h_event)
        self._pending_d2h_event = None

    def begin_window(self, iteration):
        """Mark the beginning of a new sparse checkpoint window."""
        self._window_start_iteration = iteration
        # Rotate via reference swap, not ``OrderedDict(other)``.  The copy-
        # construct form used to rebuild every key/value every boundary
        # (O(N) per window); the swap is semantically identical (no other
        # reader holds the old reference) and O(1).
        self._in_flight_snapshots = self._snapshots
        self._snapshots = OrderedDict()
        # Per-iter RNG mirrors the operator-snapshot lifecycle so persisted
        # RNG always matches persisted operator state.
        self._in_flight_rng_state_per_iter = self._rng_state_per_iter
        self._rng_state_per_iter = OrderedDict()

    def finalize_window(self):
        """Finalize the current window: promote in-flight to persisted, GC old.

        Release the previous window's flat pinned buffers back to the pool.
        Buffers currently held by the async persist worker are marked busy
        and will be returned to the pool by the worker's completion callback
        via ``release_busy``.
        """
        for snap in self._persisted_snapshots.values():
            for flat in snap._flat_buffers:
                self._pool.release(flat)
            # Do NOT reset ``snap._flat_buffers`` or ``snap.state_dict``
            # here.  Under B-async peer replication, a worker thread may
            # still hold a reference to this snapshot via its manifest;
            # in-place mutation caused the worker's ``_post_ring_send``
            # to raise ``IndexError`` on ``snap._flat_buffers[group_idx]``
            # (sibling race to the dict-swap KeyError).  Python GC
            # reclaims the snapshot naturally once the worker's ref
            # drops; pool flats are busy-guarded by
            # ``_mark_persisted_flats_busy``.
        # Reference swap (see ``begin_window``) — O(1), identical semantics.
        self._persisted_snapshots = self._in_flight_snapshots
        self._in_flight_snapshots = OrderedDict()
        # Promote per-iter RNG in lockstep with operator snapshots.
        self._persisted_rng_state_per_iter = self._in_flight_rng_state_per_iter
        self._in_flight_rng_state_per_iter = OrderedDict()
        # Inject the per-iter RNG as a pseudo-operator into ``_persisted_snapshots``
        # so save / replicate / peer-pull / cascade all carry RNG uniformly
        # without each callsite needing per-iter dict plumbing.  The
        # converter's ``initialize_from_snapshots`` filters ``_RNG_PSEUDO_OP``
        # out of the replay schedule and routes its state_dict to the
        # converter's per-iter RNG cache.  No flat buffers (RNG tensors are
        # plain CPU ByteTensors), so ``finalize_window``'s buffer-release
        # loop on the next rollover is a no-op for these entries.
        for iteration, rng_state in self._persisted_rng_state_per_iter.items():
            if not rng_state:
                continue
            snap = OperatorSnapshot(_RNG_PSEUDO_OP, iteration, is_active=False)
            for key, tensor in rng_state.items():
                snap.add_tensor(key, tensor)
            self._persisted_snapshots[(iteration, _RNG_PSEUDO_OP)] = snap

    def replicate_to_peers(self,
                           dp_group,
                           dp_rank,
                           dp_world_size,
                           device,
                           replication_factor=None,
                           persisted_snapshots=None,
                           pinned_release_fn=None):
        """Symmetric ring replication: every rank sends to r forward peers.

        Under ZeRO-1/2 each DP rank owns a non-overlapping shard of the
        optimizer state.  A primary-only scheme (old rank 0 → peers 1..r)
        would lose every non-primary rank's optimizer shard on failure,
        so we flip to a ring: rank k sends its own snapshot to ranks
        ``(k+1) .. (k+r)`` mod ``dp_world_size`` and receives the
        snapshots sent by ranks ``(k-1) .. (k-r)`` mod ``dp_world_size``.
        If rank k fails, its replacement can pull rank-k's shard from
        any of ranks ``(k+1) .. (k+r)`` — each of them has
        ``_received_snapshots[k]`` in pinned CPU memory.

        Gloo accepts CPU tensors natively so pinned snapshot buffers go
        straight to the peer's pinned memory with no GPU transit.  Sends
        use ``isend`` so the ring pattern doesn't deadlock under the
        sequential send/recv semantics of blocking gloo primitives.

        Args:
            dp_group: Gloo process group to use for sends — must be the
                replication mirror built in the coordinator, not the
                training NCCL group.
            dp_rank: This rank's index within ``dp_group``.
            dp_world_size: Total ranks in ``dp_group``.
            device: Retained for API compatibility (unused on gloo).
            replication_factor: Number of peers to replicate to.  Defaults
                to the engine's configured factor.  Clamped to
                ``dp_world_size - 1``.
        """
        del device  # gloo sends directly from CPU — no staging device needed
        if replication_factor is None:
            replication_factor = self.replication_factor
        if replication_factor <= 0 or dp_world_size <= 1:
            return

        r = min(replication_factor, dp_world_size - 1)

        # Drop the previous receive window atomically.  Every sender
        # ships a fresh snapshot each window; we clear the per-sender
        # dicts and release pooled flat buffers together so readers that
        # enumerate ``_received_snapshots`` never see a stale op mixed in
        # with a fresh one.
        stale_flats = self._received_flat_buffers
        self._received_flat_buffers = []
        for flat in stale_flats:
            self._pool.release(flat)
        self._received_snapshots.clear()
        self._received_metadata.clear()
        self._received_window_start.clear()

        # Pin the persisted-snapshots dict reference for this call.
        # Callers on the training thread (coordinator.py submit path)
        # pass the dict they captured BEFORE the next ``finalize_window``
        # could rotate it — without that pin the manifest could
        # reference iter/op pairs that vanish from
        # ``self._persisted_snapshots`` before ``_post_ring_send``
        # dereferences them, surfacing as ``KeyError`` on the worker.
        # Legacy callers (tests, non-replication paths) can omit the
        # arg; we fall back to the live attribute.
        persisted = persisted_snapshots if persisted_snapshots is not None else self._persisted_snapshots
        # Build our own manifest once and reuse across all r send peers.
        own_manifest = self._build_replication_manifest(persisted) if persisted else None
        if own_manifest is not None and own_manifest["operators"]:
            own_length, own_payload = self._frame_manifest(own_manifest)
        else:
            own_manifest = None
            own_length = torch.tensor([0], dtype=torch.int64)
            own_payload = None

        # Clone every send-source flat into a pageable CPU arena so
        # the pinned D2H flats can cycle back through training's pool
        # without waiting on the slow gloo ship.  The clone runs on
        # this worker thread as a CPU-to-CPU memcpy — faster than the
        # gloo ship it replaces on the busy-hold critical path.
        # After the clone we invoke ``pinned_release_fn`` (supplied by
        # the coordinator's submit path) so training can observe the
        # pinned flats as released before the slow wire send starts.
        # Clones live in the pool's pageable sub-key and are returned
        # at the end of this call.
        clone_flats_by_iter_key = {}
        _clone_t0 = time.perf_counter() if os.environ.get("MOEV_CLONE_TIMING", "0") == "1" else None
        _clone_op_count = 0
        _clone_byte_count = 0
        if own_manifest is not None:
            for op_entry in own_manifest["operators"]:
                iter_key = (op_entry["iteration"], op_entry["name"])
                snap = persisted[iter_key]
                per_op_clones = []
                for group_idx, group_entry in enumerate(op_entry["groups"]):
                    if group_idx < len(snap._flat_buffers):
                        flat = snap._flat_buffers[group_idx]
                        clone = self._pool.acquire(flat.shape, flat.dtype, pin=False)
                        clone.copy_(flat)
                        per_op_clones.append(clone)
                        if _clone_t0 is not None:
                            _clone_op_count += 1
                            _clone_byte_count += flat.numel() * flat.element_size()
                    else:
                        # Non-packed fallback snapshot (test path via
                        # ``add_tensor``): no flat to clone.  Leave
                        # ``per_op_clones`` short; ``_post_ring_send``
                        # falls back to its staging-pack path below.
                        per_op_clones.append(None)
                clone_flats_by_iter_key[iter_key] = per_op_clones
        if _clone_t0 is not None:
            _clone_dur_ms = (time.perf_counter() - _clone_t0) * 1000
            print(f"[clone-timing] dur_ms={_clone_dur_ms:.1f} ops={_clone_op_count} "
                  f"bytes={_clone_byte_count}",
                  flush=True)
        # Pinned D2H flats are no longer needed by this worker once
        # the clones exist — signal the coordinator to release its
        # busy refs so the training thread's next D2H acquire can
        # reuse them.  ``pinned_release_fn=None`` preserves the
        # legacy behaviour (release via done-callback after send).
        if pinned_release_fn is not None:
            pinned_release_fn()

        # One ring hop per offset.  Pair every rank with a forward
        # send peer and a backward recv peer.  ``isend`` returns a
        # handle that we wait on at the end of each offset so any
        # error surfaces before we proceed.  Non-packed staging
        # buffers (acquired from the pool to carry an unpacked op's
        # bytes over isend) are released right after wait() so they
        # don't leak into ``_pending_gpu_staging`` — that list is
        # owned by the training thread's D2H path and shouldn't be
        # touched from this worker thread.
        # Ring arithmetic produces group-local rank indices, but
        # ``dist.isend``/``dist.recv`` interpret ``dst``/``src`` as
        # global process-group ranks regardless of the ``group=``
        # argument.  When ``dp_group`` is a real subgroup (e.g. a
        # per-PP-stage DP group {2, 3}), passing a group-local index
        # straight through raises ``ValueError: Global rank N is not
        # part of group`` as soon as the group-local index doesn't
        # coincide with a global rank in the subgroup.  Translate
        # here so the subgroup path matches the
        # group-is-world-and-indices-coincide path the unit tests
        # already exercise.
        def _group_to_global(group_rank):
            # Skip translation when dist isn't up (single-rank unit
            # tests that mock the comm layer) or no real group is
            # present — the mocked tests pin group-local ring
            # arithmetic and would see different dst values if we
            # translated blindly.
            if dp_group is None or not dist.is_initialized():
                return group_rank
            return dist.get_global_rank(dp_group, group_rank)

        for offset in range(1, r + 1):
            local_sender = (dp_rank - offset) % dp_world_size
            send_dst = _group_to_global((dp_rank + offset) % dp_world_size)
            recv_src = _group_to_global(local_sender)
            send_handles, temp_buffers = self._post_ring_send(send_dst,
                                                              own_manifest,
                                                              own_length,
                                                              own_payload,
                                                              dp_group,
                                                              persisted,
                                                              clone_flats_by_iter_key=clone_flats_by_iter_key)
            # Keep ``recv_src`` (global) for the wire-level ``dist.recv``;
            # ``local_sender`` is what server-side lookups
            # (``get_received_snapshots_for``, ``serve_peer_pull_request``)
            # use to index ``_received_snapshots``, so that's what the
            # receiver keys by.  The two must not be conflated.
            self._recv_ring_peer(local_sender, recv_src, dp_group)
            for h in send_handles:
                h.wait()
            for buf in temp_buffers:
                self._pool.release(buf)

        # Clones served their purpose; return them to the pool's
        # pageable sub-key so the next window's replicate_to_peers
        # can reuse them without a fresh malloc.
        for per_op_clones in clone_flats_by_iter_key.values():
            for clone in per_op_clones:
                if clone is not None:
                    self._pool.release(clone)

        total_received_ops = sum(len(ops) for ops in self._received_snapshots.values())
        if total_received_ops:
            logger.info(f"[MoEvement] Received {total_received_ops} operator snapshots "
                        f"from {len(self._received_snapshots)} peers on the replication ring (r={r})")

    def _post_ring_send(self,
                        send_dst,
                        manifest,
                        length_tensor,
                        payload_tensor,
                        group,
                        persisted=None,
                        clone_flats_by_iter_key=None):
        """Post isend operations for our own manifest + flats to one peer.

        Returns ``(handles, temp_buffers)``.  The caller must ``wait()``
        on every handle before releasing ``temp_buffers`` back to the
        pool — those buffers are the scratch-pack allocations used on
        the non-packed fallback path, and they can't be reclaimed while
        an isend is still reading them.  An empty manifest is encoded
        as ``length=0`` so receivers short-circuit.

        ``persisted`` is the snapshot dict the manifest was built from;
        callers pin it at worker entry so a concurrent ``finalize_window``
        on the training thread can't swap ``self._persisted_snapshots``
        out before the per-op lookup below runs.  Defaults to the live
        attribute for backwards compat with non-replication callers.

        ``clone_flats_by_iter_key`` (optional): dict mapping
        ``(iteration, op_name) -> [cloned pageable flat per group]``.
        When supplied, send from the clones instead of the persisted
        ``snap._flat_buffers``.  The clones live in the pool's
        pageable sub-key and their caller (``replicate_to_peers``)
        owns the full lifecycle — releases back to the pool after all
        r offsets complete.  This decouples pinned D2H flats from
        gloo wire time so the training thread's next D2H acquire
        doesn't wait on a busy-held flat for the full ship duration.
        """
        if persisted is None:
            persisted = self._persisted_snapshots
        handles = [dist.isend(length_tensor, dst=send_dst, group=group)]
        temp_buffers = []
        if manifest is None:
            return handles, temp_buffers
        handles.append(dist.isend(payload_tensor, dst=send_dst, group=group))
        # Hot path: prefer pageable clones from
        # ``clone_flats_by_iter_key`` (decouples the send from the
        # pinned D2H lifetime).  Fall back to the persisted pinned
        # flat when no clone exists (legacy callers), then to a
        # staging pack for snapshots constructed via ``add_tensor``.
        for op_entry in manifest["operators"]:
            iter_key = (op_entry["iteration"], op_entry["name"])
            snap = persisted[iter_key]
            has_packed = len(snap._flat_buffers) == len(op_entry["groups"])
            per_op_clones = None
            if clone_flats_by_iter_key is not None:
                per_op_clones = clone_flats_by_iter_key.get(iter_key)
            for group_idx, group_entry in enumerate(op_entry["groups"]):
                clone = per_op_clones[group_idx] if per_op_clones is not None and group_idx < len(
                    per_op_clones) else None
                if clone is not None:
                    handles.append(dist.isend(clone, dst=send_dst, group=group))
                    continue
                if has_packed:
                    handles.append(dist.isend(snap._flat_buffers[group_idx], dst=send_dst, group=group))
                    continue
                flat_cpu = self._pool.acquire((group_entry["total_elems"], ),
                                              group_entry["dtype"],
                                              pin=self._get_cuda_stream() is not None)
                self._pack_group_into(snap.state_dict, group_entry, flat_cpu)
                temp_buffers.append(flat_cpu)
                handles.append(dist.isend(flat_cpu, dst=send_dst, group=group))
        return handles, temp_buffers

    def _recv_ring_peer(self, local_sender, recv_src, group):
        """Receive one peer's manifest + flats and store keyed by sender.

        ``recv_src`` is the *global* rank used for the wire-level
        ``dist.recv``; ``local_sender`` is the *group-local* DP rank
        that server-side lookups (``get_received_snapshots_for``,
        ``serve_peer_pull_request``) use to index ``_received_snapshots``.
        The two must not be conflated.

        Short-circuits when the sender had no snapshot (length==0).

        The per-flat recvs are issued as ``irecv`` batch and waited on
        together so gloo can overlap transfers across ops — the serial
        blocking-recv pattern used a single-recv-in-flight window that
        stalled between each op while the next buffer was acquired and
        the syscall / socket handshake round-tripped.
        Post-failure cleanup releases every pre-allocated flat; on
        success they're appended to ``_received_flat_buffers`` for the
        next window's release cycle.  Entry semantics are now
        all-or-nothing per sender: a mid-batch failure leaves
        ``_received_snapshots[local_sender]`` absent (the previous
        window's slot was already cleared at the top of
        ``replicate_to_peers``), which downstream peer-pull code
        tolerates the same as the empty-sender short-circuit.
        """
        length_tensor = torch.zeros(1, dtype=torch.int64)
        dist.recv(length_tensor, src=recv_src, group=group)
        length = int(length_tensor.item())
        if length == 0:
            # Peer had no persisted snapshot this window — record an
            # empty slot so a later lookup for this sender returns None.
            self._received_snapshots[local_sender] = {}
            self._received_metadata[local_sender] = {}
            self._received_window_start[local_sender] = -1
            return

        payload = torch.zeros(length, dtype=torch.uint8)
        dist.recv(payload, src=recv_src, group=group)
        # ``weights_only=True`` restricts unpickling to tensors + plain
        # containers + torch.dtype — the only types our manifest carries
        # — so a compromised peer can't inject arbitrary code.
        manifest = torch.load(io.BytesIO(payload.numpy().tobytes()), weights_only=True)

        # Pre-allocate every (op, group) flat and post its ``irecv``
        # before waiting on any.  Ordering matches the sender's isend
        # walk in ``_post_ring_send`` — gloo FIFO-matches isend/irecv
        # pairs with the same (src, tag) so same-order posting is
        # enough; tag defaults to 0 on both sides.  Pre-posting gives
        # gloo ready buffers to fill as bytes stream in so the kernel
        # socket buffer doesn't stall while userspace acquires the
        # next flat.
        op_flats = []  # list aligned with manifest["operators"]: op_flats[i][j] = flat for op i, group j
        posted_flats = []
        handles = []
        try:
            for op_entry in manifest["operators"]:
                per_op_flats = []
                for group_entry in op_entry["groups"]:
                    total = group_entry["total_elems"]
                    dtype = group_entry["dtype"]
                    # Unpinned: gloo doesn't need pinned memory, and
                    # peer-pull H2D (rare path) tolerates a sync copy.
                    # Eliminates receiver-side cudaHostAlloc pressure.
                    flat_cpu = self._pool.acquire((total, ), dtype, pin=False)
                    posted_flats.append(flat_cpu)
                    per_op_flats.append(flat_cpu)
                    handles.append(dist.irecv(flat_cpu, src=recv_src, group=group))
                op_flats.append(per_op_flats)
            for handle in handles:
                handle.wait()
        except Exception:
            for flat in posted_flats:
                self._pool.release(flat)
            raise

        # All flats have landed.  Publish per-op tensors + metadata and
        # commit the sender slot.  Layout unpacking is pure CPU
        # bookkeeping; no more wire traffic from here.
        self._received_snapshots[local_sender] = {}
        self._received_metadata[local_sender] = {}
        self._received_window_start[local_sender] = manifest["window_start_iteration"]
        for op_entry, per_op_flats in zip(manifest["operators"], op_flats):
            iter_key = (int(op_entry["iteration"]), op_entry["name"])
            op_tensors = {}
            for group_entry, flat_cpu in zip(op_entry["groups"], per_op_flats):
                self._received_flat_buffers.append(flat_cpu)
                for key, offset, shape in group_entry["layout"]:
                    n = 1
                    for s in shape:
                        n *= int(s)
                    op_tensors[key] = flat_cpu[offset:offset + n].view(shape)
            self._received_metadata[local_sender][iter_key] = {
                "is_active": op_entry["is_active"],
            }
            self._received_snapshots[local_sender][iter_key] = op_tensors

        logger.info(f"[MoEvement] Received peer snapshot for {len(manifest['operators'])} operator entries")

    def _build_replication_manifest(self, persisted=None):
        """Build the per-window replication manifest, grouping keys by dtype.

        Assumes the caller (training thread) has already synchronized the
        snapshot stream; we deliberately don't call ``self.synchronize()``
        here because this runs on the replication worker thread and the
        training thread may be issuing the next window's D2H copies into
        the shared ``_pending_gpu_staging`` list.

        Each ``operators`` entry carries its own ``iteration`` field so a
        window with per-iter FP16 captures for the same op_name (frozen in
        multiple consecutive iterations) ships every capture rather than
        collapsing them by name.

        ``persisted`` is the snapshot dict to build from — callers pin it
        at worker entry (see ``replicate_to_peers``).  Defaults to the
        live attribute for callers that don't need the race-safety pin.
        """
        if persisted is None:
            persisted = self._persisted_snapshots
        operators = []
        for (iteration, op_name), snap in persisted.items():
            by_dtype = defaultdict(list)
            for key, tensor in snap.state_dict.items():
                by_dtype[tensor.dtype].append((key, tensor))
            groups = []
            for dtype, items in by_dtype.items():
                layout = []
                offset = 0
                for key, tensor in items:
                    shape = tuple(int(s) for s in tensor.shape)
                    layout.append((key, offset, shape))
                    offset += tensor.numel()
                groups.append({"dtype": dtype, "total_elems": offset, "layout": layout})
            operators.append({
                "name": op_name,
                "is_active": snap.is_active,
                "iteration": int(iteration),
                "groups": groups,
            })
        return {
            "window_start_iteration": self._window_start_iteration,
            "operators": operators,
        }

    @staticmethod
    def _pack_group_into(state_dict, group, flat):
        """Pack the tensors of one (op, dtype) group into the given flat buffer."""
        for key, offset, shape in group["layout"]:
            src = state_dict[key]
            n = src.numel()
            flat[offset:offset + n].copy_(src.contiguous().view(-1))

    @staticmethod
    def _frame_manifest(manifest):
        """Pickle ``manifest`` once and return ``(length, payload)`` tensors.

        The caller reuses these tensors across every peer target so we avoid
        re-serializing the (identical) manifest per peer.
        """
        with trace_range("recovery/manifest_serialize"):
            buf = io.BytesIO()
            torch.save(manifest, buf)
            raw = buf.getvalue()
            payload = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            length = torch.tensor([payload.numel()], dtype=torch.int64)
        return length, payload

    def serve_peer_pull_request(self,
                                requester_rank,
                                sender_dp_rank,
                                group,
                                fault_iter=None,
                                loss_scaler_state=None,
                                engine_scalars=None,
                                protocol_version=PEER_PULL_PROTOCOL_BULK):
        """Serve a replacement rank's request for a specific sender's state.

        Called on the peer side of the pull protocol.  The replacement
        rank asks "give me what you received from rank ``sender_dp_rank``
        during the last ring replication"; we serialize that per-sender
        slice of ``_received_snapshots`` and ship it on the provided
        (typically gloo) group.  Protocol shape depends on
        ``protocol_version``: ``PEER_PULL_PROTOCOL_BULK`` (the default)
        ships one length-prefixed manifest followed by all flats;
        ``PEER_PULL_PROTOCOL_STREAMING`` ships one mini-manifest per
        iteration followed by that iter's flats, in sorted iteration
        order.  The streaming shape is the wire foundation for hiding
        pull-of-iter-N+1 behind replay-of-iter-N on the receiver.

        ``fault_iter`` is the serving rank's current ``global_step`` —
        the iteration that paused DP peers had most recently completed
        before the fault.  Threading it through the manifest lets the
        replacement drive catch-up replay past the last persisted window
        back up to this iter.  ``None`` (the default) elides the field
        and callers that don't need catch-up (tests, early windows) see
        the old two-field metadata shape.

        Sends ``length=0`` if we have nothing for that sender (the
        replacement will try the next peer from the cluster manager's
        list).
        """
        if group is not None and dist.is_initialized():
            requester_global = dist.get_global_rank(group, requester_rank)
        else:
            requester_global = requester_rank
        if sender_dp_rank not in self._received_snapshots:
            dist.send(torch.tensor([0], dtype=torch.int64), dst=requester_global, group=group)
            return

        states = self._received_snapshots[sender_dp_rank]
        meta_by_iter_op = self._received_metadata.get(sender_dp_rank, {})
        window_start = self._received_window_start.get(sender_dp_rank, -1)
        if not states:
            dist.send(torch.tensor([0], dtype=torch.int64), dst=requester_global, group=group)
            return

        if protocol_version == PEER_PULL_PROTOCOL_STREAMING:
            self._serve_streaming(requester_global, group, states, meta_by_iter_op, window_start, fault_iter,
                                  loss_scaler_state, engine_scalars)
            return

        # Build a manifest parallel to ``_build_replication_manifest``
        # but sourced from our received views (already on CPU, already
        # packed into flat pinned buffers via the ring recv path).  One
        # entry per (iteration, op_name) — the full per-iter payload
        # received from the original sender gets re-shipped intact.
        with trace_range("recovery/serve_manifest_build"):
            operators = []
            for iter_key, op_tensors in states.items():
                iteration, op_name = iter_key
                by_dtype = defaultdict(list)
                for key, tensor in op_tensors.items():
                    by_dtype[tensor.dtype].append((key, tensor))
                groups = []
                for dtype, items in by_dtype.items():
                    layout = []
                    offset = 0
                    for key, tensor in items:
                        shape = tuple(int(s) for s in tensor.shape)
                        layout.append((key, offset, shape))
                        offset += tensor.numel()
                    groups.append({"dtype": dtype, "total_elems": offset, "layout": layout})
                op_meta = meta_by_iter_op.get(iter_key, {"is_active": True})
                operators.append({
                    "name": op_name,
                    "is_active": op_meta["is_active"],
                    "iteration": int(iteration),
                    "groups": groups,
                })
            manifest = {"window_start_iteration": window_start, "operators": operators}
            if fault_iter is not None:
                manifest["fault_iter"] = int(fault_iter)
            if loss_scaler_state is not None:
                manifest["loss_scaler_state"] = loss_scaler_state
            if engine_scalars is not None:
                manifest["engine_scalars"] = engine_scalars
        length, payload = self._frame_manifest(manifest)
        with trace_range("recovery/serve_manifest_send"):
            dist.send(length, dst=requester_global, group=group)
            dist.send(payload, dst=requester_global, group=group)

        # Pack each group's tensors into a flat buffer and send.  We
        # don't reuse the flat pinned buffer from the ring recv because
        # this is on-demand, not the hot path.
        with trace_range("recovery/serve_flats_send"):
            pin_cpu = self._get_cuda_stream() is not None
            for op_entry in operators:
                op_tensors = states[(op_entry["iteration"], op_entry["name"])]
                for group_entry in op_entry["groups"]:
                    flat_cpu = self._pool.acquire((group_entry["total_elems"], ), group_entry["dtype"], pin=pin_cpu)
                    try:
                        self._pack_group_into(op_tensors, group_entry, flat_cpu)
                        dist.send(flat_cpu, dst=requester_global, group=group)
                    finally:
                        self._pool.release(flat_cpu)

    def _serve_streaming(self, requester_global, group, states, meta_by_iter_op, window_start, fault_iter,
                         loss_scaler_state, engine_scalars):
        """Serve the iter-major streaming variant of ``serve_peer_pull_request``.

        For each iter in sorted order: send a mini-manifest
        (length-prefixed, pickled) carrying that iter's per-op layout
        plus ``is_last_iter`` sentinel, then send that iter's flats.
        Bundle-level fields (``window_start_iteration``, ``fault_iter``,
        ``loss_scaler_state``, ``engine_scalars``) ride the LAST
        mini-manifest so the receiver has everything it needs by the
        time the stream ends.

        The ``is_last_iter`` sentinel lets the receiver stop posting
        recvs without a separate end-of-stream signal.
        """
        sorted_iters = sorted({iter_key[0] for iter_key in states.keys()})
        by_iter = defaultdict(list)
        for iter_key in states.keys():
            by_iter[iter_key[0]].append(iter_key)

        pin_cpu = self._get_cuda_stream() is not None
        for idx, iteration in enumerate(sorted_iters):
            iter_op_entries = []
            for iter_key in by_iter[iteration]:
                _, op_name = iter_key
                op_tensors = states[iter_key]
                by_dtype = defaultdict(list)
                for key, tensor in op_tensors.items():
                    by_dtype[tensor.dtype].append((key, tensor))
                groups = []
                for dtype, items in by_dtype.items():
                    layout = []
                    offset = 0
                    for key, tensor in items:
                        shape = tuple(int(s) for s in tensor.shape)
                        layout.append((key, offset, shape))
                        offset += tensor.numel()
                    groups.append({"dtype": dtype, "total_elems": offset, "layout": layout})
                op_meta = meta_by_iter_op.get(iter_key, {"is_active": True})
                iter_op_entries.append({
                    "name": op_name,
                    "is_active": op_meta["is_active"],
                    "iteration": int(iteration),
                    "groups": groups,
                })

            is_first = (idx == 0)
            is_last = (idx == len(sorted_iters) - 1)
            mini_manifest = {
                "iteration": int(iteration),
                "operators": iter_op_entries,
                "is_last_iter": is_last,
            }
            # Bundle-level fields ride the FIRST mini-manifest so the
            # streaming receiver (S2 coordinator threading) can commit
            # to recovery state — set replay cursor, restore the loss
            # scaler + engine scalars, freeze operators — on iter 0 and
            # return from ``load_sparse_from_peer`` without waiting for
            # the whole window to drain.  The replay loop then drains
            # subsequent iters from the pull thread on demand.
            if is_first:
                mini_manifest["window_start_iteration"] = window_start
                # Ship the full persisted-iter list so the receiver can
                # compute ``replay_iters`` via the same ``_compute_replay_iters``
                # helper the cascade path uses, keeping recovering ranks'
                # handshake counts symmetric (paused-peer release relies
                # on every recovering rank running the same number of
                # iters — see coordinator.py:_release_paused_peers).
                mini_manifest["persisted_iters"] = [int(it) for it in sorted_iters]
                if fault_iter is not None:
                    mini_manifest["fault_iter"] = int(fault_iter)
                if loss_scaler_state is not None:
                    mini_manifest["loss_scaler_state"] = loss_scaler_state
                if engine_scalars is not None:
                    mini_manifest["engine_scalars"] = engine_scalars

            length, payload = self._frame_manifest(mini_manifest)
            with trace_range("recovery/serve_iter_manifest_send"):
                dist.send(length, dst=requester_global, group=group)
                dist.send(payload, dst=requester_global, group=group)
            with trace_range("recovery/serve_iter_flats_send"):
                for op_entry in iter_op_entries:
                    op_tensors = states[(op_entry["iteration"], op_entry["name"])]
                    for group_entry in op_entry["groups"]:
                        flat_cpu = self._pool.acquire((group_entry["total_elems"], ),
                                                      group_entry["dtype"],
                                                      pin=pin_cpu)
                        try:
                            self._pack_group_into(op_tensors, group_entry, flat_cpu)
                            dist.send(flat_cpu, dst=requester_global, group=group)
                        finally:
                            self._pool.release(flat_cpu)

    def pull_snapshot_from_peer(self, peer_rank, group, protocol_version=PEER_PULL_PROTOCOL_BULK, on_iter_ready=None):
        """Pull a sparse snapshot from a named peer on the replication ring.

        Called on the replacement rank with ``peer_rank`` identifying a
        surviving peer that received this rank's replicated snapshot.
        ``peer_rank`` is a *group-local* rank on ``group``; we translate
        to the global rank before handing it to ``dist.recv``, which
        treats ``src`` as global regardless of the ``group`` kwarg.

        Returns ``(metadata, operator_states)`` shaped like
        ``load_from_disk``, or ``(None, None)`` if the peer had nothing
        (the caller can retry another peer from the cluster manager's
        list).

        Leaves the pulled state in a private dict rather than mingling
        with the ring receiver's ``_received_snapshots`` — those are
        keyed by original sender, and a replacement's pull is
        semantically "my own shard", not "what I happened to receive".
        """
        # See note in ``replicate_to_peers._group_to_global``:
        # mocked-dist unit tests pass a sentinel group and expect
        # group-local src values to flow through unchanged.
        if group is not None and dist.is_initialized():
            peer_global = dist.get_global_rank(group, peer_rank)
        else:
            peer_global = peer_rank

        if protocol_version == PEER_PULL_PROTOCOL_STREAMING:
            return self._pull_streaming(peer_global, group, on_iter_ready=on_iter_ready)

        with trace_range("recovery/pull_manifest_recv"):
            length_tensor = torch.zeros(1, dtype=torch.int64)
            dist.recv(length_tensor, src=peer_global, group=group)
            length = int(length_tensor.item())
            if length == 0:
                return None, None

            payload = torch.zeros(length, dtype=torch.uint8)
            dist.recv(payload, src=peer_global, group=group)
        with trace_range("recovery/pull_manifest_deserialize"):
            manifest = torch.load(io.BytesIO(payload.numpy().tobytes()), weights_only=True)

        with trace_range("recovery/pull_flats_recv"):
            pin_cpu = self._get_cuda_stream() is not None
            per_iter_operator_states = {}
            per_iter_active = {}
            for op_entry in manifest["operators"]:
                op_name = op_entry["name"]
                iteration = int(op_entry["iteration"])
                op_tensors = {}
                for group_entry in op_entry["groups"]:
                    dtype = group_entry["dtype"]
                    total = group_entry["total_elems"]
                    flat_cpu = self._pool.acquire((total, ), dtype, pin=pin_cpu)
                    try:
                        dist.recv(flat_cpu, src=peer_global, group=group)
                    except Exception:
                        self._pool.release(flat_cpu)
                        raise
                    # Track for release on ``clear()`` alongside the ring's
                    # received flats — same lifecycle, same pool.
                    self._received_flat_buffers.append(flat_cpu)
                    for key, offset, shape in group_entry["layout"]:
                        n = 1
                        for s in shape:
                            n *= int(s)
                        op_tensors[key] = flat_cpu[offset:offset + n].view(shape)
                per_iter_operator_states.setdefault(iteration, {})[op_name] = op_tensors
                per_iter_active.setdefault(iteration, {})[op_name] = op_entry["is_active"]

        metadata = {
            "window_start_iteration": manifest["window_start_iteration"],
            "per_iter_active": per_iter_active,
        }
        if "fault_iter" in manifest:
            metadata["fault_iter"] = int(manifest["fault_iter"])
        if "loss_scaler_state" in manifest:
            metadata["loss_scaler_state"] = manifest["loss_scaler_state"]
        if "engine_scalars" in manifest:
            metadata["engine_scalars"] = manifest["engine_scalars"]
        return metadata, per_iter_operator_states

    def _pull_streaming(self, peer_global, group, on_iter_ready=None):
        """Receive the iter-major streaming variant of the pull protocol.

        Loop: recv length-prefixed mini-manifest, then that iter's
        flats; continue until the mini-manifest's ``is_last_iter`` is
        True.  Bundle-level fields ride the FIRST mini-manifest so the
        caller can commit to recovery state (cursor, scaler restore,
        operator freeze) after iter 0 lands rather than waiting for
        the full stream.

        If ``on_iter_ready`` is provided, invoke it per-iter as that
        iter's state becomes consumable.  The callback signature is
        ``on_iter_ready(iteration, iter_op_states, iter_per_op_active,
        bundle_fields_if_first)`` — the coordinator uses this to ingest
        iters into the converter from a pull thread while the main
        thread replays the earlier iters.  ``bundle_fields_if_first``
        is a metadata-shaped dict on the first iter only, ``None``
        thereafter.  When the callback is set the receiver does **not**
        accumulate the full ``per_iter_operator_states`` (the caller
        has consumed each iter already), and the return value carries
        only bundle-level metadata with an empty states dict so any
        accidental legacy consumer loudly sees "nothing to ingest".

        Without a callback the behavior is S1-compatible: accumulate
        and return ``(metadata, per_iter_operator_states)``.
        """
        pin_cpu = self._get_cuda_stream() is not None
        per_iter_operator_states = {}
        per_iter_active = {}
        window_start = -1
        fault_iter_val = None
        loss_scaler_state_val = None
        engine_scalars_val = None
        saw_first = False

        while True:
            with trace_range("recovery/pull_iter_manifest_recv"):
                length_tensor = torch.zeros(1, dtype=torch.int64)
                dist.recv(length_tensor, src=peer_global, group=group)
                length = int(length_tensor.item())
                if length == 0:
                    # Peer had nothing to ship — caller retries next
                    # peer.  Only legal at the very start of the stream
                    # (no iters received yet); a length=0 mid-stream
                    # would be a protocol violation the peer never
                    # sends, so we don't special-case it further.
                    if not saw_first:
                        return None, None
                    break
                payload = torch.zeros(length, dtype=torch.uint8)
                dist.recv(payload, src=peer_global, group=group)
            mini_manifest = torch.load(io.BytesIO(payload.numpy().tobytes()), weights_only=True)

            # Bundle-level fields ride iter 0.  Capture them BEFORE we
            # invoke the callback so the callback sees them on its
            # first call.
            bundle_fields = None
            if not saw_first:
                saw_first = True
                window_start = mini_manifest.get("window_start_iteration", -1)
                if "fault_iter" in mini_manifest:
                    fault_iter_val = int(mini_manifest["fault_iter"])
                if "loss_scaler_state" in mini_manifest:
                    loss_scaler_state_val = mini_manifest["loss_scaler_state"]
                if "engine_scalars" in mini_manifest:
                    engine_scalars_val = mini_manifest["engine_scalars"]
                bundle_fields = {"window_start_iteration": window_start}
                if "persisted_iters" in mini_manifest:
                    bundle_fields["persisted_iters"] = list(mini_manifest["persisted_iters"])
                if fault_iter_val is not None:
                    bundle_fields["fault_iter"] = fault_iter_val
                if loss_scaler_state_val is not None:
                    bundle_fields["loss_scaler_state"] = loss_scaler_state_val
                if engine_scalars_val is not None:
                    bundle_fields["engine_scalars"] = engine_scalars_val

            with trace_range("recovery/pull_iter_flats_recv"):
                iteration = int(mini_manifest["iteration"])
                iter_op_states = {}
                iter_active = {}
                for op_entry in mini_manifest["operators"]:
                    op_name = op_entry["name"]
                    op_tensors = {}
                    for group_entry in op_entry["groups"]:
                        dtype = group_entry["dtype"]
                        total = group_entry["total_elems"]
                        flat_cpu = self._pool.acquire((total, ), dtype, pin=pin_cpu)
                        try:
                            dist.recv(flat_cpu, src=peer_global, group=group)
                        except Exception:
                            self._pool.release(flat_cpu)
                            raise
                        if on_iter_ready is None:
                            self._received_flat_buffers.append(flat_cpu)
                        else:
                            self._received_flat_buffers_by_iter.setdefault(iteration, []).append(flat_cpu)
                        for key, offset, shape in group_entry["layout"]:
                            n = 1
                            for s in shape:
                                n *= int(s)
                            op_tensors[key] = flat_cpu[offset:offset + n].view(shape)
                    iter_op_states[op_name] = op_tensors
                    iter_active[op_name] = op_entry["is_active"]

            if on_iter_ready is None:
                per_iter_operator_states[iteration] = iter_op_states
                per_iter_active[iteration] = iter_active
            else:
                on_iter_ready(iteration, iter_op_states, iter_active, bundle_fields)

            if mini_manifest.get("is_last_iter"):
                break

        metadata = {
            "window_start_iteration": window_start,
            "per_iter_active": per_iter_active,
        }
        if fault_iter_val is not None:
            metadata["fault_iter"] = fault_iter_val
        if loss_scaler_state_val is not None:
            metadata["loss_scaler_state"] = loss_scaler_state_val
        if engine_scalars_val is not None:
            metadata["engine_scalars"] = engine_scalars_val
        return metadata, per_iter_operator_states

    def received_senders(self):
        """Return the dp_ranks whose snapshots we've received.

        Useful for a replacement rank coordinating a peer pull: any of
        these ranks holds the corresponding sender's state in pinned
        CPU memory and can re-ship it on request.
        """
        return list(self._received_snapshots.keys())

    def get_received_snapshots_for(self, sender_dp_rank):
        """Return ``(metadata, per_iter_operator_states)`` received from a specific sender.

        Returns ``(None, None)`` when no snapshot has been received from
        that sender — including the case where the sender sent an empty
        window (``window_start_iteration == -1`` with no operators).

        The result has the same shape as ``load_from_disk`` so the
        converter can consume it unchanged — ``per_iter_operator_states``
        is ``{iteration: {op_name: state_dict}}`` and the metadata dict
        carries a parallel ``per_iter_active`` map.
        """
        if sender_dp_rank not in self._received_snapshots:
            return None, None
        states = self._received_snapshots[sender_dp_rank]
        meta_by_iter_op = self._received_metadata.get(sender_dp_rank, {})
        window_start = self._received_window_start.get(sender_dp_rank, -1)
        if not states:
            return None, None
        per_iter_operator_states = {}
        per_iter_active = {}
        for (iteration, op_name), tensors in states.items():
            per_iter_operator_states.setdefault(iteration, {})[op_name] = tensors
            meta = meta_by_iter_op.get((iteration, op_name))
            assert meta is not None, (f"[MoEvement] sender={sender_dp_rank} has state for ({iteration}, {op_name}) "
                                      "but no metadata — recv path broke its atomic-commit invariant; defaulting to "
                                      "is_active=True would silently restore a frozen op into the FP32 master slot")
            per_iter_active.setdefault(iteration, {})[op_name] = meta["is_active"]
        metadata = {
            "window_start_iteration": window_start,
            "per_iter_active": per_iter_active,
        }
        return metadata, per_iter_operator_states

    def get_persisted_snapshots(self):
        """Get the most recently persisted sparse checkpoint snapshots.

        Returns:
            OrderedDict of ``(iteration, operator_name) -> OperatorSnapshot``.
        """
        return self._persisted_snapshots

    def get_current_snapshots(self):
        """Get the current in-progress sparse snapshots.

        Returns:
            OrderedDict of ``(iteration, operator_name) -> OperatorSnapshot``.
        """
        return self._snapshots

    def get_all_snapshots(self):
        """Get all snapshots (persisted + in-flight + current).

        Returns:
            OrderedDict of ``(iteration, operator_name) -> OperatorSnapshot``.
        """
        all_snaps = OrderedDict()
        all_snaps.update(self._persisted_snapshots)
        all_snaps.update(self._in_flight_snapshots)
        all_snaps.update(self._snapshots)
        return all_snaps

    def clear(self):
        """Clear all snapshots and free memory."""
        # Drain the snapshot stream before releasing buffers back to the
        # pool: an in-flight D2H may still be writing into them, and the
        # pool would hand them straight back out to the next acquire.
        self.synchronize()
        self._snapshots.clear()
        self._persisted_snapshots.clear()
        self._in_flight_snapshots.clear()
        self._received_snapshots.clear()
        self._received_metadata.clear()
        self._received_window_start.clear()
        stale_flats = self._received_flat_buffers
        self._received_flat_buffers = []
        for flat in stale_flats:
            self._pool.release(flat)
        # Streaming per-iter buffers (any iters not already drop-released
        # by the coordinator on a clean replay).  Same release ordering
        # as the bulk list above — synchronize was called at the top.
        stale_by_iter = self._received_flat_buffers_by_iter
        self._received_flat_buffers_by_iter = {}
        for bufs in stale_by_iter.values():
            for flat in bufs:
                self._pool.release(flat)

    def release_iter_buffers(self, iteration):
        """Release the streaming pull-side flat buffers for one iter.

        SD-O4 S3.  The coordinator's ``_drop_replayed_iter`` calls this
        after it has CUDA-event-synced the iter's H2D copies so the
        backing pool storage is no longer being read by an in-flight
        ``non_blocking=True`` copy.  No-op when the iter wasn't a
        streaming-pulled iter (e.g., bulk path or a never-arrived
        catch-up iter); the bulk flat list is freed by ``clear()`` at
        recovery end.
        """
        bufs = self._received_flat_buffers_by_iter.pop(iteration, None)
        if not bufs:
            return
        for flat in bufs:
            self._pool.release(flat)

    def save_to_disk(self, save_dir, tag, rank=0, extra_metadata=None):
        """Persist sparse snapshots to disk via the background worker.

        Writes a single bundle file per rank (``window_rank{rank}.pt``)
        containing a JSON header and raw tensor bytes.  Returns after
        enqueueing; the training thread does not block on serialization.
        Callers that need the file to be readable (tests, graceful
        shutdown) must call ``flush_persist()`` first.

        Each rank carries a different ZeRO-1/2 shard of the optimizer
        state and a different subset of the pipeline's modules, so the
        filename is rank-qualified — a shared ``window.pt`` would let
        every rank overwrite the same file in multi-rank runs, leaving
        only the last writer's shard on disk.  Mirrors the rank-keyed
        naming ``upstream_logging.save_to_disk`` already uses for
        activation/gradient logs.

        Only the *persisted* (last fully completed) window is saved.
        In-flight and current-window snapshots are deliberately
        excluded: mixing them would produce a bundle whose operators
        span multiple iteration boundaries, breaking the
        single-``window_start_iteration`` invariant that the replay
        loop relies on.  Recovery from this file always rewinds to the
        last completed window boundary — at most ``w_sparse``
        iterations of re-executed compute.

        Args:
            save_dir: Directory to save checkpoint files.
            tag: Checkpoint tag (e.g., global_step number).
            rank: Global rank of the calling process.  Defaults to 0
                for single-rank callers and tests that predate
                rank-qualified naming.
        """
        with trace_range("bundle_write"):
            checkpoint_dir = os.path.join(save_dir, str(tag), "moevement")
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.synchronize()

            all_snaps = self._persisted_snapshots
            metadata = {
                "window_start_iteration": self._window_start_iteration,
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            # Group by iteration so the on-disk bundle preserves the per-snapshot
            # payloads the replay loop needs (one frozen-op FP16 capture per iter
            # it was frozen, plus the active-iter's FP32 + optimizer state).  The
            # writer thread walks ``per_iter_snapshots`` in sorted-iter order;
            # snapshotting the state_dict references here pins them against
            # concurrent window rollover on the main thread.  The ``__moe_rng_state__``
            # pseudo-operator (injected by ``finalize_window``) rides this same
            # per-iter byte stream so save / replicate / peer-pull all carry it
            # uniformly.
            per_iter_snapshots = {}
            busy_buffers = []
            for (iteration, name), snap in all_snaps.items():
                per_iter_snapshots.setdefault(iteration, OrderedDict())[name] = {
                    "is_active": snap.is_active,
                    "state_dict": dict(snap.state_dict),
                    "fragment_info": dict(snap.fragment_info) if snap.fragment_info else None,
                }
                busy_buffers.extend(snap._flat_buffers)

            # Mark flat buffers busy so ``finalize_window`` won't reclaim
            # them while the worker is still reading their views.  The
            # completion callback clears the busy flag and releases them.
            for flat in busy_buffers:
                self._pool.mark_busy(flat)

            def _release(bufs=busy_buffers):
                for flat in bufs:
                    self._pool.release_busy(flat)

            bundle_path = os.path.join(checkpoint_dir, BUNDLE_FILENAME.format(rank=rank))
            try:
                fsync = self._fsync_on_save
                self._persist_worker.enqueue(
                    lambda path=bundle_path, md=metadata, sn=per_iter_snapshots, f=fsync: dump_bundle(
                        path, md, sn, fsync=f),
                    callback=_release,
                    label=bundle_path,
                )
            except Exception:
                # ``mark_busy`` has already run; if ``enqueue`` fails (worker
                # shut down, queue closed, etc.) the completion callback will
                # never fire and the buffers would stay stuck busy forever.
                # Release them inline before re-raising so the pool stays
                # balanced and the error surfaces to the caller.
                _release()
                raise

            logger.info(f"[MoEvement] Enqueued sparse checkpoint write to {bundle_path} "
                        f"with {len(all_snaps)} operator-iter snapshots "
                        f"across {len(per_iter_snapshots)} iterations")

    def flush_persist(self):
        """Block until every enqueued disk write has completed."""
        self._persist_worker.flush()

    @staticmethod
    def load_from_disk(load_dir, tag, rank=0):
        """Load sparse snapshots from disk.

        Reads the ``window_rank{rank}.pt`` bundle written by
        ``save_to_disk`` on the same rank.

        Args:
            load_dir: Directory containing checkpoint files.
            tag: Checkpoint tag.
            rank: Global rank whose shard to load.  Must match the
                rank passed to ``save_to_disk`` for the paired save.

        Returns:
            Tuple of (metadata dict, dict of operator name -> state_dict).
        """
        checkpoint_dir = os.path.join(load_dir, str(tag), "moevement")
        bundle_path = os.path.join(checkpoint_dir, BUNDLE_FILENAME.format(rank=rank))
        if not os.path.exists(bundle_path):
            return None, None

        metadata, per_iter_operator_states = load_bundle(bundle_path)

        total_ops = sum(len(ops) for ops in per_iter_operator_states.values())
        logger.info(f"[MoEvement] Loaded sparse checkpoint from {bundle_path} "
                    f"with {total_ops} operator-iter snapshots across "
                    f"{len(per_iter_operator_states)} iterations")

        return metadata, per_iter_operator_states
