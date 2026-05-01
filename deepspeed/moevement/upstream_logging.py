"""Upstream logging for MoEvement localized recovery.

Logs intermediate activations and gradients at pipeline stage boundaries
during training. On failure, these logs enable localized recovery: only
the affected data-parallel group rolls back, using stored logs to replay
computations without requiring global rollback.
"""

import contextlib
import io
import os
from collections import defaultdict

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

from deepspeed.moevement.buffer_pool import PinnedPool
from deepspeed.moevement.persist_worker import PersistWorker
from deepspeed.moevement.profiling import trace_range


def _stream_context(stream):
    """Return an accelerator stream context, or a no-op if none is available."""
    if stream is None:
        return contextlib.nullcontext()
    return get_accelerator().stream(stream)


def _save_and_fsync(obj, path, fsync=True):
    """``torch.save(obj, path)`` + (optional) ``fsync`` so a rank death after
    the persist-worker callback doesn't silently lose the log.

    ``torch.save`` closes the file once its buffer reaches the page cache;
    the kernel may not flush for seconds.  Re-opening the path with
    ``os.open(..., O_RDONLY)`` and calling ``os.fsync`` on the fd before
    returning guarantees the bytes are durable.  Training blocks on the
    background writer only when the caller asks for ``flush_persist`` —
    regular per-window saves stay off the critical path.

    Pass ``fsync=False`` (driven by ``MoEvementConfig.fsync_on_save``) to
    skip the durability barrier on cloud VMs where the journaled SSD
    makes the explicit fsync redundant.
    """
    torch.save(obj, path)
    if not fsync:
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LogEntry:
    """A single logged tensor at a pipeline stage boundary.

    Attributes:
        iteration: Training iteration number.
        micro_batch_id: Microbatch index within the iteration.
        stage_id: Pipeline stage that produced this tensor.
        direction: 'activation' (forward) or 'gradient' (backward).
        tensor: The logged tensor (stored on CPU).
    """

    def __init__(self, iteration, micro_batch_id, stage_id, direction, tensor):
        self.iteration = iteration
        self.micro_batch_id = micro_batch_id
        self.stage_id = stage_id
        self.direction = direction
        self.tensor = tensor
        # Set once the tensor has been handed out (replay, serialization,
        # peer transfer).  ``garbage_collect`` skips the pool return path
        # for lent tensors so a still-held reference can't see its storage
        # reused by a later pool.acquire.
        self._lent_out = False

    def total_bytes(self):
        """Byte footprint of this log entry's tensor payload.

        Used for memory accounting when the logger reports aggregate
        retention against the paper §3.4 "< 2% of host memory" claim.
        """
        return self.tensor.nelement() * self.tensor.element_size()


class UpstreamLogger:
    """Manages logging of activations and gradients at pipeline stage boundaries.

    During training, copies tensors to pinned CPU memory using a dedicated CUDA
    stream. Logs are tagged with (iteration, microbatch_id, stage_id) for precise
    replay during recovery.
    """

    def __init__(self, max_window_iterations=10, fsync_on_save=True):
        self._logs = defaultdict(list)  # (iteration, micro_batch_id) -> [LogEntry]
        self._received_logs = defaultdict(list)  # (iteration, micro_batch_id) -> [LogEntry]
        # Per-iter staging buffer for log_d2h hot path.  Holding the
        # ``LogEntry`` alloc + dict insert + max() until end-of-iter
        # ``flush_pending`` cuts the per-microbatch python overhead
        # without changing the async D2H itself.
        self._pending = []  # list[(iteration, micro_batch_id, stage_id, direction, cpu_tensor)]
        self._cuda_stream = None
        self._max_window = max_window_iterations
        self._current_iteration = -1
        self._oldest_iteration = -1
        # Driven by ``MoEvementConfig.fsync_on_save``; passed through to
        # ``_save_and_fsync`` so log shards skip the durability barrier
        # in lockstep with bundle saves.
        self._fsync_on_save = fsync_on_save
        # Background worker keeps torch.save off the training thread.
        self._persist_worker = PersistWorker()
        # Activation/gradient shapes repeat every microbatch, so a small
        # pinned pool eliminates the per-log pinned allocation — the
        # hottest allocation path in the subsystem.  ``max_per_key`` and
        # ``grow_on_miss`` are pushed in by the coordinator after init
        # via ``MoEvementConfig.pool_max_per_key`` /
        # ``MoEvementConfig.pool_grow_on_miss_activation`` (auto-sized
        # from ``gas × num_moe_layers × w_sparse`` by default).
        self._pool = PinnedPool()

    def _get_cuda_stream(self):
        # CPU accelerators expose ``Stream`` as None; fall back to synchronous
        # copies there instead of crashing on ``None()``.
        if self._cuda_stream is None:
            stream_ctor = get_accelerator().Stream
            if stream_ctor is not None:
                self._cuda_stream = stream_ctor()
        return self._cuda_stream

    def log_activation(self, tensor, iteration, micro_batch_id, stage_id):
        """Log an activation tensor being sent to the next pipeline stage.

        The tensor is asynchronously copied to pinned CPU memory.

        Args:
            tensor: The activation tensor on GPU.
            iteration: Current training iteration.
            micro_batch_id: Current microbatch index.
            stage_id: Pipeline stage sending the activation.
        """
        self._log_tensor(tensor, iteration, micro_batch_id, stage_id, "activation")

    def log_gradient(self, tensor, iteration, micro_batch_id, stage_id):
        """Log a gradient tensor being sent to the previous pipeline stage.

        Args:
            tensor: The gradient tensor on GPU.
            iteration: Current training iteration.
            micro_batch_id: Current microbatch index.
            stage_id: Pipeline stage sending the gradient.
        """
        self._log_tensor(tensor, iteration, micro_batch_id, stage_id, "gradient")

    def _log_tensor(self, tensor, iteration, micro_batch_id, stage_id, direction):
        """Internal method to copy a tensor to CPU and store as a log entry.

        Logs every tensor regardless of dtype.  Pipeline activations often
        carry attention masks or position ids alongside the fp hidden
        state, and a future refactor that turns a tuple output into a
        single non-fp tensor — or a gradient-send path that ships
        ``(grad, mask)`` tuples — would silently break replay if we
        kept the old fp-only filter on the single-tensor branch.  The
        D2H cost of an extra mask/index tensor per iter is trivial
        relative to the float activation payload.
        """
        if tensor is None:
            return

        stream = self._get_cuda_stream()

        if isinstance(tensor, (tuple, list)):
            for i, t in enumerate(tensor):
                if t is not None and isinstance(t, torch.Tensor):
                    self._log_single_tensor(t, iteration, micro_batch_id, stage_id, f"{direction}_{i}", stream)
        elif isinstance(tensor, torch.Tensor):
            self._log_single_tensor(tensor, iteration, micro_batch_id, stage_id, direction, stream)

    def _log_single_tensor(self, tensor, iteration, micro_batch_id, stage_id, direction, stream):
        """Copy a single tensor to CPU asynchronously and record it.

        Hot path keeps only the work that has to happen per microbatch:
        the async D2H queueing onto the side stream.  ``LogEntry``
        allocation, ``self._logs`` dict insertion, and the
        ``self._current_iteration`` watermark update all run once per
        iter via ``flush_pending`` instead of per-microbatch.
        """
        with trace_range("log_d2h"):
            # Pinned memory + non_blocking copies only make sense when a CUDA-like
            # stream is available; on CPU accelerators fall back to a synchronous
            # copy so the destination tensor is valid immediately on return.
            async_copy = stream is not None
            if async_copy:
                stream.wait_stream(get_accelerator().current_stream())
            with _stream_context(stream):
                cpu_tensor = self._pool.acquire(tensor.shape, tensor.dtype, pin=async_copy)
                cpu_tensor.copy_(tensor, non_blocking=async_copy)

            self._pending.append((iteration, micro_batch_id, stage_id, direction, cpu_tensor))

    def synchronize(self):
        """Wait for all async tensor copies to complete and promote pending entries.

        Tests + callers that read ``_logs`` directly after ``synchronize``
        rely on this being the single drain point: GPU side stream
        finished + Python staging buffer flushed.  ``flush_pending`` is
        a no-op when empty, so adding it here doesn't penalise callers
        that don't use the deferred path.
        """
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
        self.flush_pending()

    def flush_pending(self):
        """Promote any per-microbatch ``_pending`` tuples into ``_logs``.

        Hot ``_log_single_tensor`` path appends to ``_pending`` without
        touching the dict / building ``LogEntry`` / updating
        ``_current_iteration``.  This method does that bookkeeping once
        per iter (called from coordinator's ``on_iteration_end``), or
        on demand from any reader that needs ``_logs`` to be current.

        Idempotent: returns immediately when there's nothing pending.
        """
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        max_iter = self._current_iteration
        for iteration, micro_batch_id, stage_id, direction, cpu_tensor in pending:
            entry = LogEntry(iteration=iteration,
                             micro_batch_id=micro_batch_id,
                             stage_id=stage_id,
                             direction=direction,
                             tensor=cpu_tensor)
            self._logs[(iteration, micro_batch_id)].append(entry)
            if iteration > max_iter:
                max_iter = iteration
        self._current_iteration = max_iter

    def get_logs_for_iteration(self, iteration):
        """Retrieve all log entries for a specific iteration.

        Args:
            iteration: The iteration number.

        Returns:
            Dict of micro_batch_id -> list of LogEntry.
        """
        self.flush_pending()
        result = defaultdict(list)
        for (iter_num, mb_id), entries in self._logs.items():
            if iter_num == iteration:
                result[mb_id].extend(entries)
        return dict(result)

    def get_activations_for_replay(self, iteration, micro_batch_id, stage_id):
        """Get activation logs for replaying a specific microbatch at a stage.

        Args:
            iteration: Iteration to replay.
            micro_batch_id: Microbatch to replay.
            stage_id: Stage whose activations to retrieve.

        Returns:
            List of CPU tensors containing the logged activations.
        """
        self.flush_pending()
        key = (iteration, micro_batch_id)
        entries = self._logs.get(key, [])
        activations = []
        for entry in entries:
            if entry.stage_id == stage_id and entry.direction.startswith("activation"):
                entry._lent_out = True
                activations.append(entry.tensor)
        return activations

    def get_gradients_for_replay(self, iteration, micro_batch_id, stage_id):
        """Get gradient logs for replaying a specific microbatch at a stage.

        Args:
            iteration: Iteration to replay.
            micro_batch_id: Microbatch to replay.
            stage_id: Stage whose gradients to retrieve.

        Returns:
            List of CPU tensors containing the logged gradients.
        """
        self.flush_pending()
        key = (iteration, micro_batch_id)
        entries = self._logs.get(key, [])
        gradients = []
        for entry in entries:
            if entry.stage_id == stage_id and entry.direction.startswith("gradient"):
                entry._lent_out = True
                gradients.append(entry.tensor)
        return gradients

    def garbage_collect(self, oldest_valid_iteration, protected_iterations=None):
        """Remove stale log entries from before the given iteration.

        Called after a new sparse checkpoint is persisted to free memory.

        ``protected_iterations`` lets the caller pin specific iterations
        that would otherwise be GC'd — needed for peer-pull recovery, where
        the live neighbour must retain logs covering the last persisted
        snapshot window (which can be ``w_sparse`` iters older than the
        current rolling window the default policy keeps).

        Args:
            oldest_valid_iteration: Logs from before this iteration are removed.
            protected_iterations: Optional iterable of iterations to retain
                regardless of ``oldest_valid_iteration``.
        """
        self.flush_pending()
        protected = set(protected_iterations or ())
        stale_keys = [key for key in self._logs if key[0] < oldest_valid_iteration and key[0] not in protected]
        if stale_keys:
            # An in-flight D2H copy from before gc was triggered can still
            # be writing to one of these buffers.  Releasing to the pool
            # without syncing would let the next ``pool.acquire`` hand that
            # same storage to a fresh log, at which point the tail of the
            # old copy clobbers the new log's payload.
            self.synchronize()
        freed_bytes = 0
        for key in stale_keys:
            for entry in self._logs[key]:
                freed_bytes += entry.total_bytes()
                # Return the pinned buffer to the pool so the next window's
                # log at this shape is served without a pin_memory alloc.
                # Lent-out tensors are left for Python's GC to reclaim —
                # the consumer may still hold a reference, and reusing
                # that storage via pool.acquire would silently corrupt it.
                if not entry._lent_out:
                    self._pool.release(entry.tensor)
            del self._logs[key]

        if stale_keys:
            self._oldest_iteration = oldest_valid_iteration
            logger.info(f"[MoEvement] Garbage collected {len(stale_keys)} log entries "
                        f"(freed {freed_bytes / (1024**2):.1f} MB)")

    def total_memory_bytes(self):
        """Compute total CPU memory used by all stored logs."""
        self.flush_pending()
        total = 0
        for entries in self._logs.values():
            for entry in entries:
                total += entry.total_bytes()
        return total

    def send_logs_to(self, target_rank, stage_id, direction_filter, iteration_range, group=None):
        """Send matching log entries to a recovering pipeline stage.

        Serialises all entries whose stage_id and direction prefix match the
        filters and whose iteration falls within iteration_range, then ships
        them as a two-message transfer: an int64 byte-count followed by the
        serialised payload as a uint8 tensor.

        Must be paired with a ``recv_logs_from`` call on the target rank.

        Args:
            target_rank: Global rank to send to.  ``dist.send`` /
                ``dist.recv`` take global ranks regardless of ``group=``
                — if your caller has group-local ranks, translate via
                ``dist.get_global_rank(group, local_rank)`` first.
            stage_id: Only include entries produced by this stage.
            direction_filter: Only include entries whose direction starts with
                this prefix (e.g. ``"activation"`` or ``"gradient"``).
            iteration_range: Iterable of iteration numbers to include.
            group: Optional process group.  The payload is a CPU tensor
                (serialised log bundle) so this must be a gloo-backed group;
                the default NCCL group rejects CPU tensors.
        """
        with trace_range("log_send"):
            # Log tensors live in pinned CPU buffers fed by an async D2H copy
            # on a side stream.  The peer-pull path runs immediately after a
            # fault is detected — often within a few iters of when the copies
            # were kicked off — so without this sync ``torch.save`` can read
            # partial bytes and ship stale data to the recovering rank.
            # ``save_to_disk`` already syncs at its top; keep the contract
            # symmetric here.
            self.synchronize()
            self.flush_pending()
            valid_iters = set(iteration_range)
            entries = []
            for (iter_num, mb_id), entry_list in self._logs.items():
                if iter_num not in valid_iters:
                    continue
                for entry in entry_list:
                    if entry.stage_id == stage_id and entry.direction.startswith(direction_filter):
                        entries.append({
                            "iteration": entry.iteration,
                            "micro_batch_id": entry.micro_batch_id,
                            "stage_id": entry.stage_id,
                            "direction": entry.direction,
                            "tensor": entry.tensor,
                        })

            buf = io.BytesIO()
            torch.save(entries, buf)
            payload = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)

            # Send length first so the receiver can allocate the right buffer.
            length_tensor = torch.tensor([payload.numel()], dtype=torch.int64)
            dist.send(length_tensor, dst=target_rank, group=group)
            dist.send(payload, dst=target_rank, group=group)

            logger.info(f"[MoEvement] Sent {len(entries)} log entries "
                        f"(stage={stage_id}, dir={direction_filter}) to rank {target_rank}")

    def recv_logs_from(self, src_rank, group=None):
        """Receive log entries sent by a neighbouring pipeline stage.

        Receives the two-message payload produced by ``send_logs_to`` and
        stores the entries in ``_received_logs`` for replay.

        Args:
            src_rank: Global rank to receive from.  ``dist.send`` /
                ``dist.recv`` take global ranks regardless of ``group=``
                — if your caller has group-local ranks, translate via
                ``dist.get_global_rank(group, local_rank)`` first.
            group: Optional process group — must be gloo-backed because
                the transfer uses CPU tensors.
        """
        length_tensor = torch.zeros(1, dtype=torch.int64)
        dist.recv(length_tensor, src=src_rank, group=group)

        payload = torch.zeros(length_tensor.item(), dtype=torch.uint8)
        dist.recv(payload, src=src_rank, group=group)

        # ``weights_only=True`` keeps unpickling limited to tensors and
        # plain containers — the entry dicts we serialised above.  Guards
        # against arbitrary-code execution from a malicious sender.
        entries = torch.load(io.BytesIO(payload.numpy().tobytes()), weights_only=True)
        for e in entries:
            key = (e["iteration"], e["micro_batch_id"])
            self._received_logs[key].append(
                LogEntry(
                    iteration=e["iteration"],
                    micro_batch_id=e["micro_batch_id"],
                    stage_id=e["stage_id"],
                    direction=e["direction"],
                    tensor=e["tensor"],
                ))

        logger.info(f"[MoEvement] Received {len(entries)} log entries from rank {src_rank}")

    def get_received_activation(self, iteration, micro_batch_id):
        """Return received activation tensors for replay, or None if unavailable.

        Args:
            iteration: Training iteration to retrieve.
            micro_batch_id: Microbatch index to retrieve.

        Returns:
            List of CPU tensors, or None if no entry exists.
        """
        key = (iteration, micro_batch_id)
        entries = self._received_logs.get(key)
        if not entries:
            return None
        tensors = [e.tensor for e in entries if e.direction.startswith("activation")]
        return tensors if tensors else None

    def get_received_gradient(self, iteration, micro_batch_id):
        """Return received gradient tensors for replay, or None if unavailable.

        Args:
            iteration: Training iteration to retrieve.
            micro_batch_id: Microbatch index to retrieve.

        Returns:
            List of CPU tensors, or None if no entry exists.
        """
        key = (iteration, micro_batch_id)
        entries = self._received_logs.get(key)
        if not entries:
            return None
        tensors = [e.tensor for e in entries if e.direction.startswith("gradient")]
        return tensors if tensors else None

    def save_to_disk(self, save_dir, tag, rank):
        """Persist this rank's output logs so they survive a whole-job restart.

        Each rank writes a single file; there is no cross-rank coordination.
        After ``load_from_disk`` the entries live in ``_logs`` and can be
        shipped to neighbouring stages by ``recovery_barrier`` the same way
        in-memory logs would be.

        Args:
            save_dir: Checkpoint directory (same one as the sparse snapshot).
            tag: Checkpoint tag (e.g., global_step number).
            rank: Global rank of this process (used to name the file).
        """
        with trace_range("log_save_to_disk"):
            self.synchronize()
            self.flush_pending()
            logs_dir = os.path.join(save_dir, str(tag), "moevement")
            os.makedirs(logs_dir, exist_ok=True)

            entries = []
            for entry_list in self._logs.values():
                for entry in entry_list:
                    entries.append({
                        "iteration": entry.iteration,
                        "micro_batch_id": entry.micro_batch_id,
                        "stage_id": entry.stage_id,
                        "direction": entry.direction,
                        "tensor": entry.tensor,
                    })

            filepath = os.path.join(logs_dir, f"upstream_logs_rank{rank}.pt")
            fsync = self._fsync_on_save
            self._persist_worker.enqueue(
                lambda p=filepath, e=entries, f=fsync: _save_and_fsync(e, p, fsync=f),
                label=filepath,
            )
            logger.info(f"[MoEvement] Enqueued {len(entries)} upstream log entries for {filepath}")

    def flush_persist(self):
        """Block until every enqueued log file has been written."""
        self._persist_worker.flush()

    def load_from_disk(self, load_dir, tag, rank):
        """Load this rank's output logs from disk.

        Returns True if a log file was found and loaded; False if the file
        does not exist (e.g. older checkpoints without persisted logs).

        Args:
            load_dir: Checkpoint directory.
            tag: Checkpoint tag.
            rank: Global rank of this process.
        """
        filepath = os.path.join(load_dir, str(tag), "moevement", f"upstream_logs_rank{rank}.pt")
        if not os.path.exists(filepath):
            return False

        # Safe unpickling: the on-disk log bundle is a list of plain
        # dicts containing tensors — no custom classes to load.
        entries = torch.load(filepath, weights_only=True)
        for e in entries:
            key = (e["iteration"], e["micro_batch_id"])
            self._logs[key].append(
                LogEntry(
                    iteration=e["iteration"],
                    micro_batch_id=e["micro_batch_id"],
                    stage_id=e["stage_id"],
                    direction=e["direction"],
                    tensor=e["tensor"],
                ))
        logger.info(f"[MoEvement] Loaded {len(entries)} upstream log entries from {filepath}")
        return True

    def clear(self):
        """Clear all log entries and free memory."""
        # Drain the log stream before dropping dict entries: an in-flight
        # D2H from _log_single_tensor may still be writing into a tensor
        # whose only reference is one of these dicts.  GC after drop would
        # let torch reuse the storage before the D2H lands.
        self.synchronize()
        self._pending = []
        self._logs.clear()
        self._received_logs.clear()
        self._current_iteration = -1
        self._oldest_iteration = -1
