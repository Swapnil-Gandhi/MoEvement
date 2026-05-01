"""Buffer pool for MoEvement snapshots and upstream logs.

Two kinds of allocations dominate the hot path:

1. Pinned CPU tensors that receive D2H copies of operator state each
   sparse window.
2. Small device-side staging buffers used to pack many tensors into one
   contiguous region before a single ``cudaMemcpyAsync``.

Both are keyed by ``(shape, dtype)`` and churn at window cadence, so a
reuse pool avoids hitting the CUDA pinned/caching allocators on every
call.  Buffers held by asynchronous consumers (the ``PersistWorker``)
can be marked busy so the pool defers returning them to the free list
until the consumer releases them.
"""

import os
import threading
from collections import defaultdict

import torch

# Surfaces silent cudaHostAlloc on the hot path when prewarm-vs-acquire shapes diverge.
_POOL_MISS_LOG = os.environ.get("MOEVEMENT_POOL_MISS_LOG", "0") == "1"


class PinnedPool:
    """Reuse pool for pinned CPU and small device-side tensors.

    ``acquire(shape, dtype)`` returns a pinned CPU tensor by default.  Pass
    ``device=<torch.device>`` for a same-device tensor (used for GPU staging
    in the snapshot path); pass ``pin=False`` for a plain CPU buffer on
    accelerators that lack a pinned allocator.

    ``max_per_key=64`` is sized for typical MoE workloads
    (``max_active_ops_per_window`` × FP32-master / FP32-optim / FP16-compute
    dtype groups) — comfortable enough that steady-state acquire/release
    stays balanced and the cap only matters for transient surges.
    """

    def __init__(self, max_per_key=64):
        self._free = defaultdict(list)
        # Refcount under id(tensor) so multiple independent owners can each
        # mark_busy without one owner's release_busy prematurely freeing the
        # buffer.  Today disk-save and peer-replication serialize via
        # flush_persist; refactoring either to fully-async would otherwise
        # race on shared pinned memory.
        self._busy = defaultdict(int)
        self._max_per_key = max_per_key
        # When > 1, a missing acquire allocates grow_on_miss buffers (returns
        # one, pools the rest) so subsequent acquires for the same key skip
        # cudaHostAlloc.  Replaces projection-based init-time prewarm —
        # runtime shapes are the source of truth, so the pool can't diverge
        # from what _batched_d2h asks for.
        self._grow_on_miss = 1
        # Async peer replication acquires from a worker thread while the
        # training thread does the same for D2H staging.
        self._lock = threading.Lock()

    def set_max_per_key(self, count):
        """Resize the per-key free-list cap.  Existing buffers are kept; only
        future ``release`` calls observe the new cap."""
        with self._lock:
            self._max_per_key = max(1, int(count))

    def set_grow_on_miss(self, count):
        """Bulk-allocate ``count`` buffers per unique key on free-list miss.
        ``count=1`` is vanilla.  Tune to expected in-flight depth
        (persisted + in-flight + replication-backlog)."""
        self._grow_on_miss = max(1, int(count))

    @staticmethod
    def _key(shape, dtype, device, pin):
        device_tag = "cpu" if device is None else str(device)
        return (tuple(int(s) for s in shape), dtype, device_tag, bool(pin))

    def acquire(self, shape, dtype, *, device=None, pin=True):
        """Hand out (or allocate) a buffer matching ``(shape, dtype, device, pin)``.

        Args:
            shape: Target shape.
            dtype: Target dtype.
            device: ``None`` (default) returns a CPU buffer; any other device
                returns a same-device buffer.  ``pin`` is ignored when
                ``device`` is non-None.
            pin: When True and ``device is None``, the returned CPU buffer is
                page-locked so D2H/H2D transfers can use ``cudaMemcpyAsync``
                without staging through a bounce buffer.
        """
        pin_cpu = pin and device is None
        key = self._key(shape, dtype, device, pin_cpu)
        with self._lock:
            free_list = self._free[key]
            if free_list:
                return free_list.pop()
            # Cap pool refill by free-list headroom so a refilled-then-missed
            # key doesn't spin cudaHostAlloc on buffers that would immediately
            # be dropped past max_per_key.
            pool_room = max(0, self._max_per_key - len(free_list))
            pool_alloc = min(max(0, self._grow_on_miss - 1), pool_room)
        if _POOL_MISS_LOG:
            print(f"[moevement pool-miss] {key} alloc={pool_alloc + 1}", flush=True)
        # Allocate outside the lock — pinned allocation can stall the caller
        # and other threads shouldn't block on it.
        if device is None:
            result = torch.empty(shape, dtype=dtype, pin_memory=pin_cpu)
            pool_fresh = [torch.empty(shape, dtype=dtype, pin_memory=pin_cpu) for _ in range(pool_alloc)]
        else:
            result = torch.empty(shape, dtype=dtype, device=device)
            pool_fresh = [torch.empty(shape, dtype=dtype, device=device) for _ in range(pool_alloc)]
        if not pool_fresh:
            return result
        with self._lock:
            free_list = self._free[key]
            for t in pool_fresh:
                if len(free_list) >= self._max_per_key:
                    break
                free_list.append(t)
        return result

    def prewarm(self, shape, dtype, count=1, *, device=None, pin=True):
        """Pre-allocate ``count`` buffers ahead of first ``acquire`` so
        cold-start snapshot passes don't pay synchronous ``cudaMallocHost``
        inside ``on_iteration_end``.

        Idempotent beyond ``count`` (appends up to ``max_per_key``).
        """
        pin_cpu = pin and device is None
        key = self._key(shape, dtype, device, pin_cpu)
        with self._lock:
            existing = len(self._free[key])
        to_alloc = max(0, count - existing)
        fresh = []
        for _ in range(to_alloc):
            if device is None:
                fresh.append(torch.empty(shape, dtype=dtype, pin_memory=pin_cpu))
            else:
                fresh.append(torch.empty(shape, dtype=dtype, device=device))
        if not fresh:
            return
        with self._lock:
            free_list = self._free[key]
            for t in fresh:
                if len(free_list) >= self._max_per_key:
                    break
                free_list.append(t)

    def release(self, tensor):
        """Return a buffer to the free list unless a consumer still holds it.

        Idempotent — two independent paths can release the same flat (e.g.,
        the replication future's done-callback via ``release_busy``, and a
        later ``finalize_window`` via its snapshot ref).  Without the
        in-free-list guard below, double-append would let two ``acquire``
        calls hand out the same storage.
        """
        pin_cpu = bool(tensor.is_pinned()) if tensor.device.type == "cpu" else False
        device = None if tensor.device.type == "cpu" else tensor.device
        key = self._key(tensor.shape, tensor.dtype, device, pin_cpu)
        with self._lock:
            if self._busy.get(id(tensor), 0) > 0:
                return
            free_list = self._free[key]
            if any(f is tensor for f in free_list):
                return
            if len(free_list) >= self._max_per_key:
                return
            free_list.append(tensor)

    def mark_busy(self, tensor):
        """Increment the busy refcount; pairs with one ``release_busy``."""
        with self._lock:
            self._busy[id(tensor)] += 1

    def release_busy(self, tensor):
        """Decrement the busy refcount and return to the free list when zero."""
        with self._lock:
            tid = id(tensor)
            count = self._busy.get(tid, 0)
            if count <= 1:
                # Drop entry entirely so id() values don't leak across the
                # process lifetime.
                self._busy.pop(tid, None)
            else:
                self._busy[tid] = count - 1
        self.release(tensor)
