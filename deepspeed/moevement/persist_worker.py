"""Background writer for MoEvement persist jobs.

Runs disk persistence on a daemon thread so the training loop never
blocks on serialization + write.  The queue is FIFO and single-consumer;
order of writes is preserved within a rank.  Callers that need
durability (tests, graceful shutdown) call ``flush()`` to block until
all enqueued jobs are on disk.

A job is just ``(writer_callable, callback)``.  The worker calls
``writer_callable()`` — whether that's ``torch.save`` for upstream logs
or the custom bundle writer for sparse snapshots is the caller's choice.
"""

import queue
import threading
import time

from deepspeed.utils import logger


class PersistWorker:
    """Single-threaded background writer for disk-persist jobs.

    The queue is bounded (default ``max_queue_size=32``): if the disk
    writer falls behind training, ``enqueue`` blocks rather than letting
    the queue grow unboundedly.  Each queued job holds references to the
    caller's flat pinned buffers (the snapshot engine's lambda closure
    captures ``per_iter_snapshots``), and those buffers stay ``mark_busy``
    until the worker's done-callback runs — so an unbounded queue would
    pin unbounded host memory.  Bounded + blocking gives natural
    backpressure: when disk can't keep up, the training thread's next
    ``save_to_disk`` waits, signalling the operator that snapshot
    frequency should drop or disk throughput improve.

    A high-water-mark warning fires once each time the queue crosses
    75% of capacity and re-arms after it drains below 50% (hysteresis
    avoids log spam during steady back-pressure).
    """

    def __init__(self, max_queue_size=32, high_water_mark_ratio=0.75, rearm_ratio=0.5):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._high_water_threshold = max(1, int(max_queue_size * high_water_mark_ratio))
        self._rearm_threshold = max(0, int(max_queue_size * rearm_ratio))
        # Hysteresis flag: flipped True when the queue first crosses
        # ``_high_water_threshold``, reset to False after it drains below
        # ``_rearm_threshold``.  Prevents re-warning on every enqueue
        # while back-pressure is sustained.  **Training-thread-only**:
        # read and written exclusively from ``enqueue``; the worker
        # thread never touches it, so no locking is needed today.  If a
        # future change adds a worker-thread reader, add a ``threading.Lock``
        # around the read/flip pair.
        self._high_water_warned = False
        # Retain the first writer/callback exception seen since the last
        # ``flush`` so ``flush`` can re-raise it.  Without this, a failed
        # bundle or log write is only logged, ``save_sparse_checkpoint``
        # still returns success, and the outer ``save_checkpoint``
        # promotes the incomplete tag to ``latest``.
        self._error_lock = threading.Lock()
        self._first_error = None
        # Sticky flag: True if any writer/callback ever raised during this
        # worker's lifetime, even after ``flush`` cleared ``_first_error``.
        # Separate from ``_first_error`` because a save-path call that
        # successfully re-raised + was handled by the caller should still
        # leave a trace visible at process exit — ``shutdown`` reads this
        # to emit the critical banner for the atexit-only failure case
        # ``flush`` never sees.
        self._had_persist_error = False
        # Serializes ``_shutting_down`` flip + sentinel enqueue with
        # ``enqueue``'s flag-check + put attempt.  Closes the narrow M4
        # window where ``enqueue`` saw ``_shutting_down=False``, lost
        # the CPU, woke up after ``shutdown`` had flipped the flag and
        # queued the sentinel, and then ``put`` succeeded — the worker
        # would drain the late job after the sentinel and the caller
        # had no way to wait for it.
        self._shutdown_lock = threading.Lock()
        # Set by ``shutdown`` if ``_thread.join`` times out — the
        # worker thread is wedged on disk I/O and will leak on process
        # exit.  Surfaced via ``writer_is_stuck`` for callers that want
        # non-zero exit on partial-write incidents.
        self._writer_stuck = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="MoEvement-Persist")
        self._thread.start()
        self._shutting_down = False

    def _record_error(self, label, exc):
        with self._error_lock:
            self._had_persist_error = True
            if self._first_error is None:
                self._first_error = (label, exc)

    def writer_is_stuck(self):
        """True if ``shutdown`` saw the worker thread wedged past the join timeout.

        Indicates a leaked daemon thread that will outlive ``shutdown``;
        most often a stuck disk I/O (lost NFS mount, full disk blocking
        ``fsync``).  Process-exit caller can use this to set a non-zero
        exit code so the leak is visible to monitoring.
        """
        return self._writer_stuck

    def had_persist_error(self):
        """True if any writer/callback raised during this worker's lifetime.

        Sticky across ``flush`` — once a failure happens, the flag stays
        set even after ``flush`` has consumed and cleared ``_first_error``.
        Callers that want non-zero exit on partial-write incidents check
        this after their main loop returns.
        """
        with self._error_lock:
            return self._had_persist_error

    def _run(self):
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                writer, callback, label = job
                try:
                    writer()
                except Exception as exc:
                    # Log and keep the worker alive — losing one write is
                    # preferable to taking the whole training process down.
                    # ``flush`` re-raises the first error so callers that
                    # promote the on-disk artefact (``save_checkpoint`` →
                    # ``latest``) abort instead of publishing a partial write.
                    logger.error(f"[MoEvement] PersistWorker failed to write {label}: {exc}")
                    self._record_error(label, exc)
                if callback is not None:
                    try:
                        callback()
                    except Exception as exc:
                        logger.error(f"[MoEvement] PersistWorker callback raised on {label}: {exc}")
                        self._record_error(label, exc)
            finally:
                self._queue.task_done()

    def enqueue(self, writer, *, callback=None, label=None):
        """Queue a writer for background execution.

        Blocks if the queue is full (``max_queue_size`` jobs queued) — that's
        the intended backpressure: when disk can't keep up with training,
        the training thread waits here rather than pinning unbounded host
        memory via queued-but-not-yet-run writer closures.

        Args:
            writer: Zero-arg callable that performs the write.
            callback: Optional zero-arg callable invoked on the worker
                thread after ``writer`` returns — used by the snapshot
                engine to return pinned buffers to the pool once the
                write no longer references them.
            label: Optional string used in error messages.
        """
        qsize = self._queue.qsize()
        if not self._high_water_warned and qsize >= self._high_water_threshold:
            logger.warning(f"[MoEvement] PersistWorker queue at {qsize}/{self._max_queue_size} "
                           f"(crossed {self._high_water_threshold}) — disk writes falling behind "
                           f"training; reduce snapshot frequency or increase disk bandwidth")
            self._high_water_warned = True
        elif self._high_water_warned and qsize <= self._rearm_threshold:
            self._high_water_warned = False
        # Hold ``_shutdown_lock`` around the ``_shutting_down`` check + put_nowait
        # so ``shutdown`` can't slip its flag-flip + sentinel between the two
        # (M4 race).  If the queue is full, drop the lock so ``shutdown`` (and
        # the worker thread's drain) can make progress, sleep briefly, retake.
        # Worker silently dying is still caught: the sleep means an unbounded
        # put can't pin the training thread forever waiting on a dead worker.
        while True:
            with self._shutdown_lock:
                if self._shutting_down:
                    raise RuntimeError("PersistWorker is shutting down; no new jobs accepted")
                try:
                    self._queue.put_nowait((writer, callback, label))
                    return
                except queue.Full:
                    pass
            time.sleep(0.01)

    def flush(self):
        """Block until every enqueued job has been written.

        Raises ``RuntimeError`` (chained to the original exception) if
        any writer or callback raised since the last ``flush``.  Clears
        the stored error on return so a later retry/flush sees a clean
        state.

        If this raises, successful writes that preceded the failed one
        remain as partial files under the checkpoint tag directory —
        ``save_checkpoint`` refuses to promote the tag to ``latest`` so
        the directory is orphaned rather than corrupting the pointer.
        """
        self._queue.join()
        with self._error_lock:
            err = self._first_error
            self._first_error = None
        if err is not None:
            label, exc = err
            raise RuntimeError(f"PersistWorker failed to persist {label}") from exc

    def shutdown(self):
        """Flush pending work and tear down the worker thread.

        Writer / callback failures are suppressed during shutdown — the
        error is already logged by the worker, and swallowing it here
        keeps teardown on the happy path so the worker thread always
        exits cleanly (otherwise the ``atexit`` hook leaks it).

        After teardown, if any writer ever raised during this worker's
        lifetime (sticky ``_had_persist_error`` flag), emit a CRITICAL
        banner so an atexit-only cleanup failure doesn't slip through
        as a silent exit-0.  The ``save_checkpoint → flush_persist``
        path already surfaces errors correctly via the raise; this
        banner is specifically for the case where the only failure was
        on the final atexit flush and no caller got to see it.
        """
        try:
            self.flush()
        except Exception:
            pass
        # Hold the shutdown lock around the flag-flip + sentinel enqueue
        # so ``enqueue`` can't slip a job in between them (M4 race).
        # ``flush`` above drained the queue, so ``put`` here doesn't
        # block — no risk of holding the lock against a full queue.
        with self._shutdown_lock:
            self._shutting_down = True
            self._queue.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            # Worker is wedged on disk I/O that won't return (lost NFS,
            # full disk blocking ``fsync``).  The daemon thread will
            # leak until process exit, holding file handles + page cache
            # locks.  Surface a CRITICAL banner now and a sticky flag
            # for callers that want non-zero exit on the leak.
            self._writer_stuck = True
            logger.critical("[MoEvement] PersistWorker thread still alive after 5s shutdown; "
                            "writer is stuck on disk I/O.  Process will leak the thread on exit; "
                            "inspect disk state (full disk, lost NFS mount).")
        if self._had_persist_error:
            logger.critical("[MoEvement] PersistWorker saw at least one writer/callback failure during "
                            "this run — inspect earlier log lines for the label and traceback.  "
                            "Checkpoints already promoted to 'latest' are durable (save_checkpoint "
                            "raises at save time on write failure); this banner covers the atexit-only "
                            "cleanup path where no save_checkpoint caller got to see the error.")
