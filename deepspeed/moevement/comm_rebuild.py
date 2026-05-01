# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Primitives for tearing down and rebuilding MoEvement process groups.

After a spare rank has joined the world (launcher-driven), existing NCCL /
gloo sub-groups hold dangling references to the departed rank.  Any
collective issued on them deadlocks.  ``destroy_process_group`` is the
normal teardown path, but it itself assumes the communicator is healthy —
if the dead rank left an in-flight op, destroy blocks forever.

``_abort_or_destroy`` closes this gap: on torch >= 2.6 and an NCCL
backend we call ``backend.abort()`` (cancels in-flight ops without a
healthy-peer round-trip); everywhere else (older torch, gloo) we fall
through to ``destroy_process_group`` wrapped in a bounded-wait thread
so a wedged destroy can't hang the caller forever.

Layer 1 only tears down gloo mirrors, so the abort path is forward-
compat for Layer 2 (which will tear down NCCL sub-groups).  The module
intentionally stays thin and backend-agnostic so both layers share it.

The torch-version gate (2.6 for ``ProcessGroup.abort`` as a stable
public API) is cribbed from ``nvidia_resiliency_ext/inprocess/abort.py``
— the NCCL-version gate mentioned in the original design doc was a
misread (``ncclCommAbort`` has been in NCCL since 2.2).
"""

import threading

import torch

import deepspeed.comm as dist
from deepspeed.utils import logger
from deepspeed.utils.torch import required_torch_version


def _supports_pg_abort():
    """True iff ``ProcessGroup`` backends expose stable ``abort()`` (torch >= 2.6).

    ``ProcessGroup.abort`` existed in torch before 2.6 but was marked
    experimental; torch 2.6 stabilised it as the public API.  Older
    torch exposes ``_shutdown`` with similar semantics — callers that
    want that fallback check :func:`_supports_pg_shutdown`.
    """
    return required_torch_version(min_version=2.6)


def _supports_pg_shutdown():
    """True on torch versions where NCCL backends expose ``_shutdown`` but not ``abort``.

    Pre-2.6 torch shipped ``_shutdown`` as the private equivalent of the
    later public ``abort``.  Kept available so the abort path doesn't
    silently fall all the way through to timeout-destroy on slightly
    older torch.
    """
    return not required_torch_version(min_version=2.6)


def _abort_or_destroy(pg, timeout_sec=120.0):
    """Best-effort teardown of a process group after world topology change.

    For NCCL backends, prefers ``backend.abort()`` (torch >= 2.6) or
    ``backend._shutdown()`` (older torch) — both cancel in-flight ops
    without needing a healthy peer round-trip, so they complete even
    when a departed rank left collectives wedged.  After that, or for
    non-NCCL backends, calls ``destroy_process_group`` wrapped in a
    bounded-wait thread so a wedged gloo destroy can't hang the caller
    forever.

    If destroy doesn't return within ``timeout_sec`` we log and give
    up — the underlying communicator leaks, but the caller moves on
    and rebuilds against the new world.

    Args:
        pg: Process group to tear down.  ``None`` is a no-op so
            callers can pass ``self._replication_group`` without
            guarding against the single-rank case upstream.
        timeout_sec: Upper bound on the destroy phase.  The abort
            phase is effectively instant and doesn't respect this.

    Returns:
        None.  Exceptions from the underlying torch APIs are caught
        and logged rather than raised — teardown failures must not
        prevent the caller from building the replacement group.
    """
    if pg is None:
        return

    _shutdown_backend_if_nccl(pg)
    _destroy_with_timeout(pg, timeout_sec)


def _shutdown_backend_if_nccl(pg):
    """Call ``abort``/``_shutdown`` on the CUDA backend if present.

    Duck-typed rather than ``isinstance(..., ProcessGroupNCCL)`` — checking
    the NCCL-specific class would require importing it from torch's
    private ``distributed_c10d`` module, which DeepSpeed's ``check-
    torchdist`` pre-commit forbids in favour of ``deepspeed.comm``.

    Gloo-only groups either have no CUDA backend (``_get_backend`` raises
    or returns a gloo-backed object without ``abort``/``_shutdown``), so
    the ``hasattr`` guards trigger the silent no-op path.
    """
    try:
        backend = pg._get_backend(torch.device("cuda"))  #ignore-cuda
    except Exception:
        return  # Not a CUDA-backed group (e.g., pure gloo).

    try:
        if _supports_pg_abort() and hasattr(backend, "abort"):
            backend.abort()
        elif _supports_pg_shutdown() and hasattr(backend, "_shutdown"):
            backend._shutdown()
    except Exception as exc:
        logger.warning(f"[MoEvement] comm_rebuild: backend abort/_shutdown raised ({exc}); "
                       f"falling through to destroy_process_group")


def _destroy_with_timeout(pg, timeout_sec):
    """Run ``destroy_process_group`` with a bounded wait.

    ``destroy_process_group`` has no native timeout; if the underlying
    communicator is wedged it can block forever.  We run it on a daemon
    helper thread and join with a timeout so a wedged destroy doesn't
    stall the caller.
    """

    def _do_destroy():
        try:
            dist.destroy_process_group(pg)
        except Exception as exc:
            logger.warning(f"[MoEvement] comm_rebuild: destroy_process_group raised ({exc})")

    thread = threading.Thread(target=_do_destroy, name="MoEvement-DestroyPG", daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        logger.error(f"[MoEvement] comm_rebuild: destroy_process_group did not return within "
                     f"{timeout_sec}s; communicator is leaked but rebuild will proceed")
