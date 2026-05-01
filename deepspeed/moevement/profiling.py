# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Lightweight profiling markers for the MoEvement hot path.

Without named markers on the iteration-end / snapshot / replay /
peer-replicate path, a profile trace from Nsight Systems or PyTorch
profiler shows only anonymous kernel launches — useful for hot-spot
hunting but not for answering ``which phase of MoEvement's state
machine is this time in?``.

``trace_range(name)`` is a coarse context manager that:

- Routes through ``torch.profiler.record_function`` so the name shows
  up in PyTorch profiler traces AND in NVTX (profiler's CUDA backend
  emits NVTX internally), matching Nsight, TensorBoard, and Chrome
  Trace viewers uniformly.
- Opt-in INFO logging when ``MOEVEMENT_TRACE=1`` is set, for local
  dev runs where launching under ``nsys`` is overkill.
- Records call names into a module-level list when ``_recording`` is
  True, which the marker-presence test toggles to verify wrap sites
  without attaching a real profiler.

Keep wraps **coarse**.  Per-parameter wrapping would multiply Python
call overhead across the tight snapshot loop; the call sites chosen
here live at operator or window granularity and cost O(1) per iter.
"""

import contextlib
import os
import time

import torch

# Dev-time opt-in log.  Keeps steady-state logs clean unless the
# operator explicitly turns tracing on.
_TRACE_LOG_ENABLED = os.environ.get("MOEVEMENT_TRACE", "0") == "1"

# Test hook: when set True the context manager appends qualified
# names to ``_record_log``.  The marker-presence test toggles this
# to verify wrap sites without launching a real profiler.
_recording = False
_record_log = []


@contextlib.contextmanager
def trace_range(name):
    """Emit a profiler-visible range around the enclosed block.

    Args:
        name: Short identifier. Prefixed with ``moevement/`` before
            emission so every MoEvement marker groups under one label
            in profiler views.
    """
    qualified = f"moevement/{name}"
    if _recording:
        _record_log.append(qualified)
    rec = torch.profiler.record_function(qualified)
    rec.__enter__()
    start = time.perf_counter() if _TRACE_LOG_ENABLED else None
    try:
        yield
    finally:
        rec.__exit__(None, None, None)
        if start is not None:
            # ``print`` instead of ``logger.info`` so the trace output
            # surfaces through ``mp.spawn``'d child processes' stdout —
            # logger-level INFO is suppressed by pytest capture and by
            # the default deepspeed logger config in fresh workers.
            print(f"[MoEvement TRACE] {qualified}: {(time.perf_counter() - start) * 1000:.2f}ms", flush=True)
