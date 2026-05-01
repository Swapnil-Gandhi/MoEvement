# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Gate experiment for the streaming-recovery design.

The streaming design hides the peer-pull wire leg behind replay by
running ``dist.recv`` for iter N+1 on a background thread while the
main thread runs replay for iter N.  That only works if gloo tolerates
recv/send calls from a non-main thread, and tolerates two concurrent
recvs on the same process group distinguished by ``tag``.  PyTorch
doesn't guarantee either: the c10d Python shim marshals ops to a
single backend work queue, and gloo's thread-safety story is
underdocumented.

This probe is the blocking precondition: if it deadlocks or corrupts
payloads, S1-S3 do not proceed and we fall back to a
serialized-pull-pool design.

Opt-in only — set ``MOEVEMENT_RUN_GLOO_THREADSAFETY=1`` to include it
in a run.  Kept out of the default suite because it's a gate
experiment, not a regression test, and it would add minutes to every
CI pass for no ongoing coverage value.
"""

import os
import threading

import pytest
import torch

import deepspeed.comm as dist

from unit.common import DistributedTest

_ENABLE_ENV = "MOEVEMENT_RUN_GLOO_THREADSAFETY"
_ITERS = int(os.environ.get("MOEVEMENT_GLOO_THREADSAFETY_ITERS", "50"))
_PAYLOAD_FLOATS = int(os.environ.get("MOEVEMENT_GLOO_THREADSAFETY_PAYLOAD", "1024"))


def _make_gloo_world_group():
    return dist.new_group(ranks=list(range(dist.get_world_size())), backend='gloo')


@pytest.mark.skipif(os.environ.get(_ENABLE_ENV, "0") != "1", reason=f"Gate experiment; set {_ENABLE_ENV}=1 to enable")
class TestGlooThreadSafetyProbe(DistributedTest):
    """Probe whether gloo tolerates the streaming-recovery thread pattern."""

    world_size = 2

    def test_background_thread_recv(self):
        """Single background-thread recv — narrower precondition.

        Verifies gloo accepts ``dist.recv`` from a non-main thread at
        all.  If this fails, the streaming design is dead on arrival
        even before we consider concurrency.
        """
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()

        for iter_idx in range(_ITERS):
            expected = torch.arange(_PAYLOAD_FLOATS, dtype=torch.float32) + iter_idx * 1000.0
            if rank == 0:
                got = torch.empty(_PAYLOAD_FLOATS, dtype=torch.float32)
                err = {}

                def _recv():
                    try:
                        dist.recv(got, src=1, group=gloo_group)
                    except Exception as e:  # noqa: BLE001 — surface the type
                        err["exc"] = e

                t = threading.Thread(target=_recv, name=f"probe-recv-{iter_idx}")
                t.start()
                t.join(timeout=30.0)
                assert not t.is_alive(), f"iter {iter_idx}: background recv deadlocked"
                assert "exc" not in err, f"iter {iter_idx}: background recv raised {err.get('exc')!r}"
                torch.testing.assert_close(got, expected)
            else:
                dist.send(expected, dst=0, group=gloo_group)

    def test_two_concurrent_recvs_with_tags(self):
        """Two concurrent recvs distinguished by tag — the full precondition.

        Rank 1 sends two payloads back-to-back with tags 1 and 2.
        Rank 0 spawns two threads, each waiting on its own tag.  If
        gloo's backend queue interleaves them correctly, each thread
        sees only its own tag's bytes; if gloo can't fan out by tag
        concurrently, we either deadlock or cross-deliver.
        """
        rank = dist.get_rank()
        gloo_group = _make_gloo_world_group()

        for iter_idx in range(_ITERS):
            base = iter_idx * 1000.0
            payload_a = torch.arange(_PAYLOAD_FLOATS, dtype=torch.float32) + base
            payload_b = torch.arange(_PAYLOAD_FLOATS, dtype=torch.float32) + base + 500.0

            if rank == 0:
                got_a = torch.empty(_PAYLOAD_FLOATS, dtype=torch.float32)
                got_b = torch.empty(_PAYLOAD_FLOATS, dtype=torch.float32)
                errs = {}

                def _recv(buf, tag, key):
                    try:
                        dist.recv(buf, src=1, group=gloo_group, tag=tag)
                    except Exception as e:  # noqa: BLE001
                        errs[key] = e

                t_a = threading.Thread(target=_recv, args=(got_a, 1, "a"), name=f"probe-a-{iter_idx}")
                t_b = threading.Thread(target=_recv, args=(got_b, 2, "b"), name=f"probe-b-{iter_idx}")
                t_a.start()
                t_b.start()
                t_a.join(timeout=30.0)
                t_b.join(timeout=30.0)
                assert not t_a.is_alive(), f"iter {iter_idx}: thread-a deadlocked"
                assert not t_b.is_alive(), f"iter {iter_idx}: thread-b deadlocked"
                assert not errs, f"iter {iter_idx}: recv raised {errs!r}"
                torch.testing.assert_close(got_a, payload_a)
                torch.testing.assert_close(got_b, payload_b)
            else:
                dist.send(payload_a, dst=0, group=gloo_group, tag=1)
                dist.send(payload_b, dst=0, group=gloo_group, tag=2)
