# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Empirical tests for ``deepspeed.moevement.comm_rebuild`` primitives.

Layer 1's unit tests (``test_sparse_checkpoint.py::TestCommGroupRebuildLayer1``)
monkeypatch ``_abort_or_destroy`` to stubs and verify the event-ordering
contract around it.  What's not pinned there is whether
``_abort_or_destroy`` actually *works* under the condition it was written
for: a real NCCL communicator wedged mid-collective because a peer stopped
participating.  This module closes that gap.

The single test here induces a genuine ring-allreduce wedge on a three-rank
subgroup (rank 2 deliberately doesn't participate), calls the primitive,
and asserts both (a) the abort branch was invoked (spy on
``_shutdown_backend_if_nccl``) and (b) the call returned well inside the
budget a wedged destroy would have exhausted.  A fresh subgroup + collective
after the abort pins that the NCCL communicator manager is still usable.

Scope is deliberately narrow — the engine-level rebuild-under-wedge case
belongs to the real-fault integration test, where a worker is actually
killed by the launcher.
"""

import os
import time

import pytest
import torch

import deepspeed.comm as dist

from unit.common import DistributedTest


class TestAbortOrDestroyUnderWedge(DistributedTest):
    """Abort fast path unblocks a real wedged NCCL collective.

    Setup: 4-rank world; create subgroup ``g = {0, 1, 2}``; ranks 0 and 1
    issue an async ``all_reduce`` on ``g`` while rank 2 stays silent.
    Ranks 0 and 1 now have an in-flight NCCL kernel that cannot progress
    (ring is missing rank 2).  The test then calls ``_abort_or_destroy(g)``
    and asserts it returns in under 10s with the NCCL-abort branch having
    fired — the destroy-timeout fallback would have consumed the full
    ``timeout_sec`` budget (30s here).

    Post-abort, a fresh subgroup across the same membership + a real
    collective verifies the NCCL communicator manager still works.  A
    regression where abort leaves NCCL in a poisoned state would land
    here rather than in mock-ordering tests.

    **Novelty vs ``TestCommGroupRebuildLayer1``:** those pin event
    ordering via monkeypatched stubs; this one exercises the real
    underlying NCCL abort path and is the only test in the repo that
    empirically proves the claim ``_abort_or_destroy`` rests on.
    """

    world_size = 4

    def test_abort_unblocks_wedged_nccl_allreduce(self, monkeypatch):
        # ``async_op=True`` becomes blocking under ``TORCH_NCCL_BLOCKING_WAIT=1``;
        # without async execution there is no wedge to abort and the test
        # premise is vacuous.  Skip cleanly rather than silently pass.
        if os.environ.get("TORCH_NCCL_BLOCKING_WAIT") == "1":
            pytest.skip("TORCH_NCCL_BLOCKING_WAIT=1 forces sync NCCL; can't induce an async wedge")

        # Defer the import so the monkeypatch target is the same module
        # object ``_abort_or_destroy`` looks up via its ``__globals__`` —
        # setting the attr on ``cr_mod`` then is effective for the call
        # from within ``_abort_or_destroy``.
        import deepspeed.moevement.comm_rebuild as cr_mod
        spy_calls = []
        orig_shutdown = cr_mod._shutdown_backend_if_nccl

        def spy_shutdown(pg):
            spy_calls.append(pg)
            return orig_shutdown(pg)

        monkeypatch.setattr(cr_mod, "_shutdown_backend_if_nccl", spy_shutdown)

        my_rank = dist.get_rank()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")  #ignore-cuda

        # ``new_group`` is itself a collective on WORLD — every rank must
        # call it even though only {0, 1, 2} are members.  Rank 3 gets a
        # non-member sentinel and must not touch the group afterwards.
        g = dist.new_group([0, 1, 2])

        # Warm up the NCCL communicator for ``g`` with every member
        # participating.  Without this, the first collective on ``g``
        # triggers a lazy ``ncclCommInitRank`` handshake that itself
        # needs all members present — rank 2 skipping that handshake
        # wedges ranks 0, 1 *inside bootstrap*, where the calling
        # thread blocks and ``async_op=True`` never returns a Work
        # handle.  A warm comm turns the subsequent rank-2-silent
        # allreduce into a kernel-level wedge (kernel launched on the
        # stream and spinning on the ring), which is the production
        # fault mode the abort primitive is supposed to handle.
        if my_rank in (0, 1, 2):
            warm = torch.zeros(1, device=device)
            dist.all_reduce(warm, group=g)
            torch.cuda.synchronize()  #ignore-cuda
            del warm
        dist.barrier()

        if my_rank in (0, 1):
            x = torch.ones(1 << 14, device=device)
            work = dist.all_reduce(x, group=g, async_op=True)
            # Give NCCL time to enqueue the kernel and start spinning on
            # the ring.  Aborting before the kernel is in flight would
            # test a pre-launch state, not the wedge-mid-collective case.
            time.sleep(1.0)
            assert not work.is_completed(), (
                f"allreduce on rank {my_rank} completed despite rank 2 not participating — "
                f"the wedge did not establish, so the abort-under-wedge claim below would be vacuous")
            # Drop the handle so the eventual ``ncclCommAbort`` isn't
            # racing against a user-visible ``.wait()``.
            del work
            del x
        elif my_rank == 2:
            # "Dead" rank: never issue the collective.  Match the 1s delay
            # so ranks 0, 1's assertion above sees a live wedge.
            time.sleep(1.0)

        if my_rank in (0, 1, 2):
            start = time.time()
            # ``timeout_sec=30`` is tight enough that the fallback path
            # (destroy on a still-wedged comm, which hits the 30s
            # timeout-thread cap) cannot masquerade as a successful
            # abort.  Abort + post-abort destroy together should complete
            # in < 1s; 10s is the CI-noise budget.
            cr_mod._abort_or_destroy(g, timeout_sec=30.0)
            elapsed = time.time() - start
            assert elapsed < 10.0, (f"_abort_or_destroy took {elapsed:.1f}s on rank {my_rank} with "
                                    f"timeout_sec=30 — exceeded 10s budget, abort fast path likely "
                                    f"did not fire and the call fell through to destroy-timeout")

        # The abort branch is the distinguishing behaviour of the primitive.
        # Ranks 0, 1 had a NCCL-backed group with a real wedge, so the
        # NCCL-backend-shutdown call must have been made.  Rank 2's group
        # had no in-flight op; the shutdown call fires there too (it's
        # idempotent on a clean comm), so this assertion is strongest on
        # the ranks where the call actually did work.
        if my_rank in (0, 1):
            assert len(spy_calls) >= 1, (f"rank {my_rank}: _shutdown_backend_if_nccl was not invoked — "
                                         f"_abort_or_destroy skipped the abort path entirely")

        # World barrier on the still-healthy default group — keeps the
        # four ranks in lockstep before the post-abort rebuild.  Using
        # WORLD here is safe because the wedged comm was a subgroup,
        # not WORLD itself.
        dist.barrier()

        # Post-abort sanity: a fresh subgroup across the same membership
        # must still produce correct results.  Proves the NCCL
        # communicator manager wasn't left in a state where subsequent
        # ``new_group`` calls inherit poisoned state from the aborted
        # comm.  A regression that leaves NCCL unhappy — e.g., torch's
        # internal group-index counter desync — would surface as a
        # hang or wrong result here.
        g2 = dist.new_group([0, 1, 2])
        if my_rank in (0, 1, 2):
            x = torch.ones(1 << 14, device=device)
            dist.all_reduce(x, group=g2)
            torch.cuda.synchronize()  #ignore-cuda
            assert torch.all(x == 3.0), (f"rank {my_rank} post-abort allreduce on fresh subgroup produced "
                                         f"{x[0].item()} != 3.0 — NCCL state was corrupted by the "
                                         f"prior wedge+abort sequence")
