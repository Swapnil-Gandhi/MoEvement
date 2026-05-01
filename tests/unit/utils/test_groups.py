# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed.comm as dist
from deepspeed.utils import groups
from deepspeed.utils.groups import _get_expert_parallel_ranks, reset_for_rebuild

from unit.common import DistributedTest


def test_get_expert_parallel_ranks():
    """
    Example - E + M + D parallel
    world_size = 16
    model_degree = 2
    expert_degree = 4 # number of experts in same group
    mp_group = [0, 1], [2,3], [4,5] ...
    data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
    expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
    expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]
    """
    expert_parallel_groups, expert_data_parallel_groups = _get_expert_parallel_ranks(world_size=16,
                                                                                     tensor_parallel_size_=2,
                                                                                     expert_parallel_size_=4)
    assert expert_parallel_groups == [
        [0, 2, 4, 6],
        [8, 10, 12, 14],
        [1, 3, 5, 7],
        [9, 11, 13, 15],
    ]
    assert expert_data_parallel_groups == [
        [0, 8],
        [2, 10],
        [4, 12],
        [6, 14],
        [1, 9],
        [3, 11],
        [5, 13],
        [7, 15],
    ]


class _SentinelGroup:
    """Placeholder for a real process group in module-global state.

    Using a named class makes failure messages more readable than raw
    ``object()``, and gives the destroy-call-log a meaningful
    ``repr()`` when a test fails.
    """

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"_SentinelGroup({self.label!r})"


def test_reset_for_rebuild_destroys_and_clears_all_group_globals(monkeypatch):
    """reset_for_rebuild aborts every cached group once and nulls the globals.

    Pins the core contract Layer 2 orchestration rests on: after the
    reset, the module-global short-circuits that gate re-init (e.g.
    ``if _ZERO_PARAM_INTRA_PARALLEL_GROUP is None``, ``if group_name
    not in _EXPERT_PARALLEL_GROUP``) all flip back to their first-init
    state, so the next call to the init helpers rebuilds against the
    fresh world instead of returning the stale cached group.
    """
    # Singletons swap cleanly via monkeypatch.setattr (restored
    # automatically when the test exits).
    monkeypatch.setattr(groups, "_TENSOR_MODEL_PARALLEL_GROUP", _SentinelGroup("tp"))
    monkeypatch.setattr(groups, "_MODEL_PARALLEL_GROUP", _SentinelGroup("mp"))
    monkeypatch.setattr(groups, "_DATA_PARALLEL_GROUP", _SentinelGroup("dp"))
    monkeypatch.setattr(groups, "_WORLD_GROUP", _SentinelGroup("world"))
    monkeypatch.setattr(groups, "_ZERO_PARAM_INTRA_PARALLEL_GROUP", _SentinelGroup("zero_intra"))

    # The dict globals are mutated in-place by ``reset_for_rebuild``
    # (``.clear()``), so ``monkeypatch.setattr`` isn't the right tool —
    # it would rebind the module attribute and leave the original dict
    # in a stale state for other tests.  Save + restore by hand via
    # try / finally.
    saved = {
        "ep": dict(groups._EXPERT_PARALLEL_GROUP),
        "ep_ranks": dict(groups._EXPERT_PARALLEL_GROUP_RANKS),
        "edp": dict(groups._EXPERT_DATA_PARALLEL_GROUP),
        "edp_ranks": dict(groups._EXPERT_DATA_PARALLEL_GROUP_RANKS),
        "a2a": dict(groups._ALL_TO_ALL_GROUP),
    }
    try:
        groups._EXPERT_PARALLEL_GROUP.clear()
        groups._EXPERT_PARALLEL_GROUP.update({"ep0": _SentinelGroup("ep0"), "ep1": _SentinelGroup("ep1")})
        groups._EXPERT_PARALLEL_GROUP_RANKS.clear()
        groups._EXPERT_PARALLEL_GROUP_RANKS.update({"ep0": [0, 1], "ep1": [2, 3]})
        groups._EXPERT_DATA_PARALLEL_GROUP.clear()
        groups._EXPERT_DATA_PARALLEL_GROUP.update({"edp0": _SentinelGroup("edp0")})
        groups._EXPERT_DATA_PARALLEL_GROUP_RANKS.clear()
        groups._EXPERT_DATA_PARALLEL_GROUP_RANKS.update({"edp0": [0, 2]})
        groups._ALL_TO_ALL_GROUP.clear()
        groups._ALL_TO_ALL_GROUP.update({"local_0": _SentinelGroup("a2a0")})

        destroyed = []
        reset_for_rebuild(destroy_fn=lambda pg: destroyed.append(pg))

        # Every seeded sub-group was passed to destroy_fn exactly once —
        # a dropped sub-group would leave a live NCCL/gloo communicator
        # hanging onto the departed rank's state post-rebuild.
        # ``_WORLD_GROUP`` ("world") is deliberately NOT in this set:
        # post the ``_clone_world_group`` simplification it caches the
        # default WORLD sentinel, which the survivor-rendezvous flow
        # owns teardown for, and which the launcher-driven rebuild
        # flow doesn't tear down at all.  Nulling without destroying
        # is verified by the separate ``groups._WORLD_GROUP is None``
        # assertion below.
        expected_labels = {"tp", "mp", "dp", "zero_intra", "ep0", "ep1", "edp0", "a2a0"}
        actual_labels = {pg.label for pg in destroyed}
        assert actual_labels == expected_labels
        assert len(destroyed) == len(expected_labels)

        # Globals cleared — the next init call must rebuild, not short-circuit.
        assert groups._TENSOR_MODEL_PARALLEL_GROUP is None
        assert groups._MODEL_PARALLEL_GROUP is None
        assert groups._DATA_PARALLEL_GROUP is None
        assert groups._WORLD_GROUP is None
        assert groups._ZERO_PARAM_INTRA_PARALLEL_GROUP is None
        assert groups._EXPERT_PARALLEL_GROUP == {}
        assert groups._EXPERT_PARALLEL_GROUP_RANKS == {}
        assert groups._EXPERT_DATA_PARALLEL_GROUP == {}
        assert groups._EXPERT_DATA_PARALLEL_GROUP_RANKS == {}
        assert groups._ALL_TO_ALL_GROUP == {}
    finally:
        groups._EXPERT_PARALLEL_GROUP.clear()
        groups._EXPERT_PARALLEL_GROUP.update(saved["ep"])
        groups._EXPERT_PARALLEL_GROUP_RANKS.clear()
        groups._EXPERT_PARALLEL_GROUP_RANKS.update(saved["ep_ranks"])
        groups._EXPERT_DATA_PARALLEL_GROUP.clear()
        groups._EXPERT_DATA_PARALLEL_GROUP.update(saved["edp"])
        groups._EXPERT_DATA_PARALLEL_GROUP_RANKS.clear()
        groups._EXPERT_DATA_PARALLEL_GROUP_RANKS.update(saved["edp_ranks"])
        groups._ALL_TO_ALL_GROUP.clear()
        groups._ALL_TO_ALL_GROUP.update(saved["a2a"])


def test_reset_for_rebuild_skips_none_entries(monkeypatch):
    """destroy_fn is only called on non-None singletons.

    First-time callers (before any init has run) have ``None`` in
    every singleton slot.  Passing ``None`` into a backend destroy
    raises; the reset must filter these out.
    """
    monkeypatch.setattr(groups, "_TENSOR_MODEL_PARALLEL_GROUP", None)
    monkeypatch.setattr(groups, "_MODEL_PARALLEL_GROUP", None)
    monkeypatch.setattr(groups, "_DATA_PARALLEL_GROUP", None)
    monkeypatch.setattr(groups, "_WORLD_GROUP", None)
    monkeypatch.setattr(groups, "_ZERO_PARAM_INTRA_PARALLEL_GROUP", None)

    destroyed = []
    reset_for_rebuild(destroy_fn=lambda pg: destroyed.append(pg))

    assert destroyed == []


class TestResetForRebuildEndToEnd(DistributedTest):
    """Single-rank empirical smoke for destroy + rebuild + collective.

    Mock tests elsewhere pin the contract (``destroy_fn`` called, globals
    cleared), but they never exercise real torch-distributed state.  This
    test runs under ``DistributedTest(world_size=1)`` with the gloo
    backend so it can call ``dist.destroy_process_group`` on a live
    group and verify a subsequent ``dist.new_group`` + collective works
    end-to-end.  The real fault-scenario verification (torchrun-elastic
    + SIGKILL, world-size change) still needs a sandbox; this is the
    closest useful check we can run locally.

    Gloo is chosen over NCCL because (a) collectives must run on CPU
    tensors for a machine without CUDA-visible devices in the worker
    subprocess, and (b) the idempotency we care about is in the
    Python-level bookkeeping (globals, group objects), not
    backend-specific semantics.
    """

    world_size = 1
    backend = 'gloo'
    reuse_dist_env = True

    def test_destroy_then_rebuild_cycle_works(self):
        """Seed globals with real gloo groups, reset via real destroy, rebuild, collective.

        Pins the L2.B prerequisite: ``reset_for_rebuild`` + fresh
        ``dist.new_group`` must produce a group on which collectives
        actually dispatch.  A bug in the ordering (e.g. destroying
        while holding a reference that gets reused) would surface here
        as a hang or backend-level error that mock-ordering tests
        can't catch.
        """
        # Save the original _WORLD_GROUP / _EXPERT_PARALLEL_GROUP state
        # so we can restore it after tearing them down mid-test.  The
        # DistributedTest harness keeps the process alive across tests
        # via reuse_dist_env; leaking our rebuild state would break
        # downstream fixtures.
        saved_world = groups._WORLD_GROUP
        saved_experts = dict(groups._EXPERT_PARALLEL_GROUP)
        try:
            groups._WORLD_GROUP = None  # force a fresh clone
            groups._EXPERT_PARALLEL_GROUP.clear()

            # Build a real subgroup and wire it into the _WORLD_GROUP slot
            # the same way first-time init does (through _clone_world_group).
            pre_rebuild_group = groups._clone_world_group()
            assert pre_rebuild_group is not None

            # Populate one expert-parallel dict slot so reset exercises
            # the dict-iteration path against a real group too.
            groups._EXPERT_PARALLEL_GROUP['smoke'] = dist.new_group(ranks=[0])

            # Run a collective on the pre-rebuild group — proves it's
            # healthy before we tear it down.  Single-rank all_reduce is
            # effectively the identity, but it exercises the full dispatch.
            tensor = torch.zeros(1)
            dist.all_reduce(tensor, group=pre_rebuild_group)

            # Now the rebuild half: real destroy on every seeded group,
            # then re-init from scratch.
            reset_for_rebuild(destroy_fn=dist.destroy_process_group)
            assert groups._WORLD_GROUP is None
            assert groups._EXPERT_PARALLEL_GROUP == {}

            # Re-clone world group — the ``is None`` short-circuit should
            # flip back to the build path and repopulate the cache with
            # the current default WORLD.  Post the ``_clone_world_group``
            # simplification (returns ``dist.group.WORLD`` rather than a
            # ``new_group``-built clone), pre_ and post_rebuild_group
            # resolve to the same sentinel — what matters is that the
            # cached reference is live and dispatches collectives.
            post_rebuild_group = groups._clone_world_group()
            assert post_rebuild_group is not None

            # The load-bearing assertion: the rebuilt cache carries a
            # collective.  If reset or re-clone left state in a bad
            # place, this would raise or hang.
            tensor = torch.zeros(1)
            dist.all_reduce(tensor, group=post_rebuild_group)
        finally:
            groups._WORLD_GROUP = saved_world
            groups._EXPERT_PARALLEL_GROUP.clear()
            groups._EXPERT_PARALLEL_GROUP.update(saved_experts)
