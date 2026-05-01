# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for ``DeepSpeedEngine.rebuild_nccl_groups`` (Layer 2.B).

Layer 2.B of the MoEvement comm-group rebuild design
(`docs/moevement/DESIGN_COMM_GROUP_REBUILD.md`) composes the L2.A
primitives (`groups.reset_for_rebuild`, `optimizer.relink_all_dp_refs`)
with a re-run of the engine-level distributed-model init so that every
cached process-group reference across DeepSpeed + MoEvement gets
repointed at fresh communicators after a spare-rank substitution.

These tests pin the orchestration CALL ORDER — the one invariant the
design says must hold:

    quiesce → abort+clear (reset_for_rebuild)
           → re-init engine groups
           → rebuild pipe-grid sub-groups (pipe override; no-op in base)
           → refresh engine-cached attrs
           → relink optimizer + lp_params
           → rebuild MoEvement gloo mirrors (Layer 1)

Reversing any two of those steps either (a) runs collectives on a
torn-down communicator (quiesce must come first) or (b) leaves some
cache linked to the old group.  The pipe-grid rebuild slot is
load-bearing: ``seq_data_parallel_group`` resolves to
``grid.dp_proc_group`` via the mpu chain, so relinking before the
grid is rebuilt binds the optimizer to a group that the subsequent
grid rebuild would destroy.  Mirror rebuild must come last overall
because it reads ``self.seq_data_parallel_group``.

The tests mock every step with a stub that appends to an ordering
list; an integration test with real groups + real fault injection is
the verification this cannot provide and is gated on sandbox
availability (see design §6 Week 3).
"""

import types

from deepspeed.runtime.engine import DeepSpeedEngine


class _FakeOptimizer:
    """Stands in for ``ZeROOptimizer`` in call-order tests."""

    def __init__(self, calls):
        self._calls = calls

    def relink_all_dp_refs(self,
                           new_dp_group,
                           new_expert_parallel_group=None,
                           new_expert_dp_groups=None,
                           new_model_parallel_group=None):
        self._calls.append(("optimizer.relink_all_dp_refs", new_dp_group, new_expert_parallel_group,
                            new_expert_dp_groups, new_model_parallel_group))


class _FakeCoordinator:
    """Stands in for ``MoEvementCoordinator`` in call-order tests."""

    def __init__(self, calls, timeout_sec=120.0):
        self._calls = calls
        self.config = types.SimpleNamespace(comm_rebuild_timeout_sec=timeout_sec)

    def _quiesce_pending_replication_and_persist(self, timeout_sec):
        self._calls.append(("coordinator._quiesce_pending_replication_and_persist", timeout_sec))

    def rebuild_comm_groups(self, new_dp_group, new_pp_group=None):
        self._calls.append(("coordinator.rebuild_comm_groups", new_dp_group, new_pp_group))


def _make_fake_engine(calls, *, with_coordinator=True, with_optimizer=True, with_relink=True):
    """Bare-minimum object with the attrs/methods ``rebuild_nccl_groups`` touches.

    Using a ``types.SimpleNamespace`` rather than a real ``DeepSpeedEngine``
    instance lets the test pin call ordering without bringing up
    distributed + the full engine construction path.  Methods we want
    recorded get replaced with stubs that append to ``calls``;
    ``rebuild_nccl_groups`` itself is fetched from the real class and
    bound to the fake.
    """
    fake = types.SimpleNamespace()
    fake.moevement_coordinator = _FakeCoordinator(calls) if with_coordinator else None
    fake.optimizer = _FakeOptimizer(calls) if (with_optimizer
                                               and with_relink) else (_StubNoRelink() if with_optimizer else None)
    # Pre-refresh values — rebuild_nccl_groups will overwrite these via
    # _refresh_group_caches.  The stub for _refresh_group_caches doesn't
    # actually touch them; it just records the call.
    fake.expert_parallel_group = None
    fake.expert_data_parallel_group = None
    fake.has_moe_layers = False
    fake.seq_data_parallel_group = "stub_dp_group"

    # Stub out the helpers that would normally re-run init + cache refresh.
    fake._reinitialize_distributed_groups = lambda: calls.append(("engine._reinitialize_distributed_groups", ))
    fake._rebuild_pipe_sub_groups = lambda: calls.append(("engine._rebuild_pipe_sub_groups", ))
    fake._refresh_group_caches = lambda: calls.append(("engine._refresh_group_caches", ))
    fake._rebuild_get_pipeline_parallel_group = lambda: (calls.append(
        ("engine._rebuild_get_pipeline_parallel_group", )) or "stub_pp_group")

    return fake


class _StubNoRelink:
    """Optimizer without ``relink_all_dp_refs`` — exercises the graceful skip path."""

    pass


def test_rebuild_nccl_groups_call_order_with_moevement_and_zero1(monkeypatch):
    """Pins the full orchestration call order under the MoEvement + ZeRO-1 config.

    Expected sequence:

    1. ``coordinator._quiesce_pending_replication_and_persist`` — drain
       in-flight replication + disk-persist before anything touches
       the communicators.
    2. ``groups.reset_for_rebuild`` — abort + null every cached
       ProcessGroup global, using ``_abort_or_destroy`` as the
       injected ``destroy_fn``.
    3. ``engine._rebuild_pipe_sub_groups`` — pipe-grid sub-group
       rebuild hook.  No-op in the base engine; ``PipelineEngine``
       overrides to tear down + re-create its four sub-group
       families.  Must precede MoE re-init so the
       ``dist.new_group`` call order on survivors matches the cold-
       start order on a spare rank (grid-then-MoE) — NCCL's store-
       key prefix is derived from the per-PG creation counter, and
       any order drift makes counter N point at different semantic
       groups across ranks.
    4. ``engine._reinitialize_distributed_groups`` — re-runs the
       MoE ``set_deepspeed_parallelism`` pass.
    5. ``engine._refresh_group_caches`` — re-reads ``groups._get_*``
       into the engine-cached attrs.
    6. ``optimizer.relink_all_dp_refs`` — swaps the optimizer's
       ``dp_process_group`` / ``expert_dp_process_group`` +
       every ``lp_param._dp_group``.
    7. ``engine._rebuild_get_pipeline_parallel_group`` — fetch the
       fresh pp group for the Layer-1 rebuild.
    8. ``coordinator.rebuild_comm_groups`` — Layer 1 gloo-mirror
       rebuild against the fresh dp + pp base groups.

    Reversing any two breaks correctness — e.g. relinking before
    reset would point the optimizer at the old (about-to-be-aborted)
    group.  Pin ordering here so a future refactor that "just tidies
    up" the method can't silently re-order the steps.
    """
    # Module-level ``groups.reset_for_rebuild`` is reached by
    # ``rebuild_nccl_groups``'s direct call; swap it for a recording
    # stub.  ``monkeypatch`` restores the real implementation after
    # the test returns.
    from deepspeed.utils import groups as groups_mod

    calls = []

    def fake_reset(destroy_fn=None):
        calls.append(("groups.reset_for_rebuild", destroy_fn))

    monkeypatch.setattr(groups_mod, "reset_for_rebuild", fake_reset)

    fake = _make_fake_engine(calls)
    DeepSpeedEngine.rebuild_nccl_groups(fake)

    # The ordering list IS the spec: every name in the expected order.
    names = [c[0] for c in calls]
    assert names == [
        "coordinator._quiesce_pending_replication_and_persist",
        "groups.reset_for_rebuild",
        "engine._rebuild_pipe_sub_groups",
        "engine._reinitialize_distributed_groups",
        "engine._refresh_group_caches",
        "optimizer.relink_all_dp_refs",
        "engine._rebuild_get_pipeline_parallel_group",
        "coordinator.rebuild_comm_groups",
    ]

    # The destroy_fn handed to reset_for_rebuild is MoEvement's
    # ``_abort_or_destroy`` — any other value would mean the rebuild
    # was using stock ``dist.destroy_process_group``, losing the
    # NCCL-abort fast path for wedged communicators.
    from deepspeed.moevement.comm_rebuild import _abort_or_destroy
    reset_call = next(c for c in calls if c[0] == "groups.reset_for_rebuild")
    assert reset_call[1] is _abort_or_destroy

    # The Layer-1 rebuild receives the post-refresh ``seq_data_parallel_group``
    # and the stage's current pp group (from _rebuild_get_pipeline_parallel_group).
    rebuild_coord_call = next(c for c in calls if c[0] == "coordinator.rebuild_comm_groups")
    assert rebuild_coord_call[1] == "stub_dp_group"
    assert rebuild_coord_call[2] == "stub_pp_group"


def test_rebuild_nccl_groups_skips_coordinator_when_moevement_disabled(monkeypatch):
    """Without a MoEvement coordinator, quiesce + Layer-1 rebuild are skipped.

    Stock-DeepSpeed users who aren't running MoEvement still need the
    rebuild path to do the engine + optimizer portion.  The
    coordinator-specific steps (1 and 7) must no-op rather than raise.
    """
    from deepspeed.utils import groups as groups_mod

    calls = []
    monkeypatch.setattr(groups_mod,
                        "reset_for_rebuild",
                        lambda destroy_fn=None: calls.append(("groups.reset_for_rebuild", destroy_fn)))

    fake = _make_fake_engine(calls, with_coordinator=False)
    DeepSpeedEngine.rebuild_nccl_groups(fake)

    names = [c[0] for c in calls]
    assert names == [
        "groups.reset_for_rebuild",
        "engine._rebuild_pipe_sub_groups",
        "engine._reinitialize_distributed_groups",
        "engine._refresh_group_caches",
        "optimizer.relink_all_dp_refs",
    ]


def test_rebuild_nccl_groups_warns_but_continues_on_optimizer_without_relink(monkeypatch, caplog):
    """Optimizers lacking ``relink_all_dp_refs`` log a warning and continue.

    Bf16 optimizer and ZeRO-3 don't implement the relink primitive yet;
    the rebuild path should skip the relink step for them rather than
    crashing, while logging a loud warning so the caller knows cached
    references may be stale.  The rest of the orchestration still runs.
    """
    from deepspeed.utils import groups as groups_mod

    calls = []
    monkeypatch.setattr(groups_mod,
                        "reset_for_rebuild",
                        lambda destroy_fn=None: calls.append(("groups.reset_for_rebuild", destroy_fn)))

    fake = _make_fake_engine(calls, with_relink=False)

    # DeepSpeed's logger has propagate=False; route its warnings into
    # caplog by attaching pytest's handler.  Without this, caplog stays
    # empty even though the warning fires.
    import logging
    from deepspeed.utils import logger as ds_logger
    caplog.set_level(logging.WARNING, logger=ds_logger.name)
    ds_logger.addHandler(caplog.handler)
    try:
        DeepSpeedEngine.rebuild_nccl_groups(fake)
    finally:
        ds_logger.removeHandler(caplog.handler)

    names = [c[0] for c in calls]
    # Relink is skipped but every other step still runs.
    assert "optimizer.relink_all_dp_refs" not in names
    assert "engine._refresh_group_caches" in names
    assert "coordinator.rebuild_comm_groups" in names

    # Warning text advertises the scope (ZeRO 0/1 only).
    assert any("relink_all_dp_refs" in rec.message and "ZeRO stage 0/1" in rec.message for rec in caplog.records)


def test_rebuild_nccl_groups_calls_world_rendezvous_fn_between_reset_and_reinit(monkeypatch):
    """``world_rendezvous_fn`` fires after ``reset_for_rebuild`` and before ``_reinitialize_distributed_groups``.

    This slot is load-bearing for the survivor-preservation flow
    (``engine.survivor_rendezvous``): the hook runs
    ``destroy_process_group`` + ``init_process_group`` inside it, so
    it must execute AFTER sub-group teardown (reset) — otherwise the
    backend-abort calls in reset hit a WORLD that's already gone —
    and BEFORE sub-group rebuild (reinit) — otherwise the rebuild
    binds to the stale WORLD instead of the new one.  Pin the exact
    ordering here so a future refactor cannot silently shift the
    hook out of the correct slot.
    """
    from deepspeed.utils import groups as groups_mod

    calls = []
    monkeypatch.setattr(groups_mod,
                        "reset_for_rebuild",
                        lambda destroy_fn=None: calls.append(("groups.reset_for_rebuild", destroy_fn)))

    fake = _make_fake_engine(calls)

    def rdv():
        calls.append(("world_rendezvous_fn", ))

    DeepSpeedEngine.rebuild_nccl_groups(fake, world_rendezvous_fn=rdv)

    names = [c[0] for c in calls]
    assert names == [
        "coordinator._quiesce_pending_replication_and_persist",
        "groups.reset_for_rebuild",
        "world_rendezvous_fn",
        "engine._rebuild_pipe_sub_groups",
        "engine._reinitialize_distributed_groups",
        "engine._refresh_group_caches",
        "optimizer.relink_all_dp_refs",
        "engine._rebuild_get_pipeline_parallel_group",
        "coordinator.rebuild_comm_groups",
    ]


def test_rebuild_nccl_groups_default_skips_world_rendezvous_fn(monkeypatch):
    """Default ``world_rendezvous_fn=None`` preserves the pre-existing 8-step sequence.

    The survivor-preservation hook is additive and opt-in — callers
    driven by a launcher that has already re-invoked
    ``init_process_group`` before calling ``rebuild_nccl_groups``
    (the original consumer) must see an unchanged call order.  If
    this regresses, the torchrun-driven rebuild path starts paying
    the cost of a survivor-rendezvous step that does nothing on its
    path but could silently mask a real bug.
    """
    from deepspeed.utils import groups as groups_mod

    calls = []
    monkeypatch.setattr(groups_mod,
                        "reset_for_rebuild",
                        lambda destroy_fn=None: calls.append(("groups.reset_for_rebuild", destroy_fn)))

    fake = _make_fake_engine(calls)
    DeepSpeedEngine.rebuild_nccl_groups(fake)  # world_rendezvous_fn omitted

    names = [c[0] for c in calls]
    assert "world_rendezvous_fn" not in names
    # Original 8-step sequence unchanged.
    assert names == [
        "coordinator._quiesce_pending_replication_and_persist",
        "groups.reset_for_rebuild",
        "engine._rebuild_pipe_sub_groups",
        "engine._reinitialize_distributed_groups",
        "engine._refresh_group_caches",
        "optimizer.relink_all_dp_refs",
        "engine._rebuild_get_pipeline_parallel_group",
        "coordinator.rebuild_comm_groups",
    ]
