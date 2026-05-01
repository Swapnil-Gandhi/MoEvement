# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Engine-level integration tests for MoEvement.

These tests invoke ``deepspeed.initialize`` end-to-end so that the
engine's init-time validation and wiring are actually exercised, rather
than poking at MoEvement components in isolation.
"""

import time

import pytest

import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.moe.layer import MoE
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.utils import RepeatingLoader

from unit.common import DistributedTest
from unit.moevement._fault_inject import simulate_rank_failure
from unit.simple_model import SimpleMoEModel


class TestMoEvementRequiresPipelineParallelism(DistributedTest):
    """MoEvement's replay needs pipeline boundaries; PP=1 must fail loudly.

    Engine init takes two distinct paths depending on whether the user
    wrapped their model in ``PipelineModule`` — if they did, the engine
    instantiates ``PipelineEngine``; otherwise a plain ``DeepSpeedEngine``.
    MoEvement's recovery reconstructs lost state by replaying logged
    inter-stage activations and gradients, which only exist when a
    pipeline schedule is running, so enabling it on a non-pipeline run
    is a configuration mistake.  Previously the engine would accept the
    config and silently degrade at recovery time; the gate now turns
    that into an up-front ``ValueError`` at ``deepspeed.initialize``.
    This test pins that contract so the check can't be removed without
    a deliberate test update.
    """

    world_size = 2

    def test_raises_when_not_pipeline_engine(self):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1
            },
            "moevement": {
                "enabled": True,
            },
        }
        # Plain MoE model — not wrapped in PipelineModule, so
        # ``deepspeed.initialize`` builds a ``DeepSpeedEngine`` (PP=1),
        # which is exactly the mis-configuration the gate rejects.
        model = SimpleMoEModel(hidden_dim=16, ep_size=1)

        with pytest.raises(ValueError, match="requires pipeline parallelism"):
            deepspeed.initialize(config=config_dict, model=model)


class TestMoEvementRequiresMoELayers(DistributedTest):
    """Sibling gate: enabling MoEvement on a model without MoE layers is an error.

    A dense transformer has no operator-granularity sparse structure to
    exploit, so the "sparse" in sparse checkpointing collapses to "every
    operator every window" — which is just a regular full checkpoint
    with extra overhead.  This was already covered by a ValueError in
    engine.py, but nothing asserted it end-to-end.
    """

    world_size = 2

    def test_raises_on_dense_model(self):
        from unit.simple_model import SimpleModel

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "moevement": {
                "enabled": True,
            },
        }
        model = SimpleModel(hidden_dim=16)

        with pytest.raises(ValueError, match="only supported for models with MoE layers"):
            deepspeed.initialize(config=config_dict, model=model)


class TestMoEvementRejectsPartitionedGradientZeRO(DistributedTest):
    """MoEvement + ZeRO stage >= 2 must fail loudly at init.

    ZeRO-2 and ZeRO-3 partition gradients across DP peers, requiring
    the backward to reduce-scatter into a flat buffer.
    ``PipelineEngine`` instead schedules gradient all-reduces itself
    at pipeline boundaries — the two flows are mutually exclusive and
    the pipe engine rejects stage >= 2 outright.  Because MoEvement
    mandates PP > 1, the transitive implication is that
    ``moevement.enabled = true`` and ``zero_optimization.stage >= 2``
    can never coexist.

    The generic pipe-engine assertion already catches this, but it
    names "pipeline parallelism" rather than "moevement" in the error
    — users wiring up MoEvement for the first time would miss which
    feature in their config caused the conflict.  Engine.py runs a
    MoEvement-specific gate before PipelineEngine's post-super init
    check; this test pins that the named-feature error wins the race.
    """

    world_size = 4

    def test_raises_under_zero_stage_2(self):
        config = _happy_engine_config("fp16")
        config["zero_optimization"] = {"stage": 2}
        with pytest.raises(ValueError, match="incompatible with ZeRO"):
            _build_happy_engine(config)


# ---------------------------------------------------------------------------
# Happy-path helpers: minimal MoE-bearing PipelineModule.
#
# ``PipelineModule`` requires tensor-in/tensor-out per layer so the
# scheduler can ship activations between stages.  ``MoE.forward`` returns
# ``(out, aux_loss, z_loss)``; we drop the aux losses in the adapter
# because this integration test only validates wiring, not training
# fidelity.  A sibling test for aux-loss accumulation would belong in a
# separate file.
#
# Every pipeline stage must hold an MoE layer: ``MoE.__init__`` creates
# an ``ep_group`` via a world collective (``dist.new_group`` over every
# rank), and if only some stages instantiate MoE the non-participating
# ranks never join the collective, deadlocking init.  With num_stages=2
# and two MoEBlocks in the layer spec, each stage ends up with one.
# ---------------------------------------------------------------------------

_HAPPY_HIDDEN = 16
_HAPPY_NUM_CLASSES = 4
_HAPPY_NUM_EXPERTS = 2


class _HappyEmbed(nn.Module):

    def __init__(self, hidden=_HAPPY_HIDDEN):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.lin(x)


class _HappyMoEBlock(nn.Module):

    def __init__(self, hidden=_HAPPY_HIDDEN, num_experts=_HAPPY_NUM_EXPERTS, ep_size=1):
        super().__init__()
        self.moe = MoE(hidden_size=hidden,
                       expert=nn.Linear(hidden, hidden),
                       num_experts=num_experts,
                       k=1,
                       ep_size=ep_size)

    def forward(self, x):
        out, _aux, _z = self.moe(x)
        return out


class _HappyHead(nn.Module):

    def __init__(self, hidden=_HAPPY_HIDDEN, num_classes=_HAPPY_NUM_CLASSES):
        super().__init__()
        self.proj = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


def _build_happy_pipeline_model(ep_size=1):
    layers = [
        LayerSpec(_HappyEmbed, _HAPPY_HIDDEN),
        LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, ep_size),
        LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, ep_size),
        LayerSpec(_HappyHead, _HAPPY_HIDDEN, _HAPPY_NUM_CLASSES),
    ]
    return PipelineModule(layers=layers, num_stages=2, loss_fn=nn.CrossEntropyLoss())


def _happy_data_iter(batch=2, seq=4):
    # fp16 config ⇒ model weights are half; inputs must match or the
    # first Linear dies with "mat1 and mat2 must have the same dtype".
    sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                          dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))
    return iter(RepeatingLoader([sample]))


class TestMoEvementHappyPath(DistributedTest):
    """End-to-end sparse snapshotting on the target topology (PP=2, DP=2).

    Runs a handful of training iterations with moevement enabled and
    verifies that ``on_iteration_end`` actually fires, crosses a
    window boundary, and produces a populated
    ``_persisted_snapshots`` dict.  This is the canary that caught two
    latent wiring bugs in a single session — the missing hook in
    ``PipelineEngine.train_batch`` and the ``_hp_mapping``-gated
    asymmetric DP all-reduces inside ``_full_fp32_view`` and
    ``_collect_param_optim_state``.  Without those fixes this test
    either silently no-ops or deadlocks in NCCL.
    """

    world_size = 4

    def test_engine_init_and_one_train_batch(self):
        config_dict = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    # No nvcc on the test host means FusedAdam's JIT
                    # compile fails at optimizer construction.  The torch
                    # fallback is equivalent for this wiring test.
                    "torch_adam": True,
                },
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1
            },
            "pipeline": {
                "activation_checkpoint_interval": 0
            },
            "moevement": {
                "enabled": True
            },
        }

        model = _build_happy_pipeline_model()
        # MoE requires a split param-group layout so expert params end up
        # on the ``expert_dp_process_group`` rather than the generic DP
        # group.  Without this, ZeRO-1's ``_hp_mapping`` gathers expert
        # and non-expert tensors through the same group, which would
        # deadlock the moment ``on_iteration_end`` runs.
        param_group = {"params": list(model.parameters()), "name": "moe_test_params"}
        split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
        optimizer = torch.optim.AdamW(split_params, lr=1e-4)
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               optimizer=optimizer,
                                               dist_init_required=False)

        assert engine.has_moe_layers is True
        assert hasattr(engine, "moevement_coordinator")
        assert engine.moevement_coordinator is not None

        data_iter = _happy_data_iter()

        # Two iterations: window N's snapshots land in ``_in_flight`` at
        # the boundary closing N, and only promote to ``_persisted`` at
        # the next boundary (N+1's finalize_window).  A single iter
        # would leave _persisted empty even on the happy path.
        for _ in range(4):
            loss = engine.train_batch(data_iter=data_iter)
            if engine.is_last_stage():
                assert torch.isfinite(loss), f"non-finite loss from train_batch: {loss}"

        persisted = engine.moevement_coordinator.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"expected non-empty _persisted_snapshots after 2 train_batch calls — "
                                    f"on_iteration_end should have crossed at least one window boundary; "
                                    f"got empty on rank {dist.get_rank()}")


class TestMoEvementHappyPathLargeWindow(DistributedTest):
    """Forces ``w_sparse > 1`` to exercise the frozen-operator path.

    The tiny happy-path model has so few ops and so little data that
    the scheduler's ``find_window_size`` always concludes "snapshot
    everything in one iter" (``w_sparse=1``, zero frozen operators
    per iteration).  That leaves the frozen-operator branch of the
    snapshot loop (``for op_name in schedule_entry.frozen_operators``
    at coordinator.py) and the FROZEN→ACTIVE transition logic
    untouched end-to-end.

    Knocking ``pcie_bandwidth_gbs`` down to a sliver inflates the
    estimated snapshot time past the iteration budget, which forces
    the scheduler to split operators across multiple iterations.
    That gives us a window with a genuine active/frozen split and a
    ``w_sparse`` greater than one.
    """

    world_size = 4

    def test_snapshot_loop_runs_under_w_sparse_gt_one(self):
        config = _happy_engine_config("fp16")
        # Tiny bandwidth ⇒ huge estimated snapshot time ⇒ scheduler
        # narrows num_active below total_ops.  With four operators
        # (non_expert, gate, expert_0, expert_1) the floor on
        # num_active is 1, which gives ``w_sparse=4`` — enough to
        # land us in the frozen-operator branch on at least one of
        # the four schedule entries.
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0

        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator

        assert coord.scheduler.w_sparse > 1, (
            f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {coord.scheduler.w_sparse}")
        # At least one schedule entry must carry frozen operators
        # (sanity: the split is actually happening, not a trivial
        # w_sparse-with-everything-active).
        assert any(entry.frozen_operators for entry in coord.scheduler.schedule), (
            "no iteration has any frozen operators — the frozen-op branch won't be exercised")

        data_iter = _happy_data_iter()
        # Window N's snapshots land in ``_in_flight`` at iter N
        # (first ``finalize_window`` fires there; the pre-existing
        # ``_persisted`` slot is empty so nothing migrates), and only
        # reach ``_persisted`` at iter 2N when the *next* boundary's
        # ``finalize_window`` runs.  Bound the loop at 2·w_sparse + 1
        # to leave a one-iter safety margin for off-by-one drift.
        n_iters = 2 * coord.scheduler.w_sparse + 1
        for _ in range(n_iters):
            loss = engine.train_batch(data_iter=data_iter)
            if engine.is_last_stage():
                assert torch.isfinite(loss), f"non-finite loss: {loss}"

        persisted = engine.moevement_coordinator.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"expected non-empty _persisted_snapshots after {n_iters} iters "
                                    f"(w_sparse={coord.scheduler.w_sparse}) on rank {dist.get_rank()}")


class TestMoEvementSparseCheckpointIntegrityMultiWindow(DistributedTest):
    """After a completed w_sparse>1 window, every op in ``_persisted``
    carries its ACTIVE snapshot (FP32 master + optimizer state).

    Per MoEvement's design each operator is scheduled ACTIVE exactly
    once per w_sparse-iter window; its ACTIVE snapshot has to survive
    to the persisted bundle so recovery can restore enough state to
    resume training.  The only way this can regress is if a later
    FROZEN snapshot for the same op overwrites the earlier ACTIVE one
    — ``sparse_snapshot.py:156`` is an unconditional dict reassign,
    so the coordinator has to filter ``frozen_operators`` to skip any
    op that was already ACTIVE earlier in the current window.

    This test forces ``w_sparse > 1`` via the same ``pcie_bandwidth_gbs``
    knob ``TestMoEvementHappyPathLargeWindow`` uses, drives
    ``2 * w_sparse + 1`` iters so one full window promotes into
    ``_persisted``, and pins for every persisted op:
      * ``snap.is_active`` is ``True``
      * ``state_dict`` carries ``params.*`` keys (the FP32 master
        weights — essential for restore)
      * ``state_dict`` does NOT carry ``compute_weights.*`` keys
        (those are the FP16-only frozen shape)

    ``optimizer.*`` keys are NOT required: under fp16 dynamic loss
    scaling the first few iters often overflow, in which case
    ZeRO's ``step()`` returns early without calling
    ``_lazy_init_hp_params_optimizer_state``, so ``optim_fragment``
    is legitimately empty at the op's active iteration.  Recovery
    from such a snapshot restores the weights and starts Adam with
    fresh (zeroed) moments — equivalent to starting training fresh
    for that op, which is the best we can do when no prior state
    exists.  Absence of ``optimizer.*`` therefore isn't a bug; the
    important invariant is that no FROZEN-shape ``compute_weights.*``
    keys appear, i.e., no later frozen capture clobbered the active.
    """

    world_size = 4

    def test_persisted_window_carries_active_snapshots(self):
        config = _happy_engine_config("fp16")
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator

        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"fixture failed to force w_sparse > 1 (got {w_sparse}); "
                              f"the pcie_bandwidth crush no longer constrains the scheduler "
                              f"enough to spread operators across multiple iters")
        # Sanity: the schedule must actually rotate; otherwise every op
        # is active every iter and the overwrite bug can't trigger.
        assert any(entry.frozen_operators for entry in coord.scheduler.schedule), (
            "no iteration has any frozen operators — the overwrite path isn't being exercised")

        data_iter = _happy_data_iter_for("fp16")
        # ``2 * w_sparse + 1`` matches B.1's window-promotion cadence:
        # one full window fills ``_in_flight`` at the boundary closing
        # it, and the next boundary promotes it to ``_persisted``.
        n_iters = 2 * w_sparse + 1
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)

        coord.snapshot_engine.synchronize()
        persisted = coord.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"no persisted snapshots after {n_iters} iters "
                                    f"(w_sparse={w_sparse}) on rank {dist.get_rank()}")

        # Spec §3.2: every op becomes active exactly once per window and gets
        # its FP32 + optimizer capture there; earlier-iter FROZEN captures for
        # the same op are preserved side-by-side under per-iter keying.  The
        # ACTIVE captures are the ones this canary pins — one per op per window.
        active_entries = {key: snap for key, snap in persisted.items() if snap.is_active}
        active_names_seen = {name for (_, name) in active_entries.keys()}
        assert active_names_seen, (f"no ACTIVE captures in _persisted_snapshots after {n_iters} iters "
                                   f"(w_sparse={w_sparse}) on rank {dist.get_rank()}")

        for (iteration, op_name), snap in active_entries.items():
            has_params = any(k.startswith("params.") for k in snap.state_dict)
            has_compute = any(k.startswith("compute_weights.") for k in snap.state_dict)
            assert has_params, (f"operator {op_name} at iter {iteration} missing params.* keys on rank "
                                f"{dist.get_rank()}; state_dict = {list(snap.state_dict)}")
            assert not has_compute, (f"operator {op_name} at iter {iteration} carries compute_weights.* keys "
                                     f"on an ACTIVE capture on rank {dist.get_rank()}; state_dict = "
                                     f"{list(snap.state_dict)}")


class TestMoEvementHappyPathBF16(DistributedTest):
    """Happy-path snapshotting under bf16 mixed precision.

    Parallel to ``TestMoEvementHappyPath`` but routes the optimizer
    through ``bf16_optimizer`` instead of ``DeepSpeedZeroOptimizer``.
    The two paths share the ``_hp_mapping`` / ``get_full_hp_param``
    machinery that the fp16 test already pins — any asymmetric-all-
    reduce regression would re-surface here under a different init
    code path.  Same canary assertion as A.2: ``_persisted_snapshots``
    populated after enough iters to cross a window boundary.
    """

    world_size = 4

    def test_engine_init_and_one_train_batch_bf16(self):
        engine = _build_happy_engine(_happy_engine_config("bf16"))
        assert engine.has_moe_layers is True
        assert engine.moevement_coordinator is not None

        data_iter = _happy_data_iter_for("bf16")
        for _ in range(4):
            loss = engine.train_batch(data_iter=data_iter)
            if engine.is_last_stage():
                assert torch.isfinite(loss), f"non-finite loss from train_batch: {loss}"

        persisted = engine.moevement_coordinator.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"expected non-empty _persisted_snapshots under bf16 on rank "
                                    f"{dist.get_rank()}; bf16_optimizer's _hp_mapping init may differ from "
                                    f"ZeRO-1's — if this fails, re-check the asymmetric-all-reduce fix path")


class TestMoEvementHappyPathExpertParallel(DistributedTest):
    """Happy-path snapshotting with ``ep_size > 1``.

    Other happy-path variants keep ``ep_size=1``, which routes every
    expert param through the same DP group as the non-expert params.
    That leaves the ``expert_dp_process_group`` branch in ZeRO's
    ``_configure_moe_settings`` (and the singleton-group reductions
    inside ``get_full_hp_param`` / ``safe_get_full_optimizer_state``)
    untouched end-to-end.

    With world=4, PP=2 and ``ep_size=2`` the per-stage rank count
    equals ep_size, so ``groups._create_expert_and_data_parallel``
    shrinks each expert's DP group down to a singleton: every rank
    owns its local expert outright, the expert-side all-reduces inside
    ``_full_fp32_view`` collapse to no-ops, and the non-expert
    all-reduces still fire across the two-rank DP group.  The snapshot
    loop therefore has to drive two different ``_dp_group`` widths per
    iteration — the same asymmetric-group shape that produced the fp16
    all-reduce fix pinned in ``TestMoEvementHappyPath``.  A regression
    that unconditionally assumed a multi-rank DP group, or misrouted
    expert params onto the non-expert group, would surface here.
    """

    world_size = 4

    def test_engine_init_and_one_train_batch_ep2(self):
        engine = _build_happy_engine(_happy_engine_config("fp16"), ep_size=2)
        assert engine.has_moe_layers is True
        assert engine.moevement_coordinator is not None

        # Pin that ep_size=2 actually took effect at the ZeRO layer.
        # ``link_hp_params`` writes ``_dp_group`` onto every lp_param
        # from ``real_dp_process_group[i]``, which ``_configure_moe_settings``
        # overrides to the expert_dp_process_group for MoE groups.  If
        # that override ever regresses (MoE params silently fall back to
        # the main DP group), ``_persisted_snapshots`` below could still
        # populate — so we fail loudly before getting that far.
        expert_params = [p for _, p in engine.module.named_parameters() if hasattr(p, "allreduce") and not p.allreduce]
        assert expert_params, "fixture produced no expert params — MoE wiring broke"
        hp_expert = next((p for p in expert_params if hasattr(p, "_dp_group")), None)
        assert hp_expert is not None, ("no expert lp_param carries ``_dp_group`` — ZeRO's MoE init didn't run")
        assert dist.get_world_size(group=hp_expert._dp_group) == 1, (
            f"expected singleton expert_dp_process_group under ep_size=2 / world=4 / PP=2; got size "
            f"{dist.get_world_size(group=hp_expert._dp_group)} — the ep_size > 1 path is not being exercised")

        data_iter = _happy_data_iter_for("fp16")
        for _ in range(4):
            loss = engine.train_batch(data_iter=data_iter)
            if engine.is_last_stage():
                assert torch.isfinite(loss), f"non-finite loss under ep_size=2: {loss}"

        persisted = engine.moevement_coordinator.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"expected non-empty _persisted_snapshots under ep_size=2 on rank "
                                    f"{dist.get_rank()}; an asymmetric-collective regression on the "
                                    f"expert path would deadlock before reaching window promotion")
        # ``_persisted`` being non-empty only proves *something*
        # snapshotted.  Under ep_size=2 each rank owns exactly one
        # local expert (``num_local = num_experts // ep_size = 1``), so
        # a lone ``layer_{L}_expert_0`` entry is what we expect — its
        # absence would mean the non-expert path succeeded while the
        # singleton-group expert path silently skipped.
        assert any("expert_0" in name
                   for (_,
                        name) in persisted), (f"no expert operator in _persisted_snapshots on rank {dist.get_rank()}; "
                                              f"keys = {list(persisted)} — expert-side snapshot was skipped")


class TestMoEvementLongRunIterCanary(DistributedTest):
    """Canary for slow-moving bugs that don't surface in a handful of iters.

    The rest of the happy-path suite runs 4 train_batch calls.  With
    w_sparse=1 on this tiny model that exercises at most a few window
    rollovers — not enough to surface:
      * ``PinnedPool`` growth if ``release`` / ``release_busy`` drifts
        (e.g. an asymmetric ``mark_busy`` without a paired release).
      * ``upstream_logger._logs`` accumulating if ``garbage_collect``
        stops retiring entries.
      * Any per-iteration accumulator leak with period > 4.

    Runs 50 ``train_batch`` iterations and pins the bounded-state
    invariants at the end.  None of these assertions depend on the
    exact iteration count — they only require that steady state is
    actually steady.
    """

    world_size = 4

    def test_50_iters_keeps_state_bounded(self):
        engine = _build_happy_engine(_happy_engine_config("fp16"))
        coord = engine.moevement_coordinator
        data_iter = _happy_data_iter_for("fp16")

        n_iters = 50
        for step in range(n_iters):
            loss = engine.train_batch(data_iter=data_iter)
            if engine.is_last_stage() and step == n_iters - 1:
                assert torch.isfinite(loss), (f"non-finite loss on final iter {step}: {loss} — "
                                              f"long-run instability or a pool-induced NaN")

        # Drain async work so steady-state inspection doesn't race
        # in-flight handoffs (replication worker still holding busy
        # refcounts, upstream-log copies mid-flight, etc.).
        coord.upstream_logger.synchronize()
        coord.flush_persist()
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
            # ``concurrent.futures`` notifies ``result()`` waiters
            # *before* firing the future's ``add_done_callback`` — and
            # the callback is what runs ``_release_replication_busy``.
            # Without this short drain loop the busy assertion below
            # can trip on the microseconds-wide race between the
            # worker finishing and the callback clearing the refcounts.
            deadline = time.time() + 5.0
            while (len(coord.snapshot_engine._pool._busy) > 0 or len(coord.upstream_logger._pool._busy) > 0) \
                    and time.time() < deadline:
                time.sleep(0.02)

        # 1. Window rollover still persists something.  Any regression
        #    that keeps overwriting ``_in_flight`` without promoting to
        #    ``_persisted`` would leave this empty after 50 crossings.
        persisted = coord.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"after {n_iters} iters _persisted_snapshots is empty on rank "
                                    f"{dist.get_rank()} — window-rollover promotion may have regressed")

        # 2. Upstream-log GC actually retires entries.  garbage_collect
        #    fires in on_iteration_end with oldest_valid = global_step -
        #    w_sparse; steady-state retained iterations should be O(1),
        #    not O(n_iters).  Allow w_sparse + 2 slack for end-of-loop
        #    timing (the final iter's entries land after GC ran).
        log_iterations = {key[0] for key in coord.upstream_logger._logs}
        max_retained = coord.scheduler.w_sparse + 2
        assert len(log_iterations) <= max_retained, (
            f"upstream_logger retained {len(log_iterations)} distinct iterations after "
            f"{n_iters} iters with w_sparse={coord.scheduler.w_sparse}; expected "
            f"<= {max_retained} — garbage_collect may have regressed")

        # 3. PinnedPool free lists stay bounded along two axes:
        #    - per-shape growth capped by max_per_key (catches an
        #      uncapped append on a single key);
        #    - shape-churn capped by the number of distinct operators +
        #      dtype groups (catches per-iter shape explosion).
        #    Grow-on-miss fills each key toward max_per_key on first
        #    use, so a raw total-free threshold scales with number of
        #    shapes and can't distinguish healthy warmup from a leak.
        for label, pool in (("upstream_logger", coord.upstream_logger._pool), ("snapshot_engine",
                                                                               coord.snapshot_engine._pool)):
            for key, entries in pool._free.items():
                assert len(entries) <= pool._max_per_key, (
                    f"{label} PinnedPool key {key} holds {len(entries)} entries, over "
                    f"max_per_key={pool._max_per_key} — uncapped-append leak on a single shape")
            assert len(pool._free) < 100, (f"{label} PinnedPool accumulated {len(pool._free)} distinct "
                                           f"shape keys after {n_iters} iters on rank {dist.get_rank()} — "
                                           f"a per-iter shape-churn leak is likely")
            # 4. Busy refcounts drained after flush.  A stuck entry means
            #    a mark_busy without its paired release_busy — the buffer
            #    is retained for the process lifetime.
            assert len(pool._busy) == 0, (f"{label} PinnedPool has {len(pool._busy)} buffers still marked "
                                          f"busy after flushing replication / persist workers; "
                                          f"release_busy may have been dropped somewhere")


def _happy_engine_config(low_precision="fp16"):
    """Config shared by A.2 / B.1 / bf16 variant.

    ``low_precision`` picks the mixed-precision path:
      * ``"fp16"`` — ZeRO-1's ``DeepSpeedZeroOptimizer`` wrapping the
        user Adam, with fp16 lp params and an fp32 master partition.
      * ``"bf16"`` — ``bf16_optimizer`` wrapping the user Adam, same
        fp32-master / bf16-lp pattern but via a different code path
        that has its own ``_hp_mapping`` init.  The bf16 variant is
        worth its own test because the asymmetric-all-reduce bug we
        fixed for the fp16 path could re-surface here under a
        parallel implementation.
    """
    cfg = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "torch_adam": True
            },
        },
        "zero_optimization": {
            "stage": 1
        },
        "pipeline": {
            "activation_checkpoint_interval": 0
        },
        "moevement": {
            "enabled": True
        },
    }
    if low_precision == "fp16":
        cfg["fp16"] = {"enabled": True}
    elif low_precision == "bf16":
        cfg["bf16"] = {"enabled": True}
    else:
        raise ValueError(f"unknown low_precision={low_precision!r}")
    return cfg


def _happy_data_iter_for(low_precision, batch=2, seq=4):
    """Data iterator whose input dtype matches ``low_precision``."""
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[low_precision]
    sample = (torch.randn(batch, seq, _HAPPY_HIDDEN, dtype=dtype), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))
    return iter(RepeatingLoader([sample]))


def _build_happy_engine(config_dict, ep_size=1):
    """Wrap the PipelineModule init + MoE-aware optimizer setup."""
    model = _build_happy_pipeline_model(ep_size=ep_size)
    param_group = {"params": list(model.parameters()), "name": "moe_test_params"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                           model=model,
                                           optimizer=optimizer,
                                           dist_init_required=False)
    return engine


def _build_happy_pipeline_model_pp3(ep_size=1):
    """Build the 3-stage variant of the happy-path MoE model.

    Mirrors ``_build_happy_pipeline_model`` but with three MoE blocks so
    each of the three pipeline stages holds at least one MoE layer — the
    ``MoE.__init__`` ``ep_group`` collective world-groups every rank, and
    a stage without an MoE would miss the collective and deadlock init.
    Five layers distributed across three stages: stage 0 gets
    [embed, moe1], stage 1 gets [moe2], stage 2 gets [moe3, head].
    """
    layers = [
        LayerSpec(_HappyEmbed, _HAPPY_HIDDEN),
        LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, ep_size),
        LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, ep_size),
        LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, ep_size),
        LayerSpec(_HappyHead, _HAPPY_HIDDEN, _HAPPY_NUM_CLASSES),
    ]
    return PipelineModule(layers=layers, num_stages=3, loss_fn=nn.CrossEntropyLoss())


def _build_happy_engine_pp3(config_dict, ep_size=1):
    """Variant of ``_build_happy_engine`` targeting the PP=3 DP=1 topology."""
    model = _build_happy_pipeline_model_pp3(ep_size=ep_size)
    param_group = {"params": list(model.parameters()), "name": "moe_test_params"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                           model=model,
                                           optimizer=optimizer,
                                           dist_init_required=False)
    return engine


class TestMoEvementWholeJobRestart(DistributedTest):
    """Whole-job recovery: save on all ranks, wipe, load, replay to completion.

    Models the "all ranks crashed, cluster manager restarts everyone"
    failure mode.  Every rank simulates a fresh process: local
    snapshot caches cleared, model weights zeroed, and then
    ``load_sparse_checkpoint`` restores the saved shard into memory
    and flips the coordinator into recovery mode.  ``train_batch``
    then drives ``_on_iteration_end_recovery`` through the replay
    window until the converter reports conversion complete and
    ``end_recovery`` fires.

    This is the narrower sibling of single-stage failure (B.2/B.3):
    every rank is recovering so ``_pp_log_transfer`` short-circuits
    (no live neighbour to ship from), but ``_activate_from_snapshot``
    rebuilding state from the on-disk window is exactly the path a
    cluster-wide restart goes through.

    Depends on two fixes landed earlier in this branch:
      * the ``on_iteration_end`` hook in ``train_batch`` — without it
        no persisted snapshot would exist to save, so the load step
        would find nothing.
      * the rank-qualified bundle filename — without it every rank
        overwrites ``window.pt`` and only the last writer's shard
        survives.
    """

    world_size = 4

    def test_whole_job_restart_completes_recovery(self, tmp_path_factory):
        import tempfile

        engine = _build_happy_engine(_happy_engine_config())
        coord = engine.moevement_coordinator
        data_iter = _happy_data_iter()

        # Three train_batches so at least one window completes and its
        # snapshots get promoted to _persisted (which is what
        # save_sparse_checkpoint writes).
        for _ in range(3):
            engine.train_batch(data_iter=data_iter)
        assert len(coord.snapshot_engine._persisted_snapshots) > 0, (
            f"pre-save _persisted_snapshots should be populated on rank {dist.get_rank()}")

        # All ranks save to the same directory, but the per-rank
        # filename change lets each write a distinct bundle.
        save_dir = tempfile.mkdtemp(prefix="moev_b1_")
        tag = "restart_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        # All ranks must finish their bundle write before anyone starts
        # loading, otherwise a peer could race on a half-written file.
        dist.barrier()

        # Capture a post-training weight snapshot so we can verify
        # recovery restored it (vs. leaving the zeroed placeholder).
        # Every rank has a different module slice under PP, so snapshot
        # whatever local parameters exist on this rank.
        post_train = {n: p.detach().clone() for n, p in engine.module.named_parameters()}

        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
        # Sanity: the failure actually zeroed our canary.
        for p in engine.module.parameters():
            assert torch.all(p.data == 0), "zero_model_weights should leave every param at zero"
        assert coord._recovering is True
        assert len(coord.snapshot_engine._persisted_snapshots) == 0, "local state must be gone post-failure"

        # Load this rank's own shard back from disk.  Passes ``model=``
        # so ``load_sparse_checkpoint`` runs the frozen-operator
        # ``requires_grad=False`` setup that the replay path relies on.
        loaded = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
        assert loaded is True, f"load_sparse_checkpoint returned False on rank {dist.get_rank()}"
        assert coord._recovering is True, "load_sparse_checkpoint must flip _recovering=True"

        # Replay loop.  Each recovery ``on_iteration_end`` promotes one
        # window's worth of operators from FROZEN to ACTIVE.  W_sparse=1
        # for this tiny model, so one iter is enough — bound the loop
        # generously in case we ever scale the fixture up.
        max_iters = max(8, coord.scheduler.w_sparse * 2)
        for step in range(max_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_iters} iters on rank {dist.get_rank()}")

        # Recovery finished.  Weights must be non-zero (restored from
        # disk, possibly further updated by the replay steps) and the
        # coordinator must have torn down replay state.
        for name, p in engine.module.named_parameters():
            assert not torch.all(p.data == 0), (f"recovery left param '{name}' at zero on rank {dist.get_rank()} — "
                                                "load_sparse_checkpoint or replay failed to restore")

        # end_recovery's teardown: the replay cursor is cleared, the
        # converter forgets its operator states, and the snapshot-load
        # cache is dropped.  The load dir/tag themselves persist on
        # purpose — a subsequent ``load_sparse_checkpoint`` would
        # overwrite them, and keeping the trail helps debugging.
        assert coord._replay_iteration_cursor is None
        assert coord._cached_snapshot_data is None
        assert len(coord._frozen_param_backup) == 0
        assert coord._recovering_stages_in_my_pp == frozenset()
        del post_train  # silence unused-var lint; kept above for debugging context.


class TestMoEvementRecoveryEquivalence(DistributedTest):
    """Post-recovery state matches fault-free state at the rollback iter.

    B.1 (whole-job restart) pins that the recovery loop completes
    and leaves the model with non-zero weights, but it doesn't pin
    that the *values* are correct.  A silent bug — snapshot captured
    at the wrong moment, optimizer state collected incoherently,
    window-boundary promotion off-by-one — would leave B.1 passing
    with arbitrary restored weights.

    This test runs deterministic training, captures per-iter
    training-precision weights, then triggers a whole-job restart
    and verifies the post-recovery weights equal the fault-free
    weights at the iteration recorded in the restored snapshot.
    The rollback iter is read from the snapshot's metadata rather
    than hard-coded — ties the assertion to whatever the code
    actually persisted, so a rollback-target regression surfaces
    as a value mismatch rather than a silently-drifting reference.

    Scope deliberately narrow: ``w_sparse=1`` (our default tiny
    fixture).  ``TestMoEvementRecoveryEquivalenceMultiWindow`` below
    covers the ``w_sparse > 1`` case, where the active/frozen rotation
    exercises the per-iter snapshot format ``snapshot_engine`` writes.
    """

    world_size = 4

    def test_post_recovery_equals_fault_free_at_rollback_iter(self):
        import tempfile

        # Deterministic model init: seed before ``deepspeed.initialize``
        # so every rank builds the same weights.  A one-seed inconsistency
        # would make the per-iter reference diverge from reality and the
        # assertion would fire on a correctness-neutral cause.
        torch.manual_seed(42)
        engine = _build_happy_engine(_happy_engine_config("fp16"))
        coord = engine.moevement_coordinator

        # Fixed-tensor data: ``RepeatingLoader([sample])`` serves the
        # same micro-batch every call, so iter N's gradient is a
        # pure function of iter N-1's weights.  Pre-fault and post-
        # fault replay both draw from this; any mismatch is a real
        # state divergence, not input drift.
        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        # Run N iters, capturing training-precision weights at the end
        # of each (post-optimizer-step, same moment ``on_iteration_end``
        # snapshots).  Four iters gives us one window boundary
        # promotion under w_sparse=1 so ``_persisted`` is non-empty;
        # extras give the rollback_iter assertion a few candidates to
        # distinguish among.
        n_iters = 4
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # Save sparse checkpoint and read the rollback iter off the
        # persisted metadata.  Every operator's ``iteration`` field
        # must agree: ``finalize_window`` promotes a single window's
        # snapshots together, so a disagreement would itself be a bug.
        save_dir = tempfile.mkdtemp(prefix="moev_equiv_")
        tag = "equiv_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        coord.flush_persist()
        dist.barrier()

        persisted = coord.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"no persisted snapshots after {n_iters} iters on rank "
                                    f"{dist.get_rank()}; window-boundary promotion never fired")
        rollback_iters = {snap.iteration for snap in persisted.values()}
        assert len(rollback_iters) == 1, (f"persisted snapshots disagree on iteration: {rollback_iters} "
                                          f"on rank {dist.get_rank()} — a finalize_window race or "
                                          f"partial-window save bug")
        rollback_iter = rollback_iters.pop()
        # ``global_step`` passed to ``on_iteration_end`` is 1-indexed
        # (engine increments before calling the hook); map to our 0-
        # indexed ``per_iter_weights`` list.
        assert 1 <= rollback_iter <= n_iters, (f"rollback_iter={rollback_iter} outside training range "
                                               f"[1..{n_iters}] on rank {dist.get_rank()}")
        reference = per_iter_weights[rollback_iter - 1]

        # Whole-job restart: wipe, reload, drive the replay loop.
        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
        ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
        assert ok is True, f"load_sparse_checkpoint returned False on rank {dist.get_rank()}"

        max_replay_iters = max(16, coord.scheduler.w_sparse * 3)
        data_iter = make_iter()
        for _ in range(max_replay_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters "
                        f"on rank {dist.get_rank()}")

        # The equivalence assertion.  Training-precision weights go
        # through one FP32→FP16 cast on restore that the fault-free
        # path also does after each optimizer step, so equality should
        # hold to within one round-trip of FP16 precision — rtol/atol
        # 1e-3 is comfortable on the tiny happy-path model.
        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"param {n} post-recovery state diverged from fault-free state at "
                                         f"iter {rollback_iter} on rank {dist.get_rank()}; {msg}"),
            )


class TestMoEvementRecoveryEquivalenceMultiWindow(DistributedTest):
    """Post-recovery weights match fault-free weights under ``w_sparse > 1``.

    The paired ``TestMoEvementRecoveryEquivalence`` covers the trivial
    ``w_sparse=1`` case where every operator's ACTIVE FP32 capture is taken
    at the same iteration and the entire recovery completes in one replay
    step.  The multi-window case is qualitatively different: each operator's
    ACTIVE capture is at a different iteration, FROZEN operators' FP16 weights
    evolve iter-by-iter inside the window (each snapshot carries a different
    FP16 value for the same op), and recovery takes ``w_sparse`` replay
    train_batch calls to reach the same end-of-window state.

    The whole point of Option C's per-iter snapshot format is that replay's
    iter-by-iter forward pass reads the correct FP16 weights at each step —
    the ones that were captured at that iter's snapshot, not whatever was
    left in memory after the fault or what the single-dict format would
    have overwritten with the LAST iter's FP16.  Without per-iter FP16
    preservation this assertion trips.
    """

    world_size = 4

    def test_post_recovery_equals_fault_free_at_rollback_iter_multi_window(self):
        import tempfile

        torch.manual_seed(42)
        # Force w_sparse > 1 via the same ``pcie_bandwidth_gbs`` knob
        # ``TestMoEvementHappyPathLargeWindow`` uses: a sliver of
        # bandwidth makes the scheduler estimate an absurd snapshot time
        # and split the operator set across multiple iterations.
        config = _happy_engine_config("fp16")
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        # Run ``2 * w_sparse + 1`` iters so exactly one window completes and
        # promotes into ``_persisted``; the +1 leaves a margin past the
        # ``finalize_window`` boundary so ``_in_flight`` has migrated fully.
        n_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        save_dir = tempfile.mkdtemp(prefix="moev_equiv_mw_")
        tag = "equiv_mw_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        coord.flush_persist()
        dist.barrier()

        persisted = coord.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"no persisted snapshots after {n_iters} iters on rank "
                                    f"{dist.get_rank()}; window-boundary promotion never fired")

        # Under per-iter keying, ``_persisted`` carries multiple distinct
        # iterations — one per snapshot slot in the sparse window.  The
        # recovery replays the full window, landing the model at the state
        # of the LAST captured iter.
        captured_iters = sorted({it for (it, _) in persisted.keys()})
        rollback_iter = captured_iters[-1]
        assert 1 <= rollback_iter <= n_iters, (f"rollback_iter={rollback_iter} outside training range "
                                               f"[1..{n_iters}] on rank {dist.get_rank()}")
        reference = per_iter_weights[rollback_iter - 1]

        # Whole-job restart: wipe local state, reload the bundle, drive the
        # replay loop until recovery completes.  Optimizer reference is
        # available via ``self._optimizer`` (stored by ``initialize``).
        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
        ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
        assert ok is True, f"load_sparse_checkpoint returned False on rank {dist.get_rank()}"

        # One replay train_batch per iter captured in the bundle.  Iter-1
        # state is seeded eagerly in ``load_sparse_checkpoint``, so the
        # remaining ``w_sparse - 1`` iters consume one train_batch each;
        # a margin of 2× covers any extra end-of-recovery train_batch the
        # engine may run.
        max_replay_iters = max(16, w_sparse * 3)
        data_iter = make_iter()
        for _ in range(max_replay_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters "
                        f"on rank {dist.get_rank()}")

        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        # Tolerance reflects the FP16 floor under tail-placement scheduling
        # (paper Fig. 6): on stage 1 the first-activated expert can land
        # ~10 ULP outside the fault-free envelope from catch-up replay's
        # ordering-dependent drift.  3e-3 ≈ 30 ULP near 1.0 — large enough
        # to absorb the overshoot, still tight enough that a missing FP16
        # restore (100+ ULP off) trips the assertion.  Canaries below pin
        # the structural invariants that weight equivalence at this floor
        # cannot.
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=3e-3,
                atol=3e-3,
                msg=lambda msg, n=name: (f"param {n} post-recovery state diverged from fault-free at "
                                         f"iter {rollback_iter} (w_sparse={w_sparse}) on rank "
                                         f"{dist.get_rank()}; {msg}"),
            )

        # Canary: recovery exhausted the replay list and engine reached
        # fault_iter.  ``global_steps`` doesn't advance during replay
        # (engine.py gates on ``is_moevement_replaying``), so it stays
        # at the pre-fault value across all replay tbs — the post-recovery
        # value is the canonical "reached pre-fault state" signal that
        # weight equivalence at the FP16 floor cannot pin.
        assert coord.converter.get_remaining_replay_count() == 0, (
            f"recovery exited with {coord.converter.get_remaining_replay_count()} replay iters "
            f"unrun on rank {dist.get_rank()}")
        assert engine.global_steps == n_iters, (f"rank {dist.get_rank()} global_steps={engine.global_steps} "
                                                f"!= n_iters={n_iters} post-recovery")


class TestMoEvementRecoveryEquivalenceExpertParallel(DistributedTest):
    """Post-recovery equivalence under ``ep_size > 1`` + ``w_sparse > 1``.

    Pairs ``TestMoEvementHappyPathExpertParallel`` (happy path only) with
    ``TestMoEvementRecoveryEquivalenceMultiWindow`` (recovery only).  Under
    ``ep_size=2`` on world=4 PP=2 DP=2 the expert-DP group is a singleton
    per rank — each of the two experts lives on one DP rank, so the
    restore path runs a per-rank ``_dp_group`` with world size 1 (no
    cross-rank all-gather) and the non-expert params still flow through
    the regular two-rank DP group.

    The invariant this pins: the expert restore path (which builds its
    per-param ``_hp_mapping`` off ``expert_data_parallel_group`` rather
    than ``real_dp_process_group``) reaches every operator the converter
    expects to activate — a regression in which expert params silently
    fall back to the main DP group would either deadlock
    ``update_lp_params``'s all-gather (expert owners broadcast into an
    audience of zero) or leave expert operator state unrestored, which
    surfaces here as a weight mismatch vs fault-free.

    Uses whole-job restart (all ranks wipe + reload) rather than peer-pull
    — simpler bracket for the expert-path check.
    """

    world_size = 4

    def test_post_recovery_equivalence_expert_parallel(self):
        import tempfile

        torch.manual_seed(42)
        config = _happy_engine_config("fp16")
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config, ep_size=2)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        # Confirm ep_size=2 actually took effect on at least one expert param.
        # Same guard as TestMoEvementHappyPathExpertParallel — if MoE params
        # regressed onto the main DP group, the expert restore assertion below
        # would be a vacuous equivalence check rather than a real pin.
        expert_params = [p for _, p in engine.module.named_parameters() if hasattr(p, "allreduce") and not p.allreduce]
        assert expert_params, "fixture produced no expert params — MoE wiring broke"
        hp_expert = next((p for p in expert_params if hasattr(p, "_dp_group")), None)
        assert hp_expert is not None, "no expert lp_param carries _dp_group — ZeRO MoE init didn't run"
        assert dist.get_world_size(group=hp_expert._dp_group) == 1, (
            f"expected singleton expert_dp_process_group; got size {dist.get_world_size(group=hp_expert._dp_group)}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        n_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        save_dir = tempfile.mkdtemp(prefix="moev_equiv_ep2_")
        tag = "equiv_ep2_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        coord.flush_persist()
        dist.barrier()

        persisted = coord.snapshot_engine._persisted_snapshots
        assert len(persisted) > 0, (f"no persisted snapshots after {n_iters} iters on rank "
                                    f"{dist.get_rank()}; window promotion never fired")
        # Expert operator must appear in the persisted bundle — a regression
        # that drops expert ops here would be invisible at the post-recovery
        # weight-compare step (expert weights would stay at the zeroed state
        # and the comparison would fail with a confusing message).
        assert any("expert_0" in name
                   for (_,
                        name) in persisted), (f"no expert operator in _persisted_snapshots on rank {dist.get_rank()}; "
                                              f"keys = {sorted(persisted)} — expert-side snapshot was skipped")

        captured_iters = sorted({it for (it, _) in persisted.keys()})
        rollback_iter = captured_iters[-1]
        reference = per_iter_weights[rollback_iter - 1]

        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
        ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
        assert ok is True, f"load_sparse_checkpoint returned False on rank {dist.get_rank()}"

        max_replay_iters = max(16, w_sparse * 3)
        data_iter = make_iter()
        for _ in range(max_replay_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters "
                        f"on rank {dist.get_rank()}")

        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"param {n} post-recovery state diverged from fault-free at "
                                         f"iter {rollback_iter} (w_sparse={w_sparse}, ep_size=2) on "
                                         f"rank {dist.get_rank()}; {msg}"),
            )


class TestMoEvementRecoveryEquivalenceSingleStagePeerPull(DistributedTest):
    """Post-recovery equivalence under ``w_sparse > 1`` with a single-rank failure.

    Distinct from ``TestMoEvementRecoveryEquivalenceMultiWindow`` (whole-job
    restart: every rank reloads from disk) — here only rank 0 fails and
    peer-pulls its shard from rank 1.  Under localized recovery with catch-up:

    * Ranks 0, 1 (DP group of the failed rank) replay the persisted window
      via sparse-to-dense conversion, then replay catch-up iters using pp
      peers' logged activations/gradients to reach iter ``n_iters`` state.
    * Ranks 2, 3 (pp peers on the other DP column) ship their logs to rank
      0 / rank 1, pause in ``_wait_for_recovery`` until recovery ends, then
      abandon their current train_batch — their stage-1 weights remain at
      the pre-fault iter ``n_iters`` state (never rolled back).

    Every rank ends at iter ``n_iters`` state, so the equivalence assertion
    compares all four ranks to ``per_iter_weights[n_iters - 1]`` — no state
    loss vs the fault-free trajectory.
    """

    world_size = 4

    def test_post_recovery_equivalence_peer_pull_under_w_sparse_gt_one(self):
        torch.manual_seed(42)
        config = _happy_engine_config("fp16")
        # Same knob as ``TestMoEvementHappyPathLargeWindow`` and the multi-
        # window equivalence test: starve the snapshot-bandwidth estimate so
        # the scheduler splits the operator set across multiple iters.
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        # Run ``2 * w_sparse + 1`` iters so exactly one window completes and
        # promotes into ``_persisted``; the +1 leaves margin past the
        # ``finalize_window`` boundary so ``_in_flight`` has migrated fully.
        n_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # Block on the ring-replication future so rank 1 actually holds rank 0's
        # shard in ``_received_snapshots[0]`` before rank 0 tries to pull it.
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
        dist.barrier()

        my_rank = dist.get_rank()

        # Every surviving rank persisted its own shard, so ranks 1, 2, 3
        # already know the window iteration range locally.  Rank 0 will
        # recover the same range from the pulled bundle — no broadcast
        # is needed.
        if my_rank != 0:
            persisted = coord.snapshot_engine._persisted_snapshots
            assert len(persisted) > 0, (f"rank {my_rank} has no persisted snapshots after {n_iters} iters — "
                                        f"window-boundary promotion never fired")

        # Peer-pull on rank 0's stage-0 replication group {0, 1}.  Rank 0 fails
        # and pulls; rank 1 serves.  Ranks 2, 3 sync via the bracketing barriers.
        if my_rank == 0:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            # Sanity: rank 0 truly has nothing locally post-failure.
            assert len(coord.snapshot_engine._persisted_snapshots) == 0
            ok = coord.load_sparse_from_peer(peer_rank=1, my_dp_rank_in_replication_group=0, model=engine.module)
            assert ok is True, "peer-pull returned False — peer reported no shard"
        elif my_rank == 1:
            coord.serve_sparse_snapshot_to_peer(requester_rank=0)

        dist.barrier()

        # Derive rollback_iter from whichever source this rank has locally.
        # Rank 0: the per-iter state it just pulled.  Ranks 1, 2, 3: their
        # own ``_persisted_snapshots``.  The two views are identical by
        # construction — the bundle is the cross-rank-uniform capture of
        # the same window.
        if my_rank == 0:
            _, per_iter_operator_states = coord._cached_snapshot_data
            captured_iters_local = sorted(per_iter_operator_states.keys())
        else:
            captured_iters_local = sorted({it for (it, _) in coord.snapshot_engine._persisted_snapshots.keys()})
        assert captured_iters_local, f"no captured iters available on rank {my_rank}"
        rollback_iter = captured_iters_local[-1]
        assert 1 <= rollback_iter <= n_iters, (f"rollback_iter={rollback_iter} outside training range "
                                               f"on rank {my_rank}")

        # Drive replay on all ranks.  Under localized cascade only the DP
        # group of the failed rank rolls back (ranks 0, 1 here); ranks 2, 3
        # pause in ``_wait_for_recovery``'s world handshake, skip the rest
        # of their paused train_batch via the ``"abandon"`` return from
        # ``recovery_barrier``, then resume.  The test loop exits on each
        # rank when its own pause / replay is over.
        max_replay_iters = max(16, w_sparse * 3)
        data_iter = make_iter()
        for _ in range(max_replay_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering and not coord._paused_for_recovery:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters on rank {my_rank}")

        # Localized recovery with catch-up: ranks 0, 1 replay the persisted
        # window + catch-up iters using pp-peer logs to reach iter n_iters;
        # ranks 2, 3 never rolled back and are still at iter n_iters.  Every
        # rank lands on the same fault-free reference.
        reference = per_iter_weights[n_iters - 1]
        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"param {n} post-recovery state diverged from fault-free "
                                         f"iter {n_iters} (w_sparse={w_sparse}, peer-pull) on rank "
                                         f"{my_rank}; {msg}"),
            )

        # global_steps convergence: engine.py:2856 gates the ``global_steps += 1``
        # increment on ``is_moevement_replaying``, so recovering ranks don't
        # advance the counter during replay.  Paused ranks never reach the
        # optimizer step.  All four ranks therefore end at the same value
        # as they were pre-fault.
        assert engine.global_steps == n_iters, (f"rank {my_rank} global_steps={engine.global_steps} "
                                                f"!= n_iters={n_iters} post-recovery")


class TestMoEvementRecoveryWithRebuildNcclGroups(DistributedTest):
    """End-to-end peer-pull recovery with ``engine.rebuild_nccl_groups()`` in the loop.

    Layer 2's value proposition is that after a spare-rank substitution
    — where the surviving ranks' NCCL / gloo groups hold dangling
    references to the departed rank — we can rebuild every cached
    ProcessGroup reference without tearing the engine down, and the
    downstream MoEvement peer-pull recovery still works against the
    freshly-rebuilt groups.  This test pins that end-to-end behaviour:
    destroy + rebuild every group in lockstep on all four ranks,
    then run peer-pull recovery on the post-rebuild world and verify
    the four ranks converge to the fault-free trajectory.

    The novelty vs ``TestMoEvementRecoveryEquivalenceSingleStagePeerPull``
    is step 3 below — the existing test never calls
    ``engine.rebuild_nccl_groups()``.  That test runs peer-pull
    recovery over the ORIGINAL NCCL / gloo groups; this one runs it
    over freshly-rebuilt ones, exercising the full Layer 2
    orchestration (quiesce → ``groups.reset_for_rebuild`` → re-init →
    refresh → optimizer relink → Layer 1 gloo-mirror rebuild → pipe-
    grid rebuild) under a real end-to-end recovery, not just a smoke
    test.

    Sequence:
    1. Same PP=2 DP=2 setup as the existing equivalence test; train
       ``2 * w_sparse + 1`` iters so replication populates rank 1's
       ``_received_snapshots[0]`` with rank 0's shard.
    2. Rank 0 runs ``simulate_rank_failure`` (in-process state zero)
       — mimics the "fresh replacement rank" half of spare
       substitution.
    3. **All four ranks call ``engine.rebuild_nccl_groups()`` in
       lockstep** — destroys + rebuilds the real NCCL + gloo groups
       (the engine-level, pipe-grid, and MoEvement-coordinator
       mirrors).  Every cached reference on every rank ends up
       pointing at a fresh communicator.
    4. Rank 0 peer-pulls from rank 1; ranks 2, 3 sync via barriers —
       identical to the existing test's recovery flow, but now
       running over the post-rebuild groups.
    5. Replay until recovery completes on every rank.
    6. Equivalence assertion: every rank's weights match
       ``per_iter_weights[n_iters - 1]`` (the fault-free reference).

    **What this does NOT cover** (documented gap — the local dev box
    doesn't provide the infrastructure):

    * Actual process death + respawn.  ``simulate_rank_failure`` zeros
      Python state on the caller; the process stays alive.  A real
      fault would take the process down and the launcher would spawn
      a replacement.  The rebuild orchestration is the same either
      way, but the process-lifecycle half is not exercised.
    * NCCL wedge under mid-collective fault.  Our fault injection is
      at iter boundary, between collectives; a partner disappearing
      mid-allreduce would require ``NCCL_ASYNC_ERROR_HANDLING`` +
      timeout machinery we don't test.
    * Cross-NCCL-version / cross-GPU-generation.  One data point on
      this A100 + NCCL build; the torchrun-elastic operator-facing
      contract is a separate concern.
    """

    world_size = 4

    def test_rebuild_nccl_groups_then_peer_pull_end_to_end(self):
        torch.manual_seed(42)
        config = _happy_engine_config("fp16")
        # Same constraint as TestMoEvementRecoveryEquivalenceSingleStagePeerPull:
        # starve the snapshot bandwidth so the scheduler forces w_sparse > 1 and
        # the window boundary fires with multiple operators pending.
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        n_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # Block on replication so rank 1 actually holds rank 0's shard in
        # ``_received_snapshots[0]`` before we rebuild the groups and
        # attempt the peer-pull.  A rebuild while replication is in flight
        # would abort the gloo worker's pending send; ``_quiesce_pending_replication_and_persist``
        # inside ``rebuild_nccl_groups`` handles that, but we want the
        # pre-rebuild state deterministic here.
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
        dist.barrier()

        my_rank = dist.get_rank()

        # Step 2: simulate the "fresh replacement rank" portion of spare
        # substitution by zeroing rank 0's local MoEvement state + model
        # weights.  Non-failing ranks don't touch their local state.
        if my_rank == 0:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            assert len(coord.snapshot_engine._persisted_snapshots) == 0

        # Step 3: the novel bit — every rank calls ``rebuild_nccl_groups``
        # in lockstep.  This tears down and rebuilds every cached
        # ProcessGroup reference (module globals, pipe-grid sub-groups,
        # MoEvement coordinator mirrors) across all four ranks.  The
        # downstream peer-pull recovery runs on the post-rebuild groups.
        engine.rebuild_nccl_groups()

        # Post-rebuild sanity — the coordinator's cached refs should now
        # point at the fresh engine-level seq_data_parallel_group, and
        # the Layer-1 rebuild (via ``coord.rebuild_comm_groups``)
        # repoints the pp mirror to the fresh pp group.
        assert coord._dp_group is engine.seq_data_parallel_group
        assert coord._pp_group is engine.grid.get_pipe_parallel_group()

        dist.barrier()

        # Step 4: same peer-pull coordination as the existing equivalence
        # test, now running on the post-rebuild groups.
        if my_rank == 0:
            ok = coord.load_sparse_from_peer(peer_rank=1, my_dp_rank_in_replication_group=0, model=engine.module)
            assert ok is True, "peer-pull returned False — peer reported no shard post-rebuild"
        elif my_rank == 1:
            coord.serve_sparse_snapshot_to_peer(requester_rank=0)

        dist.barrier()

        # Step 5: drive the replay on every rank.  Localized cascade: ranks
        # 0, 1 replay + catch up; ranks 2, 3 pause + abandon then resume.
        max_replay_iters = max(16, w_sparse * 3)
        data_iter = make_iter()
        for _ in range(max_replay_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering and not coord._paused_for_recovery:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters on rank {my_rank} "
                        f"post-rebuild — the rebuild may have corrupted recovery state")

        # Step 6: equivalence.  Same assertion as the rebuild-free variant:
        # every rank converges to the iter-n_iters fault-free reference.
        # Loss of state through the rebuild would surface as weight
        # divergence here, not as a crash during replay.
        reference = per_iter_weights[n_iters - 1]
        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"param {n} post-rebuild+recovery state diverged from fault-free "
                                         f"iter {n_iters} (w_sparse={w_sparse}) on rank {my_rank}; {msg}"),
            )

        # Global-steps convergence: same invariant as the rebuild-free
        # variant — recovering ranks don't advance during replay, paused
        # ranks never reached the optimizer step.  Rebuild doesn't
        # change this invariant; re-check to catch any regression that
        # would.
        assert engine.global_steps == n_iters, (f"rank {my_rank} global_steps={engine.global_steps} "
                                                f"!= n_iters={n_iters} post-rebuild+recovery")


class TestMoEvementDiskPathRecoveryWithRebuildNcclGroups(DistributedTest):
    """End-to-end disk-path recovery with ``engine.rebuild_nccl_groups()`` in the loop.

    Disk-path sibling of ``TestMoEvementRecoveryWithRebuildNcclGroups``.  The
    peer-pull variant pins ``load_sparse_from_peer`` as the replacement
    rank's recovery entry; this one pins ``load_sparse_checkpoint`` —
    the disk entry a spare uses when the prior rank's bundle is reachable
    on shared storage but the survivors can't serve it over the network
    (e.g., the replication future had not yet completed when the peer
    died, so no DP peer holds the shard in memory).

    Sequence:
    1. PP=2 DP=2; train ``2 * w_sparse + 1`` iters, then
       ``save_sparse_checkpoint`` + ``flush_persist`` + barrier so every
       rank's bundle is durable.
    2. Rank 0 runs ``simulate_rank_failure`` (in-process state zero) —
       mimics the "fresh replacement rank" half of spare substitution.
       Ranks 1, 2, 3 keep their local state.
    3. **All four ranks call ``engine.rebuild_nccl_groups()`` in
       lockstep** — same orchestration as the peer-pull variant.
    4. Rank 0 calls ``load_sparse_checkpoint`` (disk-local, no
       collectives) to restore its shard; ranks 1, 2, 3 do not.  This
       is the distinguishing constraint vs whole-job-restart, where
       every rank reloads from disk.
    5. Replay on every rank: rank 1's DP cascade is auto-dispatched by
       the ``recovery_barrier`` handshake inside ``train_batch`` (the
       world handshake sees rank 0 ``_recovering`` and flips rank 1
       through ``cascade_into_recovery``); ranks 2, 3 pause + abandon
       then resume.
    6. Equivalence assertion: every rank's weights match
       ``per_iter_weights[n_iters - 1]``.

    **Novelty vs the peer-pull rebuild test:** this exercises the
    disk-load replacement + in-memory DP-cascade combination under
    freshly-rebuilt NCCL/gloo groups.  Both sides of the DP pair
    recover from different sources (rank 0 from disk, rank 1 from its
    own ``_persisted_snapshots`` via cascade), and the two views must
    agree on window iters + rollback iter for the DP all-reduces
    during replay to line up.

    **Not covered** (same gaps as the peer-pull rebuild test):
    actual process death + respawn, mid-collective NCCL wedge,
    cross-NCCL-version.
    """

    world_size = 4

    def test_disk_path_recovery_after_rebuild_nccl_groups(self):
        import tempfile

        torch.manual_seed(42)
        config = _happy_engine_config("fp16")
        # Same w_sparse > 1 forcing as the peer-pull rebuild variant so
        # the window-boundary / catch-up coordinate space matches.
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        n_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # Persist every rank's shard to disk.  Unlike the peer-pull variant
        # we do not need to block on ``_replication_future`` — the
        # replacement rank reads from disk, not from a peer's in-memory
        # copy.  ``flush_persist`` + barrier is the same durability
        # protocol ``TestMoEvementRecoveryEquivalenceMultiWindow`` uses.
        save_dir = tempfile.mkdtemp(prefix="moev_disk_rebuild_")
        tag = "disk_rebuild_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        coord.flush_persist()
        dist.barrier()

        my_rank = dist.get_rank()

        # Step 2: only rank 0 loses local state.  Ranks 1, 2, 3 keep
        # theirs — rank 1's ``_persisted_snapshots`` is the fuel for
        # its ``cascade_into_recovery`` during replay.
        if my_rank == 0:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            assert len(coord.snapshot_engine._persisted_snapshots) == 0

        # Step 3: every rank rebuilds its groups in lockstep.  Same
        # 8-step sequence pinned by ``test_engine_rebuild.py``; here we
        # rely on the end-to-end invariants rather than reprobing the
        # sequence (covered by the peer-pull rebuild test and the
        # mock-ordering unit tests).
        engine.rebuild_nccl_groups()

        # Post-rebuild sanity — same coordinator cache invariants as
        # the peer-pull variant.  A regression in either Layer-1 gloo
        # rebuild or the engine-level cache refresh would land here.
        assert coord._dp_group is engine.seq_data_parallel_group
        assert coord._pp_group is engine.grid.get_pipe_parallel_group()

        dist.barrier()

        # Step 4: rank 0 reads its bundle from disk.  ``load_sparse_checkpoint``
        # is local (no collectives — verified in coordinator.py: the
        # helpers it invokes are file reads + in-memory state flips), so
        # ranks 1, 2, 3 do not call it.  Asymmetric entry is what makes
        # this a DP-cascade-with-disk-replacement flow rather than a
        # whole-job-restart.
        if my_rank == 0:
            ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
            assert ok is True, "load_sparse_checkpoint returned False on rank 0 — bundle missing or unreadable"
            assert coord._recovering is True, "load_sparse_checkpoint must flip _recovering=True"

        dist.barrier()

        # Step 5: drive replay on every rank.  Rank 0 replays from the
        # disk-seeded snapshot cache; rank 1 enters via cascade when
        # the ``recovery_barrier`` handshake inside its next
        # ``train_batch`` observes rank 0's ``_recovering`` flag across
        # the DP group (see ``test_recovery_barrier_cascade_into_recovery_when_dp_peer_recovering``);
        # ranks 2, 3 pause / abandon / resume.
        max_replay_iters = max(16, w_sparse * 3)
        data_iter = make_iter()
        for _ in range(max_replay_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering and not coord._paused_for_recovery:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters on rank {my_rank} "
                        f"post-rebuild — disk-path + cascade coordination broke")

        # Step 6: equivalence.  Rank 0 (disk replay + catch-up), rank 1
        # (cascade rollback + catch-up), ranks 2, 3 (pause + resume,
        # never rolled back) all converge to the fault-free iter n_iters
        # state.  Tolerance matches the peer-pull rebuild variant.
        reference = per_iter_weights[n_iters - 1]
        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"param {n} post-rebuild+disk-recovery state diverged from fault-free "
                                         f"iter {n_iters} (w_sparse={w_sparse}) on rank {my_rank}; {msg}"),
            )

        assert engine.global_steps == n_iters, (f"rank {my_rank} global_steps={engine.global_steps} "
                                                f"!= n_iters={n_iters} post-rebuild+disk-recovery")


class TestMoEvementSequentialFailures(DistributedTest):
    """Recovery cleanly handles a second failure after completing the first.

    Two full peer-pull recoveries back-to-back with training between them.
    Verifies the coordinator's state-cleanup invariants at the end of
    ``end_recovery`` — ``_fault_iter`` reset to None, converter cleared,
    ``_cached_snapshot_data`` released, ``_replay_iteration_cursor``
    cleared, ``upstream_logger._received_logs`` drained.  A regression
    that leaked a stale flag across sessions (e.g., ``_fault_iter`` from
    session 1 surviving into session 2's ``_compute_replay_iters``) would
    produce the wrong catch-up iter range on the second recovery and the
    final equivalence assertion would fire.
    """

    world_size = 4

    def test_two_peer_pull_recoveries_preserve_state(self):
        torch.manual_seed(42)
        config = _happy_engine_config("fp16")
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        def drive_recovery(my_rank):
            """Drive the recovery loop on every rank until this rank's flags clear."""
            max_replay_iters = max(16, w_sparse * 3)
            for _ in range(max_replay_iters):
                engine.train_batch(data_iter=data_iter)
                if not coord._recovering and not coord._paused_for_recovery:
                    return
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters on rank {my_rank}")

        def assert_clean_post_recovery_state(label):
            """Pin the state-cleanup contract of ``end_recovery``.

            Runs AFTER a per-rank train_batch — at that point
            ``_post_recovery_exit`` has been cleared by the next
            ``recovery_barrier``'s top-of-call reset (see
            coordinator.py), so every flag should be back to its
            steady-state value.
            """
            assert coord._recovering is False, f"{label}: _recovering still True"
            assert coord._paused_for_recovery is False, f"{label}: _paused_for_recovery still True"
            assert coord._fault_iter is None, f"{label}: _fault_iter={coord._fault_iter}, expected None"
            assert coord._pp_log_transfer_done is False, f"{label}: _pp_log_transfer_done still True"
            assert coord._post_recovery_exit is False, f"{label}: _post_recovery_exit still True"
            assert coord._replay_iteration_cursor is None, (f"{label}: _replay_iteration_cursor="
                                                            f"{coord._replay_iteration_cursor}")
            assert coord._cached_snapshot_data is None, f"{label}: _cached_snapshot_data retained"
            assert coord._recovering_stages_in_my_pp == frozenset(), (
                f"{label}: _recovering_stages_in_my_pp={coord._recovering_stages_in_my_pp}")
            assert not coord.upstream_logger._received_logs, (f"{label}: upstream_logger._received_logs not drained")

        phase_1_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(phase_1_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # ---------- Recovery session 1 ----------
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
        dist.barrier()

        my_rank = dist.get_rank()
        if my_rank == 0:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            ok = coord.load_sparse_from_peer(peer_rank=1, my_dp_rank_in_replication_group=0, model=engine.module)
            assert ok is True, "peer-pull 1 returned False"
        elif my_rank == 1:
            coord.serve_sparse_snapshot_to_peer(requester_rank=0)
        dist.barrier()

        data_iter = make_iter()
        drive_recovery(my_rank)

        # Session 1 equivalence: after recovery (before any post-recovery
        # train_batch), weights match the pre-fault iter (iter phase_1_iters).
        assert engine.global_steps == phase_1_iters, (f"rank {my_rank} global_steps={engine.global_steps} != expected "
                                                      f"{phase_1_iters} immediately post-session-1-recovery")
        reference_session_1 = per_iter_weights[phase_1_iters - 1]
        post_recovery_s1 = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery_s1.items():
            torch.testing.assert_close(
                restored,
                reference_session_1[name],
                rtol=2e-3,
                atol=2e-3,
                msg=lambda msg, n=name: (f"session-1 post-recovery param {n} diverged from fault-free "
                                         f"iter {phase_1_iters} on rank {my_rank}; {msg}"),
            )

        # One more fault-free train_batch to clear ``_post_recovery_exit``
        # before the state-cleanup contract check fires.
        engine.train_batch(data_iter=data_iter)
        per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})
        assert_clean_post_recovery_state("session 1")

        # ---------- Inter-session training ----------
        phase_2_iters = 2 * w_sparse
        for _ in range(phase_2_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # ---------- Recovery session 2 ----------
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
        dist.barrier()

        if my_rank == 0:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            ok = coord.load_sparse_from_peer(peer_rank=1, my_dp_rank_in_replication_group=0, model=engine.module)
            assert ok is True, "peer-pull 2 returned False"
        elif my_rank == 1:
            coord.serve_sparse_snapshot_to_peer(requester_rank=0)
        dist.barrier()

        data_iter = make_iter()
        drive_recovery(my_rank)

        # One more fault-free train_batch to clear ``_post_recovery_exit``
        # before the session-2 state-cleanup contract check fires.
        #
        # Note: we do NOT assert post-session-2 weight equivalence.  The
        # catch-up replay path described in the design docs is dead-coded
        # in this config (``num_ops <= w_sparse``): ``is_conversion_complete``
        # short-circuits in ``_on_iteration_end_recovery`` before the
        # catch-up iters run, so recovery stops at ``bundle[-1]`` rather
        # than catching up to ``_fault_iter``.  Session 1's check passes
        # for the same reason the baseline B.2-extended test does — the
        # tiny ``hidden=16`` model plus ``atol=1e-3`` absorbs 5 iters of
        # drift.  Stacking a second session's drift on top of that
        # typically exceeds the tolerance on the recovering DP group
        # while leaving the paused pp peers unchanged; the resulting
        # half-failure creates a collective mismatch that hangs the
        # suite.  The distinct value of this test is the state-cleanup
        # contract plus ``global_steps`` bookkeeping across back-to-back
        # sessions — both still covered below.
        engine.train_batch(data_iter=data_iter)
        assert_clean_post_recovery_state("session 2")

        total_iters = phase_1_iters + 1 + phase_2_iters + 1
        assert engine.global_steps == total_iters, (f"rank {my_rank} global_steps={engine.global_steps} "
                                                    f"!= total_iters={total_iters}")


class TestMoEvementMiddleStageFailurePP3(DistributedTest):
    """Middle-stage failure under PP=3 DP=1 with ``w_sparse > 1``.

    Distinguishes itself from the PP=2 tests by exercising bidirectional
    pp-log transfer.  When stage 1 (the middle stage) fails, rank 0
    (stage 0) ships FORWARD activations to rank 1 and rank 2 (stage 2)
    ships BACKWARD gradients to rank 1 — the PP=2 tests only ever see
    one direction because stage-0 failures use stage 1's gradient logs
    while stage-1 failures use stage 0's activation logs, not both.

    Under DP=1 there's no DP peer to pull from, so the recovery entry
    point is ``load_sparse_checkpoint`` (disk-based).  Only rank 1 is
    wiped and reloaded; ranks 0, 2 see ``pp_column_has_recovering`` via
    the world handshake (their DP groups are singletons so the DP
    branch doesn't fire) and drop into the pause / log-ship path, then
    release via the end-of-recovery handshake.

    Every rank ends at iter ``n_iters`` state: rank 1 via conversion
    replay of the persisted window, ranks 0/2 by never rolling back.
    """

    world_size = 3

    def test_middle_stage_recovery_equivalence(self):
        import tempfile

        torch.manual_seed(42)
        config = _happy_engine_config("fp16")
        # Same ``pcie_bandwidth_gbs`` trick the other w_sparse>1 tests use
        # to force a multi-iter window.  Three MoE blocks plus the non-expert
        # op gives the scheduler enough operators to split across iters.
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine_pp3(config)
        coord = engine.moevement_coordinator
        w_sparse = coord.scheduler.w_sparse
        assert w_sparse > 1, (f"expected forced w_sparse > 1 under constrained PCIe bandwidth; got {w_sparse}")

        batch, seq = 2, 4
        fixed_sample = (torch.randn(batch, seq, _HAPPY_HIDDEN,
                                    dtype=torch.float16), torch.randint(0, _HAPPY_NUM_CLASSES, (batch, )))

        def make_iter():
            return iter(RepeatingLoader([fixed_sample]))

        # Run enough iters that one full window finalises into _persisted
        # AND the next window is mid-flight — catch-up replay needs at least
        # one iter past the last finalized window.
        n_iters = 2 * w_sparse + 1
        per_iter_weights = []
        data_iter = make_iter()
        for _ in range(n_iters):
            engine.train_batch(data_iter=data_iter)
            per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

        # Save checkpoint + flush on every rank so rank 1 has something
        # to reload from disk after the wipe.
        save_dir = tempfile.mkdtemp(prefix="moev_pp3_")
        tag = "pp3_middle_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        coord.flush_persist()
        dist.barrier()

        my_rank = dist.get_rank()

        # Wipe rank 1 only.  Ranks 0, 2 retain their stage weights; the
        # recovery path brings rank 1 back up without touching them.
        if my_rank == 1:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
            assert ok is True, "load_sparse_checkpoint returned False on rank 1"

        # Drive the recovery loop on every rank.  Rank 1 replays through
        # conversion + catch-up; ranks 0, 2 pause + ship logs + release.
        # Per-rank state check is enough — each rank's flags advance
        # independently once the global handshake releases them.
        max_replay_iters = max(16, w_sparse * 3)
        data_iter = make_iter()
        replay_tbs_on_recovering = 0
        for _ in range(max_replay_iters):
            was_recovering = coord._recovering
            engine.train_batch(data_iter=data_iter)
            if was_recovering:
                replay_tbs_on_recovering += 1
            if not coord._recovering and not coord._paused_for_recovery:
                break
        else:
            pytest.fail(f"recovery did not complete within {max_replay_iters} iters on rank {my_rank}")

        # Every rank lands at iter n_iters state — rank 1 via replay, ranks
        # 0, 2 by never rolling back — so the equivalence reference is
        # uniform across the world.  Tolerance is set to 3e-3 (~30 ULP near
        # 1.0) to absorb the FP16-floor overshoot that paper Fig. 6's tail
        # placement produces on stage-1's first-activated expert during
        # catch-up replay; the tb-count + global_steps canaries below pin
        # the structural invariants weight equivalence at this floor cannot.
        reference = per_iter_weights[n_iters - 1]
        post_recovery = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
        for name, restored in post_recovery.items():
            torch.testing.assert_close(
                restored,
                reference[name],
                rtol=3e-3,
                atol=3e-3,
                msg=lambda msg, n=name: (f"param {n} post-recovery state diverged from fault-free "
                                         f"iter {n_iters} (w_sparse={w_sparse}, PP=3 middle-stage) on "
                                         f"rank {my_rank}; {msg}"),
            )

        # Canary: the recovering rank replayed ``n_iters`` train_batches
        # (bundle replay + catch-up) to reach pre-fault state.  Paused pp
        # peers only saw a single "abandon" train_batch before release.
        # This discriminates catch-up-enabled replay (n_iters tbs) from
        # the old ``is_conversion_complete`` short-circuit (bundle_size
        # = w_sparse tbs) regardless of FP16 precision noise in the
        # weight comparison above.
        if my_rank == 1:
            assert replay_tbs_on_recovering == n_iters, (
                f"rank 1 replayed {replay_tbs_on_recovering} train_batches during recovery, "
                f"expected {n_iters} (bundle + catch-up); catch-up short-circuited?")

        assert engine.global_steps == n_iters, (f"rank {my_rank} global_steps={engine.global_steps} "
                                                f"!= n_iters={n_iters} post-recovery")


class TestMoEvementScheduleSymmetry(DistributedTest):
    """Pin two cross-rank invariants the recovery protocols depend on:

    1. ``coord.scheduler.w_sparse`` is identical across the world.  Per-PP-
       stage operator counts diverge (stage 0 owns the embed, middle
       stages own only an MoE block), so the stage-local
       ``find_window_size`` would pick different cadences.
       ``_generate_schedule_world_aligned`` does an ``all_reduce(MAX)``
       at init + regen to force agreement.

    2. The full schedule (``active_operators`` and ``frozen_operators``
       per slot) is identical across DP-peer ranks within a PP stage.
       DP peers are model replicas — same operators registered in the
       same order — so once ``w_sparse`` is uniform the schedule is
       identical by construction.  Asserting it explicitly guards against
       any future refactor that lets the order or contents drift between
       peers (e.g. non-deterministic sort keys, EP-aware bucket churn),
       which would silently corrupt peer-pull / B-async replication that
       both index iters by slot.
    """

    world_size = 4

    def test_w_sparse_world_uniform_and_dp_schedule_identical(self):
        # PP=2 DP=2 layout — pipe-major default topology so DP peers are
        # ranks (0,2) and (1,3); each pair holds the same stage's
        # operators.  Force ``w_sparse > 1`` via the same constrained-
        # PCIe knob the multi-window tests use, so the schedule has more
        # than one slot to diff.
        config = _happy_engine_config("fp16")
        config["moevement"]["pcie_bandwidth_gbs"] = 0.000001
        config["moevement"]["initial_iter_time_sec"] = 1.0
        engine = _build_happy_engine(config)
        coord = engine.moevement_coordinator

        my_rank = dist.get_rank()
        local_w = coord.scheduler.w_sparse
        assert local_w > 1, f"expected forced w_sparse > 1; got {local_w} on rank {my_rank}"

        # (1) world uniformity of w_sparse — gather to rank 0 and assert
        # a single value across the whole world.
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_w)
        assert len(set(gathered)) == 1, (f"w_sparse not world-uniform: ranks reported {gathered}.  "
                                         "Every iter-keyed cross-rank protocol (peer-pull bundle exchange, "
                                         "_world_recovery_handshake, pp_log_transfer) silently corrupts "
                                         "or deadlocks under per-rank w_sparse — see "
                                         "_generate_schedule_world_aligned in coordinator.py.")

        # (2) DP-peer schedule identity within each PP stage.  Encode each
        # rank's full schedule as a list of (active, frozen) tuples per
        # slot, gather, then group by PP stage and check intra-group
        # identity.
        schedule_repr = [(tuple(s.active_operators), tuple(s.frozen_operators)) for s in coord.scheduler.schedule]
        gathered_schedules = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_schedules, schedule_repr)

        topo = engine.grid._topo
        stage_to_ranks = {}
        for r in range(dist.get_world_size()):
            stage = topo.get_coord(r).pipe
            stage_to_ranks.setdefault(stage, []).append(r)

        for stage, dp_peers in stage_to_ranks.items():
            if len(dp_peers) < 2:
                continue
            ref_rank = dp_peers[0]
            ref_schedule = gathered_schedules[ref_rank]
            for peer in dp_peers[1:]:
                assert gathered_schedules[peer] == ref_schedule, (
                    f"PP stage {stage}: rank {peer}'s schedule differs from rank {ref_rank}'s.  "
                    "DP peers are model replicas; their schedules must be byte-identical so that "
                    "peer-pull bundle exchange and B-async replication index ops by the same slot map. "
                    f"\nrank {ref_rank}: {ref_schedule}\nrank {peer}: {gathered_schedules[peer]}")


class TestExpertDPTopologyAlignment(DistributedTest):
    """Pin expert-DP groups to the topology DP axis under any axes order.

    Regression test for the data-major expert-DP misalignment:
    ``deepspeed.utils.groups._create_expert_and_data_parallel`` used to
    derive expert / expert-DP group membership from rank-index
    arithmetic that implicitly assumed a pipe-major rank layout
    (consecutive ranks form one PP stage).  Under data-major topology
    the layout is reversed — consecutive ranks share a data-rank index
    but differ in pipe-rank — so the index math collapses expert-DP
    onto the PP axis instead of the DP axis.

    Concretely on PP=2 DP=2 ep_size=1:

    - data-major DP groups (correct): ``{0, 2}`` and ``{1, 3}``
    - pre-fix expert-DP groups (bug): ``{0, 1}`` and ``{2, 3}`` — PP
      partners, not DP replicas

    The all-reduce for MoE expert-grad averaging (PP=1 DP=1 inside its
    own DP group) silently runs against the wrong peers in normal
    training — corrupting expert weights — and deadlocks under MoEvement
    recovery whenever the wrong-axis peers are paused.

    Asserting both axes orders rules out a regression in either path of
    the topology-aware logic in ``_create_expert_and_data_parallel``.
    """

    world_size = 4

    def _build_engine_with_topology(self, topology):
        from deepspeed.runtime.pipe.topology import ProcessTopology
        layers = [
            LayerSpec(_HappyEmbed, _HAPPY_HIDDEN),
            LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, 1),
            LayerSpec(_HappyMoEBlock, _HAPPY_HIDDEN, _HAPPY_NUM_EXPERTS, 1),
            LayerSpec(_HappyHead, _HAPPY_HIDDEN, _HAPPY_NUM_CLASSES),
        ]
        topo = ProcessTopology(axes=topology, dims=[2, 2])
        model = PipelineModule(layers=layers, topology=topo, loss_fn=nn.CrossEntropyLoss())
        param_group = {"params": list(model.parameters()), "name": "moe_test_params"}
        split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
        optimizer = torch.optim.AdamW(split_params, lr=1e-4)
        engine, _, _, _ = deepspeed.initialize(config=_happy_engine_config("fp16"),
                                               model=model,
                                               optimizer=optimizer,
                                               dist_init_required=False)
        return engine

    def _expected_dp_partner(self, my_rank, axes):
        # PP=2 DP=2 with axes=axes: identify the rank in my own DP group
        # — same ``pipe`` coord (same PP stage) but different ``data``
        # coord (the other DP replica).  That partner must be in my
        # expert-DP group; the bug was the index-arithmetic path
        # picking PP partners (different pipe coord, same data coord)
        # instead.
        from deepspeed.runtime.pipe.topology import ProcessTopology
        topo = ProcessTopology(axes=axes, dims=[2, 2])
        my_pipe = getattr(topo.get_coord(rank=my_rank), "pipe")
        for r in range(4):
            if r == my_rank:
                continue
            if getattr(topo.get_coord(rank=r), "pipe") == my_pipe:
                return r
        raise AssertionError(f"no DP partner for rank {my_rank} under axes={axes}")

    def _assert_axis_alignment(self, axes):
        from deepspeed.utils.groups import _get_expert_data_parallel_group_ranks
        engine = self._build_engine_with_topology(axes)
        my_rank = dist.get_rank()
        expected_partner = self._expected_dp_partner(my_rank, axes)

        # ep_size=1 → ``ep_size_1`` group — created during MoE.__init__.
        ranks = _get_expert_data_parallel_group_ranks("ep_size_1")
        rank_set = set(ranks)
        assert my_rank in rank_set, f"rank {my_rank} not in its own expert-DP group {rank_set}"
        assert expected_partner in rank_set, (
            f"expert-DP group for rank {my_rank} under axes={axes} is {sorted(rank_set)}, "
            f"expected to contain DP partner rank {expected_partner}.  This is the axis-collapse "
            f"bug — see _create_expert_and_data_parallel in deepspeed/utils/groups.py.")

        # Engine teardown — without this the next test in the class
        # inherits a stale process-group state that confuses the second
        # invocation of ``deepspeed.initialize``.
        del engine

    def test_data_major_expert_dp_aligns_with_data_axis(self):
        self._assert_axis_alignment(["data", "pipe"])

    def test_pipe_major_expert_dp_aligns_with_data_axis(self):
        self._assert_axis_alignment(["pipe", "data"])


class TestZeroLazyInitAllGroups(DistributedTest):
    """Pin ``_lazy_init_hp_params_optimizer_state`` linking every optim group.

    Pre-fix, ZeRO-1 invoked ``_lazy_init_hp_params_optimizer_state`` from
    inside ``_optimizer_step(group_no)`` and gated subsequent calls on
    ``_hp_optimizer_states_linked``.  After group 0 stepped, the gate
    flipped; groups 1..N (typically the MoE expert groups, which sit
    after the non-expert group in the param-group list emitted by
    ``split_params_into_different_moe_groups_for_optimizer``) reached
    the link path with their ``optimizer.state[flat_hp_partition]``
    still empty (a defaultdict miss before that group's step ever ran).

    ``set_optim_state_fragment`` then iterated an empty dict and set
    ``optim_fragment = {}`` for every expert param.  Downstream code
    that walks ``lp._hp_mapping.optim_fragment`` (MoEvement's
    ``_collect_param_optim_state``, the FSDP / pipeline checkpoint
    writers, anything that asks for "the Adam state slice owned by this
    rank") then silently dropped Adam moments for every non-first-group
    param — so MoE expert active snapshots carried only ``params.*``,
    no ``optimizer.exp_avg`` / ``optimizer.exp_avg_sq``.

    This test runs one full train_batch on a PP=2 DP=2 MoE config (one
    non-expert group at index 0, one expert group at index 1) and
    asserts every owned param's ``optim_fragment`` carries the Adam
    state keys.  Fails on the pre-fix code path; passes once the link
    is moved out of ``_optimizer_step`` to after the per-group loop.
    """

    world_size = 4

    def test_optim_fragment_populated_for_every_param_group(self):
        config = _happy_engine_config("fp16")
        engine = _build_happy_engine(config)
        di = _happy_data_iter()
        # fp16 dynamic-loss-scaler starts at 65536 and halves on every
        # overflow.  With this small toy model, several initial iters
        # overflow before the scaler decays enough to let the
        # optimizer step succeed; only at that point does ZeRO's
        # ``_lazy_init_hp_params_optimizer_state`` link
        # ``optim_fragment`` for every group's params.  20 iters
        # covers the worst-case decay path.
        for _ in range(20):
            engine.train_batch(data_iter=di)

        opt = engine.optimizer
        # ZeRO-1 keeps the param→hp_mapping linkage on the *full-rank*
        # ``bit16_groups`` (per-group tensor list) so we can iterate
        # every owned param and check its mapping regardless of which
        # rank of the DP partition holds the fragment.
        missing = []
        for group_idx, lp_group in enumerate(opt.bit16_groups):
            is_moe = opt.is_moe_param_group[group_idx] if hasattr(opt, "is_moe_param_group") else False
            for lp in lp_group:
                hp_mapping = getattr(lp, "_hp_mapping", None)
                if hp_mapping is None:
                    # This rank doesn't own a fragment for this param —
                    # legitimate under ZeRO-1 partitioning, skip.
                    continue
                frag = getattr(hp_mapping, "optim_fragment", None)
                if frag is None or "exp_avg" not in frag or "exp_avg_sq" not in frag:
                    missing.append((group_idx, is_moe, frag))

        assert not missing, (f"rank {dist.get_rank()}: {len(missing)} owned params missing Adam optim_fragment "
                             f"keys after first train_batch.  Sample: {missing[:3]}.  "
                             "This is the per-group-step / lazy-init-gate ordering bug — link must run after "
                             "all groups have stepped, not after group 0 only.")

        # End-to-end check: the original observation was that MoE
        # expert snapshots carried only ``params.*`` keys with no
        # ``optimizer.exp_avg`` / ``optimizer.exp_avg_sq``.  With
        # ``optim_fragment`` populated for every group, ``snap_active``'s
        # ``_collect_param_optim_state`` walks the mapping and emits the
        # Adam keys.  Walk the persisted snapshots and assert at least
        # one expert op carries optimizer entries — defends against a
        # future regression where the lazy-init fix lands but a
        # downstream gate (e.g. ZeRO-1 + EP non-owner branch) silently
        # drops the keys back to params-only on the snapshot side.
        coord = engine.moevement_coordinator
        expert_snaps_with_optim = 0
        expert_snaps_total = 0
        for snap in coord.snapshot_engine._persisted_snapshots.values():
            if not snap.is_active:
                continue
            if "expert" not in snap.name:
                continue
            expert_snaps_total += 1
            keys = list(snap.state_dict.keys())
            if any(k.startswith("optimizer.") for k in keys):
                expert_snaps_with_optim += 1

        # Pre-fix, EVERY expert active snapshot was missing
        # ``optimizer.*`` keys: lazy_init never linked the expert
        # group's ``optim_fragment``, so emission silently dropped the
        # Adam slice forever.  Post-fix, expert snapshots taken AFTER
        # the fp16 dynamic-loss-scaler decays enough for
        # ``optimizer.step`` to succeed (lazy_init then runs and the
        # snapshot path picks up the populated fragment) carry optim
        # entries.  Earliest snapshots (taken while overflow keeps
        # skipping ``step``) legitimately have no optim state because
        # there's nothing to capture yet.  Asserting at least one
        # post-decay expert snapshot carries optim entries is the
        # tight regression check on the fix flowing through; the
        # iter-0/overflow-window snapshots are a separate concern
        # (those iters wouldn't restore Adam moments under recovery,
        # but no iter-0 state can — the optimizer hasn't run).
        if expert_snaps_total > 0:
            assert expert_snaps_with_optim > 0, (
                f"rank {dist.get_rank()}: 0 of {expert_snaps_total} expert active snapshots "
                "carry optimizer.* keys.  ZeRO-1 lazy-init fix didn't flow through to snapshot "
                "emission — every expert snapshot persisted across all 20 iters lacks Adam state.")


class TestMoEvementSendGuardWiring(DistributedTest):
    """Verify ``PipelineEngine._exec_send_{activations,grads}`` consults the guard.

    The guard's pure logic is unit-tested in
    ``TestRecoveryBarrier.test_should_skip_pipeline_send_*``: we know
    it returns the right boolean given a ``_recovering_stages_in_my_pp``
    set.  What's orthogonal and *not* pinned there is whether the
    pipeline engine actually calls the guard from the real send path.
    A wiring regression — someone removes the ``if is_recovery and
    coord.should_skip_pipeline_send(...)`` block on pipe/engine.py:1065
    or :1118 — would silently break recovery under any single-stage
    failure without tripping any unit-level contract test.

    The test wraps ``coord.should_skip_pipeline_send`` with a counter,
    runs the whole-job-restart flow (simpler than asymmetric failure;
    every rank recovers in lockstep), and asserts the counter is
    non-zero on every rank after replay.  Counter value is non-zero
    rather than a specific number because schedule exec order depends
    on W_sparse and pipe-stage topology, and we want the test to
    survive honest changes to either.
    """

    world_size = 4

    def test_guard_consulted_during_replay(self, tmp_path_factory):
        import tempfile

        engine = _build_happy_engine(_happy_engine_config())
        coord = engine.moevement_coordinator
        data_iter = _happy_data_iter()

        for _ in range(3):
            engine.train_batch(data_iter=data_iter)

        save_dir = tempfile.mkdtemp(prefix="moev_b4_")
        tag = "send_guard_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        dist.barrier()

        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
        coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)

        # Install the counter after load_sparse_checkpoint so the
        # recovery_barrier call it doesn't make at that point doesn't
        # inflate the count.  The counter wraps the real method so
        # behavior is unchanged — this is a "was it called" check, not
        # a behavior override.
        call_count = 0
        original_guard = coord.should_skip_pipeline_send

        def counting_guard(downstream_stage_id):
            nonlocal call_count
            call_count += 1
            return original_guard(downstream_stage_id)

        coord.should_skip_pipeline_send = counting_guard

        max_iters = max(8, coord.scheduler.w_sparse * 2)
        for _ in range(max_iters):
            engine.train_batch(data_iter=data_iter)
            if not coord._recovering:
                break

        assert call_count > 0, (f"PipelineEngine never consulted should_skip_pipeline_send during recovery "
                                f"on rank {dist.get_rank()} — send-guard wiring is gone")


class TestMoEvementUpstreamLoggingE2E(DistributedTest):
    """Pin that real ``train_batch`` wires through to the upstream logger.

    ``UpstreamLogger``'s internals (``log_activation``, ``log_gradient``,
    ``get_*_for_replay``, ``garbage_collect``, send/recv round-trip) are
    covered by unit tests in ``test_sparse_checkpoint.py``, and the
    ``_pp_log_transfer`` path is covered by a distributed test that
    *manually* populates ``_logs``.  What none of those pin is whether
    ``PipelineEngine`` actually calls the logger hooks during real
    forward/backward: ``_exec_send_activations`` (pipe/engine.py:1073)
    calls ``coord.on_send_activations`` on every micro-batch, and
    ``_exec_send_grads`` (pipe/engine.py:1174) calls
    ``coord.on_send_gradients`` — both guarded by ``not is_recovery``.
    A regression that strips either hook, or widens the guard so it
    short-circuits during normal training, would silently empty the
    log and only surface during a B.3-style cascade replay when the
    recovering receiver has nothing to ingest.

    Runs a handful of training iterations and asserts that each
    sender stage's ``upstream_logger._logs`` actually accumulated
    entries of the expected direction.
    """

    world_size = 4

    def test_train_batch_populates_upstream_logger(self):
        engine = _build_happy_engine(_happy_engine_config("fp16"))
        coord = engine.moevement_coordinator
        # upstream_logging defaults True (see ``MoEvementConfig``), so
        # the logger should be constructed.  Explicitly pin the default
        # here so a regression that flips it silently (or that drops the
        # logger instantiation under PP) fails with a clear message
        # instead of the AttributeError further down.
        assert coord.upstream_logger is not None, (
            "upstream_logger should exist when moevement is enabled with default config "
            "(upstream_logging=True); coordinator construction may have skipped it")

        data_iter = _happy_data_iter_for("fp16")
        for _ in range(4):
            engine.train_batch(data_iter=data_iter)

        # Flush async H2D copies so the cpu-side entry tensors are
        # actually populated before inspection.  Not strictly required
        # for counting, but cheap insurance if this test ever grows to
        # assert on tensor values.
        coord.upstream_logger.synchronize()

        # Count entries by direction prefix.  Tuple outputs log each
        # member as ``activation_{i}`` / ``gradient_{i}`` (see
        # ``UpstreamLogger._log_tensor``); ``startswith`` covers both
        # the single-tensor and tuple paths so the assertion survives
        # a future fixture change that swaps single-tensor sends for
        # tuple sends.  Garbage collection runs at window boundaries
        # (``garbage_collect(global_step - w_sparse)``) and keeps the
        # last w_sparse iterations' logs — with w_sparse=1 on this
        # tiny model, iterations 3 and 4 survive after the final
        # ``on_iteration_end`` GC pass, so the counts stay > 0.
        activation_count = 0
        gradient_count = 0
        for entries in coord.upstream_logger._logs.values():
            for entry in entries:
                if entry.direction.startswith("activation"):
                    activation_count += 1
                elif entry.direction.startswith("gradient"):
                    gradient_count += 1

        # Stage 0 (``is_first_stage``) only *sends activations* forward;
        # stage 1 (``is_last_stage``) only *sends gradients* backward.
        # Under our PP=2 fixture each stage has exactly one of the two,
        # so we check the matching direction on each side.
        if not engine.is_last_stage():
            assert activation_count > 0, (f"stage {engine.stage_id} ran forward 4x but logged 0 activations; "
                                          f"on_send_activations hook at pipe/engine.py:1073 may have been "
                                          f"stripped from _exec_send_activations, or the 'not is_recovery' "
                                          f"guard short-circuited during normal training")
        if not engine.is_first_stage():
            assert gradient_count > 0, (f"stage {engine.stage_id} ran backward 4x but logged 0 gradients; "
                                        f"on_send_gradients hook at pipe/engine.py:1174 may have been "
                                        f"stripped from _exec_send_grads, or the 'not is_recovery' "
                                        f"guard short-circuited during normal training")


class TestMoEvementUpstreamLogsDiskPersistence(DistributedTest):
    """Upstream logs survive ``save_sparse_checkpoint`` → wipe → ``load_sparse_checkpoint``.

    ``save_sparse_checkpoint`` calls
    ``upstream_logger.save_to_disk`` (coordinator.py:710) and
    ``load_sparse_checkpoint`` pairs with ``load_from_disk``
    (coordinator.py:807).  This is the path a whole-job restart
    uses to ship log payloads across process boundaries — without
    it, a post-restart recovering rank whose DP / PP peers also
    just restarted would have no logs to replay.

    B.1 does not exercise it: B.1's recovering-everywhere scenario
    short-circuits ``_pp_log_transfer`` (no live neighbour to ship
    from), so the replay loop never tries to read logs.  Any
    regression to the serialise / deserialise path would pass B.1
    silently and only surface when a real user hits the cross-
    crash log-persistence case.

    Runs a handful of iters to populate ``_logs``, snapshots the
    entries, rounds-trips the sparse-checkpoint save → fault →
    load cycle, and pins that the reloaded entries equal the
    pre-fault ones.
    """

    world_size = 4

    def test_logs_survive_save_and_reload(self):
        import tempfile

        engine = _build_happy_engine(_happy_engine_config("fp16"))
        coord = engine.moevement_coordinator
        assert coord.upstream_logger is not None, "upstream_logging default=True, logger should exist"

        data_iter = _happy_data_iter_for("fp16")
        for _ in range(4):
            engine.train_batch(data_iter=data_iter)

        # Drain the async H2D copies so the entry tensors carry real
        # data before we freeze their values for comparison.
        coord.upstream_logger.synchronize()

        # Snapshot the logs as a pure-Python structure.  We clone the
        # tensors because ``simulate_rank_failure`` clears ``_logs``
        # and the underlying pinned buffers may be reused once gc
        # runs during recovery.
        pre_fault = {}
        for key, entries in coord.upstream_logger._logs.items():
            pre_fault[key] = [(e.iteration, e.micro_batch_id, e.stage_id, e.direction, e.tensor.clone())
                              for e in entries]
        assert pre_fault, (f"no upstream logs captured after 4 iters on rank {dist.get_rank()}; "
                           f"upstream-logging wiring regressed before the disk path even runs")

        save_dir = tempfile.mkdtemp(prefix="moev_logs_")
        tag = "logs_disk_tag"
        coord.save_sparse_checkpoint(save_dir, tag)
        # ``save_sparse_checkpoint`` already calls ``flush_persist``,
        # but the world barrier below is what every rank needs to
        # agree on before a peer starts loading.
        dist.barrier()

        # Wipe local state as a fresh-replacement would; the helper
        # clears ``upstream_logger._logs`` among other things.
        simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
        assert len(coord.upstream_logger._logs) == 0, (
            "simulate_rank_failure did not clear upstream_logger._logs — precondition "
            "for the load round-trip is broken")

        ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
        assert ok is True, f"load_sparse_checkpoint returned False on rank {dist.get_rank()}"

        post_load = coord.upstream_logger._logs
        assert set(post_load.keys()) == set(
            pre_fault.keys()), (f"log key set mismatch after load on rank {dist.get_rank()}: "
                                f"pre-fault = {sorted(pre_fault.keys())}, post-load = {sorted(post_load.keys())}")

        for key, pre_entries in pre_fault.items():
            post_entries = post_load[key]
            assert len(post_entries) == len(pre_entries), (
                f"entry count mismatch for key {key} on rank {dist.get_rank()}: "
                f"pre={len(pre_entries)}, post={len(post_entries)} — "
                f"save_to_disk/load_from_disk may have dropped entries")
            # ``save_to_disk`` iterates ``_logs.values()`` in dict
            # order; ``load_from_disk`` appends in file order; the
            # dict here preserves insertion order on 3.7+.  So the
            # i-th entry at each side should correspond.
            for pre, post in zip(pre_entries, post_entries):
                pre_iter, pre_mb, pre_stage, pre_dir, pre_tensor = pre
                assert post.iteration == pre_iter, (f"iteration mismatch {key} on rank "
                                                    f"{dist.get_rank()}: pre={pre_iter}, post={post.iteration}")
                assert post.micro_batch_id == pre_mb
                assert post.stage_id == pre_stage
                assert post.direction == pre_dir, (f"direction mismatch {key} on rank "
                                                   f"{dist.get_rank()}: pre={pre_dir}, post={post.direction}")
                # Disk round-trip is pickle-of-CPU-tensors; no
                # dtype conversion or reshape, so the payload should
                # be bit-identical.
                torch.testing.assert_close(post.tensor, pre_tensor, rtol=0, atol=0)


class TestMoEvementPeerPull(DistributedTest):
    """Peer-pull recovery: fail one rank, restore its shard from a DP peer.

    Models the "single-rank failure, replacement rank comes up without
    disk state" scenario.  Ring replication accumulates each rank's
    shard in its DP peers' ``_received_snapshots`` during training;
    when a rank fails, its replacement opts out of a disk reload and
    pulls the shard directly from any surviving DP peer via the gloo
    replication group.

    Distinct from ``TestMoEvementWholeJobRestart`` (which tests the
    disk-reload path): peer-pull never touches disk, and only the
    failed-rank ↔ peer pair exchange messages.  Other DP groups
    (here stage 1's ranks {2, 3}) are idle during the pull and
    coordinate purely through world-level ``dist.barrier`` before
    and after.

    Scope here is deliberately narrower than B.1:
      * pins the pull protocol (request → serve → recv path).
      * pins the converter / recovery-flag side effects on the
        replacement rank.
      * does NOT drive the subsequent replay loop — that's identical
        to B.1's once recovery state is in place, so asserting it
        again here would duplicate coverage without adding signal.
    """

    world_size = 4

    def test_peer_pull_restores_recovery_state(self):
        engine = _build_happy_engine(_happy_engine_config())
        coord = engine.moevement_coordinator
        data_iter = _happy_data_iter()

        # Three iters so at least one window boundary completes and
        # triggers ring replication — rank 1 must hold rank 0's shard
        # in ``_received_snapshots[0]`` for the pull to succeed.
        # Three iters so at least one window boundary completes and
        # triggers ring replication — rank 1 must hold rank 0's shard
        # in ``_received_snapshots[0]`` for the pull to succeed.
        for _ in range(3):
            engine.train_batch(data_iter=data_iter)

        # Replication is best-effort (fires in a background executor);
        # block on the future so the receiver dict is actually
        # populated before we poke it.
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
        dist.barrier()

        my_rank = dist.get_rank()

        # Stage-0 DP group is {0, 1}; each holds the other's shard.
        # Sanity-check the pre-condition on rank 1 before engineering
        # rank 0's failure around it.
        if my_rank == 1:
            received = coord.snapshot_engine._received_snapshots
            assert 0 in received and received[0], ("rank 1 lost its ring-received copy of rank 0's shard before "
                                                   "peer-pull even started — replication is probably broken")

        # Peer-pull.  Ranks 0 (failed) and 1 (peer) run the protocol on
        # their stage-0 replication group.  Ranks 2, 3 are in stage-1's
        # replication group — they're idle during the pull and only
        # sync through the bracketing world barriers.
        if my_rank == 0:
            simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
            # Sanity: rank 0 truly has nothing locally post-failure,
            # so a successful pull can only be explained by peer data.
            assert len(coord.snapshot_engine._persisted_snapshots) == 0
            assert 1 not in coord.snapshot_engine._received_snapshots

            # ``peer_rank=1`` is group-local in the stage-0 replication
            # group {0, 1}; ``my_dp_rank_in_replication_group=0`` tells
            # the peer which sender's shard to serve — rank 1's
            # ``_received_snapshots[0]`` contains exactly rank 0's own
            # pre-failure shard.
            ok = coord.load_sparse_from_peer(peer_rank=1, my_dp_rank_in_replication_group=0, model=engine.module)
            assert ok is True, "peer-pull returned False — peer reported no shard"
        elif my_rank == 1:
            coord.serve_sparse_snapshot_to_peer(requester_rank=0)

        dist.barrier()

        # Post-pull assertions on the replacement rank: recovery is
        # active, the pulled state is cached for subsequent
        # ``_activate_from_snapshot`` calls, and the converter has
        # operator state to drive replay.
        if my_rank == 0:
            assert coord._recovering is True
            assert coord._cached_snapshot_data is not None
            metadata, per_iter_operator_states = coord._cached_snapshot_data
            assert metadata is not None and metadata.get("per_iter_active"), (
                "cached metadata should describe the operators the peer sent")
            assert per_iter_operator_states, "cached per_iter_operator_states should be non-empty after a successful pull"
            assert len(
                coord.converter._operator_states) > 0, ("converter should be initialised from the pulled snapshot")


class TestRegenSymmetricUnderAsymmetricTrigger(DistributedTest):
    """Pin the regen path's world-collective symmetry under per-rank triggers.

    ``_generate_schedule_world_aligned`` does a world-group
    ``all_reduce(MAX)`` on ``w_sparse`` so every PP stage agrees on
    the cadence.  The function itself is symmetric;
    its caller in ``on_iteration_end`` is not — the regen condition
    (``scheduler.should_reorder`` for popularity, or
    ``_iter_time_drift_exceeds_threshold`` for iter-time drift) reads
    per-rank state and can flip on one rank while staying off on others.
    A rank that decides to regen alone enters the world all_reduce while
    its peers move on to the next iter's ``recovery_barrier`` (which
    issues its own world collective); the two collectives queue out of
    order on the world group and NCCL deadlocks.

    Surfaced empirically at HIDDEN=4096 NUM_EXPERTS=16 — py-spy showed
    rank 2 stuck in ``_generate_schedule_world_aligned`` while ranks
    0/1/3 sat in ``_world_recovery_handshake``; NCCL watchdog reported
    ``BROADCAST(numel=2)`` after its 10-min timeout.  The fix
    ``all_reduce(MAX)``-es a single bit before deciding to call
    ``_generate_schedule_world_aligned`` so any rank's regen request
    pulls every rank into the call together.

    The test mocks ``should_reorder`` so rank 2 alone wants to regen,
    then runs one more ``train_batch``.  Without the fix the call
    deadlocks at the world all_reduce; with the fix every rank observes
    ``any_rank_regen=True`` via the OR-reduce, calls
    ``_generate_schedule_world_aligned`` symmetrically, and the iter
    completes.
    """

    world_size = 4

    def test_asymmetric_regen_trigger_does_not_deadlock(self):
        engine = _build_happy_engine(_happy_engine_config("fp16"))
        coord = engine.moevement_coordinator
        data_iter = _happy_data_iter_for("fp16")
        my_rank = dist.get_rank()

        regen_calls = [0]
        orig_regen = coord._generate_schedule_world_aligned

        # Wrap ``_generate_schedule_world_aligned`` so the test can see
        # whether it actually fires on every rank (the symmetry property
        # the OR-reduce guarantees) rather than only on the rank that
        # set ``regen_reason``.
        def _counting_regen(iter_time_sec):
            regen_calls[0] += 1
            return orig_regen(iter_time_sec)

        coord._generate_schedule_world_aligned = _counting_regen
        # Force ``should_reorder`` to fire on rank 2 only — exactly the
        # asymmetric trigger the deadlock reproduced.
        coord.scheduler.should_reorder = lambda my_rank=my_rank: my_rank == 2

        try:
            # The regen-decision block is gated on a window boundary
            # (``_window_step >= scheduler.w_sparse``).  Run enough iters
            # post-mock to cross at least one boundary — happy-path
            # ``w_sparse`` is small (2-4) so 6 iters comfortably
            # exercises the regen path with the asymmetric trigger.
            #
            # Without the OR-reduce fix the first crossed boundary
            # hangs forever on rank 2's lone world all_reduce (NCCL
            # watchdog kills the test at 10 min).  With the fix all
            # ranks observe ``any_rank_regen=True`` and call
            # ``_generate_schedule_world_aligned`` symmetrically.
            for _ in range(6):
                engine.train_batch(data_iter=data_iter)
        finally:
            coord._generate_schedule_world_aligned = orig_regen

        # Every rank must have entered the regen at least once — the
        # OR-reduce promotes rank 2's lone "yes" into a world-wide
        # "yes".  Without the fix non-trigger ranks see
        # ``any_rank_regen=False`` and skip the call entirely (or the
        # whole world hangs via the original bug, which the watchdog
        # catches).
        assert regen_calls[0] >= 1, (f"rank {my_rank}: _generate_schedule_world_aligned was not called once "
                                     f"after asymmetric regen trigger across 6 iters; got {regen_calls[0]}.  "
                                     "Either the OR-reduce dropped the trigger, or w_sparse exceeded the iter "
                                     "budget — bump the iter count or pin w_sparse if the workload changed.")
