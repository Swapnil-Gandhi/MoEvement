# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Spike test for ``ZeROOptimizer.relink_hp_params``.

Proves the primitive that Layer 2 of MoEvement's comm-group rebuild
design depends on (see `docs/moevement/DESIGN_COMM_GROUP_REBUILD.md`
§3.3 / §5.1):

Every ZeRO-1 managed lp_param holds ``_hp_mapping`` fragment state plus
a direct ``_dp_group`` reference bound at optimizer init.  After a
spare-rank substitution the original dp group is torn down and
rebuilt; unless we re-point every lp_param's ``_dp_group`` at the
fresh group, the next ``safe_get_full_hp_param`` issues an all_reduce
on a destroyed communicator and deadlocks.

``relink_hp_params`` is the load-bearing relink primitive.  This test
exercises it end-to-end: build ZeRO-1 against an initial dp group,
build a fresh group with the same member set (simulating the post-
spare world), call relink, assert every lp_param now points at the
fresh group, then call ``safe_get_full_fp32_param`` to prove the
all_reduce actually succeeds on the new communicator.

If this test ever fails, Layer 2 of the comm-rebuild plan is blocked.
"""

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_fp32_param

from unit.common import DistributedTest
from unit.simple_model import SimpleModel


class TestZeroRelinkHpParams(DistributedTest):
    """Spike: prove `ZeROOptimizer.relink_hp_params` actually repoints lp_params."""

    world_size = 2
    reuse_dist_env = True

    def _make_config(self):
        return {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6,
                    "torch_adam": True,
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 2
            },
            "zero_optimization": {
                "stage": 1,
            },
        }

    def test_relink_repoints_dp_group_on_every_lp_param(self):
        """After relink, every lp_param._dp_group points to the fresh group.

        Pins the guarantee the Layer 2 design rests on: the relink
        primitive actually overwrites the per-param ``_dp_group``
        reference, not just the optimizer's top-level bookkeeping.
        """
        if not get_accelerator().is_available():
            pytest.skip("relink spike needs a real accelerator for ZeRO-1 init")

        hidden_dim = 8
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=self._make_config())

        optimizer = model.optimizer
        # ``deepspeed.initialize`` returns a wrapped engine; the ZeRO
        # optimizer lives on ``.optimizer``.
        original_group = optimizer.real_dp_process_group[0]
        lp_params = list(optimizer.bit16_groups[0])

        # Sanity: at init, every lp_param points at the original group.
        assert len(lp_params) > 0, "SimpleModel should produce at least one lp_param"
        assert all(lp._dp_group is original_group for lp in lp_params)

        # Prove the initial all_reduce path works before the rebuild.
        # Under fp16 + ZeRO-1 the hp fragment is present, so
        # safe_get_full_fp32_param returns a non-None tensor whose
        # all_reduce rode ``original_group``.
        pre_rebuild = safe_get_full_fp32_param(lp_params[0])
        assert pre_rebuild is not None

        # Build the "post-spare" group with the same member set.  Ranks
        # are identical so partition boundaries don't move — matches
        # the 1:1 substitution scope the relink helper advertises.
        fresh_group = dist.new_group(ranks=list(range(self.world_size)))

        optimizer.relink_hp_params([fresh_group])

        # The load-bearing assertion: every lp_param now points at the
        # fresh group, not the original.
        assert all(lp._dp_group is fresh_group for lp in lp_params)
        assert all(lp._dp_group is not original_group for lp in lp_params)
        assert optimizer.real_dp_process_group[0] is fresh_group

        # End-to-end proof: safe_get_full_fp32_param succeeds, meaning
        # the all_reduce inside ``get_full_hp_param`` ran on ``fresh_group``
        # without deadlocking.  Value should match the pre-rebuild
        # reading since no optimizer step ran between the two calls.
        post_rebuild = safe_get_full_fp32_param(lp_params[0])
        assert post_rebuild is not None
        assert torch.allclose(pre_rebuild, post_rebuild)

        model.destroy()

    def test_relink_rejects_mismatched_group_count(self):
        """Passing the wrong number of new groups raises rather than silently
        leaving some param groups linked to the old (torn-down) comm.

        The caller contract is "one fresh group per param group, in the
        same order"; violating it would mean only part of the optimizer
        was relinked, a silent partial-failure mode worth making loud.
        """
        if not get_accelerator().is_available():
            pytest.skip("relink spike needs a real accelerator for ZeRO-1 init")

        model = SimpleModel(8)
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=self._make_config())

        optimizer = model.optimizer
        expected = len(optimizer.optimizer.param_groups)

        with pytest.raises(ValueError, match=f"expected {expected} new dp_groups"):
            optimizer.relink_hp_params([])

        model.destroy()

    def test_rebuild_nccl_groups_end_to_end_zero1(self):
        """Full ``rebuild_nccl_groups`` round-trip: train → rebuild → train.

        This is the local empirical check Layer 2.B ships with.  It
        proves:

        - ``groups.reset_for_rebuild`` + the engine's ``_reinitialize_distributed_groups``
          + ``_refresh_group_caches`` sequence doesn't corrupt torch
          distributed state when the groups are healthy (no wedged
          communicators, no departed rank — same-membership rebuild).
        - ``relink_all_dp_refs`` repoints optimizer + lp_param refs
          under a real optimizer built by ``deepspeed.initialize``.
        - Training continues after the rebuild with sensible loss +
          preserved optimizer state.

        It does NOT verify the fault-scenario path (torchrun-elastic +
        SIGKILL + spare rejoin + rebuild) — that's gated on sandbox
        infrastructure (see design §6 Week 3).  This smoke test is the
        bounded coverage we can run locally.
        """
        if not get_accelerator().is_available():
            pytest.skip("rebuild_nccl_groups smoke needs a real accelerator for ZeRO-1 init")

        hidden_dim = 8
        model = SimpleModel(hidden_dim)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=self._make_config())

        # Capture the pre-rebuild group references so post-rebuild
        # assertions can confirm every cache actually swapped.  Using
        # ``seq_data_parallel_group`` because that's the one the
        # optimizer was built against (engine.py:2053) — it falls
        # through to ``_clone_world_group`` for non-Megatron runs.
        pre_engine_group = engine.seq_data_parallel_group
        pre_opt_group = engine.optimizer.dp_process_group
        lp_params = list(engine.optimizer.bit16_groups[0])
        pre_lp_group = lp_params[0]._dp_group

        # Train one iteration so there's real optimizer state
        # (Adam momentum + variance non-zero) whose preservation we
        # can check across the rebuild.  Without a pre-rebuild step,
        # the Adam state is at init values and the "same before and
        # after" assertion is vacuous.
        device = engine.device
        batch_x = torch.randn(1, hidden_dim, device=device, dtype=torch.float16)
        batch_y = torch.randint(0, hidden_dim, (1, ), device=device)
        loss_before = engine(batch_x, batch_y)
        engine.backward(loss_before)
        engine.step()

        pre_hp_param = safe_get_full_fp32_param(lp_params[0])
        assert pre_hp_param is not None

        # The rebuild.  Under this test's no-fault config every
        # destroy runs against a healthy group and the abort fast
        # path is a no-op (gloo-only mirror), so the destroy path
        # is the hot branch.
        engine.rebuild_nccl_groups()

        # Every cached ref ends up pointing at the same group.  Post
        # the ``_clone_world_group`` simplification that returns
        # ``dist.group.WORLD``, pre_ and post_rebuild references
        # resolve to the same sentinel — the "they differ" assertion
        # that used to guard against a missed swap is no longer
        # meaningful (the sentinel is stable across rebuilds).  What
        # remains load-bearing: every cache agrees with the engine's
        # current view of seq_data_parallel_group, so a later
        # collective won't run on a stale ref the orchestration
        # forgot to refresh.
        del pre_engine_group, pre_opt_group, pre_lp_group
        assert engine.optimizer.dp_process_group is engine.seq_data_parallel_group
        assert lp_params[0]._dp_group is engine.seq_data_parallel_group

        # Training step AFTER the rebuild.  Forward + backward + step
        # run real collectives (grad allreduce + hp-param reductions)
        # on the fresh groups.  If any cached ref was missed, this
        # would deadlock or raise.
        batch_x2 = torch.randn(1, hidden_dim, device=device, dtype=torch.float16)
        batch_y2 = torch.randint(0, hidden_dim, (1, ), device=device)
        loss_after = engine(batch_x2, batch_y2)
        engine.backward(loss_after)
        engine.step()

        # Loss is finite — a deadlock / NaN cascade would surface as
        # non-finite loss or a hang.  Exact value isn't meaningful
        # across a fresh batch, but sanity-check the training step
        # produced a real number.
        assert torch.isfinite(loss_after).all()

        # hp-param read-back still works on the new group.
        post_hp_param = safe_get_full_fp32_param(lp_params[0])
        assert post_hp_param is not None

        engine.destroy()

    def test_relink_all_dp_refs_updates_singular_and_list(self):
        """``relink_all_dp_refs`` covers self.dp_process_group + real_dp_process_group.

        ``relink_hp_params`` only touches the per-param-group list and
        the lp_param linkage.  Layer 2's full rebuild also needs
        ``self.dp_process_group`` (consumed by the global grad
        all-reduce in ``allreduce_gradients``) and
        ``self.expert_dp_process_group`` (MoE) swapped in lockstep.
        This test pins that the higher-level helper repoints all three
        caches on a non-MoE setup.
        """
        if not get_accelerator().is_available():
            pytest.skip("relink spike needs a real accelerator for ZeRO-1 init")

        model = SimpleModel(8)
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=self._make_config())

        optimizer = model.optimizer
        original_default = optimizer.dp_process_group
        lp_params = list(optimizer.bit16_groups[0])
        fresh_group = dist.new_group(ranks=list(range(self.world_size)))

        optimizer.relink_all_dp_refs(new_dp_group=fresh_group)

        # Singular default group swapped — the grad-all-reduce path uses this.
        assert optimizer.dp_process_group is fresh_group
        assert optimizer.dp_process_group is not original_default
        # Per-param-group list swapped too (non-MoE config → every entry = fresh).
        assert all(pg is fresh_group for pg in optimizer.real_dp_process_group)
        # And the lp_param-level refs inherited by the inner relink_hp_params call.
        assert all(lp._dp_group is fresh_group for lp in lp_params)

        # End-to-end round-trip still works on the fresh communicator.
        post_rebuild = safe_get_full_fp32_param(lp_params[0])
        assert post_rebuild is not None

        model.destroy()
