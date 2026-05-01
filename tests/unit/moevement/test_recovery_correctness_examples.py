# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Recovery correctness tests on the canonical example workloads.

The existing ``TestMoEvementRecoveryEquivalence`` family in
``test_engine_integration.py`` pins the recovery contract on a toy
fixture (``_build_happy_engine``: 16-hidden, 2-expert, 4-class).  This
file lifts the same "post-recovery weights == fault-free weights at
rollback iter" assertion onto the **canonical example workloads** —
``examples/moevement/_common.py`` (cifar) and
``examples/moevement/train_gpt_moe.py`` (transformer + MoE FFN) —
so the example surface comes with a correctness guarantee per real
workload, not just on the unit-test fixture.

Three correctness probes:
1. **Snapshot to CPU**: covered transitively — the load_sparse_checkpoint
   path reads the persisted CPU snapshot bytes and the post-recovery
   weights match, so the snapshot was correct.
2. **Replication to remote CPU**: covered by ``test_distributed.py``
   (``TestPeerReplicationDistributed``); the engine_config used here
   sets ``replication_factor=1`` so the snapshot is also gloo-shipped
   to the DP peer, exercising the wire path implicitly.
3. **Sparse-to-dense recovery**: the assertion at the bottom of each
   test method.  Loss-equivalence is the load-bearing claim.
"""

import importlib.util
import os
import sys
import tempfile

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.utils import RepeatingLoader

from unit.common import DistributedTest
from unit.moevement._fault_inject import simulate_rank_failure

_EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/moevement"))


def _import_example_module(filename, module_name):
    """Import an examples/moevement/*.py file as a module.

    The example scripts live outside the ``deepspeed`` import root and
    use bare ``from _common import ...`` lines that need
    ``examples/moevement/`` on ``sys.path``.  Push it in for the
    duration of the load, then pop so the rest of the test session
    isn't polluted.
    """
    sys.path.insert(0, _EXAMPLES_DIR)
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(_EXAMPLES_DIR, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _recovery_engine_config():
    """Engine config tuned for in-test recovery — w_sparse=1 so a
    single-iter window finalizes immediately.

    The user-facing example default ``pcie_bandwidth_gbs=1e-6`` is
    deliberately starved to force ``w_sparse > 1`` (so the example
    exercises the multi-iter snapshot path); in this test we want the
    opposite — w_sparse=1 produces a finalized window after iter 1, so
    a 4-iter run guarantees a non-empty ``_persisted_snapshots`` and
    a single, unambiguous rollback iter for the assertion.
    """
    return {
        "train_batch_size": 8,  # BATCH=4 per GPU * DP=2
        "train_micro_batch_size_per_gpu": 4,
        "steps_per_print": 100,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "torch_adam": True,
            },
        },
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 1,
        },
        "pipeline": {
            "activation_checkpoint_interval": 0,
        },
        "moevement": {
            "enabled": True,
            # Realistic bandwidth -> w_sparse=1, deterministic rollback iter.
            "pcie_bandwidth_gbs": 24.0,
            "initial_iter_time_sec": 0.05,
            "replication_factor": 1,
        },
    }


def _build_engine_with_model(config_dict, model):
    """Wrap ``deepspeed.initialize`` with the MoE-aware optimizer split.

    Mirrors ``_build_happy_engine`` from ``test_engine_integration.py``
    so the recovery probe runs the same engine wiring path as the toy
    fixture, just with the canonical example model swapped in.
    """
    param_group = {"params": list(model.parameters()), "name": "moev_recovery_test"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        config=config_dict,
        model=model,
        optimizer=optimizer,
        dist_init_required=False,
    )
    return engine


def _drive_recovery_replay(engine, coord, data_iter_factory, n_iters):
    """Run the post-load replay loop until coord exits recovery."""
    data_iter = data_iter_factory()
    max_replay = max(16, coord.scheduler.w_sparse * 3)
    for _ in range(max_replay):
        engine.train_batch(data_iter=data_iter)
        if not coord._recovering:
            return
    pytest.fail(f"recovery did not complete within {max_replay} iters "
                f"on rank {dist.get_rank()} (n_iters={n_iters}, w_sparse={coord.scheduler.w_sparse})")


def _assert_post_recovery_equivalence(engine, reference_weights, rollback_iter):
    """One round-trip of FP16 cast happens on restore; rtol/atol 2e-3 covers it."""
    post = {name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()}
    for name, restored in post.items():
        torch.testing.assert_close(
            restored,
            reference_weights[name],
            rtol=2e-3,
            atol=2e-3,
            msg=lambda msg, n=name: (f"param {n} post-recovery state diverged from fault-free state at "
                                     f"iter {rollback_iter} on rank {dist.get_rank()}; {msg}"),
        )


def _run_recovery_equivalence(engine, coord, fixed_sample, n_iters=4):
    """The shared 4-step recovery-equivalence drill.

    1. Train ``n_iters`` deterministic iters (RepeatingLoader over a single
       fixed sample).  Capture per-iter post-optimizer weights.
    2. ``coord.save_sparse_checkpoint`` -> flush -> barrier.
    3. Read the rollback iter off the persisted bundle's metadata; assert
       it points into the captured-weights range.
    4. ``simulate_rank_failure(zero_model_weights=True)`` -> load -> drive
       the replay loop -> assert restored weights match captured weights
       at the rollback iter.

    The rollback iter is read from the bundle (not hard-coded) so a
    finalize_window-target regression surfaces as a value mismatch rather
    than a silent reference drift.
    """

    def make_iter():
        return iter(RepeatingLoader([fixed_sample]))

    per_iter_weights = []
    data_iter = make_iter()
    for _ in range(n_iters):
        engine.train_batch(data_iter=data_iter)
        per_iter_weights.append({name: p.detach().clone().cpu() for name, p in engine.module.named_parameters()})

    save_dir = tempfile.mkdtemp(prefix="moev_recovery_examples_")
    tag = "recovery_examples"
    coord.save_sparse_checkpoint(save_dir, tag)
    coord.flush_persist()
    dist.barrier()

    persisted = coord.snapshot_engine._persisted_snapshots
    assert len(persisted) > 0, (f"no persisted snapshots after {n_iters} iters on rank {dist.get_rank()}; "
                                f"window-boundary promotion never fired")
    rollback_iters = {snap.iteration for snap in persisted.values()}
    assert len(rollback_iters) == 1, (f"persisted snapshots disagree on iteration: {rollback_iters} on rank "
                                      f"{dist.get_rank()} — finalize_window race or partial-window save bug")
    rollback_iter = rollback_iters.pop()
    assert 1 <= rollback_iter <= n_iters, (f"rollback_iter={rollback_iter} outside training range "
                                           f"[1..{n_iters}] on rank {dist.get_rank()}")
    reference = per_iter_weights[rollback_iter - 1]

    simulate_rank_failure(coord, model=engine.module, zero_model_weights=True)
    ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module, optimizer=engine.optimizer)
    assert ok is True, f"load_sparse_checkpoint returned False on rank {dist.get_rank()}"

    _drive_recovery_replay(engine, coord, make_iter, n_iters)
    _assert_post_recovery_equivalence(engine, reference, rollback_iter)


class TestRecoveryEquivalenceCifarExample(DistributedTest):
    """Post-recovery weights match fault-free weights on the cifar example model.

    Uses ``examples/moevement/_common.py``'s canonical CIFAR-shape MoE
    pipeline (ConvBackbone → 4× MoEBlock → Classifier).  Synthetic input
    of shape ``(B, 3, 32, 32)`` fp16 + ``(B,)`` int64 labels, same
    ``RepeatingLoader`` pattern the fixed-sample path uses elsewhere.
    """

    world_size = 4

    def test_post_recovery_equals_fault_free_at_rollback_iter(self):
        common = _import_example_module("_common.py", "_common_for_recovery_test")

        torch.manual_seed(42)
        model = common.build_pipeline_model()
        engine = _build_engine_with_model(_recovery_engine_config(), model)
        coord = engine.moevement_coordinator

        gen = torch.Generator().manual_seed(7)
        fixed_sample = (
            torch.randn(common.BATCH, common.IMG_C, common.IMG_H, common.IMG_W, dtype=torch.float16, generator=gen),
            torch.randint(0, common.NUM_CLASSES, (common.BATCH, ), generator=gen),
        )
        _run_recovery_equivalence(engine, coord, fixed_sample, n_iters=4)


class TestRecoveryEquivalenceGptMoEExample(DistributedTest):
    """Post-recovery weights match fault-free weights on the gpt-moe example model.

    Uses ``examples/moevement/train_gpt_moe.py``'s canonical transformer
    + MoE FFN model (GPTEmbedding → 4× GPTBlock → GPTHead).  Synthetic
    input of shape ``(B, S)`` int64 token IDs + ``(B, S)`` int64 targets.
    """

    world_size = 4

    def test_post_recovery_equals_fault_free_at_rollback_iter(self):
        # _common must be imported first so train_gpt_moe's
        # ``from _common import ...`` resolves.
        _import_example_module("_common.py", "_common_for_recovery_test")
        gpt = _import_example_module("train_gpt_moe.py", "train_gpt_moe_for_recovery_test")

        torch.manual_seed(42)
        model = gpt.build_gpt_pipeline_model()
        engine = _build_engine_with_model(_recovery_engine_config(), model)
        coord = engine.moevement_coordinator

        gen = torch.Generator().manual_seed(7)
        fixed_sample = (
            torch.randint(0, gpt.VOCAB_SIZE, (gpt.BATCH, gpt.SEQ_LEN), generator=gen, dtype=torch.long),
            torch.randint(0, gpt.VOCAB_SIZE, (gpt.BATCH, gpt.SEQ_LEN), generator=gen, dtype=torch.long),
        )
        _run_recovery_equivalence(engine, coord, fixed_sample, n_iters=4)
