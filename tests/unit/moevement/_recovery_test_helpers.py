# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Shared test fixtures for MoEvement recovery tests.

The pipeline model + DeepSpeed config + data iter are reused across
the single-host SIGKILL test (``test_moevement_recovery_under_sigkill.py``)
and the multi-host emulated-fault test
(``test_moevement_multihost_emulated.py``).  Lifting them keeps the
two recovery tests in lockstep on shape and avoids duplicate
maintenance when the toy model evolves.
"""

import torch
import torch.nn as nn

from deepspeed.moe.layer import MoE

# Toy-model dimensions — same numbers as the single-host SIGKILL test.
HIDDEN = 16
NUM_CLASSES = 4
NUM_EXPERTS = 2
BATCH = 2
SEQ = 4


class _Embed(nn.Module):

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.lin(x)


class _MoEBlock(nn.Module):

    def __init__(self, hidden=HIDDEN, num_experts=NUM_EXPERTS, ep_size=1):
        super().__init__()
        self.moe = MoE(hidden_size=hidden,
                       expert=nn.Linear(hidden, hidden),
                       num_experts=num_experts,
                       k=1,
                       ep_size=ep_size)

    def forward(self, x):
        out, _aux, _z = self.moe(x)
        return out


class _Head(nn.Module):

    def __init__(self, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.proj = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


def build_pipeline_model(topology=None):
    """Build the toy MoE pipeline model used by the recovery tests.

    Args:
        topology: Optional ``ProcessTopology``.  When ``None`` (default),
            ``PipelineModule`` falls back to ``PipeDataParallelTopology``
            (``axes=['pipe', 'data']``), which is the right shape for
            single-host runs — DP groups stay intra-process under
            mp.spawn so peer-pull goes over loopback exactly as the
            existing SIGKILL test expects.  Pass an explicit
            ``ProcessTopology(axes=['data', 'pipe'], ...)`` for the
            multi-host test where the launcher allocates ranks 0,1 to
            host A and 2,3 to host B and we want DP groups (and thus
            peer-pull) to traverse the network.
    """
    from deepspeed.pipe import PipelineModule, LayerSpec
    layers = [
        LayerSpec(_Embed, HIDDEN),
        LayerSpec(_MoEBlock, HIDDEN, NUM_EXPERTS, 1),
        LayerSpec(_MoEBlock, HIDDEN, NUM_EXPERTS, 1),
        LayerSpec(_Head, HIDDEN, NUM_CLASSES),
    ]
    if topology is not None:
        return PipelineModule(layers=layers, topology=topology, loss_fn=nn.CrossEntropyLoss())
    return PipelineModule(layers=layers, num_stages=2, loss_fn=nn.CrossEntropyLoss())


def engine_config(streaming_recovery=False):
    """DeepSpeed config for the recovery tests.

    fp16 + torch_adam + starved PCIe bandwidth (so the scheduler picks
    ``w_sparse > 1`` and the window boundary fires with multiple
    operators pending) — same shape the in-process rebuild test
    validates.
    """
    return {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
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
            "enabled": True,
            "pcie_bandwidth_gbs": 0.000001,
            "initial_iter_time_sec": 1.0,
            "streaming_recovery": streaming_recovery,
        },
    }


def data_iter():
    from deepspeed.utils import RepeatingLoader
    sample = (
        torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.float16),
        torch.randint(0, NUM_CLASSES, (BATCH, )),
    )
    return iter(RepeatingLoader([sample]))


def build_engine(config, skip_initial_broadcast, topology=None):
    """Mirror of ``_build_happy_engine`` with the recovery-mode flag plumbed.

    ``topology`` is forwarded to ``build_pipeline_model`` — see that
    docstring for when each axis order is appropriate.
    """
    import deepspeed
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    model = build_pipeline_model(topology=topology)
    param_group = {"params": list(model.parameters()), "name": "moe_test_params"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        optimizer=optimizer,
        dist_init_required=False,
        skip_initial_broadcast=skip_initial_broadcast,
    )
    return engine
