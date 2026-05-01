# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Shared helpers for the MoEvement example scripts.

Defines the canonical CIFAR-shaped MoE + pipeline model, the DeepSpeed
engine config with MoEvement enabled, and a CIFAR-10 / synthetic-fallback
data loader.  The example scripts (``train_cifar_moe.py``,
``resume_after_fault.py``, ``run_with_survivor_supervisor.py``) import
this module so the recovery story — not the model wiring — is the
~80% of what a reader sees.

Workload shape (matches the perf-investigation harness in
``profile_run.py`` so overhead measurements transfer between examples):

- 3-channel 32x32 input -> conv stem -> 64 spatial-token sequence ->
  Linear(64 -> hidden) -> 4x MoE blocks -> mean-pool classifier
- BATCH=4 microbatch * 64 spatial tokens = 256 routing tokens / mb
- PP=2 DP=2, fp16, ZeRO-1, 4 experts / MoE block, ep_size=2

Synthetic-tensor fallback (``cifar_data_iter(fake_data=True)``) is the
default so the scripts run without a torchvision download; pass
``fake_data=False`` (or set MOEV_DATA_DIR) to train on real CIFAR-10.
Recovery semantics are independent of the data source, so the resume
+ supervisor demos use the synthetic fallback by default.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.moe.layer import MoE

# Shape constants — pinned here so the example scripts agree.
HIDDEN = 256
NUM_CLASSES = 10  # CIFAR-10
NUM_MOE_LAYERS = 4
NUM_EXPERTS = 4
EP_SIZE = 2
BATCH = 4
# CIFAR image dims; pool ratio (32 -> 16 -> 8) is hardcoded in the conv stem.
IMG_C, IMG_H, IMG_W = 3, 32, 32
CONV_OUT_CHANNELS = 64
SPATIAL_TOKENS = (IMG_H // 4) * (IMG_W // 4)


class ConvBackbone(nn.Module):
    """Stage-0 conv stem: image -> per-spatial-token features.

    Output shape ``(B, SPATIAL_TOKENS=64, hidden)``.  Each spatial cell of
    the conv feature map becomes a routing token for the downstream MoE
    layers — same ``(B, S, H)`` shape contract MoE expects, driven by a
    real CV backbone instead of synthetic ``randn`` tokens.
    """

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.conv1 = nn.Conv2d(IMG_C, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, CONV_OUT_CHANNELS, kernel_size=3, padding=1)
        self.fc = nn.Linear(CONV_OUT_CHANNELS, hidden)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # (B, 64 channels, 8, 8) -> (B, 8*8 spatial tokens, 64 features).
        x = x.flatten(2).transpose(1, 2)
        x = F.relu(self.fc(x))
        return x  # (B, SPATIAL_TOKENS, hidden)


class MoEBlock(nn.Module):
    """Residual MoE block over the (B, SPATIAL_TOKENS, hidden) token grid."""

    def __init__(self, hidden=HIDDEN, num_experts=NUM_EXPERTS, ep_size=EP_SIZE):
        super().__init__()
        self.moe = MoE(
            hidden_size=hidden,
            expert=nn.Linear(hidden, hidden),
            num_experts=num_experts,
            k=1,
            ep_size=ep_size,
        )

    def forward(self, x):
        out, _aux, _z = self.moe(x)
        return out


class Classifier(nn.Module):
    """Final stage: token-mean-pool -> linear -> 10-class logits."""

    def __init__(self, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.proj = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


def build_pipeline_model(num_stages=2, hidden=HIDDEN):
    """Assemble the cifar PipelineModule: 1 conv + N MoE + 1 classifier.

    ``partition_method='uniform'`` is explicit so the conv backbone's FC
    layer (whose param count would dominate the default weighted
    partition) doesn't push every MoE block onto stage 1.  Uniform
    splits (1 + NUM_MOE_LAYERS + 1) layers as evenly as possible across
    ``num_stages`` so every stage carries at least one MoE — the engine
    guard requires it.
    """
    from deepspeed.pipe import LayerSpec, PipelineModule

    layers = [
        LayerSpec(ConvBackbone, hidden),
        *[LayerSpec(MoEBlock, hidden, NUM_EXPERTS, EP_SIZE) for _ in range(NUM_MOE_LAYERS)],
        LayerSpec(Classifier, hidden, NUM_CLASSES),
    ]
    return PipelineModule(
        layers=layers,
        num_stages=num_stages,
        loss_fn=nn.CrossEntropyLoss(),
        partition_method="uniform",
    )


def engine_config(train_batch_size=None):
    """DeepSpeed config with every MoEvement knob annotated.

    The ``moevement.pcie_bandwidth_gbs`` value is deliberately starved
    (1e-6) so the scheduler forces ``w_sparse > 1`` on this small model.
    Realistic hardware values would let the scheduler compute a larger
    window organically; the example pins a large window explicitly so
    every run exercises the multi-iter snapshot path.
    """
    return {
        "train_batch_size": train_batch_size or BATCH * 2,  # DP=2 → global batch = per-gpu * 2
        "train_micro_batch_size_per_gpu": BATCH,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                # ``torch_adam=True`` sidesteps the FusedAdam JIT compile — we don't
                # need nvcc on the box for this example to run.
                "torch_adam": True,
            },
        },
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            # MoEvement requires ZeRO-1 (stage 0 skips the optimizer sharding
            # MoEvement expects; stage 2/3 partition gradients, which the
            # MoEvement gate rejects).  See DESIGN_COMM_GROUP_REBUILD.md.
            "stage": 1,
        },
        "pipeline": {
            "activation_checkpoint_interval": 0,
        },
        "moevement": {
            # Feature gate — MoEvement is dormant unless this is True.
            "enabled": True,
            # PCIe bandwidth budget per iter.  Drives ``w_sparse`` selection:
            # higher values = smaller window (each iter can snapshot more).
            # Starved here to force multi-iter windows on this small model.
            "pcie_bandwidth_gbs": 0.000001,
            # Initial iter-time estimate (sec).  The runtime recalibrates
            # after ~50 iters (see AUDIT's §3.1 / iter_time_window_iters).
            "initial_iter_time_sec": 1.0,
            # How many DP peers to replicate each snapshot to.  Default 1
            # (no peer replication).  Increase to enable peer-pull recovery.
            "replication_factor": 1,
        },
    }


def cifar_data_iter(batch_size=BATCH, fake_data=True, data_dir=None, seed=123):
    """Yield ``(image_fp16, label_int64)`` micro-batches for the pipe engine.

    Two paths:

    - ``fake_data=True`` (default for examples): synthetic
      ``(B, 3, 32, 32)`` fp16 tensors and ``(B,)`` int64 labels.  Same
      shape contract as real CIFAR-10, so the model and pipeline behave
      identically.  Default for resume / supervisor demos and the smoke
      tests.
    - ``fake_data=False``: ``torchvision.datasets.CIFAR10`` with
      auto-download.  ToTensor + per-channel normalize, then cast to fp16
      so it matches the engine's FP16 mode.
    """
    from deepspeed.utils import RepeatingLoader

    if fake_data:
        gen = torch.Generator().manual_seed(seed)
        sample = (
            torch.randn(batch_size, IMG_C, IMG_H, IMG_W, dtype=torch.float16, generator=gen),
            torch.randint(0, NUM_CLASSES, (batch_size, ), generator=gen),
        )
        return iter(RepeatingLoader([sample]))

    import torchvision
    import torchvision.transforms as T

    if data_dir is None:
        data_dir = os.environ.get("MOEV_DATA_DIR", os.path.expanduser("~/.cache/moevement/cifar10"))
    os.makedirs(data_dir, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    def _collate(batch):
        imgs = torch.stack([item[0] for item in batch]).to(dtype=torch.float16)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return imgs, labels

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        collate_fn=_collate,
        persistent_workers=True,
    )
    return iter(RepeatingLoader(loader))


def build_engine(config=None, skip_initial_broadcast=False, hidden=HIDDEN):
    """Wire a DeepSpeed PipelineEngine with MoEvement enabled.

    Mirrors the test-suite ``_build_engine`` helper.  Returns the
    engine; caller drives the training loop.

    ``hidden`` defaults to ``HIDDEN`` (256); the perf-measurement scripts
    bump it (e.g., 2048) to match profile_run.py scale.
    """
    import deepspeed
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    if config is None:
        config = engine_config()

    model = build_pipeline_model(hidden=hidden)
    param_group = {"params": list(model.parameters()), "name": "moevement_example_params"}
    split_params = split_params_into_different_moe_groups_for_optimizer(param_group)
    optimizer = torch.optim.AdamW(split_params, lr=config["optimizer"]["params"]["lr"])

    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        optimizer=optimizer,
        dist_init_required=False,
        skip_initial_broadcast=skip_initial_broadcast,
    )
    return engine


def log_iter(rank, step, loss, engine):
    """Per-iter log line the example scripts share.

    Prints rank, step, loss, and a handful of MoEvement observables so
    a reader sees the same format in normal training, disk-reload, and
    peer-pull demos.  ``remaining_replay_count`` stays 0 outside
    recovery; during replay it counts down from ``len(replay_iters)``.
    """
    coord = getattr(engine, "moevement_coordinator", None)
    remaining = 0
    recovering = False
    if coord is not None:
        recovering = coord.is_recovering()
        remaining = coord.converter.get_remaining_replay_count() if recovering else 0
    marker = " [replay]" if recovering else ""
    print(
        f"[rank {rank}] step={step:4d} loss={loss:.4f} global_steps={engine.global_steps} "
        f"remaining_replay={remaining}{marker}",
        flush=True,
    )
