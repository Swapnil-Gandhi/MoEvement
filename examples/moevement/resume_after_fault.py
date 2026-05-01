# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Resume MoEvement training from a disk checkpoint.

Mirrors ``train_cifar_moe.py`` but loads an existing checkpoint instead
of starting fresh.  Demonstrates the advertised **Whole-Cluster Restart**
recovery mode: MoEvement's sparse bundle + upstream logs (written by
the saving run) are loaded, the coordinator enters recovery mode, and
the first ``W_sparse`` iterations replay the saved window's per-iter
snapshots before training continues past the saved iter.

The script deliberately has no MoEvement-specific load wiring beyond
``engine.load_checkpoint(...)``, which dispatches to the MoEvement
coordinator's ``load_sparse_checkpoint`` internally
(see ``engine.py`` around the ``moevement_coordinator.load_sparse_checkpoint``
call).  That's the point: porting to your own training loop means adding
the config knob; the checkpoint round-trip is automatic.

**Two-step launch:**

```
# 1) produce a checkpoint:
deepspeed --num_gpus=4 examples/moevement/train_cifar_moe.py \\
    --fake-data --save-dir /tmp/moevement_ckpt --num-iters 50 --save-every 20

# 2) resume from it:
deepspeed --num_gpus=4 examples/moevement/resume_after_fault.py \\
    --load-dir /tmp/moevement_ckpt --num-iters 20
```

Expected observable: the first iter prints ``[replay]`` with
``remaining_replay=N`` counting down across the replayed window; after
the window is replayed, the marker disappears and ``global_steps``
advances by one per real iter again.  Loss trajectory is continuous
across the resume — same curve as a fault-free run would produce past
the save iter.

Related:
- ``train_cifar_moe.py`` — the saving-side counterpart (canonical
  training example).
- ``run_with_survivor_supervisor.py`` — peer-pull demo (real SIGKILL).
- ``docs/moevement/README.md`` — recovery-mode taxonomy.
"""

import argparse
import os

import torch

import deepspeed

from _common import build_engine, cifar_data_iter, engine_config, log_iter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--load-dir",
        default="/tmp/moevement_ckpt",
        help="Directory containing the checkpoint written by train_cifar_moe.py "
        "(or any DeepSpeed engine using the MoEvement coordinator).",
    )
    parser.add_argument(
        "--load-tag",
        default=None,
        help="Checkpoint tag to load (e.g., 'step_40').  Defaults to whatever the "
        "'latest' file points at — matches what the saving run wrote last.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="How many post-resume training iters to run.  The first W_sparse "
        "of these are replay iters driving the sparse-to-dense conversion.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Populated automatically by the DeepSpeed launcher.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    deepspeed.init_distributed(dist_backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  #ignore-cuda

    # Same model seed as the saving-side script so the pre-load state is
    # byte-identical; the load overwrites it anyway, but identical seeds
    # keep the comparison clean for any reader who diffs.
    torch.manual_seed(42)

    engine = build_engine(config=engine_config())

    # ``engine.load_checkpoint`` dispatches to the MoEvement coordinator
    # internally when one is attached; no MoEvement-specific calls needed.
    # ``tag=None`` reads the 'latest' file so the resume picks up the
    # saving run's last-written tag by default.
    load_path, _client_state = engine.load_checkpoint(args.load_dir, tag=args.load_tag)
    if load_path is None:
        if rank == 0:
            print(
                f"[rank 0] no checkpoint found at {args.load_dir} "
                f"(tag={args.load_tag or 'latest'}); run train_cifar_moe.py --save-dir first",
                flush=True,
            )
        return

    coord = engine.moevement_coordinator
    if rank == 0:
        if coord is not None and coord.is_recovering():
            remaining = coord.converter.get_remaining_replay_count()
            fault_iter = coord._fault_iter
            print(
                f"[rank 0] resumed from {load_path} at global_steps={engine.global_steps}; "
                f"MoEvement recovery active (fault_iter={fault_iter}, {remaining} replay iters pending)",
                flush=True,
            )
        else:
            # Bundle was absent, empty, or the window had already fully converted —
            # normal continuation, no replay window.
            print(
                f"[rank 0] resumed from {load_path} at global_steps={engine.global_steps}; "
                f"no MoEvement recovery (bundle absent or already fully converted)",
                flush=True,
            )

    # Same data seed as the saving side.  The fake-data loader yields a
    # RepeatingLoader over a single synthetic CIFAR-shaped sample (see
    # ``_common.cifar_data_iter``), so every iter sees identical inputs;
    # post-resume loss continuity is a pure model-state property.  Real
    # CIFAR-10 isn't needed here — the recovery contract holds for any
    # deterministic data source of the right shape.
    data_iter = cifar_data_iter(fake_data=True, seed=123)

    for step in range(1, args.num_iters + 1):
        loss = engine.train_batch(data_iter=data_iter)
        if rank == 0:
            log_iter(rank, step, float(loss) if loss is not None else float("nan"), engine)

    if rank == 0:
        print("[rank 0] resume complete", flush=True)


if __name__ == "__main__":
    main()
