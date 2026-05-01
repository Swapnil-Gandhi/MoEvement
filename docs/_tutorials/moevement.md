---
title: "MoEvement: Sparse Checkpointing and Localized Recovery for MoE Training"
tags: MoE training fault-tolerance checkpointing
---

MoE training runs are large, long, and fragile: a single GPU or node failure
late in training can waste days of compute. The standard remedy — frequent
full-model checkpoints — is prohibitively expensive once you're training a
trillion-parameter sparse model, because most of the bytes on disk are expert
parameters that did not activate during the last window.

**MoEvement** is an opt-in DeepSpeed subsystem that exploits the sparse
activation pattern of MoE layers to make checkpointing and recovery cheap:

- **Sparse checkpointing.** Each training window, only a scheduled subset of
  operators is snapshotted at full FP32 + optimizer state precision. The rest
  snapshot just their FP16 compute weights. This keeps per-iteration
  checkpoint cost roughly constant in the total parameter count.
- **In-memory peer replication.** Persisted snapshots are replicated to a
  small, configurable number of DP peers so that a recovering rank can
  reconstruct state without touching disk.
- **Upstream logging + pipeline replay.** Each pipeline stage logs the
  activations and gradients it sends to its neighbours. When a stage fails,
  its neighbours ship their logs over, and the recovering stage replays its
  missing window using logged inputs — no need to re-run upstream stages.

## When to use it

Enable MoEvement when **all** of the following hold:

1. You're training a model with MoE layers (`deepspeed.moe.layer.MoE`).
2. You're using ZeRO-0 or ZeRO-1. (ZeRO-2 and ZeRO-3 are not supported — see
   [Limitations](#limitations).)
3. You care about recovery time from a single-rank or single-node failure
   more than you care about adding ~1–2% per-iteration overhead for
   snapshotting.

If you just want cheap periodic checkpoints without the peer-replication or
localized-recovery logic, stick with `deepspeed.save_checkpoint` — MoEvement
is designed around the failure-recovery use case.

## Enabling MoEvement

Add a `moevement` block to your DeepSpeed config:

```json
{
  "train_batch_size": 128,
  "moevement": {
    "enabled": true,
    "replication_factor": 1,
    "upstream_logging": true
  }
}
```

That's the minimum. The full set of options:

| Key                      | Default | Description |
| ------------------------ | ------- | ----------- |
| `enabled`                | `false` | Master switch. MoEvement is a no-op when disabled. |
| `replication_factor`     | `1`     | Number of DP peers to replicate each snapshot to. Set to `0` to disable peer replication (disk-only durability). |
| `reorder_threshold`      | `0.10`  | Fraction change in activation frequency that triggers the scheduler to reorder operators between active/frozen. |
| `reorder_fraction`       | `0.25`  | Minimum fraction of experts whose frequency must change before reordering kicks in. |
| `pcie_bandwidth_gbs`     | `25.0`  | Effective GPU→CPU PCIe bandwidth, used to size the sparse window. Tune down on older hardware. |
| `upstream_logging`       | `true`  | Log activations and gradients at pipeline-stage boundaries for replay-based recovery. |
| `initial_iter_time_sec`  | `1.0`   | Starting estimate for per-iteration wall-clock time. Only used until the first real window completes. |

The engine automatically wires MoEvement into the training loop: snapshots are
taken at the gradient-accumulation boundary, peer replication runs at each
window boundary, and logs are flushed on the same cadence.

## How recovery works

When a rank detects a failure (via your cluster manager, heartbeat, etc.),
trigger recovery:

```python
engine.moevement_trigger_recovery(failed_stage_id=my_stage_id)
```

This puts the engine into a recovery mode where:

1. **Sparse checkpoint is loaded** from disk if available.  Replacement ranks
   that came up with no local state pull the snapshot from a DP peer via
   the explicit `load_sparse_from_peer(peer_dp_rank)` API — each rank's own
   shard is replicated to `r` forward peers on the ring, so a replacement
   for rank `k` asks any of ranks `(k+1)..(k+r) mod dp_world_size` for their
   `_received_snapshots[k]`.
2. **Upstream logs are received** from the previous and next pipeline stages
   (for activations and gradients respectively).
3. **Replay runs for `W_sparse` iterations**: forward and backward use logged
   activations/gradients instead of the usual P2P receive, frozen operators
   use their FP16 weights with `requires_grad=False` so no weight gradients
   are computed, and each window boundary promotes a scheduled subset of
   frozen operators to active by loading their snapshotted FP32 weights and
   optimizer state.
4. Once every operator is active, `end_recovery()` runs automatically and
   normal training resumes.

From the user's perspective after `moevement_trigger_recovery`, training just
continues — no explicit replay loop to orchestrate.

## Frozen-operator semantics

During recovery, operators marked FROZEN by the schedule:

- **Do** run forward normally.
- **Do** propagate gradients through themselves to the previous stage
  (input-gradient).
- **Do not** compute weight gradients (`requires_grad=False` on their
  parameters, so autograd skips the weight-grad path).
- **Do not** receive optimizer updates. For vanilla optimizers this falls
  out naturally (`p.grad is None` ⇒ Adam short-circuits). For ZeRO-1,
  `zero_frozen_gradients()` runs before the optimizer step to zero any
  frozen-param grad slots that ZeRO pre-allocated; any resulting moment
  drift is overwritten when the operator is promoted to ACTIVE and its
  snapshotted optimizer state is loaded.

## Tuning

- **`replication_factor`.** Default is `1` (one peer backup in addition to
  disk). Increase only if you expect correlated failures within the same DP
  group. Each additional peer costs another full H2D copy of the snapshot
  per window on the primary, partially mitigated by our fan-out in the send
  loop.
- **`pcie_bandwidth_gbs`.** The window-size algorithm assumes this bandwidth
  when budgeting snapshot cost against iteration time. Wrong values just
  change how aggressive the scheduler is; they don't affect correctness.
  Profile your link (`nvidia-smi topo -m`) and set it honestly.
- **`reorder_threshold` / `reorder_fraction`.** Control how reactive the
  scheduler is to shifts in expert activation patterns. The defaults are
  tuned for typical MoE training and rarely need adjustment.

## Inspecting state

The coordinator exposes a few APIs that are useful for debugging:

```python
coord = engine.moevement_coordinator

# Which operators are currently active / frozen
coord.converter.get_active_operators()
coord.converter.get_frozen_operators()

# Memory footprint breakdown
coord.get_memory_usage()

# Inspect the current window
coord.snapshot_engine.get_persisted_snapshots()
```

## Failure matrix

What MoEvement recovers from, and what forces a full reload from the last
`save_checkpoint`:

| Failure scenario | Recovery path |
| ---------------- | ------------- |
| Single rank / node fails | Load snapshot from DP peer (or disk), pull logs from pipeline neighbours, replay. |
| Multiple non-correlated ranks fail across different DP/PP groups | Each failed rank recovers independently and in parallel via the single-rank path. |
| **K consecutive pipeline stages fail together** (bounded by live stages) | Supported. Live stage above the failed region ships activation logs to the first failed stage; live stage below ships gradient logs to the last failed stage; the replay chains through the region because each failed stage's freshly-recomputed output feeds the next failed stage's input via the normal pipeline channels during replay. |
| Full pipeline column fails (no live bounding stages) | Full reload from `save_checkpoint`. |
| Primary DP rank + all `r` peers fail together | Full reload from `save_checkpoint`. Mitigate by placing DP peers across failure domains and raising `replication_factor`. |
| Whole-cluster restart | Full reload from `save_checkpoint`. |
| Failure during an in-flight recovery | Not currently handled — the in-flight replay will stall on the now-dead neighbour. |

## Limitations

- **ZeRO-2 is not supported.** ZeRO-2 partitions gradients across DP
  peers, requiring backward to reduce-scatter grads into a flat
  buffer. `PipelineEngine` instead schedules its own gradient all-
  reduces at pipeline boundaries (it disables
  `enable_backward_allreduce` in `super().backward()`), so the two
  gradient-flow strategies are mutually exclusive — the pipe engine
  rejects `zero_optimization.stage >= 2` outright at `__init__`.
  Because MoEvement mandates PP > 1, the combination is unreachable.
  Enabling MoEvement with stage 2 or 3 fails fast at
  `deepspeed.initialize` with a named-feature error.
- **ZeRO-3 is not supported.** Even if the PP-incompatibility above
  were lifted, the snapshot and restore paths go through DeepSpeed's
  `_hp_mapping` fragment API (`safe_get_full_fp32_param` etc.) to
  handle ZeRO-1 and the bf16 optimizer's partitioned FP32 masters,
  which also happens to dispatch to ZeRO-3's `_z3_optimizer` helpers
  — but the ZeRO-3 path has not been tested with MoEvement's replay
  logic. Raise this to us if you need it.
- **NCCL liveness detection is out of scope.** MoEvement assumes some
  external mechanism (cluster manager, heartbeat, NCCL watchdog) informs it
  that a rank has failed. It does not detect failures itself.
- **No automatic peer-placement across failure domains.** DP peers are just
  dp_ranks `[1..r]`; whether those ranks live on distinct racks/hosts is up
  to your launch topology.

## File layout on disk

When `engine.save_checkpoint(save_dir, tag)` is called, MoEvement writes an
additional subdirectory alongside the regular checkpoint:

```
<save_dir>/<tag>/
  mp_rank_00_model_states.pt       # regular DeepSpeed checkpoint
  ...
  moevement/
    window.pt                      # JSON header + raw tensor bytes for
                                   # every operator in the window
    upstream_logs_rank<N>.pt       # optional, when upstream_logging=true
```

`load_sparse_checkpoint` reads from this directory automatically when present.

## Further reading

- Source: `deepspeed/moevement/` — `coordinator.py`, `sparse_snapshot.py`,
  `upstream_logging.py`, `converter.py`, `scheduler.py`.
- Tests: `tests/unit/moevement/test_sparse_checkpoint.py` and
  `tests/unit/moevement/test_distributed.py` are the best reference for how
  the pieces fit together at the API level.
