# MoEvement: Sparse Checkpointing for Fast and Reliable MoE Training

This repository contains the implementation of MoEvement, introduced in the NSDI'26 paper "Sparse Checkpointing for Fast and Reliable MoE Training". The name is pronounced "movement" — a reference to MoE models and the movement of training state that keeps a job making progress through failures. [[Paper]](https://arxiv.org/pdf/2412.15411)

---

## Why MoEvement?

Existing checkpointing techniques each fail on one of three axes when
applied to MoE training (paper §2, Table 1):

| System | Low overhead | Fast recovery | Full recovery | High ETTR |
|---|---|---|---|---|
| CheckFreq [FAST'21] | ✗ | ✗ | ✓ | ✗ |
| Gemini [SOSP'23] | ✗ | ✗ | ✓ | ✗ |
| MoC-System [ASPLOS'25] | ✗ | ✓ | ✗ (loses tokens) | ✗ |
| **MoEvement** | **✓** | **✓** | **✓** | **✓** |

- **Challenge #1 — Runtime–Recovery Tradeoff.**  Dense checkpointers
  (CheckFreq, Gemini) are forced to lengthen the interval to amortize
  the snapshot cost, which lengthens recovery proportionally.  On
  DeepSeek-16.4B/64E at MTBF = 10 min, Gemini's optimal policy still
  costs 11–15% per-iteration overhead and lands at ETTR = 0.76.
  MoEvement's sparse window hides the whole snapshot inside one
  iteration and checkpoints every iteration, so the interval is `1`.
- **Challenge #2 — Correctness–Efficiency Tension.**  MoC-System
  checkpoints only a subset of experts per iteration to reduce cost,
  but experts without a recent snapshot revert to stale state on
  recovery, dropping tokens and violating synchronous-training
  semantics.  MoEvement's **sparse-to-dense conversion** (§3.3)
  reconstructs a consistent dense checkpoint by replaying micro-batches
  across the window; no expert reverts to a stale iteration.
- **Challenge #3 — Global Rollback Scope.**  Existing systems roll
  back every rank to the last global checkpoint even when only one
  rank failed.  MoEvement's **upstream logging** (§3.4) confines
  rollback to the failed DP group; other DP groups pause in place,
  reducing recovery time by ~23% on three-stage pipelines and more as
  pipelines deepen.

---

## The three core ideas (paper → code)

| Paper § | Idea | Core module |
|---|---|---|
| §3.2 | **Sparse checkpointing** — snapshot a subset of operators per iteration; cycle through all operators in `Wsparse` iterations | `deepspeed/moevement/sparse_snapshot.py` + `scheduler.py` |
| §3.3 | **Sparse-to-dense conversion** — reconstruct a logically consistent dense checkpoint by incrementally activating operators and replaying micro-batches | `deepspeed/moevement/conversion.py` + `coordinator.py` replay loop |
| §3.4 | **Upstream logging** — log activations and gradients at pipeline-stage boundaries so a recovering stage can replay without its neighbors rolling back | `deepspeed/moevement/upstream_logging.py` |

Supporting components (Algorithm 1 in the paper, pp. 7) — the
`FindWindowSize` + `GenerateSchedule` routines that pick `Wsparse` and
operator ordering from profiled bandwidth and parameter counts — live
in `scheduler.py`.

---

## Quick start

Enable MoEvement in your DeepSpeed config:

```json
{
  "train_batch_size": 64,
  "optimizer": { "type": "Adam", "params": { "lr": 1e-4, "torch_adam": true } },
  "zero_optimization": { "stage": 1 },
  "fp16": { "enabled": true },
  "pipeline": { "activation_checkpoint_interval": 0 },
  "moevement": {
    "enabled": true,
    "pcie_bandwidth_gbs": 50.0,
    "initial_iter_time_sec": 1.0,
    "upstream_logging": true
  }
}
```

Constraints:

- Pipeline parallelism must be enabled (`PP ≥ 2`).
- ZeRO stage must be `< 2` (optimizer-state partitioning is allowed,
  gradient partitioning is not).
- Model must contain `MoE` layers (via `deepspeed.moe.layer.MoE`).
- Adam optimizer must be used with `torch_adam=True` (the path has been
  tested against FusedAdam but the lint isn't maintained against its
  quirks).

Callers that manage their own persistence (launcher / cluster manager):

```python
# Periodic disk save:
coord = engine.moevement_coordinator
coord.save_sparse_checkpoint(save_dir, tag=f"step{engine.global_steps}")
coord.flush_persist()  # optional; waits on the background writer

# On fault + whole-job restart, after a fresh DeepSpeed init:
coord = engine.moevement_coordinator
ok = coord.load_sparse_checkpoint(save_dir, tag, model=engine.module,
                                  optimizer=engine.optimizer)
# Then: run engine.train_batch(...) in a loop until coord.is_recovering() is False.
```

DP cascade and peer pull are triggered automatically from the recovery
handshake at the top of each `train_batch`; no explicit API call is
needed.

---

## Testing

The suite lives in `tests/unit/moevement/`:

- `test_sparse_checkpoint.py` — unit tests for the converter, snapshot
  engine, scheduler, and the replay state machine.
- `test_engine_integration.py` — happy-path training, recovery-
  equivalence, upstream-logging wiring, and the PP=3 middle-stage test.
- `test_distributed.py` — ring-replication and recovery-handshake
  tests that need real distributed init.
- `test_fault_inject.py` / `_fault_inject.py` — contract tests for the
  in-process fault-injection helper.

Run the full suite:

```sh
cd tests
export PATH="/path/to/deepspeed/conda-env/bin:$PATH"
export DS_IGNORE_CUDA_DETECTION=1   # if nvcc isn't on the box
python -m pytest unit/moevement/
```
---

## Citation

If you use MoEvement in your research, please cite the paper:

```bibtex
@article{moevement,
   author = {Swapnil Gandhi and Christos Kozyrakis},
   title = {Sparse Checkpointing for Fast and Reliable MoE Training},
   booktitle = {23rd {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 2026)},
   year = {2026},
   isbn = {978-1-939133-54-0},
   url = {https://www.usenix.org/conference/nsdi26/presentation/gandhi},
   publisher = {{USENIX} Association},
   month = may
}
```
---

## Acknowledgments

MoEvement is built on top of [DeepSpeed](https://github.com/deepspeedai/deepspeed), and would not exist without it. We thank the DeepSpeed maintainers and contributors for the pipeline-parallel runtime, ZeRO optimizer, and MoE layer primitives that MoEvement extends. For background on the underlying framework — installation, supported parallelism strategies, and the broader feature set — please see the [DeepSpeed README](https://github.com/deepspeedai/deepspeed#readme).
