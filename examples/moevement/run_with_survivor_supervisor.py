# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Live-recovery demo under real SIGKILL via ``SurvivorSupervisor``.

Demonstrates the advertised **DP Cascade** recovery mode the way it
actually fires in production: one rank dies mid-training, a spare
takes its slot, the recovering DP group replays the persisted window
while the unaffected DP group pauses inside ``recovery_barrier``,
and training continues past the fault.

Internally this path uses peer-pull (``load_sparse_from_peer`` /
``serve_sparse_snapshot_to_peer``) so the spare can bootstrap from
its surviving DP peer without a whole-cluster restart.  Peer-pull as
a standalone user-facing API is **not advertised**.  For disk-based
recovery that IS advertised for production, see
``resume_after_fault.py`` instead.

**Architecture:**

- This script runs in two modes gated on ``--worker``.  Default mode
  launches ``SurvivorSupervisor`` which spawns 4 subprocess workers
  (each re-invoking this script with ``--worker``).  Worker mode runs
  the actual training + recovery logic.
- Victim rank self-SIGKILLs at a configured iter so the demo is
  reproducible with a single command (no external killer needed).
- The supervisor detects the death, broadcasts the new master
  endpoint, and spawns a replacement process with
  ``DS_SURVIVOR_IS_SPARE=1`` in its env.

**Launch:**

```
deepspeed --num_gpus=4 is NOT used here — the SurvivorSupervisor
is its own launcher.  Run directly with python:

python examples/moevement/run_with_survivor_supervisor.py \\
    --fault-at-iter 18 --post-recovery-iters 8
```

**Expected observable (rank 0 victim, rank 0 spare after):**

```
[rank 0 (initial)] pre-fault training done (n=18), waiting for fault
[rank 0 (initial)] SELF-SIGKILL at step 18
[rank 1 (initial)] fault detected, new_master=127.0.0.1:NNNNN
[rank 1 (initial)] survivor_rendezvous complete
[rank 1 (initial)] served peer-pull snapshot to rank 0
[rank 0 (spare)]   SPARE cold-start, peer-pulling from rank 1
[rank 0 (spare)]   peer-pull complete (engine.global_steps=18)
[rank 0 (spare)]   recovery converged after K replay iters
[rank 0 (spare)]   post-recovery step=1 loss=... global_steps=19
...
```

**NOT advertised for production** — the supervisor here is a
minimum-viable shape, and peer-pull's asymmetric DP-collective
pattern has known edges.  Use ``resume_after_fault.py`` for the
advertised Whole-Cluster-Restart recovery mode.
"""

import argparse
import os
import signal
import sys
import time

import torch

import deepspeed

from _common import build_engine, cifar_data_iter, engine_config

# Fixed by the 4-rank PP=2 DP=2 topology ``_common.py`` builds:
#   rank 0 = PP stage 0, DP rank 0  ← victim
#   rank 1 = PP stage 0, DP rank 1  ← peer-pull source for spare
#   rank 2 = PP stage 1, DP rank 0  ← unaffected DP group
#   rank 3 = PP stage 1, DP rank 1  ← unaffected DP group
_VICTIM_RANK = 0
_PEER_PULL_SOURCE_RANK = 1
_WORLD_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal: set by the supervisor when it spawns this script as a worker.",
    )
    parser.add_argument(
        "--fault-at-iter",
        type=int,
        default=18,
        help="Iteration at which the victim rank self-SIGKILLs.  Must be large "
        "enough for at least one ``finalize_window`` to have fired so the "
        "peer-pull source (rank 1) actually has a persisted shard to serve.",
    )
    parser.add_argument(
        "--post-recovery-iters",
        type=int,
        default=8,
        help="Training iterations to run AFTER recovery completes, demonstrating "
        "the post-recovery happy path.  The replay window itself (before this "
        "counter starts) is logged with the ``[replay]`` marker the other "
        "example scripts use.",
    )
    parser.add_argument(
        "--supervisor-timeout",
        type=float,
        default=300.0,
        help="Supervisor monitor-loop timeout.  Workers that haven't "
        "``signal_done``'d within this window are treated as wedged.",
    )
    return parser.parse_args()


def _log(rank, role, msg):
    print(f"[rank {rank} ({role})] {msg}", flush=True)


def _recovery_replay_loop(engine, coord, data_iter, max_iters):
    """Drive ``train_batch`` until the coordinator reports recovery done.

    During replay the engine's ``train_batch`` re-executes saved-window
    iters (on the recovering DP group) and pauses in
    ``recovery_barrier`` (on the unaffected DP group).  When every
    rank has caught up, ``coord._recovering`` flips False on the
    recovering ranks and ``coord._paused_for_recovery`` flips False
    on the paused ranks.

    Returns the iter count at which convergence was observed, or -1
    if the loop exhausted ``max_iters`` without converging — the
    caller should treat that as a hang and fail loud.
    """
    for step in range(1, max_iters + 1):
        engine.train_batch(data_iter=data_iter)
        if not coord._recovering and not coord._paused_for_recovery:
            return step
    return -1


def run_worker(args):
    """Worker process body — runs under the supervisor's spawn env."""
    from deepspeed.launcher.survivor_supervisor import WorkerProbe
    import deepspeed.comm as dist

    probe = WorkerProbe()
    rank = probe.rank
    world_size = probe.world_size
    is_spare = probe.is_spare()
    local_rank = int(os.environ["LOCAL_RANK"])
    role = "spare" if is_spare else "initial"

    torch.cuda.set_device(local_rank)  #ignore-cuda
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

    # Same seed on every rank so every initial worker's pre-broadcast
    # weights are identical; spare skips the broadcast but that's fine
    # because peer-pull overwrites its state from rank 1 anyway.
    torch.manual_seed(42)
    engine = build_engine(config=engine_config(), skip_initial_broadcast=is_spare)
    coord = engine.moevement_coordinator
    w_sparse = coord.scheduler.w_sparse
    _log(rank, role, f"engine built (w_sparse={w_sparse}, spare={is_spare})")

    data_iter = cifar_data_iter(fake_data=True, seed=123)

    if is_spare:
        # Spare: no pre-fault training.  Its entire purpose is to take
        # the victim's slot and reconstruct state from its surviving DP
        # peer.  ``engine.initialize`` has already set up the default
        # PG on the new master (supervisor spawned us with the
        # post-fault MASTER_ADDR / MASTER_PORT + DS_SURVIVOR_IS_SPARE=1).
        _log(rank, role, f"SPARE cold-start — peer-pulling from rank {_PEER_PULL_SOURCE_RANK}")
        ok = coord.load_sparse_from_peer(
            peer_rank=_PEER_PULL_SOURCE_RANK,
            my_dp_rank_in_replication_group=0,
            model=engine.module,
            engine=engine,
        )
        assert ok, (f"peer-pull returned False — rank {_PEER_PULL_SOURCE_RANK} "
                    f"had no shard to serve (check that --fault-at-iter is past "
                    f"a window boundary so replication actually fired)")
        _log(rank, role, f"peer-pull complete (global_steps={engine.global_steps})")
    else:
        # Survivor: pre-fault training up to ``fault_at_iter``.  Match
        # the engine's data seed so the replay loop on the recovering
        # DP group sees the same sample sequence.
        torch.manual_seed(123)
        for step in range(1, args.fault_at_iter + 1):
            engine.train_batch(data_iter=data_iter)
            if rank == 0:
                print(f"[rank 0 ({role})] step={step} global_steps={engine.global_steps}", flush=True)

        # Wait for in-flight replication to land on the peer-pull
        # source before the fault fires.  Without this, the gloo
        # worker's pending send can race the SIGKILL and leave rank 1
        # with no shard to serve; the spare's peer-pull returns False.
        # ``_replication_futures`` is a FIFO deque since the
        # multi-outstanding peer-send refactor; drain in order.
        while coord._replication_futures:
            coord._replication_futures.popleft().result(timeout=30.0)
        dist.barrier()

        if rank == _VICTIM_RANK:
            _log(rank, role, f"SELF-SIGKILL at step {args.fault_at_iter}")
            # The supervisor's monitor loop polls ``proc.poll()`` every
            # 100ms; it'll see this death on the next tick and start
            # the handover.
            os.kill(os.getpid(), signal.SIGKILL)
            # unreachable

        # Non-victim survivors: wait for the supervisor to detect the
        # death and broadcast the new master endpoint via the probe.
        _log(rank, role, "pre-fault training done, waiting for fault signal")
        fault = None
        while fault is None:
            fault = probe.check_for_fault()
            if fault is None:
                time.sleep(0.05)
        new_addr, new_port, new_world_size = fault
        _log(rank, role, f"fault detected, new_master={new_addr}:{new_port}")

        engine.survivor_rendezvous(
            new_master_addr=new_addr,
            new_master_port=new_port,
            new_rank=rank,
            new_world_size=new_world_size,
        )
        _log(rank, role, "survivor_rendezvous complete")

        # Only the victim's DP peer serves the peer-pull request.  The
        # unaffected DP group (ranks 2, 3) skips this handshake — they
        # participate in the recovery via ``recovery_barrier`` during
        # the replay loop below.
        if rank == _PEER_PULL_SOURCE_RANK:
            coord.serve_sparse_snapshot_to_peer(requester_rank=_VICTIM_RANK)
            _log(rank, role, f"served peer-pull snapshot to rank {_VICTIM_RANK}")

    # Every post-recovery rank barriers here to gate the replay loop
    # on both (a) the spare having peer-pulled its state and (b) all
    # survivors having completed survivor_rendezvous + optional serve.
    dist.barrier()

    max_replay = max(16, w_sparse * 3)
    converged_at = _recovery_replay_loop(engine, coord, data_iter, max_replay)
    if converged_at < 0:
        _log(rank, role, f"ERROR: recovery did not converge within {max_replay} iters (w_sparse={w_sparse})")
        probe.signal_done()
        sys.exit(1)
    _log(rank, role, f"recovery converged after {converged_at} replay iters "
         f"(global_steps={engine.global_steps})")

    # Post-recovery training — shows that the cluster is back on the
    # happy path.  Use the same data iter so the loss trajectory is
    # continuous with the pre-fault run on the surviving DP group.
    for step in range(1, args.post_recovery_iters + 1):
        loss = engine.train_batch(data_iter=data_iter)
        if rank == 0:
            val = float(loss) if loss is not None else float("nan")
            print(
                f"[rank 0 ({role})] post-recovery step={step} loss={val:.4f} "
                f"global_steps={engine.global_steps}",
                flush=True,
            )

    _log(rank, role, "done")
    probe.signal_done()


def run_supervisor(args):
    """Supervisor process body — spawns 4 workers + monitors liveness."""
    from deepspeed.launcher.survivor_supervisor import SurvivorSupervisor

    worker_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--fault-at-iter",
        str(args.fault_at_iter),
        "--post-recovery-iters",
        str(args.post_recovery_iters),
    ]
    sup = SurvivorSupervisor(worker_cmd=worker_cmd, world_size=_WORLD_SIZE)
    exit_code = sup.run(timeout_sec=args.supervisor_timeout)
    sys.exit(exit_code)


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_supervisor(args)


if __name__ == "__main__":
    main()
