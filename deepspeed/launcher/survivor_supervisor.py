# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Minimum-viable supervisor for ``survivor_rendezvous`` — 1:1 spare substitution.

The supervisor is a stand-alone external process that spawns N training
workers, monitors their liveness, and on detecting a worker death:

  1. allocates a new master endpoint for the post-fault world,
  2. broadcasts ``(victim_rank, new_master_addr, new_master_port)`` to the
     surviving workers via a side-channel ``TCPStore`` it hosts,
  3. spawns a replacement process for the victim's rank slot with
     ``DS_SURVIVOR_IS_SPARE=1`` set in its environment.

Surviving workers opt in with a :class:`WorkerProbe` at the top of their
entrypoint.  They call :meth:`WorkerProbe.check_for_fault` at a known
safe point each training step (between ``engine.train_batch`` / collective
calls); on fault they invoke ``engine.survivor_rendezvous(...)`` with the
returned endpoint.  The spare process detects its role via
:meth:`WorkerProbe.is_spare` and passes ``skip_initial_broadcast=True``
to ``deepspeed.initialize`` — see the docstring on that flag in
``deepspeed/__init__.py`` for the Layer-3 rationale.

Scope for this first cut: **one fault, 1:1 substitution** (world size
unchanged, victim's rank slot reused).  Cascading faults, world-size
changes, and multi-victim substitutions are explicit non-goals — they'd
require a different protocol on the survivor side and are the right
place to iterate after this lands in production.

Typical user flow
-----------------

Worker entrypoint (``my_trainer.py``)::

    from deepspeed.launcher.survivor_supervisor import WorkerProbe

    def main():
        probe = WorkerProbe()
        engine, *_ = deepspeed.initialize(
            model=model, config=config,
            skip_initial_broadcast=probe.is_spare(),
        )
        for step in range(num_steps):
            fault = probe.check_for_fault()
            if fault is not None:
                new_addr, new_port, world_size = fault
                engine.survivor_rendezvous(
                    new_master_addr=new_addr,
                    new_master_port=new_port,
                    new_rank=dist.get_rank(),
                    new_world_size=world_size,
                )
            engine.train_batch(data_iter=data)
        probe.signal_done()

Supervisor launch (external to the training script)::

    from deepspeed.launcher.survivor_supervisor import SurvivorSupervisor

    sup = SurvivorSupervisor(
        worker_cmd=["python", "my_trainer.py"],
        world_size=4,
    )
    sys.exit(sup.run())

``sup.run()`` blocks until every worker has called ``signal_done()`` (→ 0),
times out (→ 1), or a second fault is observed (→ 2 — out of scope for v1).
"""

import datetime
import os
import socket
import subprocess
import sys
import time
from typing import List, Optional, Sequence, Tuple

import torch.distributed as _torch_dist
# Environment variable the supervisor sets on the spare process so the
# worker's ``WorkerProbe`` can distinguish "original boot" from "recovery
# cold-start."  Keep the name stable — users will grep for it.
_IS_SPARE_ENV = "DS_SURVIVOR_IS_SPARE"


def _get_free_port() -> int:
    """Return a currently-unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SurvivorSupervisor:
    """Spawns workers, monitors liveness, handles at most one fault.

    One supervisor hosts one training job.  Multiple jobs need multiple
    supervisors on non-conflicting (supervisor_port, master_port) pairs.

    Args:
        worker_cmd: ``subprocess.Popen``-compatible argv.  The worker
            entrypoint must instantiate a :class:`WorkerProbe` to
            participate in fault handling.
        world_size: Number of ranks.  Unchanged across fault recovery
            under 1:1 substitution.
        master_addr: Host the workers' training ``TCPStore`` binds to
            (rank 0's address).  Default: loopback.
        master_port: Port the initial training master binds to.  If
            ``None``, an unused port is picked.
        supervisor_addr: Host the supervisor's side-channel
            ``TCPStore`` binds to.  Usually same as master_addr.
        supervisor_port: Port the supervisor's side-channel
            ``TCPStore`` binds to.  If ``None``, an unused port is
            picked.  The supervisor's store is DISTINCT from the
            workers' training store — survivors connect to it
            for fault signalling only.
    """

    def __init__(
        self,
        worker_cmd: Sequence[str],
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: Optional[int] = None,
        supervisor_addr: Optional[str] = None,
        supervisor_port: Optional[int] = None,
    ):
        self.worker_cmd: List[str] = list(worker_cmd)
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port if master_port is not None else _get_free_port()
        self.supervisor_addr = supervisor_addr if supervisor_addr is not None else master_addr
        self.supervisor_port = (supervisor_port if supervisor_port is not None else _get_free_port())
        self._store: Optional[_torch_dist.TCPStore] = None
        self._procs: List[Optional[subprocess.Popen]] = []
        self._fault_handled = False

    def run(self, timeout_sec: float = 600.0) -> int:
        """Spawn workers, drive the monitor loop, return an exit code.

        Returns:
            * ``0`` — every worker reached ``signal_done()``.
            * ``1`` — ``timeout_sec`` elapsed before all workers
              finished.
            * ``2`` — a second fault was observed after the first was
              already being handled.  Explicitly out of scope for
              v1; the supervisor aborts rather than attempt a
              cascading recovery.
        """
        self._store = _torch_dist.TCPStore(
            self.supervisor_addr,
            self.supervisor_port,
            is_master=True,
            wait_for_workers=False,
            timeout=datetime.timedelta(seconds=max(60.0, timeout_sec)),
        )

        for rank in range(self.world_size):
            self._procs.append(
                self._spawn_worker(
                    rank=rank,
                    master_addr=self.master_addr,
                    master_port=self.master_port,
                    is_spare=False,
                ))

        try:
            return self._monitor(timeout_sec)
        finally:
            self._cleanup()

    def _spawn_worker(
        self,
        rank: int,
        master_addr: str,
        master_port: int,
        is_spare: bool,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(self.world_size)
        env["LOCAL_RANK"] = str(rank)
        env["MASTER_ADDR"] = master_addr
        env["MASTER_PORT"] = str(master_port)
        env["SUPERVISOR_ADDR"] = self.supervisor_addr
        env["SUPERVISOR_PORT"] = str(self.supervisor_port)
        env[_IS_SPARE_ENV] = "1" if is_spare else "0"
        return subprocess.Popen(self.worker_cmd, env=env)

    def _monitor(self, timeout_sec: float) -> int:
        deadline = time.time() + timeout_sec
        poll_interval = 0.1
        while time.time() < deadline:
            if self._all_done():
                return 0

            victim = self._find_dead_worker()
            if victim is not None:
                if self._fault_handled:
                    # A second fault after the first is out of scope.
                    # Don't try to chain recovery — abort cleanly so the
                    # operator sees the boundary.
                    return 2
                self._handle_fault(victim)
                self._fault_handled = True

            time.sleep(poll_interval)
        return 1

    def _find_dead_worker(self) -> Optional[int]:
        for rank, proc in enumerate(self._procs):
            if proc is None:
                continue
            if proc.poll() is not None:
                # Worker exited.  For v1 we treat any non-``done`` exit
                # as a fault; a clean exit before signal_done is also a
                # ``fault`` as far as the supervisor is concerned.
                if not self._store.check([f"done/rank_{rank}"]):
                    return rank
        return None

    def _all_done(self) -> bool:
        keys = [f"done/rank_{r}" for r in range(self.world_size)]
        return self._store.check(keys)

    def _handle_fault(self, victim_rank: int) -> None:
        # Allocate a fresh master endpoint for the post-fault world.
        # The old one is dead (if rank 0 was the victim, its TCPStore
        # died with it; otherwise survivors still need a rendezvous
        # point that the spare can join).  Broadcast order matters:
        # set the trigger key LAST so survivors' ``check_for_fault``
        # observes a fully-populated record on the first positive
        # check.
        new_master_addr = self.master_addr
        new_master_port = _get_free_port()

        self._store.set("fault/victim_rank", str(victim_rank))
        self._store.set("fault/new_master_addr", new_master_addr)
        self._store.set("fault/new_master_port", str(new_master_port))
        self._store.set("fault/triggered", "1")

        self._procs[victim_rank] = self._spawn_worker(
            rank=victim_rank,
            master_addr=new_master_addr,
            master_port=new_master_port,
            is_spare=True,
        )

    def _cleanup(self) -> None:
        for proc in self._procs:
            if proc is None:
                continue
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)


class WorkerProbe:
    """Worker-side client of the supervisor's side-channel ``TCPStore``.

    Instantiate once at the top of the worker entrypoint.  Call
    :meth:`check_for_fault` at a safe point each step; call
    :meth:`signal_done` when the worker has no more work to do.
    :meth:`is_spare` returns whether this process is the replacement
    for a victim (read once; pass the result to
    ``deepspeed.initialize(..., skip_initial_broadcast=...)``).

    Raises:
        KeyError: if the supervisor environment variables
            (``SUPERVISOR_ADDR`` / ``SUPERVISOR_PORT`` / ``RANK`` /
            ``WORLD_SIZE``) are not set — usually means the worker was
            launched outside a :class:`SurvivorSupervisor`.
    """

    def __init__(self, connect_timeout_sec: float = 60.0):
        self._rank = int(os.environ["RANK"])
        self._world_size = int(os.environ["WORLD_SIZE"])
        addr = os.environ["SUPERVISOR_ADDR"]
        port = int(os.environ["SUPERVISOR_PORT"])
        self._store = _torch_dist.TCPStore(
            addr,
            port,
            is_master=False,
            timeout=datetime.timedelta(seconds=connect_timeout_sec),
        )
        self._fault_consumed = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def is_spare(self) -> bool:
        """True iff this process was spawned as a post-fault replacement.

        Read-once at worker startup.  Pass the result as
        ``skip_initial_broadcast=probe.is_spare()`` to
        ``deepspeed.initialize``.
        """
        return os.environ.get(_IS_SPARE_ENV, "0") == "1"

    def check_for_fault(self) -> Optional[Tuple[str, int, int]]:
        """Non-blocking fault poll.

        Returns ``None`` if no fault has been signalled or the fault
        on this probe has already been consumed.  On the first
        positive check, returns
        ``(new_master_addr, new_master_port, world_size)`` — pass the
        addr/port straight to ``engine.survivor_rendezvous``.  Sets an
        internal consumed flag so subsequent calls return ``None``
        until a new fault fires (v1 supports one fault, so in practice
        it's one-shot).

        Safe to call every step.  Under the hood does a single
        ``TCPStore.check`` on a single key (fast).
        """
        if self._fault_consumed:
            return None
        if not self._store.check(["fault/triggered"]):
            return None
        new_addr = self._store.get("fault/new_master_addr").decode()
        new_port = int(self._store.get("fault/new_master_port").decode())
        self._fault_consumed = True
        return (new_addr, new_port, self._world_size)

    def signal_done(self) -> None:
        """Tell the supervisor this rank has finished its work."""
        self._store.set(f"done/rank_{self._rank}", "ok")


def _main_cli() -> int:
    """``python -m deepspeed.launcher.survivor_supervisor -- worker.py`` CLI.

    Parses ``--world-size`` and forwards the remaining argv after
    ``--`` as the worker command.  Kept minimal — richer configuration
    should instantiate :class:`SurvivorSupervisor` programmatically.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("worker_cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.worker_cmd:
        parser.error("worker command required after --")
    # argparse REMAINDER picks up a leading "--" if present; strip it.
    cmd = args.worker_cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("worker command required after --")

    sup = SurvivorSupervisor(
        worker_cmd=cmd,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    return sup.run(timeout_sec=args.timeout_sec)


if __name__ == "__main__":
    sys.exit(_main_cli())
