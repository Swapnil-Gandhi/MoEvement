# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Two helper functions an external orchestrator calls on each rank after a fault.

The MoEvement coordinator already exposes the building blocks for
peer-pull recovery (``survivor_rendezvous``, ``load_sparse_from_peer``,
``serve_sparse_snapshot_to_peer``, ``recovery_barrier``).  These helpers
package those calls into the two shapes a fault orchestrator actually
needs: one body for surviving ranks and one body for the replacement
process that takes the failed rank's slot.

Out of scope: detecting the fault, electing a new master endpoint,
launching the replacement process.  Those are the orchestrator's job
(supervisor, slurm, kubernetes, custom launcher).  These helpers run
*after* the orchestrator has surfaced ``new_master_addr``,
``new_master_port``, and the peer-pull source rank to every rank that
needs them.

The helpers do NOT drive ``train_batch`` themselves — they prepare
state for the caller's training loop, which then runs ``train_batch``
until ``coord._recovering`` and ``coord._paused_for_recovery`` both
clear (see ``run_until_recovered``).  Splitting the prepare step
(rendezvous + peer-pull / cascade) from the drive step (replay loop)
lets a caller log, instrument, or otherwise interleave their own
per-iter logic without the helpers swallowing the loop.
"""

from typing import Optional


def _world_to_replication_local_rank(coord, world_rank: int) -> int:
    """Convert a world rank to its index within the replication group.

    The MoEvement primitives (``serve_sparse_snapshot_to_peer``,
    ``load_sparse_from_peer``) take **group-local** ranks for the
    replication group, which is a 2-rank gloo subgroup mirroring the
    DP group.  Callers think in world ranks; this helper bridges.
    """
    import deepspeed.comm as dist
    if coord._replication_group is None:
        raise RuntimeError("coord._replication_group is None — set_topology not yet called?")
    members = dist.get_all_ranks_from_group(coord._replication_group)
    try:
        return list(members).index(world_rank)
    except ValueError as exc:
        raise ValueError(f"world rank {world_rank} is not a member of replication group "
                         f"{list(members)}") from exc


def run_as_survivor(
    engine,
    coord,
    *,
    victim_rank: int,
    peer_pull_source_rank: int,
    new_master_addr: Optional[str] = None,
    new_master_port: Optional[int] = None,
    new_rank: Optional[int] = None,
    new_world_size: Optional[int] = None,
    rendezvous: bool = True,
):
    """Prepare a surviving rank to participate in MoEvement recovery.

    Two-step: optional ``engine.survivor_rendezvous`` to leave the
    WORLD that was bound to the dead peer and re-init on a new master
    endpoint, then — if this rank is the donor for the spare's
    peer-pull — serve its persisted shard via
    ``coord.serve_sparse_snapshot_to_peer``.  Other survivors return
    immediately; the cascade trigger and pp-log transfer fire
    automatically inside the next ``train_batch``'s
    ``recovery_barrier`` round.

    The ``rendezvous`` flag distinguishes the two fault models the
    helpers support:

    * ``True`` (default): real fault — the dead peer's process exited,
      the WORLD bound to that peer is broken, survivors must re-init
      on a fresh master.  Caller supplies ``new_master_*`` and
      ``new_rank`` / ``new_world_size``.
    * ``False``: emulated fault — the "victim" process is alive but
      its local state was zeroed in-place (see
      ``simulate_rank_failure``); the WORLD is intact, no rendezvous
      needed.  ``new_master_*`` and ``new_rank`` are ignored.

    Args:
        engine: The DeepSpeed engine for this rank.  Must already be
            built; this helper does not construct it.
        coord: The engine's ``MoEvementCoordinator``.
        victim_rank: World rank of the rank whose state was lost.
            Used by the donor to dispatch its serve call.
        peer_pull_source_rank: World rank of the survivor that holds
            the persisted shard the spare will pull.  When
            ``dist.get_rank() == peer_pull_source_rank``, this helper
            serves; otherwise it returns after the optional rendezvous.
        new_master_addr / new_master_port / new_rank / new_world_size:
            Required when ``rendezvous=True``.  Endpoint elected by
            the orchestrator post-fault and this rank's slot in the
            post-fault world (equal to pre-fault rank under the
            "preserve survivors" model).
        rendezvous: See above.  Default ``True`` matches the real-
            SIGKILL flow; pass ``False`` for emulated-fault tests.
    """
    import deepspeed.comm as dist
    from deepspeed.accelerator import get_accelerator

    if rendezvous:
        if new_master_addr is None or new_master_port is None or new_rank is None or new_world_size is None:
            raise ValueError("rendezvous=True requires new_master_addr, new_master_port, "
                             "new_rank, and new_world_size to be set")
        get_accelerator().synchronize()
        engine.survivor_rendezvous(
            new_master_addr=new_master_addr,
            new_master_port=new_master_port,
            new_rank=new_rank,
            new_world_size=new_world_size,
        )

    if dist.get_rank() == peer_pull_source_rank:
        # The MoEvement primitive takes the requester's rank within the
        # replication group, not its world rank — convert here so the
        # caller can think in world ranks consistently.
        victim_local = _world_to_replication_local_rank(coord, victim_rank)
        coord.serve_sparse_snapshot_to_peer(requester_rank=victim_local)


def run_as_spare(engine, coord, *, peer_pull_source_rank: int) -> bool:
    """Peer-pull recovery state on the rank that took the failed slot.

    No rendezvous: this helper assumes ``engine`` is already bound to
    the post-fault WORLD.  Two valid setups:

    * **Real fault.** A fresh process was launched on the new master
      endpoint and called ``deepspeed.init_distributed`` against it,
      so its WORLD is correct from the start; survivors meanwhile
      called ``engine.survivor_rendezvous`` to leave the broken old
      WORLD.  No rendezvous needed on the spare.
    * **Emulated fault.** The "victim" process is alive but its local
      state was zeroed in-place (see ``simulate_rank_failure``); the
      WORLD is intact for everyone.  No rendezvous needed.

    The caller must have built ``engine`` via
    ``deepspeed.initialize(..., skip_initial_broadcast=True)`` so the
    spare doesn't try to participate in the WORLD's initial broadcast
    when its DP peer's snapshot isn't available yet — the peer-pull
    inside this helper is what actually places the right weights.

    Args:
        engine: The freshly-built DeepSpeed engine.
        coord: ``engine.moevement_coordinator``.
        peer_pull_source_rank: World rank of the donor survivor.  The
            helper converts this and ``dist.get_rank()`` to their
            indices within the replication group internally before
            calling ``coord.load_sparse_from_peer``.

    Returns:
        True if the peer-pull succeeded.  False indicates the donor
        had no shard for this rank (shouldn't happen under the normal
        DP-replication invariant); caller should escalate.
    """
    import deepspeed.comm as dist

    peer_local = _world_to_replication_local_rank(coord, peer_pull_source_rank)
    my_local = _world_to_replication_local_rank(coord, dist.get_rank())
    return bool(
        coord.load_sparse_from_peer(
            peer_rank=peer_local,
            my_dp_rank_in_replication_group=my_local,
            model=engine.module,
            engine=engine,
        ))


def run_until_recovered(engine, coord, data_iter, max_iters: int) -> int:
    """Drive ``train_batch`` until this rank exits the recovering state.

    Replay on the recovering DP group and the wait loop on the paused
    DP group both clear when the world handshake observes
    ``recovering=0`` everywhere.  Caller passes a budget so a divergent
    recovery surfaces as a bounded failure rather than a hang; the
    return value is the number of iters consumed (useful for
    instrumentation).

    Args:
        engine: The engine to drive.
        coord: Its coordinator (used to observe recovery flags).
        data_iter: Caller's data iterator, passed through to
            ``train_batch``.
        max_iters: Hard upper bound on iters consumed before raising.

    Raises:
        RuntimeError: If recovery hasn't completed within ``max_iters``.
    """
    consumed = 0
    for _ in range(max_iters):
        engine.train_batch(data_iter=data_iter)
        consumed += 1
        if not coord._recovering and not coord._paused_for_recovery:
            return consumed
    raise RuntimeError(f"recovery did not complete within {max_iters} iters "
                       f"(_recovering={coord._recovering}, _paused_for_recovery={coord._paused_for_recovery})")


__all__ = ["run_as_survivor", "run_as_spare", "run_until_recovered"]
