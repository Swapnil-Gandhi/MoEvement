MOEVEMENT = "moevement"
MOEVEMENT_ENABLED = "enabled"
MOEVEMENT_ENABLED_DEFAULT = False
MOEVEMENT_REPLICATION_FACTOR = "replication_factor"
MOEVEMENT_REPLICATION_FACTOR_DEFAULT = 1
MOEVEMENT_REORDER_THRESHOLD = "reorder_threshold"
MOEVEMENT_REORDER_THRESHOLD_DEFAULT = 0.10
MOEVEMENT_REORDER_FRACTION = "reorder_fraction"
MOEVEMENT_REORDER_FRACTION_DEFAULT = 0.25
MOEVEMENT_PCIE_BANDWIDTH_GBS = "pcie_bandwidth_gbs"
MOEVEMENT_PCIE_BANDWIDTH_GBS_DEFAULT = 25.0
MOEVEMENT_UPSTREAM_LOGGING = "upstream_logging"
MOEVEMENT_UPSTREAM_LOGGING_DEFAULT = True
MOEVEMENT_INITIAL_ITER_TIME_SEC = "initial_iter_time_sec"
MOEVEMENT_INITIAL_ITER_TIME_SEC_DEFAULT = 1.0
MOEVEMENT_ACTIVATION_COUNT_WINDOW_ITERS = "activation_count_window_iters"
MOEVEMENT_ACTIVATION_COUNT_WINDOW_ITERS_DEFAULT = 100
MOEVEMENT_ITER_TIME_WINDOW_ITERS = "iter_time_window_iters"
MOEVEMENT_ITER_TIME_WINDOW_ITERS_DEFAULT = 50
MOEVEMENT_COMM_REBUILD_TIMEOUT_SEC = "comm_rebuild_timeout_sec"
MOEVEMENT_COMM_REBUILD_TIMEOUT_SEC_DEFAULT = 120.0
MOEVEMENT_FSYNC_ON_SAVE = "fsync_on_save"
MOEVEMENT_FSYNC_ON_SAVE_DEFAULT = True
MOEVEMENT_REPLICATION_QUEUE_WARN_THRESHOLD = "replication_queue_warn_threshold"
MOEVEMENT_REPLICATION_QUEUE_WARN_THRESHOLD_DEFAULT = 64
MOEVEMENT_REPLICATION_QUEUE_MAX_OUTSTANDING = "replication_queue_max_outstanding"
MOEVEMENT_REPLICATION_QUEUE_MAX_OUTSTANDING_DEFAULT = 256
MOEVEMENT_STREAMING_RECOVERY = "streaming_recovery"
MOEVEMENT_STREAMING_RECOVERY_DEFAULT = False
MOEVEMENT_MAX_PREFETCHED_ITERS = "max_prefetched_iters"
MOEVEMENT_MAX_PREFETCHED_ITERS_DEFAULT = 8
MOEVEMENT_POOL_GROW_ON_MISS_ACTIVATION = "pool_grow_on_miss_activation"
# 0 means "auto-size from gas × num_moe_layers × w_sparse × 2";
# otherwise treat as the literal grow_on_miss for the upstream
# logger pool.  See ``coordinator._derive_activation_pool_grow``.
MOEVEMENT_POOL_GROW_ON_MISS_ACTIVATION_DEFAULT = 0
MOEVEMENT_POOL_MAX_PER_KEY = "pool_max_per_key"
MOEVEMENT_POOL_MAX_PER_KEY_DEFAULT = 4096
MOEVEMENT_SNAPSHOT_OVERLAP_TARGET = "snapshot_overlap_target"
# Fraction of one iteration's PCIe budget the per-iter snapshot is
# allowed to consume.  ``find_window_size`` picks w_sparse such that
# per-iter snapshot bytes / pcie_bw <= overlap_target × iter_time.
# Smaller value = larger w_sparse = more compute headroom for the
# side stream to drain (less likely to block optim_step), but worse
# recovery cost per fault.  Default 1.0 preserves the historical
# recovery-optimal MIN-w_sparse behavior; 0.5 is the recommended
# perf-optimal setting for production training.
MOEVEMENT_SNAPSHOT_OVERLAP_TARGET_DEFAULT = 1.0
MOEVEMENT_W_SPARSE_OVERRIDE = "w_sparse_override"
# 0 means "use scheduler.find_window_size"; otherwise force this w_sparse.
# The default scheduler MINIMIZES w_sparse subject to "snapshot fits in
# one iter PCIe budget" (paper §3.5 Algorithm 1) — the right choice for
# recovery-frequency goals but suboptimal for steady-state perf when
# compute headroom is plenty (boundary cost amortizes inversely with
# w_sparse, and per-iter D2H drain has more compute time to overlap).
# Set this to a larger value (e.g. 16 or 32) at production scale to
# trade recovery responsiveness for steady-state throughput.
MOEVEMENT_W_SPARSE_OVERRIDE_DEFAULT = 0
MOEVEMENT_PERSIST_TO_DISK = "persist_to_disk"
# Whether ``engine.save_checkpoint`` writes the MoEvement bundle (sparse
# snapshot + upstream logs) to disk alongside the regular DeepSpeed
# checkpoint.  Default ``False``: peer-pull from a surviving DP peer's
# pinned host snapshot is the primary recovery target, so the disk
# bundle is opt-in.  Setting this ``True`` restores the historical
# behavior — the bundle is written on every ``save_checkpoint`` call,
# enabling the §3 "Whole-Cluster Restart" recovery path
# (``resume_after_fault.py``).
MOEVEMENT_PERSIST_TO_DISK_DEFAULT = False


class MoEvementConfig:
    """Configuration for MoEvement sparse checkpointing system.

    Args:
        enabled: Whether MoEvement sparse checkpointing is enabled.
        replication_factor: Number of peer nodes to replicate sparse snapshots to.
        reorder_threshold: Relative frequency change (over the rolling
            activation-count window's older-half vs. newer-half rates) that
            marks one expert as "changed enough to matter" for reorder
            detection. Paper §3.5 specifies "over 10%".
        reorder_fraction: Fraction of experts whose frequency must change to trigger reordering.
        pcie_bandwidth_gbs: Effective GPU-to-CPU PCIe bandwidth in GB/s for scheduling.
        upstream_logging: Whether to enable upstream logging for localized recovery.
        initial_iter_time_sec: Initial estimate of per-iteration wall-clock time in
            seconds, used to compute the sparse-window size before real measurements
            are available. The schedule is re-generated at window boundaries.
        activation_count_window_iters: Length (in training iterations) of
            the rolling window over which per-expert token counts are
            aggregated for the §3.5 reorder-trigger rate comparison.
            Larger values smooth more but delay detection; smaller values
            react faster but increase false-positive reorders on noise.
        iter_time_window_iters: Length (in training iterations) of the
            rolling window over which per-iteration wall-clock durations
            are collected to drive runtime ``w_sparse`` recalibration.
            Once full, the median duration is fed back into
            ``find_window_size`` at every window boundary; the schedule
            is regenerated only when the new value differs from the
            current ``w_sparse`` by ≥ 1 iteration (hysteresis).
        comm_rebuild_timeout_sec: Upper bound (seconds) on the gloo /
            NCCL ``destroy_process_group`` fallback inside
            ``rebuild_comm_groups``.  A spare-rank substitution can
            leave a group's communicator wedged on the dead rank's
            in-flight op; the timeout prevents the rebuild path from
            blocking forever waiting for a destroy that will never
            return.  The NCCL ``abort()`` fast path (torch >= 2.6) is
            effectively instant and ignores this value.
        fsync_on_save: Whether to ``os.fsync`` bundle + log files on
            save (default ``True``).  When ``False``, ``torch.save``
            still writes to kernel page cache but skips the explicit
            durability barrier — saves return faster, since the fsync
            dominates ``save_sparse_checkpoint`` cost on conventional
            disks.  Lossy on OS crash / power loss; opt-in only on
            cloud VMs with journaled local SSDs where the lost-bytes
            window is acceptable.
        replication_queue_warn_threshold: Number of in-flight peer
            replication jobs that triggers a one-shot warning log
            (default 64).  Peer replication runs on a background
            worker so training never blocks at the window boundary;
            the outstanding queue is a signal that replication is
            falling behind window cadence and pinned CPU memory is
            accumulating.  A healthy run should not approach this in
            steady state — crossing it means the next
            ``replication_queue_max_outstanding`` is on the horizon
            and that the gloo wire is not keeping up.  Debounced:
            re-armed once depth drops to half the warn threshold,
            so a single slow boundary doesn't spam.
        replication_queue_max_outstanding: Hard cap on in-flight peer
            replication jobs (default 256).  Once hit, the window
            boundary blocks on the oldest outstanding future so
            pinned CPU memory doesn't grow unbounded — this is the
            ONLY path by which replication can block training and
            should be treated as a host-OOM safety rail, not a
            steady-state cliff.  Each outstanding window holds the
            window's pinned-CPU snapshot, so 256 windows can sit on
            hundreds of GB of pinned host memory; lower the cap on
            smaller hosts and raise it on larger ones if you'd rather
            buffer than block.  If a run consistently hits the cap,
            the gloo wire genuinely can't keep up — the right response
            is faster transport (NCCL/IB) or larger ``w_sparse``, not
            a higher cap.
        streaming_recovery: Opt-in wire-level change to the peer-pull
            recovery protocol (default ``False``).  When ``True``, a
            replacement rank negotiates the per-iter streaming
            protocol during the handshake so the serving peer ships
            the manifest + flats iter-by-iter in sorted order rather
            than as one bulk manifest followed by all flats.  The
            iter-major wire is the foundation for overlapping the
            pull-of-iter-N+1 with the replay-of-iter-N on the
            replacement side.  The on-the-wire handshake changes
            whether this knob is on or off — both sides of a
            deployment must upgrade together.
        max_prefetched_iters: Cap on the number of pulled-but-not-yet-
            replayed iters held in the streaming pull queue (default
            8).  Pull thread blocks on ``Queue.put`` once the cap
            is hit so peak CPU memory stays predictable: with a
            faster network than replay, an unbounded queue would
            balloon to ``W_sparse * per_iter_bytes``.  Only consulted
            when ``streaming_recovery`` is ``True``.
    """

    def __init__(self, param_dict=None):
        if param_dict is None:
            param_dict = {}

        moevement_dict = param_dict.get(MOEVEMENT, {})

        self.enabled = moevement_dict.get(MOEVEMENT_ENABLED, MOEVEMENT_ENABLED_DEFAULT)
        self.replication_factor = moevement_dict.get(MOEVEMENT_REPLICATION_FACTOR,
                                                     MOEVEMENT_REPLICATION_FACTOR_DEFAULT)
        self.reorder_threshold = moevement_dict.get(MOEVEMENT_REORDER_THRESHOLD, MOEVEMENT_REORDER_THRESHOLD_DEFAULT)
        self.reorder_fraction = moevement_dict.get(MOEVEMENT_REORDER_FRACTION, MOEVEMENT_REORDER_FRACTION_DEFAULT)
        self.pcie_bandwidth_gbs = moevement_dict.get(MOEVEMENT_PCIE_BANDWIDTH_GBS,
                                                     MOEVEMENT_PCIE_BANDWIDTH_GBS_DEFAULT)
        self.upstream_logging = moevement_dict.get(MOEVEMENT_UPSTREAM_LOGGING, MOEVEMENT_UPSTREAM_LOGGING_DEFAULT)
        self.initial_iter_time_sec = moevement_dict.get(MOEVEMENT_INITIAL_ITER_TIME_SEC,
                                                        MOEVEMENT_INITIAL_ITER_TIME_SEC_DEFAULT)
        self.activation_count_window_iters = moevement_dict.get(MOEVEMENT_ACTIVATION_COUNT_WINDOW_ITERS,
                                                                MOEVEMENT_ACTIVATION_COUNT_WINDOW_ITERS_DEFAULT)
        self.iter_time_window_iters = moevement_dict.get(MOEVEMENT_ITER_TIME_WINDOW_ITERS,
                                                         MOEVEMENT_ITER_TIME_WINDOW_ITERS_DEFAULT)
        self.comm_rebuild_timeout_sec = moevement_dict.get(MOEVEMENT_COMM_REBUILD_TIMEOUT_SEC,
                                                           MOEVEMENT_COMM_REBUILD_TIMEOUT_SEC_DEFAULT)
        self.fsync_on_save = moevement_dict.get(MOEVEMENT_FSYNC_ON_SAVE, MOEVEMENT_FSYNC_ON_SAVE_DEFAULT)
        self.replication_queue_warn_threshold = moevement_dict.get(MOEVEMENT_REPLICATION_QUEUE_WARN_THRESHOLD,
                                                                   MOEVEMENT_REPLICATION_QUEUE_WARN_THRESHOLD_DEFAULT)
        self.replication_queue_max_outstanding = moevement_dict.get(
            MOEVEMENT_REPLICATION_QUEUE_MAX_OUTSTANDING, MOEVEMENT_REPLICATION_QUEUE_MAX_OUTSTANDING_DEFAULT)
        self.streaming_recovery = moevement_dict.get(MOEVEMENT_STREAMING_RECOVERY,
                                                     MOEVEMENT_STREAMING_RECOVERY_DEFAULT)
        self.max_prefetched_iters = moevement_dict.get(MOEVEMENT_MAX_PREFETCHED_ITERS,
                                                       MOEVEMENT_MAX_PREFETCHED_ITERS_DEFAULT)
        self.pool_grow_on_miss_activation = moevement_dict.get(MOEVEMENT_POOL_GROW_ON_MISS_ACTIVATION,
                                                               MOEVEMENT_POOL_GROW_ON_MISS_ACTIVATION_DEFAULT)
        self.pool_max_per_key = moevement_dict.get(MOEVEMENT_POOL_MAX_PER_KEY, MOEVEMENT_POOL_MAX_PER_KEY_DEFAULT)
        self.w_sparse_override = moevement_dict.get(MOEVEMENT_W_SPARSE_OVERRIDE, MOEVEMENT_W_SPARSE_OVERRIDE_DEFAULT)
        self.snapshot_overlap_target = float(
            moevement_dict.get(MOEVEMENT_SNAPSHOT_OVERLAP_TARGET, MOEVEMENT_SNAPSHOT_OVERLAP_TARGET_DEFAULT))
        self.persist_to_disk = moevement_dict.get(MOEVEMENT_PERSIST_TO_DISK, MOEVEMENT_PERSIST_TO_DISK_DEFAULT)

    @property
    def pcie_bandwidth_bytes_per_sec(self):
        """Convenience view of ``pcie_bandwidth_gbs`` in bytes/second.

        The scheduler does its ``find_window_size`` math in bytes to
        compare snapshot size (bytes) with iter time (seconds); users
        configure the knob in the more intuitive GiB/s unit.
        """
        return self.pcie_bandwidth_gbs * (1024**3)


def get_moevement_config(param_dict):
    """Parse a DeepSpeed param dict into a :class:`MoEvementConfig`.

    Thin helper used by DeepSpeed's config loader so the engine can
    instantiate the MoEvement config via the same pattern as other
    subsystems (``get_<name>_config(config)``).
    """
    return MoEvementConfig(param_dict)
