"""Single-file bundle format for MoEvement sparse checkpoints.

Replaces ``torch.save`` per operator with a compact ``window.pt`` file
containing a JSON header followed by raw tensor bytes.  Two wins on
the persist path:

- **One file per window** instead of ``metadata.pt`` + one per operator.
  A 256-operator window drops from 257 open/close/fsync round-trips to
  one.
- **No pickle.** Tensor payloads are written as raw bytes; shapes and
  dtypes live in the JSON header.  Skips pickle's single-threaded Python
  loop and makes the serialized form independent of PyTorch's version.

File layout::

    [4 bytes: magic b"MOEV"]
    [1 byte:  format version]
    [8 bytes little-endian uint64: header length N]
    [N bytes: UTF-8 JSON header]
    [raw tensor bytes, concatenated in header-declared order]

The header carries the ``metadata`` dict ``save_to_disk`` emits plus a
per-iteration list ``iterations``, each element carrying that snapshot's
``{op_name: {is_active, tensors: [...]}}`` where every tensor descriptor
is ``{key, dtype, shape, offset, nbytes}``.  Preserving iteration
grouping on disk matches the spec's per-snapshot FP16 captures — the
recovery loop replays iteration-by-iteration and must see each
iteration's frozen-op FP16 weights, not just the last window-iter's
overwrite.

A ``byte_order`` field makes loads on a host with opposite endianness
fail loudly instead of silently returning garbage.
"""

import json
import os
import struct
import sys

import torch

BUNDLE_FILENAME = "window_rank{rank}.pt"
_MAGIC = b"MOEV"
_VERSION = 2

# Short, stable names keep the header tiny even with hundreds of tensors.
_DTYPE_TO_STR = {
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def dump_bundle(path, metadata, per_iter_snapshots, fsync=True):
    """Write ``metadata`` + per-iteration snapshots to a single bundle file.

    Args:
        path: Destination filename.  Parent directory must already exist.
        metadata: The window metadata dict (``window_start_iteration`` and
            any aggregates ``save_to_disk`` emits for the window as a whole).
        per_iter_snapshots: ``{iteration: {op_name: {"is_active": bool,
            "state_dict": {key: tensor}, "fragment_info": ... | None}}}``.
            Iterations are written in ascending order so a reader can
            stream per-iter blocks without seeking.  An op absent from an
            iteration's dict means nothing was captured for that op at
            that iteration — either because the op had already been
            active earlier in the window (spec's "nothing is captured
            after active") or because it was not in the schedule yet.

            ``fragment_info`` is an optional dict
            ``{key: {"full_shape": [...], "fragment_numel": int}}`` that
            declares per-entry which tensors in ``state_dict`` are
            per-rank fragments of a larger full-shape param.  Present
            under ZeRO-1/2 fragment-snapshot (each rank captured only
            its own HP / optim_fragment slice, no DP all-reduce);
            absent on the non-ZeRO path and in tests that build
            ``OperatorSnapshot`` directly via ``add_tensor``.  Restore
            uses the fragment metadata to copy bytes directly into
            ``_hp_mapping.hp_fragment`` / ``optim_fragment[key]``
            without the ``safe_set_full_*`` narrow-from-full path.
        fsync: When True (default), force bytes to durable storage via
            ``os.fsync`` before returning.  When False, bytes hit kernel
            page cache only — saves return faster but a rank death before
            the kernel flushes loses the bundle.  Opt-in only on cloud
            VMs / journaled local SSDs where the lost-bytes window is
            acceptable.  See ``MoEvementConfig.fsync_on_save``.
    """
    iterations_meta = []
    ordered = []  # [(tensor_for_writing, nbytes)] preserving header order
    offset = 0
    for iteration in sorted(per_iter_snapshots.keys()):
        op_descriptors = {}
        for op_name, entry in per_iter_snapshots[iteration].items():
            fragment_info = entry.get("fragment_info") or {}
            tensors_meta = []
            for key, tensor in entry["state_dict"].items():
                dtype_str = _DTYPE_TO_STR.get(tensor.dtype)
                if dtype_str is None:
                    raise TypeError(f"[MoEvement] unsupported dtype {tensor.dtype} "
                                    f"for iter {iteration} op {op_name} key {key}")
                nbytes = tensor.numel() * tensor.element_size()
                desc = {
                    "key": key,
                    "dtype": dtype_str,
                    "shape": [int(s) for s in tensor.shape],
                    "offset": offset,
                    "nbytes": nbytes,
                }
                # Fragment entries carry the full param shape + this
                # rank's slice numel in the descriptor so the loader
                # can route them to the fragment-direct restore path.
                frag = fragment_info.get(key)
                if frag is not None:
                    desc["full_shape"] = [int(s) for s in frag["full_shape"]]
                    desc["fragment_numel"] = int(frag["fragment_numel"])
                tensors_meta.append(desc)
                ordered.append((tensor, nbytes))
                offset += nbytes
            op_descriptors[op_name] = {
                "is_active": bool(entry["is_active"]),
                "tensors": tensors_meta,
            }
        iterations_meta.append({
            "iteration": int(iteration),
            "operators": op_descriptors,
        })

    header = {
        "metadata": metadata,
        "iterations": iterations_meta,
        "byte_order": sys.byteorder,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(bytes([_VERSION]))
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for tensor, nbytes in ordered:
            if nbytes == 0:
                continue
            # ``view(uint8)`` reinterprets bytes without copying data; the
            # preceding contiguous()/cpu() handle dtype-agnostic byte
            # extraction (including bfloat16, which numpy can't name).
            # ``memoryview(ndarray)`` exposes the buffer to ``file.write``
            # without the intermediate copy ``tobytes()`` would incur.
            raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
            f.write(memoryview(raw.numpy()))
        # Force bytes to durable storage before returning.  Without this,
        # ``torch.save`` / ``file.close`` only hand the buffer to the OS
        # page cache, and a rank death before the kernel flushes loses
        # the "persisted" bundle — the caller already believes the write
        # succeeded (flush_persist returns, the replication group's
        # coordinator callback fires).  CheckFreq and DeepSpeed's own
        # ``save_checkpoint`` take the same precaution.  Opt-out via
        # ``fsync=False`` (driven by ``MoEvementConfig.fsync_on_save``)
        # for cloud VMs where the journaled SSD makes the explicit
        # barrier redundant — the fsync dominates ``save_sparse_checkpoint``
        # cost on conventional disks.
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def load_bundle(path):
    """Read a bundle written by :func:`dump_bundle`.

    Returns ``(metadata, per_iter_operator_states)`` where
    ``per_iter_operator_states`` is ``{iteration: {op_name: {key: tensor}}}``.
    Callers needing the per-op ``is_active`` flag read it from
    ``metadata["per_iter_active"][iteration][op_name]``, which
    :func:`load_bundle` populates as a parallel dict.

    Fragment entries (per-rank slices of a full-shape param captured
    under ZeRO-1/2) surface their metadata as
    ``metadata["per_iter_fragment_info"][iteration][op_name][key] =
    {"full_shape": [...], "fragment_numel": int}``.  Entries absent
    from this map are full-shape (non-ZeRO path or test-built
    snapshots); restore handles both formats.
    """
    with open(path, "rb") as f:
        magic = f.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f"[MoEvement] bad bundle magic in {path!r}: {magic!r}")
        version = f.read(1)
        if not version or version[0] > _VERSION:
            raise ValueError(f"[MoEvement] unsupported bundle version {version!r} in {path!r}")
        (header_len, ) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
        stored_order = header.get("byte_order", "little")
        if stored_order != sys.byteorder:
            raise ValueError(f"[MoEvement] bundle {path!r} was written on a {stored_order}-endian host; "
                             f"this host is {sys.byteorder}-endian")
        # readinto avoids the intermediate ``bytes`` object that
        # ``bytearray(f.read())`` would allocate and copy.
        data_start = f.tell()
        remaining = os.fstat(f.fileno()).st_size - data_start
        blob = bytearray(remaining)
        f.readinto(blob)

    metadata = dict(header["metadata"])
    per_iter_operator_states = {}
    per_iter_active = {}
    per_iter_fragment_info = {}
    for iter_block in header["iterations"]:
        iteration = int(iter_block["iteration"])
        op_states = {}
        op_active = {}
        op_fragment_info = {}
        for op_name, entry in iter_block["operators"].items():
            tensors = {}
            fragment_info = {}
            for desc in entry["tensors"]:
                dtype = _STR_TO_DTYPE[desc["dtype"]]
                nbytes = desc["nbytes"]
                shape = desc["shape"]
                if nbytes == 0:
                    tensors[desc["key"]] = torch.empty(shape, dtype=dtype)
                    continue
                raw = torch.frombuffer(blob, dtype=torch.uint8, count=nbytes, offset=desc["offset"])
                # ``frombuffer`` shares storage with ``blob``, so every returned
                # tensor would pin the whole bundle (potentially multi-GB) until
                # the last reference drops.  ``clone`` detaches into its own
                # storage so individual tensors can be freed as the caller
                # copies them to device.
                tensors[desc["key"]] = raw.view(dtype).reshape(shape).clone()
                if "fragment_numel" in desc:
                    fragment_info[desc["key"]] = {
                        "full_shape": list(desc.get("full_shape", shape)),
                        "fragment_numel": int(desc["fragment_numel"]),
                    }
            op_states[op_name] = tensors
            op_active[op_name] = bool(entry["is_active"])
            if fragment_info:
                op_fragment_info[op_name] = fragment_info
        per_iter_operator_states[iteration] = op_states
        per_iter_active[iteration] = op_active
        if op_fragment_info:
            per_iter_fragment_info[iteration] = op_fragment_info
    metadata["per_iter_active"] = per_iter_active
    metadata["per_iter_fragment_info"] = per_iter_fragment_info
    return metadata, per_iter_operator_states
