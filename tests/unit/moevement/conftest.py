# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Test-level noise suppression for MoEvement tests.

Two sources of loud-but-cosmetic output show up on every MoEvement
test run and make real failures hard to spot:

1. NCCL's eager-init P2P serialization warning fires on every
   unbatched ``dist.send`` / ``dist.recv`` we use in ring replication
   and upstream-log transfer.  The serialization behaviour is
   intentional (we only need a few P2P ops per window, so batching
   them would add complexity for no throughput win); NCCL just
   wants us to know.  Setting
   ``TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=false``
   quiets it.

2. MoE's ``_one_hot_to_float`` / ``_top_idx`` helpers hit
   ``aten._local_scalar_dense`` inside dynamo tracing under torch
   2.11, which inductor can't compile and falls back on with a
   multi-line graph-break warning per-rank, per-iteration.  The
   MoEvement tests don't use ``torch.compile``, so disabling dynamo
   globally via ``TORCHDYNAMO_DISABLE=1`` is safe and the cheapest
   way to silence the spam.

Both are set with ``setdefault`` so an explicit env override in the
test runner still wins.
"""

import os

os.environ.setdefault("TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
