"""Regression test: PartitionWrapper must implement every method the NBD
server (and bridge) invoke on its backend, so partitioned mode doesn't fail
a command at runtime (e.g. FLUSH -> durability_barrier, which regressed).

Run: python -m pytest tests/test_partition_wrapper_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.partition_wrapper import PartitionWrapper

# Methods the NBD server / bridge call on their backend mapper.
REQUIRED = ['read', 'write', 'flush', 'flush_all', 'durability_barrier',
            'clear_dirty_bit', 'rescan_mft', 'get_size']


def test_wrapper_implements_backend_api():
    missing = [m for m in REQUIRED
               if not callable(getattr(PartitionWrapper, m, None))]
    assert not missing, f"PartitionWrapper is missing NBD-backend methods: {missing}"
