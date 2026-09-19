#!/usr/bin/env python3
"""Virtual MFT slots must come from the real MFT, not a fixed range.

The scan used to stop at record 120. On any volume with more than ~96 real
files every record below that is taken, the free list came back empty, and
virtual mode silently added nothing.

    python3 tests/test_virtual_mft_slots.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.virtual_files import VirtualFileManager

failures = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        failures.append(name)


class FakeMapper:
    """Enough of ClusterMapper for set_mapper()."""

    def __init__(self, used_files, used_dirs, total_records):
        self.mft_record_to_source = {r: f"/src/f{r}" for r in used_files}
        self.mft_record_to_dir = {r: f"d{r}" for r in used_dirs}
        self._mft_total_records = total_records
        self.path_to_mft_record = {}


def manager(mapper):
    m = VirtualFileManager(tempfile.gettempdir())
    m.set_mapper(mapper)
    return m


def main():
    print("virtual MFT slot allocation")

    # A volume like the real one: every record from 24 up past 120 is a real
    # file, and the MFT is far larger than the old hardcoded cap.
    used = range(24, 5000)
    m = manager(FakeMapper(used, [], total_records=20000))

    check("finds slots beyond the old cap", len(m._available_mft_slots) > 0, True)
    check("first slot is past the real files", m._available_mft_slots[0] >= 5000, True)
    check("no slot collides with a real record",
          any(s in set(used) for s in m._available_mft_slots), False)
    check("never hands out a system record",
          any(s < 24 for s in m._available_mft_slots), False)
    check("free list is capped", len(m._available_mft_slots),
          VirtualFileManager.MAX_FREE_SLOTS)

    # Allocation draws from the free list and stays disjoint from real records.
    first = m._allocate_mft_record()
    check("allocates the first free slot", first >= 5000, True)
    check("allocation is not a real record", first in set(used), False)

    # Small volume: gaps below the old cap are still used.
    m2 = manager(FakeMapper([24, 25, 27], [26], total_records=66))
    check("small volume finds the gap", 28 in m2._available_mft_slots, True)
    check("small volume skips used records",
          any(s in (24, 25, 26, 27) for s in m2._available_mft_slots), False)
    check("stays within the real MFT",
          all(s < 66 for s in m2._available_mft_slots), True)

    # Genuinely full MFT: no slots, and no invented fallback record.
    m3 = manager(FakeMapper(range(24, 66), [], total_records=66))
    check("full MFT yields no slots", m3._available_mft_slots, [])
    check("no fabricated fallback record", m3._next_mft_record, None)
    check("allocation refuses rather than colliding",
          m3._allocate_mft_record(), None)

    # A mapper that never reported an MFT size must not invent slots.
    m4 = manager(FakeMapper([], [], total_records=0))
    check("unknown MFT size yields no slots", m4._available_mft_slots, [])
    check("unknown MFT size has no fallback", m4._next_mft_record, None)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
