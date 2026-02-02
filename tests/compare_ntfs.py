#!/usr/bin/env python3
"""Compare NTFS structures."""
import struct
import sys

def analyze_ntfs(filename, label, partition_offset=0):
    data = open(filename, "rb").read()
    boot = data[partition_offset:partition_offset+512]

    print(f"\n=== {label} ===")

    # Boot sector analysis
    bps = struct.unpack("<H", boot[0x0B:0x0D])[0]
    spc = boot[0x0D]
    cluster_size = bps * spc
    mft_cluster = struct.unpack("<Q", boot[0x30:0x38])[0]
    hidden = struct.unpack("<I", boot[0x1C:0x20])[0]
    total_sectors = struct.unpack("<Q", boot[0x28:0x30])[0]

    print(f"Hidden sectors: {hidden}")
    print(f"Total sectors: {total_sectors}")
    print(f"Bytes per sector: {bps}")
    print(f"Sectors per cluster: {spc}")
    print(f"Cluster size: {cluster_size}")
    print(f"MFT cluster: {mft_cluster}")

    mft_offset = partition_offset + (mft_cluster * cluster_size)

    # Check MFT record 0
    mft0 = data[mft_offset:mft_offset+1024]
    print(f"\nMFT Record 0:")
    print(f"  Signature: {mft0[0:4]}")

    usa_offset = struct.unpack("<H", mft0[4:6])[0]
    usa_count = struct.unpack("<H", mft0[6:8])[0]
    print(f"  USA offset: {usa_offset}, count: {usa_count}")

    seq_num = struct.unpack("<H", mft0[16:18])[0]
    hard_links = struct.unpack("<H", mft0[18:20])[0]
    first_attr = struct.unpack("<H", mft0[20:22])[0]
    flags = struct.unpack("<H", mft0[22:24])[0]
    used_size = struct.unpack("<I", mft0[24:28])[0]
    alloc_size = struct.unpack("<I", mft0[28:32])[0]

    print(f"  Sequence: {seq_num}")
    print(f"  Hard links: {hard_links}")
    print(f"  First attr offset: {first_attr}")
    print(f"  Flags: 0x{flags:04X}")
    print(f"  Used size: {used_size}")
    print(f"  Alloc size: {alloc_size}")

    # Check USA (fixup) values
    usn = struct.unpack("<H", mft0[usa_offset:usa_offset+2])[0]
    sec1_end = struct.unpack("<H", mft0[510:512])[0]
    sec2_end = struct.unpack("<H", mft0[1022:1024])[0]
    print(f"  USN value: 0x{usn:04X}")
    print(f"  Sector 1 end (510-511): 0x{sec1_end:04X} (should match USN)")
    print(f"  Sector 2 end (1022-1023): 0x{sec2_end:04X} (should match USN)")

    # Check fixup array entries
    for i in range(usa_count):
        val = struct.unpack("<H", mft0[usa_offset + i*2:usa_offset + i*2 + 2])[0]
        print(f"  Fixup[{i}]: 0x{val:04X}")

    # Dump attributes
    print(f"  Attributes:")
    off = first_attr
    while off < 1024 - 8:
        attr_type = struct.unpack("<I", mft0[off:off+4])[0]
        if attr_type == 0xFFFFFFFF:
            print(f"    END at offset {off}")
            break
        attr_len = struct.unpack("<I", mft0[off+4:off+8])[0]
        non_res = mft0[off+8]
        attr_names = {0x10: "$STD_INFO", 0x30: "$FILE_NAME", 0x80: "$DATA", 0x90: "$INDEX_ROOT"}
        name = attr_names.get(attr_type, f"0x{attr_type:X}")
        print(f"    {name}: len={attr_len}, non_res={non_res} at off {off}")
        if attr_len == 0 or attr_len > 1024:
            break
        off += attr_len

if __name__ == "__main__":
    # Reference NTFS (no partition)
    analyze_ntfs("/tmp/ref-ntfs.raw", "REFERENCE NTFS", 0)

    # Our NTFS (with partition at 1MB)
    analyze_ntfs("/tmp/ntfs-test.raw", "OUR NTFS", 1048576)
