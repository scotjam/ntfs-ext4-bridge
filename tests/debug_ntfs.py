#!/usr/bin/env python3
"""Debug NTFS structures comprehensively."""
import struct
import sys

def parse_data_runs(data):
    """Parse NTFS data runs and return list of (length, lcn, delta) tuples."""
    runs = []
    offset = 0
    current_lcn = 0

    while offset < len(data):
        header = data[offset]
        if header == 0:
            break

        length_size = header & 0x0F
        offset_size = (header >> 4) & 0x0F

        if length_size == 0:
            break

        offset += 1

        # Read length (unsigned)
        length = int.from_bytes(data[offset:offset + length_size], "little", signed=False)
        offset += length_size

        # Read offset (signed)
        if offset_size > 0:
            lcn_delta = int.from_bytes(data[offset:offset + offset_size], "little", signed=True)
            offset += offset_size
            current_lcn += lcn_delta
        else:
            lcn_delta = 0

        runs.append((length, current_lcn, lcn_delta))

    return runs

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ntfs-test.raw"
    data = open(filename, "rb").read()
    partition_start = 1048576

    print("=== BOOT SECTOR ===")
    boot = data[partition_start:partition_start+512]
    print(f"OEM ID: {boot[3:11]}")
    bytes_per_sector = struct.unpack("<H", boot[0x0B:0x0D])[0]
    sectors_per_cluster = boot[0x0D]
    print(f"Bytes per sector: {bytes_per_sector}")
    print(f"Sectors per cluster: {sectors_per_cluster}")
    cluster_size = bytes_per_sector * sectors_per_cluster
    print(f"Cluster size: {cluster_size}")

    total_sectors = struct.unpack("<Q", boot[0x28:0x30])[0]
    print(f"Total sectors: {total_sectors}")
    print(f"Total size: {total_sectors * bytes_per_sector} bytes")

    mft_cluster = struct.unpack("<Q", boot[0x30:0x38])[0]
    mft_mirror = struct.unpack("<Q", boot[0x38:0x40])[0]
    print(f"MFT cluster: {mft_cluster}")
    print(f"MFT mirror cluster: {mft_mirror}")

    clusters_per_mft = boot[0x40]
    if clusters_per_mft > 127:
        mft_record_size = 1 << (256 - clusters_per_mft)
    else:
        mft_record_size = clusters_per_mft * cluster_size
    print(f"Clusters per MFT record (raw): {clusters_per_mft}")
    print(f"MFT record size: {mft_record_size}")

    serial = struct.unpack("<Q", boot[0x48:0x50])[0]
    print(f"Volume serial: 0x{serial:016X}")

    signature = struct.unpack("<H", boot[0x1FE:0x200])[0]
    print(f"Boot signature: 0x{signature:04X} (should be 0xAA55)")

    print("\n=== MFT RECORD 0 ($MFT) ===")
    mft_offset = partition_start + (mft_cluster * cluster_size)
    print(f"MFT byte offset in image: {mft_offset}")

    mft0 = data[mft_offset:mft_offset+1024]
    print(f"Signature: {mft0[0:4]} (should be b'FILE')")

    first_attr = struct.unpack("<H", mft0[20:22])[0]

    # Find $DATA attribute
    offset = first_attr
    while offset < 1024 - 8:
        attr_type = struct.unpack("<I", mft0[offset:offset+4])[0]
        if attr_type == 0xFFFFFFFF:
            break
        attr_len = struct.unpack("<I", mft0[offset+4:offset+8])[0]
        non_resident = mft0[offset+8]

        if attr_type == 0x80:  # $DATA
            print(f"\n$DATA attribute at offset {offset}:")
            print(f"  Non-resident: {non_resident}")
            if non_resident:
                # Non-resident attribute header
                data_runs_offset = struct.unpack("<H", mft0[offset+32:offset+34])[0]
                alloc_size = struct.unpack("<Q", mft0[offset+40:offset+48])[0]
                real_size = struct.unpack("<Q", mft0[offset+48:offset+56])[0]
                init_size = struct.unpack("<Q", mft0[offset+56:offset+64])[0]
                print(f"  Allocated size: {alloc_size}")
                print(f"  Real size: {real_size}")
                print(f"  Initialized size: {init_size}")
                print(f"  Data runs offset: {data_runs_offset}")

                # Parse data runs
                runs_start = offset + data_runs_offset
                runs_data = mft0[runs_start:runs_start+64]
                print(f"  Data runs bytes: {runs_data[:20].hex()}")

                runs = parse_data_runs(runs_data)
                print(f"  Parsed runs:")
                for i, (length, lcn, delta) in enumerate(runs):
                    print(f"    Run {i}: {length} clusters at LCN {lcn} (delta {delta})")
                    expected_offset = partition_start + (lcn * cluster_size)
                    print(f"           -> byte offset {expected_offset} in image")

        if attr_len == 0 or attr_len > 1024:
            break
        offset += attr_len

    print("\n=== VERIFICATION ===")
    print(f"MFT should be at cluster {mft_cluster}")
    print(f"MFT byte offset in partition: {mft_cluster * cluster_size}")
    print(f"MFT byte offset in image: {partition_start + mft_cluster * cluster_size}")
    print(f"Actual MFT offset used: {mft_offset}")

    # Check if MFT data runs point back to MFT location
    print(f"\nIf $MFT $DATA runs point to cluster {mft_cluster}, that is correct.")

if __name__ == "__main__":
    main()
