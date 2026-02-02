#!/usr/bin/env python3
"""Debug MFT record structure."""
import struct
import sys

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ntfs-test.raw"
    data = open(filename, "rb").read()
    partition_start = 1048576
    boot = data[partition_start:partition_start+512]

    mft_cluster = struct.unpack("<Q", boot[0x30:0x38])[0]
    cluster_size = boot[0x0D] * 512
    mft_offset = partition_start + (mft_cluster * cluster_size)
    print("MFT at offset", mft_offset, "(cluster", mft_cluster, ")")

    # Read MFT record 0 ($MFT)
    mft0 = data[mft_offset:mft_offset+1024]
    print("Signature:", mft0[0:4])
    fixup_offset = struct.unpack("<H", mft0[4:6])[0]
    fixup_count = struct.unpack("<H", mft0[6:8])[0]
    print("Fixup offset:", fixup_offset)
    print("Fixup count:", fixup_count)
    first_attr = struct.unpack("<H", mft0[20:22])[0]
    print("First attr offset:", first_attr)
    flags = struct.unpack("<H", mft0[22:24])[0]
    print("Flags:", flags, "in_use:", flags & 1, "is_dir:", (flags >> 1) & 1)
    used_size = struct.unpack("<I", mft0[24:28])[0]
    print("Used size:", used_size)

    # Walk attributes
    offset = first_attr
    print("\nAttributes:")
    attr_names = {0x10: "$STANDARD_INFO", 0x30: "$FILE_NAME", 0x80: "$DATA", 0x90: "$INDEX_ROOT"}
    while offset < 1024 - 8:
        attr_type = struct.unpack("<I", mft0[offset:offset+4])[0]
        if attr_type == 0xFFFFFFFF:
            print("  END marker at offset", offset)
            break
        attr_len = struct.unpack("<I", mft0[offset+4:offset+8])[0]
        non_resident = mft0[offset+8]
        name = attr_names.get(attr_type, "0x%X" % attr_type)
        print("  %s: len=%d, non_resident=%d at offset %d" % (name, attr_len, non_resident, offset))

        if attr_type == 0x80 and non_resident == 0:
            # Resident $DATA - show content
            val_len = struct.unpack("<I", mft0[offset+16:offset+20])[0]
            val_off = struct.unpack("<H", mft0[offset+20:offset+22])[0]
            print("    Resident data: len=%d" % val_len)

        if attr_len == 0 or attr_len > 1024:
            print("  ERROR: Invalid attribute length")
            break
        offset += attr_len

    # Check fixups
    print("\nFixup array at offset", fixup_offset, ":")
    for i in range(fixup_count):
        val = struct.unpack("<H", mft0[fixup_offset + i*2:fixup_offset + i*2 + 2])[0]
        print("  [%d]: 0x%04X" % (i, val))

    # Check sector end values
    print("\nSector end values (should match fixup[0]):")
    print("  Sector 1 end (offset 510):", "0x%04X" % struct.unpack("<H", mft0[510:512])[0])

if __name__ == "__main__":
    main()
