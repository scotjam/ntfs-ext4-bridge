#!/usr/bin/env python3
import struct

def get_filename_info(rec):
    """Extract parent MFT reference and filename from $FILE_NAME attribute."""
    first_attr = struct.unpack("<H", rec[20:22])[0]
    off = first_attr
    while off < 1024 - 8:
        attr_type = struct.unpack("<I", rec[off:off+4])[0]
        if attr_type == 0xFFFFFFFF:
            return None
        attr_len = struct.unpack("<I", rec[off+4:off+8])[0]
        if attr_type == 0x30:  # $FILE_NAME
            val_off = struct.unpack("<H", rec[off+20:off+22])[0]
            fn_data = rec[off+val_off:off+val_off+68]
            parent_ref = struct.unpack("<Q", fn_data[0:8])[0]
            parent_rec = parent_ref & 0xFFFFFFFFFFFF
            parent_seq = (parent_ref >> 48) & 0xFFFF
            name_len = fn_data[64]
            name = fn_data[66:66+name_len*2].decode("utf-16-le", errors="ignore")
            return (parent_rec, parent_seq, name)
        if attr_len == 0 or attr_len > 1024:
            return None
        off += attr_len
    return None

# Our NTFS
data = open("/tmp/ntfs-test.raw", "rb").read()
partition_start = 1048576
mft_offset = partition_start + (8 * 4096)

print("OUR NTFS:")
for i in [0, 1, 5, 9]:
    rec = data[mft_offset + i*1024:mft_offset + (i+1)*1024]
    info = get_filename_info(rec)
    print(f"  Record {i}: parent={info}")

# Reference NTFS
ref_data = open("/tmp/ref-ntfs.raw", "rb").read()
ref_mft_offset = 4 * 4096

print("\nREFERENCE NTFS:")
for i in [0, 1, 5, 9]:
    rec = ref_data[ref_mft_offset + i*1024:ref_mft_offset + (i+1)*1024]
    info = get_filename_info(rec)
    print(f"  Record {i}: parent={info}")
