#!/usr/bin/env python3
"""Check what entries are in the root directory index."""
import struct

def check_index(data, mft_offset, label):
    root_rec = data[mft_offset + 5*1024:mft_offset + 6*1024]
    print(f"\n=== {label} Root Directory Index ===")

    # Find $INDEX_ROOT attribute with name $I30
    first_attr = struct.unpack('<H', root_rec[20:22])[0]
    off = first_attr
    while off < 1024 - 8:
        attr_type = struct.unpack('<I', root_rec[off:off+4])[0]
        if attr_type == 0xFFFFFFFF:
            break
        attr_len = struct.unpack('<I', root_rec[off+4:off+8])[0]
        name_len = root_rec[off+9]

        if attr_type == 0x90:  # $INDEX_ROOT
            name_off = struct.unpack('<H', root_rec[off+10:off+12])[0]
            name = root_rec[off+name_off:off+name_off+name_len*2].decode('utf-16-le') if name_len else ""
            print(f"Found $INDEX_ROOT (name='{name}')")

            val_len = struct.unpack('<I', root_rec[off+16:off+20])[0]
            val_off = struct.unpack('<H', root_rec[off+20:off+22])[0]
            index_data = root_rec[off+val_off:off+val_off+val_len]

            # Parse index root header (16 bytes)
            indexed_type = struct.unpack('<I', index_data[0:4])[0]
            collation = struct.unpack('<I', index_data[4:8])[0]
            block_size = struct.unpack('<I', index_data[8:12])[0]
            print(f"  Indexed type: 0x{indexed_type:X}, Collation: {collation}, Block size: {block_size}")

            # Parse node header (starts at offset 16)
            entries_off = struct.unpack('<I', index_data[16:20])[0]
            total_size = struct.unpack('<I', index_data[20:24])[0]
            alloc_size = struct.unpack('<I', index_data[24:28])[0]
            flags = index_data[28]
            print(f"  Entries offset: {entries_off}, Total: {total_size}, Alloc: {alloc_size}, Flags: {flags}")

            # Walk index entries (start at offset 16 + entries_off)
            entry_off = 16 + entries_off
            entry_num = 0
            while entry_off < val_len:
                mft_ref = struct.unpack('<Q', index_data[entry_off:entry_off+8])[0]
                entry_len = struct.unpack('<H', index_data[entry_off+8:entry_off+10])[0]
                key_len = struct.unpack('<H', index_data[entry_off+10:entry_off+12])[0]
                entry_flags = struct.unpack('<I', index_data[entry_off+12:entry_off+16])[0]

                if entry_flags & 2:  # End entry
                    print(f"  Entry {entry_num}: END (flags=0x{entry_flags:X})")
                    break

                mft_rec_num = mft_ref & 0xFFFFFFFFFFFF
                mft_seq = (mft_ref >> 48) & 0xFFFF

                # The key is a $FILE_NAME structure
                if key_len >= 66:
                    fn_data = index_data[entry_off+16:entry_off+16+key_len]
                    fn_name_len = fn_data[64]
                    fn_name = fn_data[66:66+fn_name_len*2].decode('utf-16-le', errors='ignore')
                    print(f"  Entry {entry_num}: MFT {mft_rec_num} (seq {mft_seq}), name='{fn_name}'")
                else:
                    print(f"  Entry {entry_num}: MFT {mft_rec_num} (seq {mft_seq}), key_len={key_len}")

                if entry_len == 0:
                    break
                entry_off += entry_len
                entry_num += 1

        if attr_len == 0 or attr_len > 1024:
            break
        off += attr_len

# Reference NTFS
ref = open('/tmp/ref-ntfs.raw', 'rb').read()
check_index(ref, 4 * 4096, "REFERENCE")

# Our NTFS
our = open('/tmp/ntfs-test.raw', 'rb').read()
check_index(our, 1048576 + 8 * 4096, "OUR")
