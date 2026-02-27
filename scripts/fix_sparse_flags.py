#!/usr/bin/env python3
"""Clear the SPARSE flag from $STANDARD_INFORMATION for all files in the MFT.

This fixes ntfs-3g refusing to read files that were originally created as sparse
but later had their data runs replaced by allocate_file_direct().
"""
import struct
import mmap
import sys

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "/export/media/bridge-test/image.raw"

    f = open(image_path, "r+b")
    m = mmap.mmap(f.fileno(), 0)

    boot = m[0:512]
    bps = struct.unpack("<H", boot[0x0B:0x0D])[0]
    spc = boot[0x0D]
    cs = bps * spc
    mft_cluster = struct.unpack("<Q", boot[0x30:0x38])[0]
    mft_offset = mft_cluster * cs

    fixed = 0
    offset = mft_offset
    record_num = 0

    while offset + 1024 <= len(m):
        record = bytearray(m[offset:offset + 1024])
        if record[:4] != b"FILE":
            break

        # Undo fixups for inspection
        usa_off = struct.unpack("<H", record[4:6])[0]
        usa_cnt = struct.unpack("<H", record[6:8])[0]
        for i in range(1, usa_cnt):
            se = i * 512 - 2
            if usa_off + i * 2 + 2 <= 1024 and se + 2 <= 1024:
                orig = struct.unpack("<H", record[usa_off + i*2:usa_off + i*2 + 2])[0]
                struct.pack_into("<H", record, se, orig)

        flags = struct.unpack("<H", record[22:24])[0]
        if flags & 0x01 and not (flags & 0x02):  # In-use file
            first_attr = struct.unpack("<H", record[20:22])[0]
            off = first_attr
            stdinfo_flags_offset = None
            filename = None

            while off < 1024 - 8:
                atype = struct.unpack("<I", record[off:off+4])[0]
                if atype == 0xFFFFFFFF:
                    break
                alen = struct.unpack("<I", record[off+4:off+8])[0]
                if alen == 0 or alen > 1024:
                    break

                if atype == 0x10:  # STANDARD_INFORMATION
                    val_off = struct.unpack("<H", record[off+20:off+22])[0]
                    si_flags = struct.unpack("<I", record[off + val_off + 32:off + val_off + 36])[0]
                    if si_flags & 0x0200:
                        stdinfo_flags_offset = off + val_off + 32

                if atype == 0x30:  # FILENAME
                    val_off = struct.unpack("<H", record[off+20:off+22])[0]
                    nlen = record[off + val_off + 64]
                    name = record[off + val_off + 66:off + val_off + 66 + nlen * 2].decode(
                        "utf-16-le", errors="replace"
                    )
                    filename = name

                off += alen

            if stdinfo_flags_offset is not None:
                # Read current flags from raw mmap (fixups don't affect first 510 bytes)
                abs_offset = mft_offset + record_num * 1024 + stdinfo_flags_offset
                old_flags = struct.unpack("<I", m[abs_offset:abs_offset + 4])[0]
                new_flags = old_flags & ~0x0200  # Clear SPARSE
                struct.pack_into("<I", m, abs_offset, new_flags)
                print(f"  Record {record_num}: {filename} - cleared sparse ({old_flags:#06x} -> {new_flags:#06x})")
                fixed += 1

        offset += 1024
        record_num += 1

    m.flush()
    m.close()
    f.close()
    print(f"\nCleared sparse flag on {fixed} files")

if __name__ == "__main__":
    main()
