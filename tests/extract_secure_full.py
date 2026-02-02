#!/usr/bin/env python3
"""Extract all $Secure components from reference NTFS."""
import struct

ref = open('/tmp/ref-ntfs.raw', 'rb').read()
mft_offset = 4 * 4096
secure_rec = ref[mft_offset + 9*1024:mft_offset + 10*1024]

print("=== $Secure Record Analysis ===")
print(f"Signature: {secure_rec[0:4]}")
flags = struct.unpack('<H', secure_rec[22:24])[0]
print(f"Flags: 0x{flags:04X}")

first_attr = struct.unpack('<H', secure_rec[20:22])[0]
off = first_attr
print("\nAttributes:")
while off < 1024 - 8:
    attr_type = struct.unpack('<I', secure_rec[off:off+4])[0]
    if attr_type == 0xFFFFFFFF:
        print(f"  END at offset {off}")
        break
    attr_len = struct.unpack('<I', secure_rec[off+4:off+8])[0]
    non_res = secure_rec[off+8]
    name_len = secure_rec[off+9]
    name = ""
    if name_len > 0:
        name_off = struct.unpack('<H', secure_rec[off+10:off+12])[0]
        name = secure_rec[off+name_off:off+name_off+name_len*2].decode('utf-16-le')

    attr_names = {0x10: "$STD_INFO", 0x30: "$FILE_NAME", 0x80: "$DATA", 0x90: "$INDEX_ROOT"}
    type_name = attr_names.get(attr_type, f"0x{attr_type:X}")
    print(f"  {type_name} (name='{name}'): len={attr_len}, non_res={non_res}, at offset {off}")

    # Extract attribute bytes
    attr_bytes = secure_rec[off:off+attr_len]
    with open(f'/tmp/secure_attr_{off}_{type_name.replace("$","")}.bin', 'wb') as f:
        f.write(attr_bytes)

    if attr_type == 0x90:  # INDEX_ROOT - dump the index structure
        if non_res == 0:
            val_len = struct.unpack('<I', secure_rec[off+16:off+20])[0]
            val_off = struct.unpack('<H', secure_rec[off+20:off+22])[0]
            index_data = secure_rec[off+val_off:off+val_off+val_len]
            print(f"    Index data ({val_len} bytes): {index_data[:64].hex()}")

    if attr_len == 0 or attr_len > 1024:
        break
    off += attr_len

# Save the entire record
with open('/tmp/secure_mft_record.bin', 'wb') as f:
    f.write(secure_rec)
print(f"\nSaved complete MFT record to /tmp/secure_mft_record.bin")
