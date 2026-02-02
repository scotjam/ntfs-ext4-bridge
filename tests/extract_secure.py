#!/usr/bin/env python3
"""Extract $Secure data from reference NTFS."""
import struct

ref = open('/tmp/ref-ntfs.raw', 'rb').read()
mft_offset = 4 * 4096
secure_rec = ref[mft_offset + 9*1024:mft_offset + 10*1024]

# Write the raw $Secure MFT record
with open('/tmp/secure_record.bin', 'wb') as f:
    f.write(secure_rec)
print("Saved $Secure MFT record (1024 bytes)")

# Find and extract $SDS data
first_attr = struct.unpack('<H', secure_rec[20:22])[0]
off = first_attr
while off < 1024 - 8:
    attr_type = struct.unpack('<I', secure_rec[off:off+4])[0]
    if attr_type == 0xFFFFFFFF:
        break
    attr_len = struct.unpack('<I', secure_rec[off+4:off+8])[0]
    non_res = secure_rec[off+8]
    name_len = secure_rec[off+9]

    if attr_type == 0x80 and name_len > 0:  # Named DATA
        name_off = struct.unpack('<H', secure_rec[off+10:off+12])[0]
        name = secure_rec[off+name_off:off+name_off+name_len*2].decode('utf-16-le')
        print(f"Found named $DATA: '{name}', non_res={non_res}")

        if name == '$SDS' and non_res:
            runs_off = struct.unpack('<H', secure_rec[off+32:off+34])[0]
            real_size = struct.unpack('<Q', secure_rec[off+48:off+56])[0]

            # Parse data run
            runs = secure_rec[off+runs_off:off+attr_len]
            header = runs[0]
            len_size = header & 0xF
            off_size = (header >> 4) & 0xF
            run_len = int.from_bytes(runs[1:1+len_size], 'little')
            run_off = int.from_bytes(runs[1+len_size:1+len_size+off_size], 'little', signed=True)

            # Extract actual $SDS data
            sds_offset = run_off * 4096
            sds_data = ref[sds_offset:sds_offset+real_size]
            with open('/tmp/sds_data.bin', 'wb') as f:
                f.write(sds_data)
            print(f"Extracted $SDS: {real_size} bytes from cluster {run_off}")

    if attr_len == 0 or attr_len > 1024:
        break
    off += attr_len

print("Done")
