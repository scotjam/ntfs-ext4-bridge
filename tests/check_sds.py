#!/usr/bin/env python3
import struct

ref = open('/tmp/ref-ntfs.raw', 'rb').read()
mft_offset = 4 * 4096
secure_rec = ref[mft_offset + 9*1024:mft_offset + 10*1024]

# Find $SDS data attribute
first_attr = struct.unpack('<H', secure_rec[20:22])[0]
off = first_attr
while off < 1024 - 8:
    attr_type = struct.unpack('<I', secure_rec[off:off+4])[0]
    if attr_type == 0xFFFFFFFF:
        break
    attr_len = struct.unpack('<I', secure_rec[off+4:off+8])[0]
    non_res = secure_rec[off+8]
    name_len = secure_rec[off+9]
    if attr_type == 0x80 and name_len > 0:  # Named $DATA
        name_off = struct.unpack('<H', secure_rec[off+10:off+12])[0]
        name = secure_rec[off+name_off:off+name_off+name_len*2].decode('utf-16-le')
        print(f"Found $DATA stream: '{name}', non_res={non_res}, attr_len={attr_len}")
        if non_res:
            # Non-resident - get data run info
            runs_off = struct.unpack('<H', secure_rec[off+32:off+34])[0]
            alloc = struct.unpack('<Q', secure_rec[off+40:off+48])[0]
            real = struct.unpack('<Q', secure_rec[off+48:off+56])[0]
            print(f"  alloc={alloc}, real={real}")
            runs = secure_rec[off+runs_off:off+attr_len]
            print(f"  data runs: {runs[:20].hex()}")

            # Parse data runs to find the actual data
            header = runs[0]
            if header:
                len_size = header & 0xF
                off_size = (header >> 4) & 0xF
                run_len = int.from_bytes(runs[1:1+len_size], 'little')
                run_off = int.from_bytes(runs[1+len_size:1+len_size+off_size], 'little', signed=True)
                print(f"  First run: {run_len} clusters at LCN {run_off}")

                # Read the actual $SDS data
                sds_offset = run_off * 4096
                sds_data = ref[sds_offset:sds_offset+min(256, real)]
                print(f"  $SDS data (first 256 bytes): {sds_data.hex()}")
    if attr_len == 0 or attr_len > 1024:
        break
    off += attr_len
