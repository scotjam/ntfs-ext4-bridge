#!/usr/bin/env python3
"""Rebuild directories whose $ATTRIBUTE_LIST points at extension records that
no longer exist, so ntfs-3g can list them and populate can refill them.

    repair_attrlist_dirs.py IMAGE            # report only
    repair_attrlist_dirs.py IMAGE --apply    # rewrite the broken records

Run with the bridge STOPPED. IMAGE is the bare NTFS volume the bridge serves
(the partition table is added in memory, not in the file).

What "broken" means here: a directory outgrew its 1024-byte MFT record, so
NTFS moved $INDEX_ROOT into an extension record and left an $ATTRIBUTE_LIST
saying where. The cluster holding that list was later handed to a file (the
bridge's $Bitmap cache did not know ntfs-3g had allocated it) and overwritten,
so the list now points at garbage. ntfs-3g reports "Index root attribute
missing" and refuses every operation on the directory, including adding the
children back.

The repair rewrites each such record as an empty directory: the original
$STANDARD_INFORMATION and $FILE_NAME are kept, everything that lived beyond
the base record is dropped, and a fresh empty $INDEX_ROOT is installed. The
children live on ext4; the next bridge start repopulates them. Clusters the
dropped attributes held are left marked used in $Bitmap - leaking a few
clusters is harmless, freeing one still in use is not.
"""
import struct
import sys

REC = 1024


def undo_fixups(rec):
    rec = bytearray(rec)
    usa_off, usa_cnt = struct.unpack_from('<HH', rec, 4)
    for i in range(1, usa_cnt):
        end = i * 512 - 2
        if end + 2 > len(rec) or usa_off + i * 2 + 2 > len(rec):
            break
        rec[end:end + 2] = rec[usa_off + i * 2:usa_off + i * 2 + 2]
    return rec


def apply_fixups(rec):
    usa_off, usa_cnt = struct.unpack_from('<HH', rec, 4)
    usn = (struct.unpack_from('<H', rec, usa_off)[0] + 1) & 0xFFFF or 1
    struct.pack_into('<H', rec, usa_off, usn)
    for i in range(1, usa_cnt):
        end = i * 512 - 2
        struct.pack_into('<H', rec, usa_off + i * 2, struct.unpack_from('<H', rec, end)[0])
        struct.pack_into('<H', rec, end, usn)
    return rec


def parse_runs(rb):
    runs, pos, lcn = [], 0, 0
    while pos < len(rb) and rb[pos]:
        h = rb[pos]; pos += 1
        ls, os_ = h & 0xF, h >> 4
        cnt = int.from_bytes(rb[pos:pos + ls], 'little'); pos += ls
        if os_:
            lcn += int.from_bytes(rb[pos:pos + os_], 'little', signed=True); pos += os_
            runs.append((lcn, cnt))
        else:
            runs.append((-1, cnt))
    return runs


def attrs(rec):
    """Yield (type, name, resident, value_or_runs, attr_off, attr_len)."""
    off = struct.unpack_from('<H', rec, 20)[0]
    while off + 8 <= REC:
        t, l = struct.unpack_from('<II', rec, off)
        if t == 0xFFFFFFFF or l == 0 or l > REC - off:
            return
        nonres = rec[off + 8]
        nl, no = rec[off + 9], struct.unpack_from('<H', rec, off + 10)[0]
        name = rec[off + no:off + no + nl * 2].decode('utf-16le', 'replace') if nl else ''
        if nonres:
            ro = struct.unpack_from('<H', rec, off + 32)[0]
            yield t, name, False, parse_runs(rec[off + ro:off + l]), off, l
        else:
            vl, vo = struct.unpack_from('<IH', rec, off + 16)
            yield t, name, True, bytes(rec[off + vo:off + vo + vl]), off, l
        off += l


def empty_index_root(cluster_size, instance):
    """A resident $INDEX_ROOT('$I30') holding one end entry: an empty directory."""
    blk = 4096
    cpb = blk // cluster_size if cluster_size <= blk else blk // 512
    value = (struct.pack('<IIIB3x', 0x30, 1, blk, cpb)          # INDEX_ROOT
             + struct.pack('<IIIB3x', 0x10, 0x20, 0x20, 0)       # INDEX_HEADER (empty)
             + struct.pack('<QHHH2x', 0, 0x10, 0, 2))            # end entry, LAST
    name = '$I30'.encode('utf-16le')
    hdr = struct.pack('<IIBBHHHIHBB', 0x90, 0x20 + len(value), 0, 4, 0x18, 0, instance,
                      len(value), 0x20, 0, 0)
    return hdr + name + value


def main():
    path = sys.argv[1]
    apply = '--apply' in sys.argv[2:]
    f = open(path, 'r+b' if apply else 'rb')

    def pread(off, n):
        f.seek(off); return f.read(n)

    boot = pread(0, 512)
    bps = struct.unpack_from('<H', boot, 0x0B)[0]; spc = boot[0x0D]; csz = bps * spc
    mft_lcn = struct.unpack_from('<Q', boot, 0x30)[0]
    rec0 = undo_fixups(pread(mft_lcn * csz, REC))
    mft_runs = next(v for t, n, r, v, *_ in attrs(rec0) if t == 0x80 and not r)
    mft_map, vcn = [], 0
    for lcn, cnt in mft_runs:
        mft_map.append((vcn, vcn + cnt, lcn)); vcn += cnt
    per = csz // REC
    nrec = vcn * per

    def rec_off(n):
        v = n // per
        for a, b, lcn in mft_map:
            if a <= v < b:
                return (lcn + (v - a)) * csz + (n % per) * REC
        return None

    # $MFTMirr: where the first records are mirrored (record 1's $DATA)
    rec1 = undo_fixups(pread(rec_off(1), REC))
    mirror_runs = next((v for t, n, r, v, *_ in attrs(rec1) if t == 0x80 and not r), [])
    mirror_lcn = mirror_runs[0][0] if mirror_runs else -1
    mirror_records = (mirror_runs[0][1] * csz) // REC if mirror_runs else 0

    recs = {}
    for n in range(nrec):
        o = rec_off(n)
        if o is None:
            continue
        raw = pread(o, REC)
        if raw[:4] != b'FILE':
            continue
        r = undo_fixups(raw)
        flags = struct.unpack_from('<H', r, 22)[0]
        base = struct.unpack_from('<Q', r, 32)[0] & 0xFFFFFFFFFFFF
        recs[n] = (r, flags, base)

    names, parents = {}, {}
    for n, (r, flags, base) in recs.items():
        for t, name, res, val, *_ in attrs(r):
            if t == 0x30 and res and len(val) >= 66:
                pref = struct.unpack_from('<Q', val, 0)[0] & 0xFFFFFFFFFFFF
                ln, ns = val[64], val[65]
                if n not in names or ns != 2:
                    names[n] = val[66:66 + ln * 2].decode('utf-16le', 'replace')
                    parents[n] = pref

    def fullpath(n, d=0):
        if n == 5 or d > 40:
            return ''
        p = fullpath(parents.get(n, 5), d + 1)
        return (p + '/' if p else '') + names.get(n, f'<{n}>')

    broken = []
    for n, (r, flags, base) in recs.items():
        if not (flags & 1) or not (flags & 2) or base:
            continue
        for t, name, res, val, *_ in attrs(r):
            if t != 0x20:
                continue
            data = val if res else b''.join(
                pread(lcn * csz, cnt * csz) for lcn, cnt in val if lcn >= 0)
            pos, ok, exts = 0, True, set()
            while pos + 26 <= len(data):
                et, el, nl, no, lv, ref, ei = struct.unpack_from('<IHBBQQH', data, pos)
                if el == 0:
                    break
                rn = ref & 0xFFFFFFFFFFFF
                if rn != n:
                    exts.add(rn)
                    ext = recs.get(rn)
                    if ext is None or not (ext[1] & 1) or ext[2] != n:
                        ok = False
                pos += el
            if not ok:
                broken.append((n, sorted(exts)))

    print(f"{len(broken)} broken attribute-list director{'y' if len(broken) == 1 else 'ies'}")
    for n, exts in broken:
        print(f"  rec {n:6}  {fullpath(n)}")

    if not apply:
        print("\nReport only. Re-run with --apply to rebuild them as empty directories.")
        return

    for n, exts in broken:
        r, flags, base = recs[n]
        usa_off, usa_cnt = struct.unpack_from('<HH', r, 4)
        new = bytearray(REC)
        new[:48] = r[:48]                                  # header incl. USA
        attr_off = (usa_off + usa_cnt * 2 + 7) & ~7
        struct.pack_into('<H', new, 18, 1)                 # hard links
        struct.pack_into('<H', new, 20, attr_off)
        struct.pack_into('<H', new, 22, 0x03)              # in use | directory
        struct.pack_into('<I', new, 28, REC)
        struct.pack_into('<Q', new, 32, 0)                 # base record: none
        off, inst = attr_off, 0
        kept = []
        for t, name, res, val, aoff, alen in attrs(r):
            if t == 0x10 and res:
                kept.insert(0, bytes(r[aoff:aoff + alen]))
            elif t == 0x30 and res:
                kept.append(bytes(r[aoff:aoff + alen]))
        if not any(a[:4] == b'\x10\x00\x00\x00' for a in kept) or len(kept) < 2:
            print(f"  rec {n}: base record lacks resident $STANDARD_INFORMATION/$FILE_NAME, skipping")
            continue
        for a in kept:
            a = bytearray(a)
            struct.pack_into('<H', a, 14, inst)             # renumber instances
            new[off:off + len(a)] = a
            off += len(a); inst += 1
        ir = empty_index_root(csz, inst); inst += 1
        new[off:off + len(ir)] = ir; off += len(ir)
        struct.pack_into('<I', new, off, 0xFFFFFFFF); off += 8
        struct.pack_into('<I', new, 24, off)               # bytes used
        struct.pack_into('<H', new, 40, inst)              # next attribute id
        apply_fixups(new)
        o = rec_off(n)
        f.seek(o); f.write(new)
        if n < mirror_records and mirror_lcn >= 0:
            f.seek(mirror_lcn * csz + n * REC); f.write(new)
        # release orphaned extension records that still claim this base
        for rn in exts:
            ext = recs.get(rn)
            if ext and ext[2] == n and (ext[1] & 1):
                er = bytearray(ext[0])
                struct.pack_into('<H', er, 22, ext[1] & ~1)
                apply_fixups(er)
                f.seek(rec_off(rn)); f.write(er)
        print(f"  rebuilt rec {n} as empty directory: {fullpath(n)}")
    f.flush()
    print("\nDone. Verify with: ntfsfix -n IMAGE ; then start the bridge so populate refills them.")


if __name__ == '__main__':
    main()
