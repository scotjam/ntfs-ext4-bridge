#!/usr/bin/env python3
"""Census of an ntfs-bridge image: cross-linked clusters and broken
attribute-list directories. Read-only. Usage: ntfs_census.py IMAGE [OFFSET]

OFFSET is the partition offset in bytes (0 for a bare image, 1048576 for
the partitioned images the bridge serves to the VM).
"""
import struct
import sys
from collections import defaultdict

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


def parse_runs(rb):
    """Mapping pairs -> [(lcn, count)], lcn=-1 for sparse."""
    runs, pos, lcn = [], 0, 0
    while pos < len(rb) and rb[pos]:
        h = rb[pos]; pos += 1
        ls, os_ = h & 0xF, h >> 4
        cnt = int.from_bytes(rb[pos:pos + ls], 'little'); pos += ls
        if os_:
            d = int.from_bytes(rb[pos:pos + os_], 'little', signed=True); pos += os_
            lcn += d
            runs.append((lcn, cnt))
        else:
            runs.append((-1, cnt))
    return runs


def attrs(rec):
    """Yield (type, name, resident, value_or_runs, instance, attr_off, attr_len)."""
    off = struct.unpack_from('<H', rec, 20)[0]
    while off + 8 <= REC:
        t, l = struct.unpack_from('<II', rec, off)
        if t == 0xFFFFFFFF or l == 0 or l > REC - off:
            return
        nonres = rec[off + 8]
        nl, no = rec[off + 9], struct.unpack_from('<H', rec, off + 10)[0]
        inst = struct.unpack_from('<H', rec, off + 14)[0]
        name = rec[off + no:off + no + nl * 2].decode('utf-16le', 'replace') if nl else ''
        if nonres:
            ro = struct.unpack_from('<H', rec, off + 32)[0]
            yield t, name, False, parse_runs(rec[off + ro:off + l]), inst, off, l
        else:
            vl, vo = struct.unpack_from('<IH', rec, off + 16)
            yield t, name, True, bytes(rec[off + vo:off + vo + vl]), inst, off, l
        off += l


def main():
    path = sys.argv[1]
    part = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    f = open(path, 'rb')

    def pread(off, n):
        f.seek(part + off)
        return f.read(n)

    boot = pread(0, 512)
    bps = struct.unpack_from('<H', boot, 0x0B)[0]
    spc = boot[0x0D]
    csz = bps * spc
    mft_lcn = struct.unpack_from('<Q', boot, 0x30)[0]
    total_clusters = struct.unpack_from('<Q', boot, 0x28)[0] // spc
    print(f"cluster={csz} mft_lcn={mft_lcn} clusters={total_clusters}")

    # $MFT runs from record 0
    rec0 = undo_fixups(pread(mft_lcn * csz, REC))
    mft_runs = None
    for t, name, res, val, *_ in attrs(rec0):
        if t == 0x80 and not res:
            mft_runs = val
    assert mft_runs, "no $MFT data runs"
    # vcn -> file offset
    mft_map = []
    vcn = 0
    for lcn, cnt in mft_runs:
        mft_map.append((vcn, vcn + cnt, lcn))
        vcn += cnt
    nrec = vcn * csz // REC
    per_cluster = csz // REC

    def rec_off(n):
        v = n // per_cluster
        for a, b, lcn in mft_map:
            if a <= v < b:
                return (lcn + (v - a)) * csz + (n % per_cluster) * REC
        return None

    print(f"$MFT: {len(mft_runs)} runs, {nrec} records")

    # ---- pass 1: read every record --------------------------------------
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

    names, parents, is_dir = {}, {}, {}
    for n, (r, flags, base) in recs.items():
        for t, name, res, val, *_ in attrs(r):
            if t == 0x30 and res and len(val) >= 66:
                pref = struct.unpack_from('<Q', val, 0)[0] & 0xFFFFFFFFFFFF
                ln, ns = val[64], val[65]
                nm = val[66:66 + ln * 2].decode('utf-16le', 'replace')
                if n not in names or ns != 2:      # prefer non-DOS name
                    names[n] = nm; parents[n] = pref
        is_dir[n] = bool(flags & 2)

    def fullpath(n, depth=0):
        if n == 5 or depth > 40:
            return ''
        p = fullpath(parents.get(n, 5), depth + 1)
        return (p + '/' if p else '') + names.get(n, f'<{n}>')

    # ---- pass 2: ownership -----------------------------------------------
    owner = {}                       # cluster -> label (metadata)
    file_runs = {}                   # rec -> [(lcn,cnt)]
    def claim_meta(lcn, cnt, label):
        for c in range(lcn, lcn + cnt):
            owner.setdefault(c, label)

    for n, (r, flags, base) in recs.items():
        in_use = flags & 1
        for t, name, res, val, inst, *_ in attrs(r):
            if res:
                continue
            if n < 16 or t in (0xA0, 0x20, 0x50) or is_dir.get(n) or base:
                for lcn, cnt in val:
                    if lcn >= 0:
                        claim_meta(lcn, cnt, f"meta:{n}:{name or ''}:{t:#x}")
            elif t == 0x80 and not name and in_use:
                file_runs.setdefault(n, []).extend(
                    (lcn, cnt) for lcn, cnt in val if lcn >= 0)

    # ---- report A: cross-links ------------------------------------------
    cross = defaultdict(list)        # file rec -> [(cluster, meta label)]
    for n, runs in file_runs.items():
        for lcn, cnt in runs:
            for c in range(lcn, lcn + cnt):
                if c in owner:
                    cross[n].append((c, owner[c]))
    print()
    print(f"=== A. cross-linked clusters: {sum(len(v) for v in cross.values())} "
          f"clusters across {len(cross)} files ===")
    hit_meta = defaultdict(int)
    for n, lst in sorted(cross.items(), key=lambda kv: -len(kv[1]))[:12]:
        metas = sorted({m for _, m in lst})
        print(f"  file rec {n:6} {len(lst):5} clusters  {fullpath(n)[:70]}")
        for m in metas[:3]:
            mn = int(m.split(':')[1])
            print(f"        overlaps {m}  ({'dir' if is_dir.get(mn) else 'sys'} {fullpath(mn)[:60]})")
    for n, lst in cross.items():
        for _, m in lst:
            hit_meta[int(m.split(':')[1])] += 1
    print(f"  metadata records hit: {len(hit_meta)}")
    for mn, k in sorted(hit_meta.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    rec {mn:6} {k:4} clusters  {'DIR ' if is_dir.get(mn) else 'SYS '}{fullpath(mn)[:70]}")

    # ---- report B: attribute-list directories ----------------------------
    print()
    print("=== B. directories with $ATTRIBUTE_LIST ===")
    bad = 0
    for n, (r, flags, base) in recs.items():
        if not (flags & 1) or not is_dir.get(n) or base:
            continue
        for t, name, res, val, *_ in attrs(r):
            if t != 0x20:
                continue
            if res:
                data = val
            else:
                data = b''.join(pread(lcn * csz, cnt * csz) for lcn, cnt in val if lcn >= 0)
            # find the real size: resident value length or runs; parse entries
            pos, refs, ok = 0, [], True
            while pos + 26 <= len(data):
                et, el, nl, no, lv, ref, ei = struct.unpack_from('<IHBBQQH', data, pos)
                if el == 0:
                    break
                rn = ref & 0xFFFFFFFFFFFF
                refs.append((et, rn))
                if rn != n:
                    ext = recs.get(rn)
                    if ext is None or not (ext[1] & 1) or ext[2] != n:
                        ok = False
                pos += el
            state = 'ok' if ok else 'BROKEN (extension record missing/freed/rebased)'
            if not ok:
                bad += 1
            print(f"  rec {n:6} {state:45} exts={sorted({rn for _, rn in refs if rn != n})}  {fullpath(n)[:60]}")
    print(f"  broken: {bad}")

    # ---- report C: extension records not in use but referenced ----------
    print()
    ext_free = [(n, b) for n, (r, fl, b) in recs.items() if b and not (fl & 1)]
    print(f"=== C. extension records (base!=0): {sum(1 for _,(r,fl,b) in recs.items() if b)}, "
          f"of which NOT in use: {len(ext_free)} ===")
    for n, b in ext_free[:10]:
        print(f"  rec {n} base {b} ({fullpath(b)[:60]})")


if __name__ == '__main__':
    main()
