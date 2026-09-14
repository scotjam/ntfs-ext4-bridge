"""Structural checks and repairs for a bridge NTFS image, straight from the
bytes - no ntfs-3g, no bridge state.

Used by tests/check_image_integrity.py (report) and
scripts/repair_directories.py (rewrite). Everything here reads the volume
as it is on disk, so run it against a copy, a backup, or with the bridge
stopped; through a running bridge the hot cache may hold newer MFT records
than the file does.

What it finds:

  cross-links      a cluster claimed by a file's $DATA and by NTFS structure
                   (a directory's index block, an attribute list, a security
                   descriptor, a system file). The bridge serves the file's
                   bytes there, so the structure reads as garbage.
  broken attrlists a directory whose $ATTRIBUTE_LIST points at extension
                   records that are gone - typically because the list's own
                   cluster was cross-linked and overwritten. ntfs-3g reports
                   "Index root attribute missing" and refuses every
                   operation on the directory.
  broken indexes   a directory whose $I30 index holds an entry that resolves
                   to nothing (listed, but stat fails) or the same name twice.
                   Left behind when an index block was served as file data
                   and written back.

Directories in the last two classes are repaired the same way: rebuilt as
an empty directory keeping their $STANDARD_INFORMATION and $FILE_NAME, so
the bridge's populate step refills them from ext4.
"""
import struct
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

REC = 1024
SECTOR = 512


def undo_fixups(buf: bytes, usa_at: int = 4) -> bytearray:
    """Undo the update-sequence fixups of an MFT record or INDX block."""
    b = bytearray(buf)
    usa_off, usa_cnt = struct.unpack_from('<HH', b, usa_at)
    for i in range(1, usa_cnt):
        end = i * SECTOR - 2
        src = usa_off + i * 2
        if end + 2 > len(b) or src + 2 > len(b):
            break
        b[end:end + 2] = b[src:src + 2]
    return b


def apply_fixups(b: bytearray) -> bytearray:
    usa_off, usa_cnt = struct.unpack_from('<HH', b, 4)
    usn = (struct.unpack_from('<H', b, usa_off)[0] + 1) & 0xFFFF or 1
    struct.pack_into('<H', b, usa_off, usn)
    for i in range(1, usa_cnt):
        end = i * SECTOR - 2
        struct.pack_into('<H', b, usa_off + i * 2, struct.unpack_from('<H', b, end)[0])
        struct.pack_into('<H', b, end, usn)
    return b


def parse_runs(rb: bytes) -> List[Tuple[int, int]]:
    """Mapping pairs -> [(lcn, count)], lcn == -1 for a sparse run."""
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


class Attr:
    __slots__ = ('type', 'name', 'resident', 'value', 'runs', 'off', 'length', 'instance')

    def __init__(self, type_, name, resident, value, runs, off, length, instance):
        self.type, self.name, self.resident = type_, name, resident
        self.value, self.runs, self.off, self.length, self.instance = value, runs, off, length, instance


def iter_attrs(rec: bytes) -> Iterator[Attr]:
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
            yield Attr(t, name, False, b'', parse_runs(rec[off + ro:off + l]), off, l, inst)
        else:
            vl, vo = struct.unpack_from('<IH', rec, off + 16)
            yield Attr(t, name, True, bytes(rec[off + vo:off + vo + vl]), [], off, l, inst)
        off += l


class Record:
    __slots__ = ('num', 'rec', 'flags', 'base', 'seq')

    def __init__(self, num, rec, flags, base, seq):
        self.num, self.rec, self.flags, self.base, self.seq = num, rec, flags, base, seq

    @property
    def in_use(self):
        return bool(self.flags & 1)

    @property
    def is_dir(self):
        return bool(self.flags & 2)


class Volume:
    """A bare NTFS volume file (or block device), parsed enough to reason
    about ownership of clusters and the shape of directories."""

    def __init__(self, path: str, writable: bool = False, part_offset: int = 0):
        self.path = path
        self.f = open(path, 'r+b' if writable else 'rb')
        self.part = part_offset
        boot = self.pread(0, 512)
        bps = struct.unpack_from('<H', boot, 0x0B)[0]
        self.spc = boot[0x0D]
        self.csz = bps * self.spc
        self.mft_lcn = struct.unpack_from('<Q', boot, 0x30)[0]
        self.total_clusters = struct.unpack_from('<Q', boot, 0x28)[0] // self.spc

        rec0 = undo_fixups(self.pread(self.mft_lcn * self.csz, REC))
        mft_runs = next(a.runs for a in iter_attrs(rec0) if a.type == 0x80 and not a.resident)
        self.mft_map, vcn = [], 0
        for lcn, cnt in mft_runs:
            self.mft_map.append((vcn, vcn + cnt, lcn)); vcn += cnt
        self.per_cluster = self.csz // REC
        self.nrec = vcn * self.per_cluster

        rec1 = undo_fixups(self.pread(self.rec_off(1), REC))
        mr = next((a.runs for a in iter_attrs(rec1) if a.type == 0x80 and not a.resident), [])
        self.mirror_lcn = mr[0][0] if mr else -1
        self.mirror_records = (mr[0][1] * self.csz) // REC if mr else 0

        self.recs: Dict[int, Record] = {}
        for n in range(self.nrec):
            o = self.rec_off(n)
            if o is None:
                continue
            raw = self.pread(o, REC)
            if raw[:4] != b'FILE':
                continue
            r = undo_fixups(raw)
            self.recs[n] = Record(n, r, struct.unpack_from('<H', r, 22)[0],
                                  struct.unpack_from('<Q', r, 32)[0] & 0xFFFFFFFFFFFF,
                                  struct.unpack_from('<H', r, 16)[0])

        self.names: Dict[int, str] = {}
        self.parents: Dict[int, int] = {}
        self.all_parents: Dict[int, set] = defaultdict(set)
        for n, r in self.recs.items():
            for a in iter_attrs(r.rec):
                if a.type == 0x30 and a.resident and len(a.value) >= 66:
                    pref = struct.unpack_from('<Q', a.value, 0)[0] & 0xFFFFFFFFFFFF
                    ln, ns = a.value[64], a.value[65]
                    self.all_parents[n].add(pref)
                    if n not in self.names or ns != 2:          # prefer the long name
                        self.names[n] = a.value[66:66 + ln * 2].decode('utf-16le', 'replace')
                        self.parents[n] = pref

    # -- raw access -------------------------------------------------------
    def pread(self, off: int, n: int) -> bytes:
        self.f.seek(self.part + off)
        return self.f.read(n)

    def pwrite(self, off: int, data: bytes):
        self.f.seek(self.part + off)
        self.f.write(data)

    def rec_off(self, n: int) -> Optional[int]:
        v = n // self.per_cluster
        for a, b, lcn in self.mft_map:
            if a <= v < b:
                return (lcn + (v - a)) * self.csz + (n % self.per_cluster) * REC
        return None

    def read_runs(self, runs, length: Optional[int] = None) -> bytes:
        out = bytearray()
        for lcn, cnt in runs:
            out += (b'\x00' * (cnt * self.csz) if lcn < 0 else self.pread(lcn * self.csz, cnt * self.csz))
        return bytes(out[:length]) if length is not None else bytes(out)

    def fullpath(self, n: int, depth: int = 0) -> str:
        if n == 5 or depth > 40:
            return ''
        p = self.fullpath(self.parents.get(n, 5), depth + 1)
        return (p + '/' if p else '') + self.names.get(n, f'<{n}>')

    def write_record(self, n: int, rec: bytearray):
        """Write a fixed-up record, and its $MFTMirr copy when it has one."""
        self.pwrite(self.rec_off(n), rec)
        if n < self.mirror_records and self.mirror_lcn >= 0:
            self.pwrite(self.mirror_lcn * self.csz + n * REC, rec)

    def close(self):
        self.f.flush()
        self.f.close()


# ---------------------------------------------------------------- findings

def find_cross_links(vol: Volume):
    """-> (per-file {rec: [(cluster, meta_label)]}, per-metadata-record hit counts)."""
    owner: Dict[int, str] = {}
    file_runs: Dict[int, List[Tuple[int, int]]] = {}
    for n, r in vol.recs.items():
        if not r.in_use:
            continue
        whole = n < 16 or r.is_dir or r.base != 0
        for a in iter_attrs(r.rec):
            if a.resident:
                continue
            if whole or not (a.type == 0x80 and not a.name):
                for lcn, cnt in a.runs:
                    if lcn >= 0:
                        for c in range(lcn, lcn + cnt):
                            owner.setdefault(c, f"{n}:{a.type:#x}")
            else:
                file_runs.setdefault(n, []).extend((l, c) for l, c in a.runs if l >= 0)
    cross = defaultdict(list)
    hits = defaultdict(int)
    for n, runs in file_runs.items():
        for lcn, cnt in runs:
            for c in range(lcn, lcn + cnt):
                if c in owner:
                    cross[n].append((c, owner[c]))
                    hits[int(owner[c].split(':')[0])] += 1
    return cross, hits


def find_broken_attrlist_dirs(vol: Volume) -> List[Tuple[int, List[int]]]:
    out = []
    for n, r in vol.recs.items():
        if not (r.in_use and r.is_dir and r.base == 0):
            continue
        for a in iter_attrs(r.rec):
            if a.type != 0x20:
                continue
            data = a.value if a.resident else vol.read_runs(a.runs)
            pos, ok, exts = 0, True, set()
            while pos + 26 <= len(data):
                et, el, nl, no, lv, ref, ei = struct.unpack_from('<IHBBQQH', data, pos)
                if el == 0:
                    break
                rn = ref & 0xFFFFFFFFFFFF
                if rn != n:
                    exts.add(rn)
                    ext = vol.recs.get(rn)
                    if ext is None or not ext.in_use or ext.base != n:
                        ok = False
                pos += el
            if not ok:
                out.append((n, sorted(exts)))
    return out


class IndexWalk:
    """One directory's $I30 index, read both ways NTFS reads it.

    `entries` come from descending the B+tree from $INDEX_ROOT - how a
    lookup (stat, open, unlink) finds a name. `bitmap_blocks` are the INDX
    blocks $INDEX_BITMAP says are in use - how readdir enumerates. The two
    must agree: a block that is in the bitmap but not in the tree is listed
    by readdir yet unfindable by lookup, which is exactly "ls shows it,
    stat says No such file, rm cannot remove it".
    """
    __slots__ = ('entries', 'reachable_blocks', 'bitmap_blocks', 'bad_blocks',
                 'block_size', 'vcn_unit', 'found')

    def __init__(self):
        self.entries: List[Tuple[str, int, int, str]] = []   # name, rec, seq, where
        self.reachable_blocks: set = set()                    # block indexes in the tree
        self.bitmap_blocks: set = set()                       # block indexes in $INDEX_BITMAP
        self.bad_blocks: List[int] = []
        self.block_size = 4096
        self.vcn_unit = SECTOR
        self.found = False


def walk_index(vol: Volume, n: int) -> IndexWalk:
    w = IndexWalk()
    root = alloc = bitmap = None
    holders = [vol.recs[n]] + [rr for rr in vol.recs.values() if rr.base == n and rr.in_use]
    for rr in holders:                      # base first, then extension records
        for a in iter_attrs(rr.rec):
            if a.name != '$I30':
                continue
            if a.type == 0x90 and a.resident and root is None:
                root = a.value
            elif a.type == 0xA0 and not a.resident and alloc is None:
                alloc = a.runs
            elif a.type == 0xB0 and bitmap is None:
                bitmap = a.value if a.resident else vol.read_runs(a.runs)
    if root is None or len(root) < 32:
        return w
    w.found = True
    w.block_size = struct.unpack_from('<I', root, 8)[0] or 4096
    w.vcn_unit = vol.csz if w.block_size >= vol.csz else SECTOR
    alloc_data = vol.read_runs(alloc) if alloc else b''
    if bitmap:
        for i in range(len(bitmap) * 8):
            if bitmap[i // 8] & (1 << (i % 8)) and i * w.block_size < len(alloc_data):
                w.bitmap_blocks.add(i)

    def entries(buf: bytes, hdr_at: int, where: str):
        eo, il = struct.unpack_from('<II', buf, hdr_at)
        pos, end = hdr_at + eo, min(hdr_at + il, len(buf))
        while pos + 16 <= end:
            ref, el, kl, fl = struct.unpack_from('<QHHH', buf, pos)
            if el < 16:
                break
            if fl & 1 and pos + el - 8 >= 0:
                block(struct.unpack_from('<Q', buf, pos + el - 8)[0])
            if kl >= 66 and pos + 16 + kl <= len(buf):
                key = buf[pos + 16:pos + 16 + kl]
                ln = key[64]
                w.entries.append((key[66:66 + ln * 2].decode('utf-16le', 'replace'),
                                  ref & 0xFFFFFFFFFFFF, ref >> 48, where))
            if fl & 2:
                break
            pos += el

    def block(vcn: int):
        idx = (vcn * w.vcn_unit) // w.block_size
        if idx in w.reachable_blocks:
            return
        w.reachable_blocks.add(idx)
        p = idx * w.block_size
        buf = alloc_data[p:p + w.block_size]
        if len(buf) < w.block_size or buf[:4] != b'INDX':
            w.bad_blocks.append(idx)
            return
        entries(undo_fixups(buf), 24, f'block{idx}')

    entries(root, 16, 'root')
    return w


def _index_entries(vol: Volume, n: int):
    """Compatibility: yield (name, rec, seq, where) from the tree walk."""
    yield from walk_index(vol, n).entries


def find_broken_index_dirs(vol: Volume) -> List[Tuple[int, List[str]]]:
    out = []
    for n, r in vol.recs.items():
        if not (r.in_use and r.is_dir and r.base == 0) or n == 5:
            continue
        reasons, seen = [], {}
        try:
            w = walk_index(vol, n)
            for idx in w.bad_blocks:
                reasons.append(f"unreadable INDX block {idx}")
            for idx in sorted(w.bitmap_blocks - w.reachable_blocks):
                reasons.append(f"orphan INDX block {idx}: in $INDEX_BITMAP but not in the "
                               f"tree (readdir lists its names, lookup cannot find them)")
            for idx in sorted(w.reachable_blocks - w.bitmap_blocks - set(w.bad_blocks)):
                reasons.append(f"INDX block {idx} in the tree but clear in $INDEX_BITMAP")
            for name, rn, seq, where in w.entries:
                tgt = vol.recs.get(rn)
                if tgt is None or not tgt.in_use:
                    reasons.append(f"dangling: {name!r} -> record {rn} not in use")
                elif seq and tgt.seq != seq:
                    reasons.append(f"stale: {name!r} -> record {rn} seq {seq} != {tgt.seq}")
                elif n not in vol.all_parents.get(rn, ()):
                    reasons.append(f"foreign: {name!r} -> record {rn} whose parent is not this dir")
                if name in seen and seen[name] != (rn, seq):
                    reasons.append(f"duplicate: {name!r} (records {seen[name][0]} and {rn})")
                elif name in seen:
                    reasons.append(f"duplicate: {name!r} (same record {rn} listed twice)")
                seen.setdefault(name, (rn, seq))
        except Exception as e:  # noqa: BLE001 - a directory we cannot parse is a finding
            reasons.append(f"unparseable index: {e}")
        if reasons:
            out.append((n, reasons))
    return out


# ------------------------------------------------------------------ repair

def empty_index_root(cluster_size: int, instance: int) -> bytes:
    """A resident $INDEX_ROOT('$I30') with one end entry: an empty directory."""
    blk = 4096
    cpb = blk // cluster_size if cluster_size <= blk else blk // SECTOR
    value = (struct.pack('<IIIB3x', 0x30, 1, blk, cpb)
             + struct.pack('<IIIB3x', 0x10, 0x20, 0x20, 0)
             + struct.pack('<QHHH2x', 0, 0x10, 0, 2))
    name = '$I30'.encode('utf-16le')
    hdr = struct.pack('<IIBBHHHIHBB', 0x90, 0x20 + len(value), 0, 4, 0x18, 0, instance,
                      len(value), 0x20, 0, 0)
    return hdr + name + value


def release_subtree(vol: Volume, n: int) -> int:
    """Mark every record whose $FILE_NAME parent chain leads to directory n
    as not in use, so nothing is left orphaned when n is rebuilt empty and
    populate recreates the tree from ext4. Returns how many were released.
    Clusters stay marked used in $Bitmap - see rebuild_dir_as_empty()."""
    children: Dict[int, List[int]] = defaultdict(list)
    for m, parents in vol.all_parents.items():
        for p in parents:
            children[p].append(m)
    released, stack = 0, list(children.get(n, []))
    seen = set()
    while stack:
        m = stack.pop()
        if m in seen or m == n:
            continue
        seen.add(m)
        r = vol.recs.get(m)
        if r is None or not r.in_use or r.base:
            continue
        stack.extend(children.get(m, []))
        rec = bytearray(r.rec)
        struct.pack_into('<H', rec, 22, r.flags & ~1)
        vol.write_record(m, apply_fixups(rec))
        r.flags &= ~1
        released += 1
    return released


def rebuild_dir_as_empty(vol: Volume, n: int) -> bool:
    """Rewrite directory record n as an empty directory, keeping its
    $STANDARD_INFORMATION and $FILE_NAME(s). Extension records that still
    claim n as their base are released, and so is everything below n (see
    release_subtree), because populate recreates the whole tree from ext4
    and a record that survives here would be an orphan with no index entry.
    Clusters the dropped attributes held are left marked used: leaking a few
    is harmless, freeing one that is still referenced is not."""
    release_subtree(vol, n)
    r = vol.recs[n]
    usa_off, usa_cnt = struct.unpack_from('<HH', r.rec, 4)
    kept = []
    for a in iter_attrs(r.rec):
        if a.type == 0x10 and a.resident:
            kept.insert(0, bytearray(r.rec[a.off:a.off + a.length]))
        elif a.type == 0x30 and a.resident:
            kept.append(bytearray(r.rec[a.off:a.off + a.length]))
    if len(kept) < 2 or kept[0][:4] != b'\x10\x00\x00\x00':
        return False
    new = bytearray(REC)
    new[:48] = r.rec[:48]
    attr_off = (usa_off + usa_cnt * 2 + 7) & ~7
    struct.pack_into('<H', new, 18, 1)
    struct.pack_into('<H', new, 20, attr_off)
    struct.pack_into('<H', new, 22, 0x03)
    struct.pack_into('<I', new, 28, REC)
    struct.pack_into('<Q', new, 32, 0)
    off, inst = attr_off, 0
    for a in kept:
        struct.pack_into('<H', a, 14, inst)
        new[off:off + len(a)] = a
        off += len(a); inst += 1
    ir = empty_index_root(vol.csz, inst); inst += 1
    new[off:off + len(ir)] = ir; off += len(ir)
    struct.pack_into('<I', new, off, 0xFFFFFFFF); off += 8
    struct.pack_into('<I', new, 24, off)
    struct.pack_into('<H', new, 40, inst)
    vol.write_record(n, apply_fixups(new))
    for m, rr in vol.recs.items():
        if rr.base == n and rr.in_use:
            er = bytearray(rr.rec)
            struct.pack_into('<H', er, 22, rr.flags & ~1)
            vol.write_record(m, apply_fixups(er))
    return True
