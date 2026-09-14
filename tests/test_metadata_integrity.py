"""A cluster is either a file's data or NTFS structure, never both.

Exercises ClusterMapper against a real ntfs-3g volume:

  1. metadata clusters (a directory's index block) are collected, kept out
     of the free-run index, and set in $Bitmap;
  2. a read of such a cluster returns the image bytes even when a file's
     data runs claim it - the read path that made "even lezen" unreadable;
  3. a client write into $Bitmap through the NBD path is mirrored into the
     RAM bitmap cache, so a later _write_bitmap() cannot erase it - the
     origin of every cross-linked cluster;
  4. a file already sitting on a metadata cluster is moved off it at startup,
     the directory lists again, and the image checks clean.

Needs root, mkfs.ntfs and ntfs-3g; skips otherwise.
"""
import bisect
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ntfs_bridge.cluster_mapper import ClusterMapper, MFT_RECORD_SIZE  # noqa: E402


def sh(*cmd, check=True):
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True)


def indx_cluster_of(mapper, rel_dir):
    """First $INDEX_ALLOCATION cluster of a directory, via the mapper's parser."""
    rec = mapper.path_to_mft_record[rel_dir]
    off = mapper._rec_offset(rec)
    record = mapper._undo_fixups(bytearray(mapper.image[off:off + MFT_RECORD_SIZE]))
    for t, name, runs in mapper._iter_nonresident_runs(record):
        if t == 0xA0:
            return next(lcn for lcn, cnt in runs if lcn >= 0)
    raise AssertionError(f"{rel_dir} has no $INDEX_ALLOCATION - make it bigger")


def is_free_in_index(mapper, c):
    return any(s <= c < s + n for s, n in mapper._free_run_index)


def main():
    if os.geteuid() != 0 or not shutil.which('mkfs.ntfs') or not shutil.which('ntfs-3g'):
        print("SKIP test_metadata_integrity: needs root, mkfs.ntfs and ntfs-3g")
        return

    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, 'src')
    image = os.path.join(tmp, 'image.raw')
    mnt = os.path.join(tmp, 'mnt')
    os.makedirs(os.path.join(src, 'Share', 'sub'))
    os.makedirs(mnt)

    a_bytes = os.urandom(1024 * 1024)                 # non-resident, mapped
    with open(os.path.join(src, 'Share', 'a.bin'), 'wb') as fh:
        fh.write(a_bytes)
    for i in range(60):                               # enough entries to need an INDX block
        with open(os.path.join(src, 'Share', 'sub', f'file-{i:03}.txt'), 'wb') as fh:
            fh.write(b'x' * 10)

    sh('truncate', '-s', '256M', image)
    sh('mkfs.ntfs', '-F', '-q', image)
    sh('mount', '-t', 'ntfs-3g', '-o', 'rw', image, mnt)
    try:
        shutil.copytree(os.path.join(src, 'Share'), os.path.join(mnt, 'Share'))
    finally:
        sh('umount', mnt)

    cs = None
    try:
        m = ClusterMapper(image, src, protected_roots=['Share'])
        cs = m.cluster_size

        # ---- 1. metadata collected and reserved ----------------------------
        indx = indx_cluster_of(m, os.path.join('Share', 'sub'))
        assert indx in m._metadata_clusters, "directory INDX cluster not collected as metadata"
        assert not is_free_in_index(m, indx), "metadata cluster still in the free-run index"
        bm = m._read_bitmap()
        assert bm[indx // 8] & (1 << (indx % 8)), "metadata cluster is FREE in $Bitmap"
        print("  ok: directory index block is metadata: reserved in index and $Bitmap")

        # ---- 2. read precedence ---------------------------------------------
        a_src = os.path.join(src, 'Share', 'a.bin')
        bisect.insort(m._direct_run_map, (indx, indx + 1, a_src, 0))   # a stale claim
        data = m.read(indx * cs, 4096)
        assert data[:4] == b'INDX', f"metadata cluster read as file data: {data[:8]!r}"
        m._direct_run_map = [r for r in m._direct_run_map if r[0] != indx]
        print("  ok: a metadata cluster reads from the image even when a file claims it")

        # ---- 3. $Bitmap write coherence -------------------------------------
        free_start, free_cnt = m._free_run_index[-1]
        c = free_start + free_cnt // 2
        bm_start = m.bitmap_clusters[0][0] * cs
        byte_off = bm_start + c // 8
        old = m.image[byte_off:byte_off + 1][0]
        assert not old & (1 << (c % 8))
        m.write(byte_off, bytes([old | (1 << (c % 8))]))     # what ntfs-3g does over NBD
        assert m._bitmap_cache[c // 8] & (1 << (c % 8)), "client $Bitmap write not mirrored to cache"
        assert not is_free_in_index(m, c), "free-run index still offers a cluster the client took"
        m._write_bitmap(m._read_bitmap())                    # the write that used to clobber
        assert m.image[byte_off:byte_off + 1][0] & (1 << (c % 8)), \
            "_write_bitmap erased the client's allocation"
        print("  ok: a client write to $Bitmap survives the next _write_bitmap()")

        # ---- 4. cross-link repair at startup --------------------------------
        rec_a = next(r for r, p in m.mft_record_to_source.items() if p == a_src)
        size_a = len(a_bytes)
        need = (size_a + cs - 1) // cs
        # Point a.bin's runs at a range starting on the INDX cluster.
        assert m._update_mft_data_runs(rec_a, [(need, indx)], size_a)
        m.flush()
        m.close()

        m2 = ClusterMapper(image, src, protected_roots=['Share'])
        off = m2._rec_offset(rec_a)
        record = m2._undo_fixups(bytearray(m2.image[off:off + MFT_RECORD_SIZE]))
        runs = m2._extract_data_runs(record)
        assert runs and not any(lcn >= 0 and m2._overlaps_metadata(lcn, n) for lcn, n in runs), \
            f"a.bin still overlaps metadata after startup: {runs}"
        assert m2.read(indx * cs, 4096)[:4] == b'INDX'
        new_lcn = next(lcn for lcn, n in runs if lcn >= 0)
        assert m2.read(new_lcn * cs, 4096) == a_bytes[:4096], "relocated file reads wrong bytes"
        m2.flush()
        m2.close()
        print("  ok: a file sitting on a metadata cluster is moved off it at startup")

        # The volume is consistent and the directory lists through plain ntfs-3g.
        r = sh('ntfsfix', '-n', image, check=False)
        assert 'does not match' not in r.stdout + r.stderr, r.stdout + r.stderr
        sh('mount', '-t', 'ntfs-3g', '-o', 'ro', image, mnt)
        try:
            assert len(os.listdir(os.path.join(mnt, 'Share', 'sub'))) == 60
        finally:
            sh('umount', mnt)
        census = os.path.join(HERE, 'check_image_integrity.py')
        if os.path.exists(census):
            out = sh(sys.executable, census, image).stdout
            assert 'cross-linked clusters: 0 ' in out, out
            print("  ok: integrity census reports no cross-links")
        print("  ok: image consistent, directory lists via ntfs-3g")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("PASS test_metadata_integrity")


if __name__ == '__main__':
    main()
