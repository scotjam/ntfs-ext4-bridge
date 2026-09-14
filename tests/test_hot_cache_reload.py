"""The hot metadata cache must be reloaded after anything rewrites the image
behind it - and until it is, flushing it destroys that rewrite.

This is the mechanism behind "$MFTMirr does not match $MFT (record 3)":
ntfsfix repaired the image on disk, the 64MB cache still held the old $MFT,
the NBD server served the two halves from different generations, and stop()
flushed the stale half back over the repair. No root or ntfs-3g needed - the
cache is exercised directly over a temp file.
"""
import mmap
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.cluster_mapper import _HotImageCache  # noqa: E402

HOT = 64 * 1024                      # small "metadata region" for the test
SIZE = 4 * HOT
OFF = 1000                           # inside the hot region, like $MFT is


def external_write(path, off, data):
    """What ntfsfix does: a plain write to the file, not through the cache."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.pwrite(fd, data, off)
        os.fsync(fd)
    finally:
        os.close(fd)


def file_bytes(path, off, n):
    with open(path, 'rb') as f:
        f.seek(off)
        return f.read(n)


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'image.raw')
        with open(path, 'wb') as f:
            f.write(b'\x00' * SIZE)
        fh = open(path, 'r+b')
        cache = _HotImageCache(mmap.mmap(fh.fileno(), 0), HOT)

        # 1. The hazard: an external write is invisible to the cache.
        external_write(path, OFF, b'REPAIRED')
        assert file_bytes(path, OFF, 8) == b'REPAIRED'
        assert cache[OFF:OFF + 8] == b'\x00' * 8, \
            "expected the cache to be stale here - the test premise is wrong"
        print("  ok: an external write is not seen by the hot cache (the hazard)")

        # 2. The damage: flushing the stale cache reverts the repair on disk.
        cache.flush()
        assert file_bytes(path, OFF, 8) == b'\x00' * 8, \
            "flush() of a stale cache no longer clobbers disk?"
        print("  ok: flushing a stale cache clobbers the repair on disk (the damage)")

        # 3. The fix: reload() makes the cache see the external write, and a
        #    flush afterwards preserves it.
        external_write(path, OFF, b'REPAIRED')
        cache.reload()
        assert cache[OFF:OFF + 8] == b'REPAIRED', "reload() did not pick up disk"
        cache.flush()
        assert file_bytes(path, OFF, 8) == b'REPAIRED', \
            "a reloaded cache should preserve the repair through flush()"
        print("  ok: reload() then flush() preserves the external write (the fix)")

        # 4. Nothing beyond the hot region is affected either way.
        far = HOT + 5000
        external_write(path, far, b'FARAWAY')
        assert cache[far:far + 7] == b'FARAWAY', \
            "beyond the hot region reads should come straight from the mmap"
        print("  ok: reads past the hot region are always live")

        cache.close()
        fh.close()
    print("PASS test_hot_cache_reload")


if __name__ == '__main__':
    main()
