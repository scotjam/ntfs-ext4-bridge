"""Durable, replayable journal of MFT->ext4 sync operations.

The bridge acknowledges an NBD MFT write as soon as it has updated the image
and queued the ext4 materialization to a background worker. If the process
dies (SIGKILL/OOM/power loss) between that ack and the worker running, the
ext4 side would silently be left behind the (potentially already-durable)
NTFS image — the "ack-before-durable" gap.

This journal closes it: every op is appended here before it is handed to the
worker, and marked done after the worker materializes it to ext4. On restart
the un-done ops are replayed, so ext4 is never left behind the image.

Durability model (matches the NBD contract):
- Appends are buffered. A SIGKILL/OOM leaves them in the OS buffer, so they
  survive a process crash and are replayed.
- `sync()` (called from the NBD FLUSH durability barrier, before the image
  is msync'd) fsyncs the journal, so everything acknowledged before a
  successful FLUSH is durable across a power loss too. The barrier also
  drains the worker queue first, so at FLUSH time the journal is empty.
- Between FLUSHes a power loss can lose an un-fsync'd op — but the image's
  own change for that op is equally un-durable (hot-cache RAM / un-msync'd
  page), so the two sides stay consistent (both lack it).

On-disk format (append-only, little-endian):
    op:   b'O' + seq(u64) + offset(u64) + length(u32) + data(length)
    done: b'D' + seq(u64)
The file is truncated to empty whenever the worker catches up (no ops
pending), which keeps it small across a long backup.
"""

import os
import struct
import threading

_OP = b'O'
_DONE = b'D'
_OP_HDR = struct.Struct('<QQI')   # seq, offset, length
_DONE_REC = struct.Struct('<Q')   # seq


def log(msg):
    print(f"[MftOpJournal] {msg}", flush=True)


def recover(path):
    """Read a pre-existing journal and return un-done ops in seq order.

    Called at startup BEFORE the journal is reopened for appending (which
    truncates it). Returns a list of (seq, offset, data). Tolerates a
    truncated tail (last record partially written before a crash).
    """
    ops = {}
    done = set()
    try:
        with open(path, 'rb') as f:
            blob = f.read()
    except OSError:
        return []
    pos = 0
    n = len(blob)
    while pos < n:
        tag = blob[pos:pos + 1]
        pos += 1
        if tag == _OP:
            if pos + _OP_HDR.size > n:
                break  # torn tail
            seq, offset, length = _OP_HDR.unpack_from(blob, pos)
            pos += _OP_HDR.size
            if pos + length > n:
                break  # torn tail
            ops[seq] = (offset, blob[pos:pos + length])
            pos += length
        elif tag == _DONE:
            if pos + _DONE_REC.size > n:
                break
            (seq,) = _DONE_REC.unpack_from(blob, pos)
            pos += _DONE_REC.size
            done.add(seq)
        else:
            break  # corrupt / unknown tag — stop
    pending = [(seq, off, data) for seq, (off, data) in sorted(ops.items())
               if seq not in done]
    return pending


class MftOpJournal:
    """Thread-safe persistent journal for MFT->ext4 sync ops."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._next_seq = 1
        self._pending = set()
        # Truncate any stale file (its ops were captured by recover() first).
        self._f = open(path, 'wb+', buffering=0)

    def append_op(self, offset, data):
        """Append an op and return its seq (called on the NBD write path)."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            self._f.write(_OP + _OP_HDR.pack(seq, offset, len(data)) + data)
            self._pending.add(seq)
            return seq

    def append_done(self, seq):
        """Mark an op materialized (called by the worker after ext4 sync).

        When nothing is pending, truncate the file to empty so it never grows
        unbounded across a long backup.
        """
        with self._lock:
            self._f.write(_DONE + _DONE_REC.pack(seq))
            self._pending.discard(seq)
            if not self._pending:
                self._f.seek(0)
                self._f.truncate(0)

    def sync(self):
        """Flush + fsync the journal (durability barrier / FLUSH)."""
        with self._lock:
            try:
                self._f.flush()
                os.fsync(self._f.fileno())
            except OSError as e:
                log(f"sync error: {e}")

    def close(self):
        with self._lock:
            try:
                self._f.close()
            except OSError:
                pass
