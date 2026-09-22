"""Two-way live sync, end to end, against a throwaway tree.

Everything here lives under SANDBOX on the root disk. The library disks are
never named and guard() refuses any path outside the sandbox.

What runs:
  * the bridge in --two-way mode (no local mount; it expects a guest agent)
  * a "guest": nbd-client + ntfs-3g on the NBD export, standing in for the VM
  * an agent emulator speaking the same /v1 protocol as bridge-agent.ps1 and
    executing ops on that mount with POSIX calls (mkdir, rm, mv, truncate,
    utime); gate_begin is emulated as unmount + disconnect, online as
    reconnect + mount
  * a reader that parses NTFS straight off the NBD device with O_DIRECT, so
    "what the guest sees" is what the bridge serves right now - not the
    kernel's page cache, and not ntfs-3g's zero-fill of uninitialised data
    (fsutil setvaliddata has no ntfs-3g equivalent, so the reader emulates
    a guest that has raised VDL: it reads the mapped clusters)

Each op is done on one side and the harness waits until the other side
mirrors it (names, sizes, bytes), recording the latency. After each phase
the ext4 tree is compared with a baseline: every path the scenario did not
touch must be byte-identical. That is the corruption check.

Usage: test_two_way_live.py [all|ntfs|ext4|mixed|restart] [--keep]
"""
import hashlib
import http.client
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import mmap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ntfs_bridge.data_runs import decode_data_runs  # noqa: E402

SANDBOX = "/root/bridge-test-2way"
NBD_PORT = 10812
CTRL_PORT = 10813
CLUSTER = 65536
MIRROR_TIMEOUT = 45.0

SOURCE = os.path.join(SANDBOX, "source")
IMAGE = os.path.join(SANDBOX, "image.raw")
MOUNT = os.path.join(SANDBOX, "mnt")          # bridge's own (unused in two-way)
GUEST = os.path.join(SANDBOX, "guest")        # our ntfs-3g mount
OVERFLOW = os.path.join(SANDBOX, "overflow")
TOKEN = os.path.join(SANDBOX, "agent.token")
LOG = os.path.join(SANDBOX, "bridge.log")

FAILS = []
LAT = []


def log(msg):
    print(msg, flush=True)


def fail(msg):
    FAILS.append(msg)
    log("  FAIL  " + msg)


def ok(msg):
    log("  ok    " + msg)


def guard(path):
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(SANDBOX) + os.sep) and real != os.path.realpath(SANDBOX):
        raise SystemExit("REFUSING: %s is outside %s" % (real, SANDBOX))
    return real


def sha(data):
    return hashlib.sha256(data).hexdigest()[:16]


def pattern(n, seed=1):
    return bytes(((i * 7 + seed * 13) & 0xFF) for i in range(n))


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

def build_tree(root):
    os.makedirs(root, exist_ok=True)

    def f(rel, data):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)

    for n in (0, 1, 511, 512, 700, 701, 4096, CLUSTER - 1, CLUSTER, CLUSTER + 1,
              3 * CLUSTER + 17):
        f("alpha/sizes/size_%d.bin" % n, pattern(n))
    f("alpha/content/all_zero.bin", bytes(CLUSTER))
    f("alpha/content/zero_then_data.bin", bytes(CLUSTER) + b"REAL" * 100)
    f("alpha/content/looks_like_file_record.bin", b"FILE\x30\x00\x03\x00" + bytes(1016))
    for i in range(300):
        f("alpha/many/entry_%03d.txt" % i, b"x%03d" % i)
    f("alpha/names/with spaces.txt", b"spaces")
    f("alpha/names/éèê-accents.txt", b"accents")
    f("alpha/names/日本語.txt", b"jp")
    f("alpha/names/dots...txt", b"dots")
    f("alpha/names/.hidden", b"hidden")
    deep = "alpha/deep"
    for i in range(12):
        deep += "/lvl%02d" % i
    f(deep + "/bottom.txt", b"deep")
    os.makedirs(os.path.join(root, "alpha/empty_dir"), exist_ok=True)
    f("beta/movies/film_a.bin", pattern(2 * CLUSTER + 5, 2))
    f("beta/movies/film_b.bin", pattern(CLUSTER + 100, 3))
    f("beta/movies/notes.txt", b"notes " * 50)
    f("beta/untouched/keep_a.bin", pattern(CLUSTER + 1, 4))
    f("beta/untouched/keep_b.txt", b"keep")
    f("beta/untouched/sub/keep_c.bin", pattern(4097, 5))


def ext4_snapshot(root, hash_prefix=None):
    """rel -> (kind, size, sha). With hash_prefix, only files under that
    prefix are hashed; the rest carry size only (real-scale trees)."""
    out = {}
    for dp, dn, fn in os.walk(root):
        rel_d = os.path.relpath(dp, root)
        if rel_d != ".":
            out[rel_d] = ("dir", 0, "")
        for n in fn:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, root)
            try:
                if hash_prefix is not None and not rel.startswith(hash_prefix):
                    out[rel] = ("file", os.stat(p).st_size, "")
                    continue
                with open(p, "rb") as fh:
                    data = fh.read()
                out[rel] = ("file", len(data), sha(data))
            except OSError as e:
                out[rel] = ("file", -1, "ERR:%s" % e)
    return out


# ---------------------------------------------------------------------------
# NTFS reader over the NBD device, O_DIRECT
# ---------------------------------------------------------------------------

class ReadError(Exception):
    pass


class DirectDev:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.buf = mmap.mmap(-1, 4 * 1024 * 1024)

    def pread(self, off, n):
        # align to 4096
        a_off = off & ~4095
        a_end = (off + n + 4095) & ~4095
        out = bytearray()
        pos = a_off
        while pos < a_end:
            chunk = min(len(self.buf), a_end - pos)
            os.lseek(self.fd, pos, os.SEEK_SET)
            got = os.readv(self.fd, [memoryview(self.buf)[:chunk]])
            if got <= 0:
                break
            out += self.buf[:got]
            pos += got
        return bytes(out[off - a_off:off - a_off + n])

    def close(self):
        os.close(self.fd)


def fixup(rec):
    rec = bytearray(rec)
    usa_off = struct.unpack_from("<H", rec, 4)[0]
    usa_cnt = struct.unpack_from("<H", rec, 6)[0]
    for i in range(1, usa_cnt):
        end = i * 512
        if end > len(rec):
            break
        rec[end - 2:end] = rec[usa_off + 2 * i:usa_off + 2 * i + 2]
    return bytes(rec)


class NtfsView:
    """Parses $MFT off the device; resolves paths; reads file bytes."""

    def __init__(self, dev_path):
        self.dev = DirectDev(dev_path)
        bs = self.dev.pread(0, 512)
        if bs[3:11] != b"NTFS    ":
            raise RuntimeError("not NTFS at %s" % dev_path)
        bps = struct.unpack_from("<H", bs, 11)[0]
        spc = bs[13]
        self.cs = bps * spc
        mft_lcn = struct.unpack_from("<Q", bs, 48)[0]
        v = struct.unpack_from("<b", bs, 64)[0]
        self.rs = (1 << -v) if v < 0 else v * self.cs
        rec0 = fixup(self.dev.pread(mft_lcn * self.cs, self.rs))
        runs = None
        for a in self.attrs(rec0):
            if a["type"] == 0x80 and not a["name"]:
                runs = a["runs"]
                self.mft_size = a["data_size"]
        if runs is None:
            raise RuntimeError("no $MFT data runs")
        self.mft_runs = runs

    def attrs(self, rec):
        out = []
        off = struct.unpack_from("<H", rec, 20)[0]
        while off + 8 <= len(rec):
            t = struct.unpack_from("<I", rec, off)[0]
            if t == 0xFFFFFFFF:
                break
            ln = struct.unpack_from("<I", rec, off + 4)[0]
            if ln == 0 or off + ln > len(rec):
                break
            nonres = rec[off + 8]
            nlen = rec[off + 9]
            noff = struct.unpack_from("<H", rec, off + 10)[0]
            name = rec[off + noff:off + noff + 2 * nlen].decode("utf-16-le", "replace")
            a = {"type": t, "name": name, "nonres": nonres}
            if nonres:
                a["data_size"] = struct.unpack_from("<Q", rec, off + 48)[0]
                a["init_size"] = struct.unpack_from("<Q", rec, off + 56)[0]
                roff = struct.unpack_from("<H", rec, off + 32)[0]
                a["runs"] = decode_data_runs(rec[off + roff:off + ln])
            else:
                vlen = struct.unpack_from("<I", rec, off + 16)[0]
                voff = struct.unpack_from("<H", rec, off + 20)[0]
                a["value"] = rec[off + voff:off + voff + vlen]
            out.append(a)
            off += ln
        return out

    def read_runs(self, runs, size):
        out = bytearray()
        for count, lcn in runs:
            n = count * self.cs
            if lcn is None:
                out += bytes(n)
            else:
                try:
                    out += self.dev.pread(lcn * self.cs, n)
                except OSError as e:
                    # the bridge refused (EIO): surface it as content, so the
                    # mirror wait retries instead of the harness dying
                    raise ReadError("EIO at cluster %d: %s" % (lcn, e))
            if len(out) >= size:
                break
        return bytes(out[:size])

    def record(self, num):
        # the whole $MFT is read once per view; records are slices of it
        if not hasattr(self, "_mft"):
            self._mft = self.read_runs(self.mft_runs, self.mft_size)
        off = num * self.rs
        if off + self.rs > len(self._mft):
            return None
        return fixup(self._mft[off:off + self.rs])

    def snapshot(self, hash_prefix=None):
        """rel path -> (kind, size, sha) for every live file/dir under root.
        With hash_prefix, only files under it are read and hashed."""
        nrec = self.mft_size // self.rs
        names = {}     # rec -> list of (parent, name)
        isdir = {}
        data = {}      # rec -> (size, runs|value)
        for num in range(nrec):
            rec = self.record(num)
            if rec is None or rec[0:4] != b"FILE":
                continue
            flags = struct.unpack_from("<H", rec, 22)[0]
            if not flags & 1:
                continue
            if struct.unpack_from("<H", rec, 20)[0] == 0:
                continue
            isdir[num] = bool(flags & 2)
            for a in self.attrs(rec):
                if a["type"] == 0x30 and not a["nonres"]:
                    v = a["value"]
                    if len(v) < 0x42:
                        continue
                    parent = struct.unpack_from("<Q", v, 0)[0] & 0xFFFFFFFFFFFF
                    ns = v[0x41]
                    nm = v[0x42:0x42 + 2 * v[0x40]].decode("utf-16-le", "replace")
                    if ns == 2:      # DOS short name
                        continue
                    names.setdefault(num, []).append((parent, nm))
                elif a["type"] == 0x80 and not a["name"]:
                    if a["nonres"]:
                        data[num] = (a["data_size"], a["runs"], True)
                    else:
                        data[num] = (len(a["value"]), a["value"], False)
        paths = {}

        def path_of(num, depth=0):
            if num == 5:
                return ""
            if depth > 64 or num not in names:
                return None
            parent, nm = names[num][0]
            pp = path_of(parent, depth + 1)
            if pp is None:
                return None
            return nm if pp == "" else pp + "/" + nm

        out = {}
        for num in names:
            if num < 16:
                continue
            for parent, nm in names[num]:
                pp = path_of(parent)
                if pp is None:
                    continue
                rel = nm if pp == "" else pp + "/" + nm
                if rel.startswith("System Volume Information") or rel.startswith("$"):
                    continue
                if isdir.get(num):
                    out[rel] = ("dir", 0, "")
                else:
                    size, payload, nonres = data.get(num, (0, b"", False))
                    if hash_prefix is not None and not rel.startswith(hash_prefix):
                        out[rel] = ("file", size, "")
                        continue
                    if nonres:
                        try:
                            b = self.read_runs(payload, size)
                        except ReadError as e:
                            out[rel] = ("file", size, str(e))
                            continue
                    else:
                        b = payload[:size]
                    out[rel] = ("file", size, sha(b))
        return out

    def close(self):
        self.dev.close()


# ---------------------------------------------------------------------------
# bridge + guest + agent
# ---------------------------------------------------------------------------

class Bridge:
    def __init__(self):
        self.proc = None
        self.fh = None

    def start(self, timeout=600):
        roots = ",".join(sorted(d for d in os.listdir(SOURCE)
                                if os.path.isdir(os.path.join(SOURCE, d))))
        cmd = [sys.executable, "-m", "ntfs_bridge.bridge",
               "--source", SOURCE, "--image", IMAGE, "--mount", MOUNT,
               "--port", str(NBD_PORT), "--partitioned", "--lazy",
               "--dealloc-timeout", "31536000", "--roots", roots,
               "--overflow-dir", OVERFLOW, "--two-way",
               "--control-host", "127.0.0.1", "--control-port", str(CTRL_PORT),
               "--agent-token-file", TOKEN]
        log("  bridge: %s" % " ".join(cmd[2:]))
        self.fh = open(LOG, "ab")
        self.fh.write(b"\n==== start %s ====\n" % time.ctime().encode())
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=self.fh,
                                     stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit("bridge exited early rc=%s; see %s" % (self.proc.returncode, LOG))
            try:
                tok = open(TOKEN).read().strip()
                c = http.client.HTTPConnection("127.0.0.1", CTRL_PORT, timeout=3)
                c.request("GET", "/v1/health", headers={"X-Bridge-Token": tok})
                r = c.getresponse()
                if r.status == 200:
                    r.read()
                    self.token = tok
                    log("  bridge control endpoint up")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise SystemExit("bridge control endpoint never came up; see %s" % LOG)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=60)
        if self.fh:
            self.fh.close()
        log("  bridge stopped")

    def health(self):
        c = http.client.HTTPConnection("127.0.0.1", CTRL_PORT, timeout=5)
        c.request("GET", "/v1/health", headers={"X-Bridge-Token": self.token})
        r = c.getresponse()
        return json.loads(r.read())


class Guest:
    """nbd-client + ntfs-3g, standing in for the VM."""

    def __init__(self):
        self.dev = None
        self.part = None
        self.mounted = False

    def connect(self):
        for i in range(16):
            d = "/dev/nbd%d" % i
            if not os.path.exists(d):
                continue
            r = subprocess.run(["nbd-client", "-c", d], capture_output=True)
            if r.returncode != 0:
                self.dev = d
                break
        if not self.dev:
            raise SystemExit("no free nbd device")
        r = subprocess.run(["nbd-client", "-N", "", "127.0.0.1", str(NBD_PORT), self.dev],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("nbd-client failed: %s" % r.stderr)
        time.sleep(1)
        subprocess.run(["partprobe", self.dev], capture_output=True)
        time.sleep(1)
        self.part = self.dev + "p1"
        if not os.path.exists(self.part):
            self.part = self.dev + "1"
        os.makedirs(GUEST, exist_ok=True)
        r = subprocess.run(["mount", "-t", "ntfs-3g", "-o", "rw,big_writes",
                            self.part, GUEST], capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(["nbd-client", "-d", self.dev], capture_output=True)
            raise SystemExit("guest mount failed: %s" % r.stderr.strip())
        self.mounted = True
        log("  guest: %s mounted at %s" % (self.part, GUEST))

    def flush(self):
        subprocess.run(["sync", "-f", GUEST], capture_output=True)
        os.sync()
        # ntfs-3g's fallocate writes zero pages through the device and they
        # stay in the kernel's block cache; a later edit of that page would
        # be applied to the cached zeros instead of the bytes the bridge
        # serves. Windows never writes those pages (createnew + setvaliddata
        # allocate without writing), so drop them here to stay faithful.
        if self.part:
            subprocess.run(["blockdev", "--flushbufs", self.part], capture_output=True)

    def disconnect(self):
        if self.mounted:
            self.flush()
            subprocess.run(["umount", GUEST], capture_output=True)
            if os.path.ismount(GUEST):
                subprocess.run(["umount", "-l", GUEST], capture_output=True)
            self.mounted = False
        if self.dev:
            subprocess.run(["nbd-client", "-d", self.dev], capture_output=True)
            time.sleep(1)
        log("  guest: disconnected")
        self.dev = None

    def remount(self):
        self.flush()
        subprocess.run(["umount", GUEST], capture_output=True)
        r = subprocess.run(["mount", "-t", "ntfs-3g", "-o", "rw,big_writes",
                            self.part, GUEST], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("guest remount failed: %s" % r.stderr.strip())

    def view(self):
        # the O_DIRECT view goes through the bridge; drop nothing, cache nothing
        return NtfsView(self.part)


class Agent(threading.Thread):
    """bridge-agent.ps1, in Python, on the guest mount."""

    def __init__(self, bridge, guest):
        super().__init__(daemon=True, name="agent")
        self.bridge = bridge
        self.guest = guest
        self.cursor = 0
        self.epoch = None
        self.running = True
        self.executed = []   # (op, path, status)
        self.gates = 0
        self.errors = []

    def call(self, ep, body, timeout=60):
        c = http.client.HTTPConnection("127.0.0.1", CTRL_PORT, timeout=timeout)
        c.request("POST", ep, body=json.dumps(body),
                  headers={"X-Bridge-Token": self.bridge.token,
                           "Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError("%s -> %d %s" % (ep, r.status, data[:200]))
        return json.loads(data or b"{}")

    def gpath(self, p):
        return os.path.join(GUEST, p.replace("\\", "/"))

    def execute(self, op):
        kind = op["op"]
        path = self.gpath(op["path"]) if op.get("path") else None
        if kind == "mkdir":
            os.makedirs(path, exist_ok=True)
        elif kind == "rm":
            if os.path.lexists(path):
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        elif kind == "mv":
            dst = self.gpath(op["dst"])
            if not os.path.lexists(path):
                if os.path.lexists(dst):
                    return
                raise FileNotFoundError(path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(path, dst)
        elif kind in ("create_sized", "resize"):
            if kind == "resize" and not os.path.exists(path):
                # same rule as bridge-agent.ps1: never resurrect a file
                raise FileNotFoundError("ENOENT: " + path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if op.get("data_b64") is not None:
                # small (resident) file: bytes travel in the op
                import base64 as _b64
                with open(path, "wb") as f:
                    f.write(_b64.b64decode(op["data_b64"]))
                if op.get("mtime_ms"):
                    t = op["mtime_ms"] / 1000.0
                    os.utime(path, (t, t))
                return
            size = int(op["size"])
            # fsutil createnew / SetLength allocate real clusters and
            # setvaliddata raises VDL without writing. ntfs-3g's truncate
            # would leave a sparse hole (nothing for the bridge to map);
            # posix_fallocate allocates clusters and sets init_size = size,
            # which is the closest emulation available.
            fd = os.open(path, os.O_RDWR | os.O_CREAT)
            try:
                cur = os.fstat(fd).st_size
                if size < cur:
                    os.ftruncate(fd, size)
                elif size > cur:
                    # ntfs-3g zero-fills the new clusters through the device
                    # (Windows does not); the batch's flush_volume op pushes
                    # those writes out before the ack
                    os.posix_fallocate(fd, cur, size - cur)
            finally:
                os.close(fd)
            if op.get("mtime_ms"):
                t = op["mtime_ms"] / 1000.0
                os.utime(path, (t, t))
        elif kind == "set_mtime":
            if os.path.exists(path) and op.get("mtime_ms"):
                t = op["mtime_ms"] / 1000.0
                os.utime(path, (t, t))
        elif kind == "flush_volume":
            self.guest.flush()
        elif kind == "gate_begin":
            self.gates += 1
            gid = op["gate_id"]
            log("  agent: gate %s begin -> going offline" % gid[:8])
            self.guest.disconnect()
            self.call("/v1/gate", {"gate_id": gid, "phase": "offline_confirmed"})
            while True:
                time.sleep(1)
                r = self.call("/v1/gate", {"gate_id": gid, "phase": "await_end"})
                if r.get("done"):
                    break
            self.guest.connect()
            self.call("/v1/gate", {"gate_id": gid, "phase": "online_confirmed"})
            log("  agent: gate %s done -> back online" % gid[:8])
        else:
            raise RuntimeError("unknown op %s" % kind)

    def run(self):
        while self.running:
            try:
                if not self.epoch:
                    h = self.call("/v1/hello", {"agent_version": "emu-1", "hostname": "emu"})
                    self.epoch = h["epoch"]
                    self.cursor = 0
                resp = self.call("/v1/poll", {"cursor": self.cursor}, timeout=90)
                if resp["epoch"] != self.epoch:
                    log("  agent: epoch change -> remount, cursor 0")
                    if self.guest.mounted:
                        self.guest.remount()
                    self.epoch = resp["epoch"]
                    self.cursor = 0
                    continue
                ops = resp.get("ops") or []
                if not ops:
                    continue
                results = []
                for op in ops:
                    try:
                        self.execute(op)
                        results.append({"seq": op["seq"], "status": "ok"})
                        self.executed.append((op["op"], op.get("path", ""), "ok"))
                    except Exception as e:
                        results.append({"seq": op["seq"], "status": "error",
                                        "code": "EFAIL", "message": str(e)})
                        self.executed.append((op["op"], op.get("path", ""), "error:%s" % e))
                        self.errors.append("%s %s: %s" % (op["op"], op.get("path"), e))
                    self.cursor = op["seq"]
                self.call("/v1/ack", {"epoch": self.epoch, "results": results})
            except Exception as e:
                if self.running:
                    if "409" in str(e) and "stale epoch" in str(e):
                        # a gate reset the epoch under a batch in flight:
                        # re-hello, same as bridge-agent.ps1's loop error path
                        self.epoch = None
                        continue
                    self.errors.append("loop: %s" % e)
                    time.sleep(2)
                    self.epoch = None


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def diff_snapshots(a, b, label_a, label_b):
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            out.append("only in %s: %s %s" % (label_b, b[k][0], k))
        elif k not in b:
            out.append("only in %s: %s %s" % (label_a, a[k][0], k))
        elif a[k] != b[k]:
            out.append("differs: %s  %s=%s  %s=%s" % (k, label_a, a[k], label_b, b[k]))
    return out


def wait_mirror(guest, label, expect=None, timeout=None):
    """Wait until the NTFS view equals ext4. Returns latency or None.

    expect: for a host-side op, the ext4 snapshot taken right after the op.
    ext4 must stay identical to it while the bridge catches up - comparing
    the volume to ext4 alone cannot see the bridge overwriting ext4 with
    bytes it then serves back.
    """
    if timeout is None:
        timeout = MIRROR_TIMEOUT
    t0 = time.time()
    last = []
    while time.time() - t0 < timeout:
        e = ext4_snapshot(SOURCE)
        if expect is not None:
            drift = diff_snapshots(expect, e, "written", "ext4-now")
            if drift:
                fail("%s: the bridge ALTERED ext4 after a host-side op; %d difference(s):"
                     % (label, len(drift)))
                for d in drift[:12]:
                    log("        " + d)
                return None
        try:
            v = guest.view()
            try:
                n = v.snapshot()
            finally:
                v.close()
        except (ReadError, OSError, RuntimeError) as err:
            # the bridge refused or the device is mid-reconnect (a gate,
            # a restart): not a mirror result, retry until the budget ends
            last = ["volume unreadable: %s" % err]
            time.sleep(1.0)
            continue
        last = diff_snapshots(e, n, "ext4", "ntfs")
        if not last:
            lat = time.time() - t0
            LAT.append((label, lat))
            ok("%-58s mirrored in %5.1fs" % (label, lat))
            return lat
        time.sleep(0.5)
    fail("%s: not mirrored after %.0fs; %d difference(s):" % (label, timeout, len(last)))
    for d in last[:12]:
        log("        " + d)
    return None


def host_op(guest, label, fn, timeout=None):
    """Run an ext4-side change and wait for the volume to mirror it."""
    try:
        fn()
    except Exception as e:
        fail("%s: host op raised %s" % (label, e))
        return None
    expect = ext4_snapshot(SOURCE)
    return wait_mirror(guest, "ext4> " + label, expect=expect, timeout=timeout)


def collateral_check(baseline, touched, phase):
    """Every ext4 path the phase did not touch must be byte-identical."""
    now = ext4_snapshot(SOURCE)
    bad = []
    for rel, v in baseline.items():
        if any(rel == t or rel.startswith(t + "/") for t in touched):
            continue
        if rel not in now:
            bad.append("VANISHED  %s" % rel)
        elif now[rel] != v:
            bad.append("CHANGED   %s  was=%s now=%s" % (rel, v, now[rel]))
    if bad:
        fail("%s: %d untouched ext4 path(s) damaged" % (phase, len(bad)))
        for b in bad[:20]:
            log("        " + b)
    else:
        ok("%s: %d untouched ext4 paths byte-identical" % (
            phase, sum(1 for r in baseline if not any(
                r == t or r.startswith(t + "/") for t in touched))))
    return not bad


# ---------------------------------------------------------------------------
# scenario ops
# ---------------------------------------------------------------------------

def g(rel):
    return os.path.join(GUEST, rel)


def s(rel):
    return os.path.join(SOURCE, rel)


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def patch(path, off, data):
    with open(path, "r+b") as f:
        f.seek(off)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def phase_ntfs(guest, touched):
    """Guest-side changes must land on ext4."""
    log("\n=== phase: NTFS -> ext4 (ops through the guest mount) ===")
    ops = [
        ("create 0-byte file", lambda: write(g("beta/new/empty.bin"), b""), ["beta/new"]),
        ("create resident file (100 B)", lambda: write(g("beta/new/small.bin"), pattern(100, 7)), ["beta/new"]),
        ("create 4096 B file", lambda: write(g("beta/new/page.bin"), pattern(4096, 8)), ["beta/new"]),
        ("create cluster+1 file", lambda: write(g("beta/new/big.bin"), pattern(CLUSTER + 1, 9)), ["beta/new"]),
        ("create 3-cluster file", lambda: write(g("beta/new/huge.bin"), pattern(3 * CLUSTER + 3, 10)), ["beta/new"]),
        ("create unicode name", lambda: write(g("beta/new/café 日本.txt"), b"unicode"), ["beta/new"]),
        ("edit: overwrite middle of existing big file",
         lambda: patch(g("beta/movies/film_a.bin"), CLUSTER + 10, b"EDITED-BY-GUEST" * 10), ["beta/movies/film_a.bin"]),
        ("edit: append to existing file",
         lambda: patch(g("beta/movies/film_b.bin"), CLUSTER + 100, pattern(5000, 11)), ["beta/movies/film_b.bin"]),
        ("edit: truncate existing file to 1000 B",
         lambda: os.truncate(g("beta/movies/film_a.bin"), 1000), ["beta/movies/film_a.bin"]),
        ("edit: resident -> non-resident (grow small file)",
         lambda: write(g("beta/movies/notes.txt"), pattern(9000, 12)), ["beta/movies/notes.txt"]),
        ("edit: overwrite guest-created file in place",
         lambda: patch(g("beta/new/huge.bin"), 5, b"XYZ"), ["beta/new"]),
        ("rename file", lambda: os.rename(g("beta/new/page.bin"), g("beta/new/page_renamed.bin")), ["beta/new"]),
        ("move file to another dir", lambda: os.rename(g("beta/new/big.bin"), g("beta/movies/big_moved.bin")),
         ["beta/new", "beta/movies/big_moved.bin"]),
        ("move file across roots", lambda: os.rename(g("beta/new/huge.bin"), g("alpha/sizes/huge_from_beta.bin")),
         ["beta/new", "alpha/sizes/huge_from_beta.bin"]),
        ("delete file", lambda: os.remove(g("beta/new/small.bin")), ["beta/new"]),
        ("delete pre-existing file", lambda: os.remove(g("alpha/sizes/size_512.bin")), ["alpha/sizes/size_512.bin"]),
        ("mkdir", lambda: os.mkdir(g("beta/new/sub")), ["beta/new"]),
        ("mkdir nested + file", lambda: write(g("beta/new/sub/deeper/x.txt"), b"nested"), ["beta/new"]),
        ("rename dir with contents", lambda: os.rename(g("beta/new/sub"), g("beta/new/sub_renamed")), ["beta/new"]),
        ("move dir into another root", lambda: os.rename(g("beta/new/sub_renamed"), g("alpha/moved_sub")),
         ["beta/new", "alpha/moved_sub"]),
        ("rename pre-existing dir", lambda: os.rename(g("alpha/names"), g("alpha/names_renamed")),
         ["alpha/names", "alpha/names_renamed"]),
        ("delete dir tree", lambda: shutil.rmtree(g("alpha/moved_sub")), ["alpha/moved_sub"]),
        ("delete pre-existing empty dir", lambda: os.rmdir(g("alpha/empty_dir")), ["alpha/empty_dir"]),
    ]
    for label, fn, t in ops:
        touched.update(t)
        try:
            fn()
        except Exception as e:
            fail("%s: guest op raised %s" % (label, e))
            continue
        guest.flush()
        wait_mirror(guest, "ntfs> " + label)


def phase_ext4(guest, touched):
    """ext4-side changes must appear on the volume, live."""
    log("\n=== phase: ext4 -> NTFS (ops on the ext4 source) ===")
    ops = [
        ("create 0-byte file", lambda: write(s("beta/host/empty.bin"), b""), ["beta/host"]),
        ("create resident file (100 B)", lambda: write(s("beta/host/small.bin"), pattern(100, 21)), ["beta/host"]),
        ("create cluster+1 file", lambda: write(s("beta/host/big.bin"), pattern(CLUSTER + 1, 22)), ["beta/host"]),
        ("create 3-cluster file", lambda: write(s("beta/host/huge.bin"), pattern(3 * CLUSTER + 3, 23)), ["beta/host"]),
        ("create unicode name", lambda: write(s("beta/host/café 日本.txt"), b"unicode"), ["beta/host"]),
        ("edit: overwrite middle of existing big file (same size)",
         lambda: patch(s("beta/untouched/keep_a.bin"), 100, b"EDITED-ON-HOST" * 10), ["beta/untouched/keep_a.bin"]),
        ("edit: append (grow) existing file",
         lambda: patch(s("alpha/sizes/size_4096.bin"), 4096, pattern(3000, 24)), ["alpha/sizes/size_4096.bin"]),
        ("edit: truncate existing file",
         lambda: os.truncate(s("alpha/sizes/size_65537.bin"), 100), ["alpha/sizes/size_65537.bin"]),
        ("edit: grow resident file past 700 B",
         lambda: write(s("alpha/sizes/size_511.bin"), pattern(20000, 25)), ["alpha/sizes/size_511.bin"]),
        ("edit: rewrite host-created file (new content, new size)",
         lambda: write(s("beta/host/huge.bin"), pattern(2 * CLUSTER, 26)), ["beta/host"]),
        ("rename file", lambda: os.rename(s("beta/host/small.bin"), s("beta/host/small_renamed.bin")), ["beta/host"]),
        ("move file to another dir", lambda: os.rename(s("beta/host/big.bin"), s("beta/movies/big_from_host.bin")),
         ["beta/host", "beta/movies/big_from_host.bin"]),
        ("move file across roots", lambda: os.rename(s("beta/host/huge.bin"), s("alpha/content/huge_from_host.bin")),
         ["beta/host", "alpha/content/huge_from_host.bin"]),
        ("delete file", lambda: os.remove(s("beta/host/empty.bin")), ["beta/host"]),
        ("delete pre-existing file", lambda: os.remove(s("alpha/sizes/size_701.bin")), ["alpha/sizes/size_701.bin"]),
        ("mkdir", lambda: os.mkdir(s("beta/host/sub")), ["beta/host"]),
        ("mkdir nested + file", lambda: write(s("beta/host/sub/deeper/y.txt"), b"nested-host"), ["beta/host"]),
        ("rename dir with contents", lambda: os.rename(s("beta/host/sub"), s("beta/host/sub_renamed")), ["beta/host"]),
        ("move dir into another root", lambda: os.rename(s("beta/host/sub_renamed"), s("alpha/host_sub")),
         ["beta/host", "alpha/host_sub"]),
        ("rename pre-existing dir", lambda: os.rename(s("alpha/content"), s("alpha/content_renamed")),
         ["alpha/content", "alpha/content_renamed"]),
        ("delete dir tree", lambda: shutil.rmtree(s("alpha/host_sub")), ["alpha/host_sub"]),
        ("bulk: 50 files in a new dir",
         lambda: [write(s("beta/host/bulk/f%02d.bin" % i), pattern(1000 + i, 30 + i)) for i in range(50)], ["beta/host"]),
        ("bulk: delete the dir", lambda: shutil.rmtree(s("beta/host/bulk")), ["beta/host"]),
    ]
    for label, fn, t in ops:
        touched.update(t)
        # the agent executes one op at a time; 50 creates through ntfs-3g
        # over NBD take ~1s each, so the bulk steps get a proportional budget
        host_op(guest, label, fn, timeout=240 if label.startswith("bulk") else None)


def phase_mixed(guest, touched):
    """Rapid sequences and both sides at once."""
    log("\n=== phase: mixed / rapid ===")
    touched.update(["beta/mix", "alpha/mix"])

    def rapid_guest():
        write(g("beta/mix/tmp.bin"), pattern(3000, 40))
        os.rename(g("beta/mix/tmp.bin"), g("beta/mix/tmp2.bin"))
        os.remove(g("beta/mix/tmp2.bin"))
        write(g("beta/mix/final.bin"), pattern(CLUSTER + 7, 41))
    rapid_guest()
    guest.flush()
    wait_mirror(guest, "ntfs> create->rename->delete->create, one burst")

    def rapid_host():
        write(s("alpha/mix/tmp.bin"), pattern(3000, 42))
        os.rename(s("alpha/mix/tmp.bin"), s("alpha/mix/tmp2.bin"))
        os.remove(s("alpha/mix/tmp2.bin"))
        write(s("alpha/mix/final.bin"), pattern(CLUSTER + 7, 43))
    host_op(guest, "create->rename->delete->create, one burst", rapid_host)

    # same name, recreated with different content on each side in turn
    host_op(guest, "recreate same name, smaller, resident",
            lambda: write(s("alpha/mix/final.bin"), pattern(500, 44)))
    write(g("beta/mix/final.bin"), pattern(2 * CLUSTER, 45))
    guest.flush()
    wait_mirror(guest, "ntfs> recreate same name, bigger")

    # both sides at once, different files
    def both():
        t = threading.Thread(target=lambda: [write(s("alpha/mix/host_%d.bin" % i), pattern(700 + i * 300, 50 + i))
                                             for i in range(10)])
        t.start()
        for i in range(10):
            write(g("beta/mix/guest_%d.bin" % i), pattern(700 + i * 300, 60 + i))
        t.join()
    both()
    guest.flush()
    wait_mirror(guest, "both> 10 files each side concurrently")
    # the host-written files must be exactly what the host wrote
    for i in range(10):
        want = pattern(700 + i * 300, 50 + i)
        got = open(s("alpha/mix/host_%d.bin" % i), "rb").read()
        if got != want:
            fail("both> host_%d.bin on ext4 is not what the host wrote (%d B, sha %s vs %s)"
                 % (i, len(got), sha(got), sha(want)))

    # cross-side: guest renames what host just created, host deletes what guest created
    os.rename(g("beta/mix/guest_0.bin"), g("beta/mix/guest_0_renamed_by_guest.bin"))
    guest.flush()
    wait_mirror(guest, "ntfs> rename a guest-created file")
    host_op(guest, "rename a host-created file",
            lambda: os.rename(s("alpha/mix/host_0.bin"), s("alpha/mix/host_0_renamed_by_host.bin")))
    host_op(guest, "host deletes a guest-created file",
            lambda: os.remove(s("beta/mix/guest_1.bin")))
    os.remove(g("alpha/mix/host_1.bin"))
    guest.flush()
    wait_mirror(guest, "ntfs> guest deletes a host-created file")
    host_op(guest, "host edits a guest-created file in place",
            lambda: patch(s("beta/mix/guest_2.bin"), 10, b"HOST-EDIT"))
    patch(g("alpha/mix/host_2.bin"), 10, b"GUEST-EDIT")
    guest.flush()
    wait_mirror(guest, "ntfs> guest edits a host-created file in place")
    # what the guest wrote must be exactly what ext4 now holds
    want = bytearray(pattern(700 + 2 * 300, 52))
    want[10:10 + len(b"GUEST-EDIT")] = b"GUEST-EDIT"
    got = open(s("alpha/mix/host_2.bin"), "rb").read()
    if got != bytes(want):
        fail("guest in-place edit: ext4 content is not the expected bytes")


def phase_edge(bridge, guest, agent, touched):
    """Edge cases: conflicts, overwrites, case-only renames, fragmentation,
    links, a consistency-gate cycle."""
    log("\n=== phase: edge cases ===")
    touched.update(["alpha/edge", "beta/edge", "alpha/links"])
    import signal

    # E1: both sides edit the same file at the same moment, disjoint ranges.
    write(s("alpha/edge/conflict.bin"), pattern(3 * CLUSTER, 80))
    host_op(guest, "create conflict file", lambda: None)
    # Guest writes to a file inside the short grace after the host's
    # create/resize of it are dropped by design (trailing driver writes for
    # that op must not reach ext4). This case is about steady state, so wait
    # until the coordinator holds no window at all (the grace is 2 s after
    # the ack, but under a loaded machine the ack itself can take longer).
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            if bridge.health().get("coordinator", {}).get("suppressed", 1) == 0:
                break
        except Exception:
            pass
        time.sleep(1)
    time.sleep(1)
    t = threading.Thread(target=lambda: patch(s("alpha/edge/conflict.bin"), 10, b"HOST-SIDE"))
    t.start()
    patch(g("alpha/edge/conflict.bin"), CLUSTER + 10, b"GUEST-SIDE")
    t.join()
    guest.flush()
    wait_mirror(guest, "both> disjoint edits of one file at once")
    got = open(s("alpha/edge/conflict.bin"), "rb").read()
    base = pattern(3 * CLUSTER, 80)
    if len(got) != len(base):
        fail("conflict: ext4 length changed (%d -> %d)" % (len(base), len(got)))
    host_ok = got[10:19] == b"HOST-SIDE"
    guest_ok = got[CLUSTER + 10:CLUSTER + 20] == b"GUEST-SIDE"
    rest_ok = (got[:10] == base[:10] and got[19:CLUSTER + 10] == base[19:CLUSTER + 10]
               and got[CLUSTER + 20:] == base[CLUSTER + 20:])
    if not rest_ok:
        fail("conflict: bytes outside the two edits changed")
    else:
        ok("conflict: untouched bytes intact; host edit %s, guest edit %s"
           % ("kept" if host_ok else "LOST", "kept" if guest_ok else "LOST"))
        if not (host_ok and guest_ok):
            fail("conflict: an edit was lost (host=%s guest=%s)" % (host_ok, guest_ok))

    # E2/E3: rename over an existing file, each side
    write(g("beta/edge/a.bin"), pattern(2000, 81))
    write(g("beta/edge/b.bin"), pattern(3000, 82))
    guest.flush()
    wait_mirror(guest, "ntfs> create a and b")
    os.replace(g("beta/edge/a.bin"), g("beta/edge/b.bin"))
    guest.flush()
    wait_mirror(guest, "ntfs> rename a over existing b")
    if os.path.exists(s("beta/edge/a.bin")) or open(s("beta/edge/b.bin"), "rb").read() != pattern(2000, 81):
        fail("rename-over: ext4 b.bin is not a's content, or a survived")
    write(s("beta/edge/c.bin"), pattern(2500, 83))
    host_op(guest, "create c", lambda: None)
    host_op(guest, "rename c over existing b", lambda: os.replace(s("beta/edge/c.bin"), s("beta/edge/b.bin")))

    # E4: case-only rename on the guest
    write(g("beta/edge/Case.TXT"), b"case")
    guest.flush()
    wait_mirror(guest, "ntfs> create Case.TXT")
    os.rename(g("beta/edge/Case.TXT"), g("beta/edge/case.txt"))
    guest.flush()
    wait_mirror(guest, "ntfs> case-only rename")

    # E5: fragmentation - fill, punch holes, then a big guest file
    for i in range(40):
        write(g("beta/edge/frag_%02d.bin" % i), pattern(CLUSTER, 90 + i))
    guest.flush()
    wait_mirror(guest, "ntfs> 40 cluster-sized files")
    for i in range(0, 40, 2):
        os.remove(g("beta/edge/frag_%02d.bin" % i))
    guest.flush()
    wait_mirror(guest, "ntfs> delete every other one")
    write(g("beta/edge/bigfrag.bin"), pattern(30 * CLUSTER + 123, 140))
    guest.flush()
    lat = wait_mirror(guest, "ntfs> 30-cluster file into the holes")
    if lat is not None and open(s("beta/edge/bigfrag.bin"), "rb").read() != pattern(30 * CLUSTER + 123, 140):
        fail("fragmented file: ext4 bytes differ from what the guest wrote")

    # E8: grow a guest file across cluster boundaries by repeated appends
    p = g("beta/edge/grow.bin")
    write(p, b"")
    acc = b""
    for i in range(10):
        chunk = pattern(20000, 150 + i)
        with open(p, "ab") as f:
            f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        acc += chunk
    guest.flush()
    lat = wait_mirror(guest, "ntfs> 10 appends across cluster boundaries")
    if lat is not None and open(s("beta/edge/grow.bin"), "rb").read() != acc:
        fail("append growth: ext4 bytes differ from the appended sequence")

    # E9/E10: rename chain and delete+recreate with a different size
    write(g("beta/edge/chain.bin"), pattern(5000, 160))
    os.rename(g("beta/edge/chain.bin"), g("beta/edge/chain1.bin"))
    os.rename(g("beta/edge/chain1.bin"), g("beta/edge/chain2.bin"))
    os.rename(g("beta/edge/chain2.bin"), g("beta/edge/chain3.bin"))
    guest.flush()
    wait_mirror(guest, "ntfs> rename chain a->b->c->d in one burst")
    os.remove(g("beta/edge/chain3.bin"))
    write(g("beta/edge/chain3.bin"), pattern(CLUSTER + 1, 161))
    guest.flush()
    wait_mirror(guest, "ntfs> delete + recreate same name, bigger")

    # E6: links and read-only on ext4
    os.makedirs(s("alpha/links"), exist_ok=True)
    write(s("alpha/links/target.txt"), b"target")
    os.link(s("alpha/links/target.txt"), s("alpha/links/hard.txt"))
    os.symlink("target.txt", s("alpha/links/sym.txt"))
    write(s("alpha/links/ro.txt"), b"ro")
    os.chmod(s("alpha/links/ro.txt"), 0o444)
    host_op(guest, "hardlink, symlink, read-only file appear", lambda: None)
    patch(g("alpha/links/ro.txt"), 0, b"RW")
    guest.flush()
    wait_mirror(guest, "ntfs> guest edits a read-only ext4 file")

    # E7: a consistency-gate cycle, then more traffic
    log("  sending SIGUSR1 (consistency gate)")
    gates_before = agent.gates
    bridge.proc.send_signal(signal.SIGUSR1)
    t0 = time.time()
    while agent.gates == gates_before and time.time() - t0 < 120:
        time.sleep(1)
    while not guest.mounted and time.time() - t0 < 180:
        time.sleep(1)
    if agent.gates == gates_before:
        fail("gate: agent never saw gate_begin")
    else:
        ok("gate cycled (%.0fs)" % (time.time() - t0))
    wait_mirror(guest, "gate> volume equals ext4 after the gate")
    write(g("beta/edge/after_gate_guest.bin"), pattern(777, 170))
    guest.flush()
    wait_mirror(guest, "ntfs> create after the gate")
    host_op(guest, "create after the gate", lambda: write(s("beta/edge/after_gate_host.bin"), pattern(778, 171)))


def phase_restart(bridge, guest, agent, touched):
    """Stop everything, restart on the reused image, verify, then keep going."""
    log("\n=== phase: restart on the reused image ===")
    agent.running = False
    guest.disconnect()
    bridge.stop()
    # change ext4 while everything is down: the restart must pick these up
    touched.update(["beta/down", "alpha/many/entry_000.txt", "alpha/many/entry_001.txt"])
    write(s("beta/down/added_while_down.bin"), pattern(CLUSTER + 9, 70))
    os.remove(s("alpha/many/entry_000.txt"))
    write(s("alpha/many/entry_001.txt"), b"rewritten while down")
    bridge.start()
    guest.connect()
    agent2 = Agent(bridge, guest)
    agent2.start()
    wait_mirror(guest, "restart> volume equals ext4 after restart")
    write(g("beta/down/after_restart_guest.bin"), pattern(1234, 71))
    guest.flush()
    wait_mirror(guest, "ntfs> create after restart")
    host_op(guest, "create after restart",
            lambda: write(s("beta/down/after_restart_host.bin"), pattern(4321, 72)))
    return agent2


# ---------------------------------------------------------------------------

def setup(fresh=True):
    guard(SANDBOX)
    if fresh:
        subprocess.run(["umount", "-l", GUEST], capture_output=True)
        subprocess.run(["umount", "-l", MOUNT], capture_output=True)
        if os.path.isdir(SANDBOX):
            shutil.rmtree(SANDBOX)
    for d in (SOURCE, MOUNT, GUEST, OVERFLOW):
        os.makedirs(d, exist_ok=True)
    build_tree(SOURCE)


def main():
    args = sys.argv[1:]
    which = args[0] if args and not args[0].startswith("--") else "all"
    keep = "--keep" in args
    setup(fresh=True)
    baseline = ext4_snapshot(SOURCE)
    log("baseline: %d ext4 paths" % len(baseline))
    bridge = Bridge()
    guest = Guest()
    agent = None
    try:
        bridge.start()
        guest.connect()
        agent = Agent(bridge, guest)
        agent.start()
        wait_mirror(guest, "startup> volume equals ext4")
        touched = set()
        if which in ("all", "ntfs"):
            phase_ntfs(guest, touched)
            collateral_check(baseline, touched, "after NTFS->ext4 phase")
        if which in ("all", "ext4"):
            phase_ext4(guest, touched)
            collateral_check(baseline, touched, "after ext4->NTFS phase")
        if which in ("all", "mixed"):
            phase_mixed(guest, touched)
            collateral_check(baseline, touched, "after mixed phase")
        if which in ("all", "edge"):
            phase_edge(bridge, guest, agent, touched)
            collateral_check(baseline, touched, "after edge phase")
        if which in ("all", "restart"):
            agent = phase_restart(bridge, guest, agent, touched)
            collateral_check(baseline, touched, "after restart phase")
        # final: overflow must not have swallowed anything
        stray = [os.path.join(dp, f) for dp, dn, fn in os.walk(OVERFLOW) for f in fn]
        if stray:
            fail("%d file(s) landed in the overflow dir: %s" % (len(stray), stray[:5]))
        else:
            ok("overflow dir empty")
        if agent and agent.errors:
            fail("agent reported %d error(s): %s" % (len(agent.errors), agent.errors[:5]))
        h = bridge.health()
        log("  bridge health: %s" % json.dumps(h)[:300])
        if h.get("journal", {}).get("dirty"):
            fail("journal has %d dirty path(s)" % h["journal"]["dirty"])
    finally:
        if agent:
            agent.running = False
        guest.disconnect()
        bridge.stop()
    log("\nlatency: %d ops, max %.1fs, mean %.1fs" % (
        len(LAT), max((l for _, l in LAT), default=0),
        (sum(l for _, l in LAT) / len(LAT)) if LAT else 0))
    if FAILS:
        log("\nFAILURES (%d):" % len(FAILS))
        for f in FAILS:
            log("  - " + f)
        log("\nTWO-WAY VERDICT: FAIL")
        return 1
    log("\nTWO-WAY VERDICT: every change mirrored, no collateral damage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
