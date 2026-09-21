"""Two-way sync with the REAL Windows VM as the guest, on the real library
through the read-only overlay.

Same idea as test_real_library_two_way.py, with ntfs.sys and the PowerShell
agent instead of ntfs-3g and the emulator:

  * the library is the lowerdir of an overlayfs: the kernel never writes to
    it; the upperdir is the ledger of everything that would have been written
  * the bridge runs in --two-way on the overlay farm, NBD on the host, the
    agent control endpoint on the libvirt NAT gateway
  * a virtio NBD disk is hot-plugged into the VM (the production bridge disk
    stays detached; the production service stays inactive)
  * bridge-agent.ps1 is installed in the VM (scheduled task, SYSTEM) pointed
    at this bridge's endpoint and token
  * guest-side changes are made by PowerShell over WinRM; host-side changes
    on the overlay's merged view; the volume is then read back through
    Windows (names, sizes, SHA-256 of the scratch files)

Checks: every change mirrored within a budget; the real tree's stat manifest
unchanged; the upperdir holds only the scratch subtree and one renamed small
file; a consistency-gate cycle (the agent takes the disk offline/online);
the agent survives a bridge restart (disk detach/attach).

Usage: test_windows_two_way.py [--keep-agent]
Credentials: /root/.bridge-winrm.json {url, user, password} (mode 600).
"""
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_real_library_overlay as ov  # noqa: E402
import test_two_way_live as tw           # noqa: E402

VM = "windows-vm"
NBD_PORT = 10820
CTRL_HOST = "192.168.122.1"
CTRL_PORT = 10821
IMAGE = ov.IMAGE
TOKEN = IMAGE + ".agent-token"
LOG = ov.LOGP
SCRATCH = ov.SHARE + "/_twoway_scratch"
CLUSTER = 65536
BUDGET = 120.0
DISK_XML = "/root/bridge-test-disk.xml"
FAILS = []
LAT = []


def log(m):
    print(m, flush=True)


def fail(m):
    FAILS.append(m)
    log("  FAIL  " + m)


def ok(m):
    log("  ok    " + m)


def sha_full(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# WinRM
# ---------------------------------------------------------------------------

class Win:
    def __init__(self):
        import winrm
        c = json.load(open("/root/.bridge-winrm.json"))
        self.s = winrm.Session(c["url"], auth=(c["user"], c["password"]),
                               transport="ntlm", read_timeout_sec=300,
                               operation_timeout_sec=240)
        self.drive = None

    def ps(self, script, check=True):
        r = self.s.run_ps(script)
        out = r.std_out.decode("utf-8", "replace")
        err = r.std_err.decode("utf-8", "replace")
        if check and r.status_code != 0:
            raise RuntimeError("powershell rc=%d: %s" % (r.status_code, err[:500]))
        return out

    def path(self, rel):
        return self.drive + ":\\" + rel.replace("/", "\\")

    def put_file(self, winpath, data, chunk=1200):
        """Write bytes to a guest path. WinRM's command line is limited to
        ~8 KB, so the base64 goes over in pieces into a temp file that the
        guest decodes at the end."""
        # stage on C: so the bridge volume never sees the temp file
        tmp = "C:\\Windows\\Temp\\bridge-put-%d.b64" % (abs(hash(winpath)) % 100000)
        b64 = base64.b64encode(data).decode()
        self.ps("$p = '%s'; New-Item -ItemType Directory -Force -Path (Split-Path $p) | Out-Null; "
                "if (Test-Path -LiteralPath '%s') { Remove-Item -LiteralPath '%s' }" % (winpath, tmp, tmp))
        for i in range(0, len(b64), chunk):
            self.ps("[IO.File]::AppendAllText('%s', '%s')" % (tmp, b64[i:i + chunk]))
        self.ps("[IO.File]::WriteAllBytes('%s', [Convert]::FromBase64String([IO.File]::ReadAllText('%s'))); "
                "Remove-Item -LiteralPath '%s'" % (winpath, tmp, tmp))

    def write_bytes(self, rel, data):
        self.put_file(self.path(rel), data)

    def patch_bytes(self, rel, offset, data):
        b64 = base64.b64encode(data).decode()
        self.ps("$fs = [IO.File]::Open('%s', 'Open', 'ReadWrite'); $fs.Seek(%d, 'Begin') | Out-Null; "
                "$b = [Convert]::FromBase64String('%s'); $fs.Write($b, 0, $b.Length); $fs.Close()"
                % (self.path(rel), offset, b64))

    def snapshot(self, hash_prefix, subtree=None):
        """rel -> (kind, size, sha) as Windows sees the volume; hashes only
        under hash_prefix; with subtree, only that directory is walked (rel
        paths stay volume-relative)."""
        root = self.drive + ":\\"
        hp = hash_prefix.replace("/", "\\")
        start = root + subtree.replace("/", "\\") if subtree else root
        out = self.ps(r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = '%s'
$hp = '%s'
$start = '%s'
$res = New-Object System.Collections.Generic.List[string]
if (-not (Test-Path -LiteralPath $start)) { return }
Get-ChildItem -LiteralPath $start -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
  $rel = $_.FullName.Substring($root.Length)
  if ($rel -like 'System Volume Information*' -or $rel -like '$RECYCLE.BIN*') { return }
  if ($_.PSIsContainer) { $res.Add("D|" + $rel + "|0|") }
  else {
    $h = ''
    if ($rel.StartsWith($hp)) { $h = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower() }
    $res.Add("F|" + $rel + "|" + $_.Length + "|" + $h)
  }
}
$res -join "`n"
""" % (root, hp, start))
        snap = {}
        for line in out.splitlines():
            parts = line.rstrip("\r").split("|", 3)
            if len(parts) != 4:
                continue
            kind, rel, size, h = parts
            rel = rel.replace("\\", "/")
            snap[rel] = ("dir" if kind == "D" else "file", int(size), h)
        return snap


# ---------------------------------------------------------------------------
# bridge + disk + agent
# ---------------------------------------------------------------------------

def start_bridge():
    roots = ",".join(sorted(d for d in os.listdir(ov.SOURCE)
                            if os.path.isdir(os.path.join(ov.SOURCE, d))))
    cmd = [sys.executable, "-m", "ntfs_bridge.bridge",
           "--source", ov.SOURCE, "--image", IMAGE, "--mount", ov.MNT,
           "--host", "127.0.0.1", "--port", str(NBD_PORT), "--partitioned", "--lazy",
           "--dealloc-timeout", "31536000", "--roots", roots,
           "--overflow-dir", ov.OVERFLOW, "--two-way",
           "--control-host", CTRL_HOST, "--control-port", str(CTRL_PORT),
           "--agent-token-file", TOKEN]
    log("  bridge: %s" % " ".join(cmd[2:]))
    fh = open(LOG, "ab")
    fh.write(b"\n==== start %s ====\n" % time.ctime().encode())
    proc = subprocess.Popen(cmd, cwd=ov.REPO, stdout=fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 1800
    import http.client
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("bridge exited early rc=%s; see %s" % (proc.returncode, LOG))
        try:
            tok = open(TOKEN).read().strip()
            c = http.client.HTTPConnection(CTRL_HOST, CTRL_PORT, timeout=3)
            c.request("GET", "/v1/health", headers={"X-Bridge-Token": tok})
            r = c.getresponse()
            if r.status == 200:
                r.read()
                log("  bridge control endpoint up")
                return proc, tok
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("bridge control endpoint never came up; see %s" % LOG)


def health(tok):
    import http.client
    c = http.client.HTTPConnection(CTRL_HOST, CTRL_PORT, timeout=5)
    c.request("GET", "/v1/health", headers={"X-Bridge-Token": tok})
    return json.loads(c.getresponse().read())


def stop_bridge(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
    log("  bridge stopped")


TARGET = {"dev": None}


def vm_targets():
    out = subprocess.run(["virsh", "domblklist", VM], capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip() and line.split()[0].startswith("vd")}


def attach_disk(win=None):
    # a ghost from an earlier unplug that Windows never released keeps its
    # target name; take the first free one rather than fight it
    used = vm_targets()
    dev = next("vd" + c for c in "abcdefghijklmnopqrstuvwxyz" if "vd" + c not in used)
    with open(DISK_XML, "w") as f:
        f.write("""<disk type='network' device='disk'>
  <driver name='qemu' type='raw' cache='none'/>
  <source protocol='nbd'><host name='127.0.0.1' port='%d'/></source>
  <target dev='%s' bus='virtio'/>
</disk>
""" % (NBD_PORT, dev))
    r = subprocess.run(["virsh", "attach-device", VM, DISK_XML, "--live"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("attach-device failed: %s" % r.stderr.strip())
    TARGET["dev"] = dev
    log("  disk attached to %s as %s (nbd 127.0.0.1:%d)" % (VM, dev, NBD_PORT))


def detach_disk(win=None):
    """virtio hot-unplug needs the guest to let go: take the disk offline in
    Windows first, then detach, then wait for the target to disappear."""
    if win is not None:
        try:
            win.ps("Get-Disk | Where-Object { $_.FriendlyName -match 'VirtIO' } | "
                   "ForEach-Object { Set-Disk -Number $_.Number -IsOffline $true -ErrorAction SilentlyContinue }",
                   check=False)
        except Exception as e:
            log("  (could not offline the disk in Windows: %s)" % e)
    dev = TARGET["dev"]
    if not dev:
        return True
    r = subprocess.run(["virsh", "detach-disk", VM, dev, "--live"],
                       capture_output=True, text=True)
    for _ in range(40):
        if dev not in vm_targets():
            log("  disk %s detached from %s" % (dev, VM))
            TARGET["dev"] = None
            return True
        time.sleep(3)
    log("  disk detach did not complete: %s" % (r.stdout.strip() or r.stderr.strip()))
    return False


def bring_disk_online(win, serial_hex):
    """Windows may leave a hot-plugged disk offline (SAN policy). Find it by
    NTFS volume serial (low 32 bits) after bringing every offline non-boot
    disk online, then return the drive letter."""
    low32 = int(serial_hex[8:16], 16)
    for _ in range(60):
        out = win.ps(r"""
Get-Disk | Where-Object { $_.Number -ne 0 -and $_.OperationalStatus -ne 'Online' } | ForEach-Object { Set-Disk -Number $_.Number -IsOffline $false -ErrorAction SilentlyContinue; Set-Disk -Number $_.Number -IsReadOnly $false -ErrorAction SilentlyContinue }
$v = Get-CimInstance Win32_Volume | Where-Object { $_.FileSystem -eq 'NTFS' -and $_.DriveLetter -and ([uint32]$_.SerialNumber -eq %d) }
if ($v) { $v.DriveLetter.TrimEnd(':') } else { '' }
""" % low32).strip()
        if out:
            return out
        time.sleep(3)
    return None


def install_agent(win, tok):
    ps1 = open(os.path.join(ov.REPO, "guest_agent", "bridge-agent.ps1"), "rb").read()
    inst = open(os.path.join(ov.REPO, "guest_agent", "install-agent.ps1"), "rb").read()
    win.put_file(r"C:\ProgramData\BridgeAgent\src\bridge-agent.ps1", ps1)
    win.put_file(r"C:\ProgramData\BridgeAgent\src\install-agent.ps1", inst)
    out = win.ps(r"""
Set-Location 'C:\ProgramData\BridgeAgent\src'
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-agent.ps1 -ControlUrl 'http://%s:%d' -Token '%s' 2>&1 | Out-String
""" % (CTRL_HOST, CTRL_PORT, tok))
    log("  agent install: " + " ".join(out.split())[:200])


def uninstall_agent(win):
    win.ps("Stop-ScheduledTask -TaskName BridgeAgent -ErrorAction SilentlyContinue; "
           "Unregister-ScheduledTask -TaskName BridgeAgent -Confirm:$false -ErrorAction SilentlyContinue; "
           "Remove-Item -Recurse -Force C:\\ProgramData\\BridgeAgent -ErrorAction SilentlyContinue", check=False)
    log("  agent removed from the VM")


def agent_log_tail(win, n=8):
    return win.ps("if (Test-Path C:\\ProgramData\\BridgeAgent\\agent.log) { Get-Content C:\\ProgramData\\BridgeAgent\\agent.log -Tail %d }" % n, check=False)


def wait_hello(log_path, since_pos, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with open(log_path, "rb") as f:
            f.seek(since_pos)
            if b"agent hello" in f.read():
                return True
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def ext4_scoped(root, subtree=None):
    out = {}
    start = os.path.join(root, subtree) if subtree else root
    if subtree and not os.path.isdir(start):
        return out
    for dp, dn, fn in os.walk(start, followlinks=True):
        rel_d = os.path.relpath(dp, root)
        if rel_d != "." and dp != start:
            out[rel_d] = ("dir", 0, "")
        for n in fn:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, root)
            try:
                if not rel.startswith(SCRATCH):
                    out[rel] = ("file", os.stat(p).st_size, "")
                    continue
                with open(p, "rb") as fh:
                    data = fh.read()
                out[rel] = ("file", len(data), sha_full(data))
            except OSError as e:
                out[rel] = ("file", -1, "ERR:%s" % e)
    return out


class Gate:
    def __init__(self, win):
        self.win = win
        self.ignore = set()

    def wait(self, label, expect=None, timeout=BUDGET, subtree=SCRATCH):
        """subtree=None compares the whole volume (slow on a cold Windows
        cache: one full walk through the bridge); the default compares the
        scratch subtree only."""
        t0 = time.time()
        last = []
        while time.time() - t0 < timeout:
            e = ext4_scoped(ov.SOURCE, subtree)
            if expect is not None:
                drift = tw.diff_snapshots(expect, e, "written", "ext4-now")
                if drift:
                    fail("%s: the bridge ALTERED ext4 after a host-side op; %d difference(s):" % (label, len(drift)))
                    for d in drift[:12]:
                        log("        " + d)
                    return None
            try:
                w = self.win.snapshot(SCRATCH, subtree)
            except Exception as err:
                last = ["volume unreadable from Windows: %s" % err]
                time.sleep(3)
                continue
            last = [d for d in tw.diff_snapshots(e, w, "ext4", "windows")
                    if not any(k in d for k in self.ignore)]
            if not last:
                lat = time.time() - t0
                LAT.append((label, lat))
                ok("%-58s mirrored in %5.1fs" % (label, lat))
                return lat
            time.sleep(3)
        fail("%s: not mirrored after %.0fs; %d difference(s):" % (label, timeout, len(last)))
        for d in last[:12]:
            log("        " + d)
        return None


def upper_entries():
    import stat as st_
    out = {}
    base = os.path.join(ov.BASE, "upper")
    for dp, dn, fn in os.walk(base):
        for d in dn:
            out[os.path.relpath(os.path.join(dp, d), base)] = "dir"
        for f in fn:
            p = os.path.join(dp, f)
            st = os.lstat(p)
            out[os.path.relpath(p, base)] = "whiteout" if st_.S_ISCHR(st.st_mode) and st.st_rdev == 0 else "file"
    return out


def s(rel):
    return os.path.join(ov.SOURCE, rel)


def smallest_real_file():
    best = None
    root = os.path.join(ov.SOURCE, ov.SHARE)
    for dp, dn, fn in os.walk(root, followlinks=True):
        if "_twoway_scratch" in dp:
            continue
        for f in fn:
            p = os.path.join(dp, f)
            try:
                sz = os.stat(p).st_size
            except OSError:
                continue
            if 0 < sz < 512 * 1024 and "." in f and not f.startswith("."):
                if best is None or sz < best[1]:
                    best = (os.path.relpath(p, ov.SOURCE), sz)
    return best


# ---------------------------------------------------------------------------

def main():
    keep_agent = "--keep-agent" in sys.argv
    if not ov.mounted(ov.MERGED):
        ov.cmd_setup()
    before = ov.stat_manifest(ov.LOWER)
    log("real library baseline: %d files (stat only)" % len(before))
    upper0 = upper_entries()
    log("upperdir before: %d entries" % len(upper0))
    import shutil
    if os.path.isdir(s(SCRATCH)):
        shutil.rmtree(s(SCRATCH))
    for p in (IMAGE, IMAGE + ".op-journal.jsonl", TOKEN):
        try:
            os.remove(p)
        except OSError:
            pass
    os.makedirs(ov.OVERFLOW, exist_ok=True)

    win = Win()
    log("  winrm: " + win.ps("whoami").strip())
    proc = None
    attached = False
    try:
        proc, tok = start_bridge()
        serial = health(tok)  # warms; serial read below from the boot sector
        with open(IMAGE, "rb") as f:
            import struct
            serial_hex = "%016X" % struct.unpack_from("<Q", f.read(512), 72)[0]
        log("  volume serial %s" % serial_hex)
        log_pos = os.path.getsize(LOG)
        attach_disk(win)
        attached = True
        win.drive = bring_disk_online(win, serial_hex)
        if not win.drive:
            raise SystemExit("the bridge volume did not appear in Windows with a drive letter")
        log("  Windows sees the volume as %s:" % win.drive)
        install_agent(win, tok)
        if wait_hello(LOG, log_pos):
            ok("agent said hello")
        else:
            fail("agent never said hello; agent.log tail:\n" + agent_log_tail(win))

        gate = Gate(win)
        e = ext4_scoped(ov.SOURCE)
        w = win.snapshot(SCRATCH)
        initial = tw.diff_snapshots(e, w, "ext4", "windows")
        for d in initial:
            gate.ignore.add(d.split(": ", 1)[1].split("  ")[0] if ": " in d else d)
        log("startup: %d ext4 paths, %d seen by Windows; %d not exposed (ignored from here on)"
            % (len(e), len(w), len(initial)))
        for d in initial[:5]:
            log("        " + d)
        gate.wait("startup> Windows view equals ext4 (whole volume)", subtree=None, timeout=1800)

        G = SCRATCH
        log("\n=== guest-side (Windows) changes inside the scratch subtree ===")
        win.write_bytes(G + "/g/a.bin", tw.pattern(100, 1))
        win.write_bytes(G + "/g/b.bin", tw.pattern(CLUSTER + 5, 2))
        win.write_bytes(G + "/g/c.txt", b"guest text")
        gate.wait("win> mkdir + 3 files")
        win.patch_bytes(G + "/g/b.bin", 10, b"GUEST-EDIT")
        gate.wait("win> edit in place")
        win.ps("Add-Content -LiteralPath '%s' -Value 'appended' -NoNewline -Encoding ascii" % win.path(G + "/g/c.txt"))
        gate.wait("win> append")
        win.ps("$fs=[IO.File]::Open('%s','Open','ReadWrite'); $fs.SetLength(1000); $fs.Close()" % win.path(G + "/g/b.bin"))
        gate.wait("win> truncate to 1000 B")
        win.ps("Rename-Item -LiteralPath '%s' -NewName 'a2.bin'" % win.path(G + "/g/a.bin"))
        win.ps("New-Item -ItemType Directory -Force -Path '%s' | Out-Null; Move-Item -LiteralPath '%s' -Destination '%s'"
               % (win.path(G + "/g2"), win.path(G + "/g/c.txt"), win.path(G + "/g2/c.txt")))
        gate.wait("win> rename + move")
        win.ps("Remove-Item -LiteralPath '%s'" % win.path(G + "/g/b.bin"))
        gate.wait("win> delete file")
        win.ps("Rename-Item -LiteralPath '%s' -NewName 'g2_renamed'" % win.path(G + "/g2"))
        gate.wait("win> rename dir with contents")
        win.ps("Remove-Item -Recurse -Force -LiteralPath '%s'" % win.path(G + "/g2_renamed"))
        gate.wait("win> delete dir tree")
        for i in range(20):
            win.write_bytes(G + "/bulk/w%02d.bin" % i, tw.pattern(2000 + i, 40 + i))
        gate.wait("win> 20 files in a burst", timeout=600)

        log("\n=== host-side changes inside the scratch subtree ===")

        def host(label, fn, timeout=BUDGET):
            fn()
            gate.wait("ext4> " + label, expect=ext4_scoped(ov.SOURCE, SCRATCH), timeout=timeout)
        host("mkdir + 2 files", lambda: [tw.write(s(G + "/h/x.bin"), tw.pattern(4096, 3)),
                                        tw.write(s(G + "/h/y.bin"), tw.pattern(2 * CLUSTER + 1, 4))])
        host("append (grow)", lambda: tw.patch(s(G + "/h/x.bin"), 4096, tw.pattern(3000, 5)))
        host("truncate", lambda: os.truncate(s(G + "/h/y.bin"), 100))
        host("rename", lambda: os.rename(s(G + "/h/x.bin"), s(G + "/h/x2.bin")))
        host("delete", lambda: os.remove(s(G + "/h/x2.bin")))
        host("rename dir", lambda: os.rename(s(G + "/h"), s(G + "/h_renamed")))
        host("delete dir tree", lambda: shutil.rmtree(s(G + "/h_renamed")))
        # the real agent spends several seconds per create (fsutil createnew,
        # Write-VolumeCache, fsutil setvaliddata, each a process): budget it
        host("30 files in a burst", lambda: [tw.write(s(G + "/hbulk/f%02d.bin" % i), tw.pattern(1000 + i, 60 + i)) for i in range(30)], timeout=900)

        log("\n=== one small real file: Windows renames it and renames it back ===")
        real = smallest_real_file()
        if real is None:
            fail("no small real file found")
        else:
            rel, sz = real
            log("  candidate: %d bytes (path withheld)" % sz)
            rdir = os.path.dirname(rel)
            win.ps("Rename-Item -LiteralPath '%s' -NewName '%s'" % (win.path(rel), os.path.basename(rel) + ".twoway-renamed"))
            gate.wait("win> rename a real file", subtree=rdir)
            win.ps("Rename-Item -LiteralPath '%s' -NewName '%s'" % (win.path(rel + ".twoway-renamed"), os.path.basename(rel)))
            gate.wait("win> rename it back", subtree=rdir)
            lower_path = os.path.join(ov.LOWER, os.path.relpath(rel, ov.SHARE))
            if not os.path.exists(s(rel)):
                fail("the renamed-back file is not at its original ext4 path")
            elif open(s(rel), "rb").read() != open(lower_path, "rb").read():
                fail("the renamed-back file differs from the real one")
            else:
                ok("renamed-back file byte-identical to the real one")

        log("\n=== consistency gate: the agent takes the disk offline and back ===")
        log_pos = os.path.getsize(LOG)
        proc.send_signal(signal.SIGUSR1)
        t0 = time.time()
        done = False
        while time.time() - t0 < 600:
            with open(LOG, "rb") as f:
                f.seek(log_pos)
                blob = f.read()
            if b"gate complete" in blob:
                done = True
                break
            if b"GATE FAILED" in blob:
                break
            time.sleep(3)
        if done:
            ok("gate cycled (%.0fs)" % (time.time() - t0))
        else:
            fail("gate did not complete; agent.log tail:\n" + agent_log_tail(win))
        win.drive = bring_disk_online(win, serial_hex) or win.drive
        gate.wait("gate> Windows view equals ext4 after the gate (whole volume)", subtree=None, timeout=1800)
        win.write_bytes(G + "/after_gate_win.bin", tw.pattern(777, 170))
        gate.wait("win> create after the gate")
        host("create after the gate", lambda: tw.write(s(G + "/after_gate_host.bin"), tw.pattern(778, 171)))

        log("\n=== bridge restart: detach, restart on the reused image, re-attach ===")
        detach_disk(win)
        attached = False
        stop_bridge(proc)
        tw.write(s(G + "/added_while_down.bin"), tw.pattern(CLUSTER + 9, 70))
        proc, tok2 = start_bridge()
        log_pos = os.path.getsize(LOG)
        attach_disk(win)
        attached = True
        win.drive = bring_disk_online(win, serial_hex) or win.drive
        if wait_hello(LOG, log_pos, timeout=600):
            ok("agent re-hello after restart")
        else:
            fail("agent did not re-hello after restart; agent.log tail:\n" + agent_log_tail(win))
        gate.wait("restart> Windows view equals ext4 after restart (whole volume)", subtree=None, timeout=1800)
        win.write_bytes(G + "/after_restart_win.bin", tw.pattern(1234, 71))
        gate.wait("win> create after restart")
        host("create after restart", lambda: tw.write(s(G + "/after_restart_host.bin"), tw.pattern(4321, 72)))
    finally:
        try:
            if not keep_agent:
                uninstall_agent(win)
        except Exception as e:
            log("  agent cleanup failed: %s" % e)
        if attached:
            detach_disk(win)      # while the bridge still serves it
        stop_bridge(proc)

    log("\n=== what reached the real disk, and what the bridge would have written ===")
    after = ov.stat_manifest(ov.LOWER)
    d = ov.diff_manifest(before, after)
    if d:
        fail("REAL LIBRARY CHANGED: %d difference(s)" % len(d))
    else:
        ok("REAL LIBRARY UNCHANGED: %d files, every stat field identical" % len(before))
    upper1 = upper_entries()
    new = {k: v for k, v in upper1.items() if k not in upper0}
    stray = []
    for k, v in sorted(new.items()):
        if k.startswith(SCRATCH) or SCRATCH.startswith(k + "/"):
            continue
        if real and (k == real[0] or k == real[0] + ".twoway-renamed" or real[0].startswith(k + "/")):
            continue
        if "System Volume Information" in k or "$RECYCLE.BIN" in k:
            stray.append((k, v))   # Windows' own folders: reported, counted as stray
            continue
        stray.append((k, v))
    log("  upperdir entries added: %d" % len(new))
    if stray:
        fail("%d upperdir entr(ies) outside the scratch subtree and the renamed file:" % len(stray))
        for k, v in stray[:10]:
            log("        %-9s %s" % (v, k))
    else:
        ok("nothing outside the scratch subtree and the renamed file was written")
    stray_ov = [os.path.join(dp, f) for dp, dn, fn in os.walk(ov.OVERFLOW) for f in fn]
    log("  overflow dir: %d file(s)%s" % (len(stray_ov), "" if not stray_ov else " (Windows' own volume-root files land here by design)"))
    if LAT:
        log("latency: %d ops, max %.1fs, mean %.1fs" % (len(LAT), max(l for _, l in LAT), sum(l for _, l in LAT) / len(LAT)))
    if FAILS:
        log("\nFAILURES (%d):" % len(FAILS))
        for f in FAILS:
            log("  - " + f.splitlines()[0])
        log("\nWINDOWS TWO-WAY VERDICT: FAIL")
        return 1
    log("\nWINDOWS TWO-WAY VERDICT: mirrored both ways with the real Windows guest; the real library untouched; ledger holds only the scratch subtree and the renamed file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
