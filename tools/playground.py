"""Two-way playground: try the bridge by hand, with the real library kept
out of reach.

    sudo python3 tools/playground.py start     # ~30 min first time (populate)
    sudo python3 tools/playground.py status
    sudo python3 tools/playground.py stop

What it sets up:
  * the library as the READ-ONLY lower layer of an overlayfs (the kernel
    cannot write to it, whatever happens); the merged view is your ext4-side
    workbench, the upperdir is the ledger of everything written
  * the bridge in --two-way on that merged view (NBD 127.0.0.1:10820, agent
    endpoint 192.168.122.1:10821), logging to the overlay dir
  * the Windows VM booted with its production bridge disk detached, this
    bridge's volume hot-plugged as a virtio disk, the agent installed as the
    BridgeAgent scheduled task (over WinRM, creds in /root/.bridge-winrm.json)

Then:
  * in Windows: the drive letter printed below - add, edit, rename, delete
  * on ext4: the merged path printed below - same
  * watch: `tail -f <overlay>/bridge.log`, and `find <upperdir>` for what
    would have been written to the real disk

`stop` unplugs the disk, stops the bridge, removes the agent task, unmounts
the overlay (upperdir kept), shuts the VM down and restores its production
disk definition. The production ntfs-bridge service is never touched.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tests"))
import test_real_library_overlay as ov   # noqa: E402
import test_windows_two_way as w         # noqa: E402

VM = w.VM
PIDFILE = os.path.join(ov.BASE, "playground.pid")
XML_BACKUP = "/root/windows-vm.xml.playground-backup"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def vm_state():
    return sh(["virsh", "domstate", VM]).stdout.strip()


def ensure_vm_running():
    if vm_state() == "running":
        return
    xml = sh(["virsh", "dumpxml", "--inactive", VM]).stdout
    open(XML_BACKUP, "w").write(xml)
    if "protocol='nbd'" in xml:
        # the production bridge disk cannot be connected (its bridge is not
        # running); boot without it, restore on stop
        r = sh(["virsh", "detach-disk", VM, "vda", "--config"])
        print("  production bridge disk detached from the VM config:", (r.stdout or r.stderr).strip())
    sh(["virsh", "start", VM])
    print("  VM starting; waiting for WinRM...")
    time.sleep(45)
    win = None
    for _ in range(60):
        try:
            win = w.Win()
            win.ps("whoami")
            break
        except Exception:
            time.sleep(10)
    if win is None:
        raise SystemExit("WinRM never answered; check the VM")
    print("  VM up")


def cmd_start():
    if os.path.exists(PIDFILE):
        raise SystemExit("already started (%s exists); run stop first" % PIDFILE)
    if not ov.mounted(ov.MERGED):
        ov.cmd_setup()
    for p in (w.IMAGE + ".op-journal.jsonl", w.TOKEN):
        try:
            os.remove(p)
        except OSError:
            pass
    os.makedirs(ov.OVERFLOW, exist_ok=True)
    ensure_vm_running()
    win = w.Win()
    proc, tok = w.start_bridge()
    open(PIDFILE, "w").write(str(proc.pid))
    import struct
    with open(w.IMAGE, "rb") as f:
        serial_hex = "%016X" % struct.unpack_from("<Q", f.read(512), 72)[0]
    log_pos = os.path.getsize(w.LOG)
    w.attach_disk(win)
    win.drive = w.bring_disk_online(win, serial_hex)
    if not win.drive:
        raise SystemExit("the volume did not appear in Windows")
    w.install_agent(win, tok)
    hello = w.wait_hello(w.LOG, log_pos)
    print("""
================ playground is up ================
Windows:   drive %s:  (share folder %s)
ext4:      %s
           (this is the overlay's merged view of the real library; changes
            you make here are what the guest sees; nothing reaches the real
            disk)
ledger:    %s
           (everything written by either side, incl. whiteouts for deletes)
bridge:    log %s
           health: curl -s -H "X-Bridge-Token: $(cat %s)" http://%s:%d/v1/health
agent:     %s   (Windows: C:\\ProgramData\\BridgeAgent\\agent.log)

Notes: a host-side create/resize opens a ~2 s window in which guest writes to
that same file are dropped; cross-share moves on ext4 show up as delete +
create; the agent applies one op every few seconds.
==================================================
""" % (win.drive, ov.SHARE, os.path.join(ov.SOURCE, ov.SHARE), ov.UPPER, w.LOG,
       w.TOKEN, w.CTRL_HOST, w.CTRL_PORT,
       "hello received" if hello else "NO HELLO YET - check agent.log"))
    # detach from the child so the bridge keeps running after this script exits
    proc_pid = proc.pid
    print("bridge pid %d (kept running); stop with: sudo python3 tools/playground.py stop" % proc_pid)
    os._exit(0)


def cmd_status():
    print("VM:", vm_state())
    print("overlay mounted:", ov.mounted(ov.MERGED))
    pid = open(PIDFILE).read().strip() if os.path.exists(PIDFILE) else None
    alive = pid and os.path.exists("/proc/%s" % pid)
    print("bridge:", "running (pid %s)" % pid if alive else "not running")
    print("VM disks:", [l.split()[0] for l in sh(["virsh", "domblklist", VM]).stdout.splitlines() if l.startswith(" vd")])
    n = sum(len(f) for _, _, f in os.walk(os.path.join(ov.BASE, "upper")))
    print("ledger entries:", n)


def cmd_stop():
    win = None
    try:
        win = w.Win()
        w.uninstall_agent(win)
    except Exception as e:
        print("  (agent cleanup skipped: %s)" % e)
    # unplug every playground target
    for l in sh(["virsh", "domblklist", VM]).stdout.splitlines():
        if l.startswith(" vd"):
            w.TARGET["dev"] = l.split()[0]
            w.detach_disk(win)
    if os.path.exists(PIDFILE):
        pid = int(open(PIDFILE).read().strip())
        try:
            os.kill(pid, 15)
            for _ in range(60):
                if not os.path.exists("/proc/%d" % pid):
                    break
                time.sleep(2)
        except OSError:
            pass
        os.remove(PIDFILE)
        print("  bridge stopped")
    ov.cmd_teardown()
    sh(["virsh", "shutdown", VM])
    for _ in range(48):
        if vm_state() == "shut off":
            break
        time.sleep(5)
    print("  VM:", vm_state())
    if os.path.exists(XML_BACKUP):
        xml = open(XML_BACKUP).read()
        m = re.search(r"<disk type=.network.[\s\S]*?</disk>", xml)
        if m and "protocol='nbd'" not in sh(["virsh", "dumpxml", "--inactive", VM]).stdout:
            tmp = "/root/vda-restore.xml"
            open(tmp, "w").write(m.group(0))
            r = sh(["virsh", "attach-device", VM, tmp, "--config"])
            print("  production bridge disk restored in the VM config:", (r.stdout or r.stderr).strip())
            os.remove(tmp)
        os.remove(XML_BACKUP)
    print("stopped; ledger kept at", ov.UPPER)


if __name__ == "__main__":
    if os.geteuid() != 0:
        raise SystemExit("run as root")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[cmd]()
