"""Probe: host creates a small (resident) file; the guest edits it; does the
edit reach ext4?"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_two_way_live as h  # noqa: E402

h.setup(fresh=True)
bridge = h.Bridge()
guest = h.Guest()
agent = None
REL = "beta/small/note.txt"
try:
    bridge.start()
    guest.connect()
    agent = h.Agent(bridge, guest)
    agent.start()
    h.wait_mirror(guest, "startup")

    h.write(h.s(REL), b"from the host\n")
    h.host_op(guest, "host creates a 14-byte file", lambda: None)
    time.sleep(6)   # let the window close
    print("guest sees:", repr(open(h.g(REL), "rb").read()))

    with open(h.g(REL), "ab") as f:
        f.write(b"and from the guest\n")
    guest.flush()
    t0 = time.time()
    while time.time() - t0 < 60:
        cur = open(h.s(REL), "rb").read()
        if b"guest" in cur:
            print("ok: ext4 has the guest edit after %.0fs: %r" % (time.time() - t0, cur))
            break
        time.sleep(2)
    else:
        print("FAIL: ext4 still %r" % open(h.s(REL), "rb").read())

    # and the reverse once more: host same-size edit of the now-larger file
    h.host_op(guest, "host edits the file in place", lambda: h.write(h.s(REL), b"HOST\n"))
    print("guest sees after host edit:", repr(open(h.g(REL), "rb").read()))
    # and a second guest edit
    with open(h.g(REL), "ab") as f:
        f.write(b"guest again\n")
    guest.flush()
    t0 = time.time()
    while time.time() - t0 < 60:
        cur = open(h.s(REL), "rb").read()
        if b"guest again" in cur:
            print("ok: second guest edit on ext4 after %.0fs: %r" % (time.time() - t0, cur))
            break
        time.sleep(2)
    else:
        print("FAIL: second guest edit missing; ext4 %r" % open(h.s(REL), "rb").read())
finally:
    if agent:
        agent.running = False
    guest.disconnect()
    bridge.stop()
os.system("grep -n 'note.txt' %s | grep -v FileWatcher | cut -c1-150 | tail -14" % h.LOG)
