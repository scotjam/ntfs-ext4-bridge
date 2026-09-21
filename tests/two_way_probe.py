"""Probe: 50 host-side creates in one dir; where do records f30+ go?"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_two_way_live as h  # noqa: E402

h.setup(fresh=True)
bridge = h.Bridge()
guest = h.Guest()
agent = None
try:
    bridge.start()
    guest.connect()
    agent = h.Agent(bridge, guest)
    agent.start()
    h.wait_mirror(guest, "startup")

    def mft_info(tag):
        v = guest.view()
        try:
            nrec = v.mft_size // v.rs
            print("[%s] $MFT data_size=%d records=%d runs=%s" % (tag, v.mft_size, nrec, v.mft_runs))
        finally:
            v.close()

    mft_info("before")
    for i in range(50):
        h.write(h.s("beta/host/bulk/f%02d.bin" % i), h.pattern(1000 + i, 30 + i))
    t0 = time.time()
    while time.time() - t0 < 90:
        time.sleep(5)
        try:
            n_guest = len(os.listdir(h.g("beta/host/bulk")))
        except OSError as e:
            n_guest = "ERR %s" % e
        v = guest.view()
        try:
            snap = v.snapshot()
        finally:
            v.close()
        n_dev = sum(1 for k in snap if k.startswith("beta/host/bulk/"))
        print("t+%2.0fs guest-mount sees %s, device view sees %d, agent executed %d ops" % (
            time.time() - t0, n_guest, n_dev, len(agent.executed)))
        if n_dev == 50:
            break
    mft_info("after")
    guest.flush()
    time.sleep(3)
    mft_info("after flush")
    v = guest.view()
    try:
        snap = v.snapshot()
        missing = sorted(k for k in ["beta/host/bulk/f%02d.bin" % i for i in range(50)] if k not in snap)
        print("missing from device view:", missing[:5], "... total", len(missing))
    finally:
        v.close()
    os.system("grep -n 'MFT has\\|MFT write: records 4[3-9][0-9]\\|MFT write: records [5-9][0-9][0-9]\\|MFT write: records 1[0-9][0-9][0-9]' %s | tail -8" % h.LOG)
    os.system("ls %s | head -3; ls %s | wc -l" % (h.g("beta/host/bulk"), h.g("beta/host/bulk")))
    print("agent errors:", agent.errors[:3])
finally:
    if agent:
        agent.running = False
    guest.disconnect()
    bridge.stop()
