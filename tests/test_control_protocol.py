"""Protocol tests for the two-way sync control server.

Runs a real ControlServer + OpJournal on an ephemeral localhost port and
exercises it with a fake Python agent (urllib). No root, no VM.

Run: python -m pytest tests/test_control_protocol.py -v
"""

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.op_journal import OpJournal
from ntfs_bridge.control_server import ControlServer
from ntfs_bridge.sync_coordinator import SyncCoordinator

TOKEN = 'test-token-123'


class FakeMapper:
    def __init__(self):
        self.ntfs_sync_in_progress = set()
        self.ntfs_sync_timestamps = {}
        self.path_to_mft_record = {}
        self.known_root_entries = {'Share'}
        self._sync_lock = threading.Lock()
        self.ext4_sync_in_progress = set()
        self.echo_observed_callback = None
        # Boot sector stand-in: volume serial 0xDEADBEEF00112233 at offset 72
        boot = bytearray(512)
        import struct
        struct.pack_into('<Q', boot, 72, 0xDEADBEEF00112233)
        self.image = bytes(boot)


class FakeGate:
    def __init__(self):
        self.calls = []

    def on_agent_phase(self, gate_id, phase):
        self.calls.append((gate_id, phase))
        if phase == 'await_end':
            return {'done': True}
        return {}

    def stats(self):
        return {'phase': 'idle'}


@pytest.fixture
def server(tmp_path):
    source = tmp_path / 'source'
    (source / 'Share').mkdir(parents=True)
    mapper = FakeMapper()
    journal = OpJournal(str(tmp_path / 'j.jsonl'), str(source), mapper,
                        quiesce_s=0.0, max_hold_s=0.0)
    coordinator = SyncCoordinator(mapper, journal)
    srv = ControlServer('127.0.0.1', 0, TOKEN, journal, coordinator, mapper)
    gate = FakeGate()
    srv.gate = gate
    srv.start()
    port = srv._httpd.server_address[1]
    yield srv, journal, coordinator, mapper, gate, port
    coordinator.stop()
    srv.stop()


def call(port, endpoint, body=None, token=TOKEN, method='POST', timeout=10):
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{endpoint}',
        data=json.dumps(body or {}).encode() if method == 'POST' else None,
        headers={'Content-Type': 'application/json',
                 'X-Bridge-Token': token},
        method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def test_rejects_bad_token(server):
    _, _, _, _, _, port = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        call(port, '/v1/hello', {}, token='wrong')
    assert exc.value.code == 401


def test_hello_returns_epoch_and_serial(server):
    _, journal, _, _, _, port = server
    resp = call(port, '/v1/hello', {'agent_version': '1.0',
                                    'hostname': 'testvm'})
    assert resp['epoch'] == journal.epoch
    assert resp['volume_serial'] == 'DEADBEEF00112233'
    assert resp['batch_max'] > 0


def test_poll_ack_roundtrip_with_suppression(server):
    _, journal, coordinator, mapper, _, port = server
    journal.inject_op({'op': 'mkdir', 'path': 'Share\\newdir',
                       '_rel': 'Share/newdir'})

    resp = call(port, '/v1/poll', {'cursor': 0})
    assert len(resp['ops']) == 1
    op = resp['ops'][0]
    assert op['op'] == 'mkdir'
    assert '_rel' not in op, "private keys must not reach the agent"

    # Dispatch opened a suppression window
    assert 'Share/newdir' in mapper.ext4_sync_in_progress

    call(port, '/v1/ack', {'epoch': journal.epoch,
                           'results': [{'seq': op['seq'], 'status': 'ok'}]})
    assert journal.stats()['acked_seq'] == op['seq']

    # Echo observation releases suppression immediately
    mapper.echo_observed_callback('Share/newdir')
    assert 'Share/newdir' not in mapper.ext4_sync_in_progress


def test_ack_with_stale_epoch_rejected(server):
    _, journal, _, _, _, port = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        call(port, '/v1/ack', {'epoch': 'stale', 'results': []})
    assert exc.value.code == 409


def test_redelivery_until_acked(server):
    _, journal, _, _, _, port = server
    journal.inject_op({'op': 'flush_volume', 'path': '', '_rel': ''})
    first = call(port, '/v1/poll', {'cursor': 0})['ops']
    second = call(port, '/v1/poll', {'cursor': 0})['ops']
    assert [o['seq'] for o in first] == [o['seq'] for o in second], \
        "unacked ops must be redelivered"
    call(port, '/v1/ack', {'epoch': journal.epoch,
                           'results': [{'seq': first[0]['seq'],
                                        'status': 'ok'}]})
    # An empty queue long-polls for POLL_TIMEOUT_S before replying []
    third = call(port, '/v1/poll', {'cursor': first[0]['seq']},
                 timeout=35)['ops']
    assert third == []


def test_gate_phase_dispatch(server):
    _, _, _, _, gate, port = server
    call(port, '/v1/gate', {'gate_id': 'g1', 'phase': 'offline_confirmed'})
    resp = call(port, '/v1/gate', {'gate_id': 'g1', 'phase': 'await_end'})
    assert resp['done'] is True
    assert ('g1', 'offline_confirmed') in gate.calls


def test_health_endpoint(server):
    _, journal, _, _, _, port = server
    resp = call(port, '/v1/health', method='GET')
    assert resp['journal']['epoch'] == journal.epoch
    assert 'coordinator' in resp


def test_suppression_timeout_sweep(server, monkeypatch):
    _, journal, coordinator, mapper, _, port = server
    monkeypatch.setattr(coordinator, 'echo_timeout_s', 0.1)
    journal.inject_op({'op': 'rm', 'path': 'Share\\x', 'recurse': True,
                       '_rel': 'Share/x'})
    resp = call(port, '/v1/poll', {'cursor': 0})
    op = resp['ops'][0]
    assert 'Share/x' in mapper.ext4_sync_in_progress
    call(port, '/v1/ack', {'epoch': journal.epoch,
                           'results': [{'seq': op['seq'], 'status': 'ok'}]})
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if 'Share/x' not in mapper.ext4_sync_in_progress:
            break
        time.sleep(0.2)
    assert 'Share/x' not in mapper.ext4_sync_in_progress, \
        "suppression must expire after ack + timeout"
