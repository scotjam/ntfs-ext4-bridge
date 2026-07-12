"""HTTP control endpoint for the two-way sync guest agent.

Stdlib-only (http.server). Binds to the libvirt NAT gateway address so it is
reachable from the guest but never from outside the host. All requests carry
a shared token in X-Bridge-Token, compared with hmac.compare_digest.

Endpoints (JSON bodies):
  POST /v1/hello  {agent_version, hostname}
                  -> {epoch, volume_serial, poll_timeout_s, batch_max}
  POST /v1/poll   {cursor}          -> {epoch, ops: [...]}   (long-poll)
  POST /v1/ack    {epoch, results: [{seq, status, code, message}]}
  POST /v1/gate   {gate_id, phase}  -> {}
  GET  /v1/health -> journal/coordinator/gate stats
"""

import hmac
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

POLL_TIMEOUT_S = 25
BATCH_MAX = 64


def log(msg):
    print(f"[ControlServer] {msg}", flush=True)


class ControlServer:
    """Runs the agent-facing HTTP endpoint in a background thread."""

    def __init__(self, host: str, port: int, token: str, journal,
                 coordinator, mapper, gate=None):
        self.host = host
        self.port = port
        self.token = token
        self.journal = journal
        self.coordinator = coordinator
        self.mapper = mapper
        self.gate = gate  # set later by bridge (circular dependency)

        self.volume_serial = self._read_volume_serial()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _read_volume_serial(self) -> str:
        """NTFS volume serial from the boot sector (offset 72, 8 bytes)."""
        try:
            serial = struct.unpack_from('<Q', self.mapper.image[0:512], 72)[0]
            return f"{serial:016X}"
        except Exception:
            return ""

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, fmt, *args):
                pass  # route through our logger only for errors

            def _reply(self, code: int, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                supplied = self.headers.get('X-Bridge-Token', '')
                return hmac.compare_digest(supplied, server.token)

            def _body(self) -> dict:
                length = int(self.headers.get('Content-Length', 0) or 0)
                if length <= 0 or length > 1024 * 1024:
                    return {}
                try:
                    return json.loads(self.rfile.read(length))
                except (ValueError, OSError):
                    return {}

            def do_GET(self):
                if not self._authed():
                    self._reply(401, {'error': 'unauthorized'})
                    return
                if self.path == '/v1/health':
                    payload = {
                        'journal': server.journal.stats(),
                        'coordinator': server.coordinator.stats()
                        if server.coordinator else {},
                        'gate': server.gate.stats() if server.gate else {},
                    }
                    self._reply(200, payload)
                else:
                    self._reply(404, {'error': 'not found'})

            def do_POST(self):
                if not self._authed():
                    self._reply(401, {'error': 'unauthorized'})
                    return
                body = self._body()

                if self.path == '/v1/hello':
                    log(f"agent hello: {body.get('hostname')} "
                        f"v{body.get('agent_version')}")
                    self._reply(200, {
                        'epoch': server.journal.epoch,
                        'volume_serial': server.volume_serial,
                        'poll_timeout_s': POLL_TIMEOUT_S,
                        'batch_max': BATCH_MAX,
                    })

                elif self.path == '/v1/poll':
                    cursor = int(body.get('cursor', 0))
                    server.journal.wait_for_ops(cursor, POLL_TIMEOUT_S)
                    ops = server.journal.ops_after(cursor, BATCH_MAX)
                    # Strip private keys before sending to the guest
                    wire_ops = [{k: v for k, v in op.items()
                                 if not k.startswith('_')} for op in ops]
                    self._reply(200, {'epoch': server.journal.epoch,
                                      'ops': wire_ops})

                elif self.path == '/v1/ack':
                    if body.get('epoch') != server.journal.epoch:
                        self._reply(409, {'error': 'stale epoch',
                                          'epoch': server.journal.epoch})
                        return
                    results = body.get('results', [])
                    server.journal.ack(results)
                    if server.coordinator:
                        server.coordinator.on_ack(results)
                    self._reply(200, {})

                elif self.path == '/v1/gate':
                    if server.gate:
                        resp = server.gate.on_agent_phase(
                            body.get('gate_id'), body.get('phase'))
                        self._reply(200, resp or {})
                    else:
                        self._reply(400, {'error': 'no gate configured'})

                else:
                    self._reply(404, {'error': 'not found'})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="ControlServer")
        self._thread.start()
        log(f"listening on {self.host}:{self.port} "
            f"(volume serial {self.volume_serial})")

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None
