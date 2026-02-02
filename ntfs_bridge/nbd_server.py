"""NBD server for live NTFS-to-ext4 bridge.

Serves a synthesized NTFS volume over NBD protocol, with writes going back to ext4.
DEBUG VERSION - Heavy logging to trace ntfs-3g write crashes.
"""
import os
import socket
import struct
import threading
import traceback
import sys
from typing import Optional

from .template_synth import TemplateSynthesizer

# NBD protocol constants
NBD_REQUEST_MAGIC = 0x25609513
NBD_REPLY_MAGIC = 0x67446698
NBD_INIT_MAGIC = b'NBDMAGIC'
NBD_OPTS_MAGIC = 0x49484156454F5054  # IHAVEOPT
NBD_OPT_EXPORT_NAME = 1
NBD_OPT_ABORT = 2
NBD_OPT_LIST = 3
NBD_OPT_INFO = 6
NBD_OPT_GO = 7

# NBD_INFO types
NBD_INFO_EXPORT = 0
NBD_INFO_NAME = 1
NBD_INFO_DESCRIPTION = 2
NBD_INFO_BLOCK_SIZE = 3

NBD_FLAG_HAS_FLAGS = (1 << 0)
NBD_FLAG_READ_ONLY = (1 << 1)
NBD_FLAG_SEND_FLUSH = (1 << 2)
NBD_FLAG_SEND_FUA = (1 << 3)
NBD_FLAG_SEND_TRIM = (1 << 5)

NBD_CMD_READ = 0
NBD_CMD_WRITE = 1
NBD_CMD_DISC = 2
NBD_CMD_FLUSH = 3
NBD_CMD_TRIM = 4

NBD_REP_ACK = 1
NBD_REP_SERVER = 2
NBD_REP_INFO = 3
NBD_REP_ERR_UNSUP = (1 << 31) + 1

def log(msg):
    """Thread-safe logging with flush."""
    thread_id = threading.current_thread().name
    print(f"[{thread_id}] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


class NBDServer:
    """NBD server serving synthesized NTFS with ext4 backend."""

    def __init__(self, template_path: str, source_dir: str,
                 host: str = '0.0.0.0', port: int = 10809,
                 read_only: bool = False):
        self.host = host
        self.port = port
        self.read_only = read_only
        self.running = False
        self.server_socket: Optional[socket.socket] = None

        # Initialize synthesizer
        self.synth = TemplateSynthesizer(template_path, source_dir)
        self.size = self.synth.get_size()

        # Write overlay for unmapped clusters
        self.write_overlay: dict = {}
        self.overlay_lock = threading.Lock()

        log(f"NBD Server initialized:")
        log(f"  Volume size: {self.size} bytes ({self.size / 1024 / 1024:.1f} MB)")
        log(f"  Read-only: {self.read_only}")

    def start(self):
        """Start the NBD server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        log(f"NBD server listening on {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                log(f"Connection from {addr}")
                thread = threading.Thread(
                    target=self._handle_client_wrapper,
                    args=(client_socket, addr),
                    name=f"Client-{addr[1]}"
                )
                thread.daemon = True
                thread.start()
            except socket.error as e:
                if self.running:
                    log(f"Socket error in accept: {e}")
                    raise
                break

    def stop(self):
        """Stop the NBD server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _handle_client_wrapper(self, sock: socket.socket, addr):
        """Wrapper to catch ALL exceptions from client handler."""
        try:
            self._handle_client(sock, addr)
        except Exception as e:
            log(f"FATAL: Unhandled exception in client handler: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            log(f"Client handler thread exiting for {addr}")

    def _handle_client(self, sock: socket.socket, addr):
        """Handle a client connection."""
        try:
            # Set socket timeout to detect hung connections
            sock.settimeout(300)  # 5 minute timeout
            log(f"Socket timeout set to 300s")

            # NBD handshake
            log("Starting handshake...")
            self._do_handshake(sock)
            log("Handshake complete, entering command loop")

            # Main command loop
            request_count = 0
            while True:
                request_count += 1
                log(f"--- Request #{request_count} ---")
                try:
                    if not self._handle_request(sock):
                        log("Request handler returned False, closing")
                        break
                except socket.timeout:
                    log("Socket timeout waiting for request")
                    break
                except Exception as e:
                    log(f"Exception in request handler: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    break

        except socket.timeout:
            log("Socket timeout during handshake")
        except ConnectionResetError as e:
            log(f"Connection reset: {e}")
        except BrokenPipeError as e:
            log(f"Broken pipe: {e}")
        except Exception as e:
            log(f"Client error: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            try:
                sock.close()
            except:
                pass
            log("Client disconnected")

    def _do_handshake(self, sock: socket.socket):
        """Perform NBD newstyle handshake."""
        log("Sending NBDMAGIC...")
        sock.sendall(NBD_INIT_MAGIC)

        log("Sending IHAVEOPT...")
        sock.sendall(struct.pack('>Q', NBD_OPTS_MAGIC))

        handshake_flags = NBD_FLAG_HAS_FLAGS
        log(f"Sending handshake flags: {handshake_flags}")
        sock.sendall(struct.pack('>H', handshake_flags))

        log("Waiting for client flags...")
        client_flags_data = sock.recv(4)
        if len(client_flags_data) < 4:
            raise ValueError(f"Short client flags: {len(client_flags_data)} bytes")
        client_flags = struct.unpack('>I', client_flags_data)[0]
        log(f"Client flags: {client_flags}")

        # Option haggling
        while True:
            log("Waiting for option...")
            opt_magic_data = sock.recv(8)
            if len(opt_magic_data) < 8:
                raise ValueError(f"Short option magic: {len(opt_magic_data)} bytes")
            opt_magic = struct.unpack('>Q', opt_magic_data)[0]

            if opt_magic != NBD_OPTS_MAGIC:
                raise ValueError(f"Bad option magic: {opt_magic:x}")

            opt_type_data = sock.recv(4)
            opt_len_data = sock.recv(4)
            opt_type = struct.unpack('>I', opt_type_data)[0]
            opt_len = struct.unpack('>I', opt_len_data)[0]

            opt_data = b''
            if opt_len > 0:
                opt_data = self._recv_exact(sock, opt_len)

            log(f"Option: type={opt_type} len={opt_len}")

            if opt_type == NBD_OPT_EXPORT_NAME:
                export_flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH
                if self.read_only:
                    export_flags |= NBD_FLAG_READ_ONLY

                log(f"OPT_EXPORT_NAME: sending size={self.size}, flags={export_flags}")
                sock.sendall(struct.pack('>Q', self.size))
                sock.sendall(struct.pack('>H', export_flags))
                sock.sendall(b'\x00' * 124)
                log("OPT_EXPORT_NAME complete")
                return

            elif opt_type == NBD_OPT_GO or opt_type == NBD_OPT_INFO:
                # Parse OPT_GO/OPT_INFO data properly:
                # 4 bytes: export name length
                # N bytes: export name
                # 2 bytes: number of info requests
                # 2*N bytes: info request types
                export_name = ''
                requested_info = []

                if len(opt_data) >= 4:
                    name_len = struct.unpack('>I', opt_data[:4])[0]
                    if len(opt_data) >= 4 + name_len:
                        export_name = opt_data[4:4+name_len].decode('utf-8', errors='replace')
                        remaining = opt_data[4+name_len:]
                        if len(remaining) >= 2:
                            nrinfos = struct.unpack('>H', remaining[:2])[0]
                            for i in range(nrinfos):
                                if len(remaining) >= 4 + i*2:
                                    info_type = struct.unpack('>H', remaining[2+i*2:4+i*2])[0]
                                    requested_info.append(info_type)

                opt_name = "OPT_GO" if opt_type == NBD_OPT_GO else "OPT_INFO"
                log(f"{opt_name}: export_name='{export_name}', requested_info={requested_info}")

                export_flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH
                if self.read_only:
                    export_flags |= NBD_FLAG_READ_ONLY

                # Always send NBD_INFO_EXPORT (required)
                info_data = struct.pack('>HQH', NBD_INFO_EXPORT, self.size, export_flags)
                log(f"{opt_name}: sending NBD_INFO_EXPORT (size={self.size}, flags={export_flags})")
                self._send_option_reply(sock, opt_type, NBD_REP_INFO, info_data)

                # Send NBD_INFO_BLOCK_SIZE if requested or always (some clients expect it)
                # Format: type(2) + min_block(4) + preferred_block(4) + max_payload(4) = 14 bytes
                if NBD_INFO_BLOCK_SIZE in requested_info or True:  # Always send it
                    min_block = 1
                    preferred_block = self.synth.cluster_size  # 4096
                    max_payload = 32 * 1024 * 1024  # 32MB max payload
                    block_info = struct.pack('>HIII', NBD_INFO_BLOCK_SIZE, min_block, preferred_block, max_payload)
                    log(f"{opt_name}: sending NBD_INFO_BLOCK_SIZE (min={min_block}, pref={preferred_block}, max={max_payload})")
                    self._send_option_reply(sock, opt_type, NBD_REP_INFO, block_info)

                # Send NBD_INFO_NAME if requested
                if NBD_INFO_NAME in requested_info:
                    name_bytes = export_name.encode('utf-8') if export_name else b''
                    name_info = struct.pack('>H', NBD_INFO_NAME) + name_bytes
                    log(f"{opt_name}: sending NBD_INFO_NAME")
                    self._send_option_reply(sock, opt_type, NBD_REP_INFO, name_info)

                log(f"{opt_name}: sending NBD_REP_ACK")
                self._send_option_reply(sock, opt_type, NBD_REP_ACK, b'')
                log(f"{opt_name} complete")

                # Only enter transmission phase for OPT_GO, not OPT_INFO
                if opt_type == NBD_OPT_GO:
                    return
                # For OPT_INFO, continue option haggling

            elif opt_type == NBD_OPT_ABORT:
                self._send_option_reply(sock, opt_type, NBD_REP_ACK, b'')
                raise ConnectionAbortedError("Client aborted")

            elif opt_type == NBD_OPT_LIST:
                export_name = b''
                reply_data = struct.pack('>I', len(export_name)) + export_name
                self._send_option_reply(sock, opt_type, NBD_REP_SERVER, reply_data)
                self._send_option_reply(sock, opt_type, NBD_REP_ACK, b'')

            else:
                log(f"Unknown option {opt_type}, sending ERR_UNSUP")
                self._send_option_reply(sock, opt_type, NBD_REP_ERR_UNSUP, b'')

    def _recv_exact(self, sock: socket.socket, length: int) -> bytes:
        """Receive exactly `length` bytes from socket."""
        data = b''
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise ConnectionError(f"Socket closed, got {len(data)}/{length} bytes")
            data += chunk
        return data

    def _send_option_reply(self, sock: socket.socket, opt_type: int,
                          reply_type: int, data: bytes):
        """Send an option reply."""
        sock.sendall(struct.pack('>Q', 0x3e889045565a9))  # NBD_REP_MAGIC
        sock.sendall(struct.pack('>I', opt_type))
        sock.sendall(struct.pack('>I', reply_type))
        sock.sendall(struct.pack('>I', len(data)))
        if data:
            sock.sendall(data)

    def _handle_request(self, sock: socket.socket) -> bool:
        """Handle a single NBD request. Returns False if connection should close."""
        # Read request header
        log("Waiting for 28-byte request header...")
        try:
            header = self._recv_exact(sock, 28)
        except ConnectionError as e:
            log(f"Failed to read header: {e}")
            return False
        except socket.timeout:
            log("Timeout reading header")
            return False

        log(f"Got header: {header.hex()}")

        magic, flags, cmd, handle, offset, length = struct.unpack('>IHHQQI', header)
        log(f"Parsed: magic={magic:x} flags={flags} cmd={cmd} handle={handle} offset={offset} len={length}")

        if magic != NBD_REQUEST_MAGIC:
            log(f"BAD MAGIC: expected {NBD_REQUEST_MAGIC:x}, got {magic:x}")
            return False

        cmd_names = {0: 'READ', 1: 'WRITE', 2: 'DISC', 3: 'FLUSH', 4: 'TRIM'}
        log(f"REQ: {cmd_names.get(cmd, f'UNKNOWN({cmd})')} offset={offset} len={length}")

        error = 0
        data = b''

        try:
            if cmd == NBD_CMD_READ:
                log(f"Processing READ...")
                data = self._do_read(offset, length)
                log(f"READ complete, {len(data)} bytes")

            elif cmd == NBD_CMD_WRITE:
                log(f"Processing WRITE, need to read {length} bytes of data...")
                try:
                    write_data = self._recv_exact(sock, length)
                    log(f"Got {len(write_data)} bytes of write data")
                except Exception as e:
                    log(f"FAILED to read write data: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    return False

                if self.read_only:
                    log("Read-only mode, returning EPERM")
                    error = 1  # EPERM
                else:
                    log("Calling _do_write...")
                    try:
                        self._do_write(offset, write_data)
                        log("_do_write complete")
                    except Exception as e:
                        log(f"_do_write FAILED: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        error = 5  # EIO

            elif cmd == NBD_CMD_DISC:
                log("DISCONNECT requested")
                return False

            elif cmd == NBD_CMD_FLUSH:
                log("Processing FLUSH...")
                self._do_flush()
                log("FLUSH complete")

            elif cmd == NBD_CMD_TRIM:
                log("Processing TRIM (ignored)")
                pass

            else:
                log(f"Unknown command {cmd}, returning EINVAL")
                error = 22  # EINVAL

        except Exception as e:
            log(f"Exception during command processing: {type(e).__name__}: {e}")
            traceback.print_exc()
            error = 5  # EIO

        # Send reply
        log(f"Sending reply: error={error}, handle={handle}")
        try:
            reply = struct.pack('>IIQ', NBD_REPLY_MAGIC, error, handle)
            sock.sendall(reply)
            log(f"Reply header sent")

            if data:
                log(f"Sending {len(data)} bytes of data...")
                sock.sendall(data)
                log(f"Data sent")

            log("Request complete")

        except Exception as e:
            log(f"FAILED to send reply: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

        return True

    def _do_read(self, offset: int, length: int) -> bytes:
        """Handle a read request."""
        result = bytearray(length)
        pos = 0

        while pos < length:
            byte_offset = offset + pos
            remaining = length - pos

            # Check overlay first
            with self.overlay_lock:
                overlay_data = self._get_overlay(byte_offset, remaining)

            if overlay_data:
                result[pos:pos+len(overlay_data)] = overlay_data
                pos += len(overlay_data)
            else:
                cluster_size = self.synth.cluster_size
                cluster_offset = byte_offset % cluster_size
                chunk_len = min(remaining, cluster_size - cluster_offset)

                synth_data = self.synth.read(byte_offset, chunk_len)
                result[pos:pos+len(synth_data)] = synth_data
                pos += chunk_len

        return bytes(result)

    def _get_overlay(self, offset: int, max_len: int) -> Optional[bytes]:
        """Get data from write overlay if present."""
        for ovl_offset, ovl_data in self.write_overlay.items():
            if ovl_offset <= offset < ovl_offset + len(ovl_data):
                rel_offset = offset - ovl_offset
                available = len(ovl_data) - rel_offset
                return ovl_data[rel_offset:rel_offset + min(max_len, available)]
        return None

    def _do_write(self, offset: int, data: bytes):
        """Handle a write request."""
        cluster_size = self.synth.cluster_size
        log(f"_do_write: offset={offset}, len={len(data)}, cluster_size={cluster_size}")

        # Check if this write affects the MFT - if so, update cluster mappings
        if self.synth.is_mft_region(offset, len(data)):
            log(f"  MFT region affected - updating cluster mappings")
            old_cluster_map = dict(self.synth.cluster_map)  # Copy before update
            self.synth.handle_mft_write(offset, data)

            # Check for newly mapped clusters that have data in overlay
            # Copy overlay data to source files for these clusters
            self._sync_overlay_to_sources(old_cluster_map)

        pos = 0
        while pos < len(data):
            byte_offset = offset + pos
            cluster = byte_offset // cluster_size
            cluster_offset = byte_offset % cluster_size

            remaining = len(data) - pos
            chunk_len = min(remaining, cluster_size - cluster_offset)
            chunk_data = data[pos:pos + chunk_len]

            log(f"  Processing chunk: cluster={cluster}, offset_in_cluster={cluster_offset}, len={chunk_len}")

            # Re-check cluster_map (may have been updated by MFT tracking)
            if cluster in self.synth.cluster_map:
                source_path, file_offset = self.synth.cluster_map[cluster]
                write_offset = file_offset + cluster_offset

                log(f"  MAPPED: {source_path} @ {write_offset}")
                try:
                    with open(source_path, 'r+b') as f:
                        f.seek(write_offset)
                        f.write(chunk_data)
                    log(f"  -> Write successful")
                except OSError as e:
                    log(f"  -> Write error: {e}, using overlay")
                    with self.overlay_lock:
                        self.write_overlay[byte_offset] = chunk_data
            else:
                log(f"  UNMAPPED: storing in overlay")
                with self.overlay_lock:
                    self.write_overlay[byte_offset] = chunk_data

            pos += chunk_len

        log(f"_do_write complete, processed {pos} bytes")

    def _sync_overlay_to_sources(self, old_cluster_map: dict):
        """Sync overlay data to source files for newly mapped clusters."""
        cluster_size = self.synth.cluster_size

        log(f"  Checking overlay sync: {len(self.write_overlay)} overlay entries")
        log(f"  Old cluster_map had {len(old_cluster_map)} entries, new has {len(self.synth.cluster_map)}")

        with self.overlay_lock:
            # Find overlay entries that now map to source files
            synced_offsets = []

            for ovl_offset, ovl_data in self.write_overlay.items():
                cluster = ovl_offset // cluster_size
                cluster_offset = ovl_offset % cluster_size

                log(f"  Overlay entry at offset {ovl_offset} (cluster {cluster})")
                log(f"    In new cluster_map: {cluster in self.synth.cluster_map}")
                log(f"    In old cluster_map: {cluster in old_cluster_map}")

                # Check if this cluster is NOW mapped (but wasn't before)
                if cluster in self.synth.cluster_map and cluster not in old_cluster_map:
                    source_path, file_offset = self.synth.cluster_map[cluster]
                    write_offset = file_offset + cluster_offset

                    log(f"  Syncing overlay cluster {cluster} to {os.path.basename(source_path)} @ {write_offset}")
                    try:
                        with open(source_path, 'r+b') as f:
                            f.seek(write_offset)
                            f.write(ovl_data)
                        log(f"  -> Sync successful")
                        synced_offsets.append(ovl_offset)
                    except OSError as e:
                        log(f"  -> Sync error: {e}")

            # Remove synced entries from overlay
            for off in synced_offsets:
                del self.write_overlay[off]

            if synced_offsets:
                log(f"  Synced {len(synced_offsets)} overlay entries to source files")

    def _do_flush(self):
        """Flush pending writes."""
        log("_do_flush called")
        pass


def main():
    """Run the NBD server."""
    import argparse

    parser = argparse.ArgumentParser(description='NTFS-ext4 bridge NBD server (DEBUG)')
    parser.add_argument('template', help='Path to NTFS template image')
    parser.add_argument('source', help='Path to ext4 source directory')
    parser.add_argument('-p', '--port', type=int, default=10809, help='Port to listen on')
    parser.add_argument('-H', '--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('-r', '--read-only', action='store_true', help='Read-only mode')

    args = parser.parse_args()

    log("Starting NBD server (DEBUG VERSION)...")

    server = NBDServer(
        template_path=args.template,
        source_dir=args.source,
        host=args.host,
        port=args.port,
        read_only=args.read_only
    )

    try:
        server.start()
    except KeyboardInterrupt:
        log("Shutting down...")
        server.stop()


if __name__ == '__main__':
    main()
