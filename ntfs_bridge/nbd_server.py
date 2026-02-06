"""NBD server for NTFS-ext4 bridge.

Serves a synthesized NTFS volume over NBD protocol.
Reads route through ClusterMapper (data from ext4, metadata from image).
Writes route through ClusterMapper (data to ext4, metadata to image).
"""
import os
import socket
import struct
import threading
import traceback
import sys
from typing import Optional, Union, Protocol

# Protocol for NBD backend (ClusterMapper or PartitionWrapper)
class NBDBackend(Protocol):
    cluster_size: int
    def get_size(self) -> int: ...
    def read(self, offset: int, length: int) -> bytes: ...
    def write(self, offset: int, data: bytes) -> None: ...
    def flush(self) -> None: ...

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

NBD_INFO_EXPORT = 0
NBD_INFO_NAME = 1
NBD_INFO_DESCRIPTION = 2
NBD_INFO_BLOCK_SIZE = 3

# Handshake flags (server to client during initial handshake)
NBD_FLAG_FIXED_NEWSTYLE = (1 << 0)  # Use fixed newstyle protocol
NBD_FLAG_NO_ZEROES = (1 << 1)       # Client doesn't need to send 124 zeros

# Transmission flags (in export info)
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
    print(f"[NBD] {msg}", flush=True)


class NBDServer:
    """NBD server backed by ClusterMapper or PartitionWrapper."""

    def __init__(self, mapper: NBDBackend,
                 host: str = '0.0.0.0', port: int = 10809,
                 read_only: bool = False):
        self.host = host
        self.port = port
        self.read_only = read_only
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.mapper = mapper
        self.size = mapper.get_size()

        log(f"Initialized: {self.size} bytes ({self.size / 1024 / 1024:.1f} MB), "
            f"read_only={self.read_only}")

    def start(self):
        """Start the NBD server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        log(f"Listening on {self.host}:{self.port}")

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
                    log(f"Socket error: {e}")
                    raise
                break

    def stop(self):
        """Stop the NBD server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _handle_client_wrapper(self, sock: socket.socket, addr):
        try:
            self._handle_client(sock, addr)
        except Exception as e:
            log(f"Client error: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            log(f"Client disconnected: {addr}")

    def _handle_client(self, sock: socket.socket, addr):
        try:
            sock.settimeout(300)
            self._do_handshake(sock)

            while True:
                try:
                    if not self._handle_request(sock):
                        break
                except socket.timeout:
                    log("Socket timeout")
                    break
                except Exception as e:
                    log(f"Request error: {type(e).__name__}: {e}")
                    break
        except socket.timeout:
            log("Handshake timeout")
        except (ConnectionResetError, BrokenPipeError) as e:
            log(f"Connection lost: {e}")
        except Exception as e:
            log(f"Client error: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            try:
                sock.close()
            except:
                pass

    def _do_handshake(self, sock: socket.socket):
        """Perform NBD newstyle handshake."""
        sock.sendall(NBD_INIT_MAGIC)
        sock.sendall(struct.pack('>Q', NBD_OPTS_MAGIC))

        # Send handshake flags: fixed newstyle + no zeros required
        handshake_flags = NBD_FLAG_FIXED_NEWSTYLE | NBD_FLAG_NO_ZEROES
        sock.sendall(struct.pack('>H', handshake_flags))

        client_flags_data = sock.recv(4)
        if len(client_flags_data) < 4:
            raise ValueError(f"Short client flags: {len(client_flags_data)} bytes")
        client_flags = struct.unpack('>I', client_flags_data)[0]

        # Option haggling
        while True:
            opt_magic_data = sock.recv(8)
            if len(opt_magic_data) < 8:
                raise ValueError(f"Short option magic")
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

            if opt_type == NBD_OPT_EXPORT_NAME:
                export_flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH
                if self.read_only:
                    export_flags |= NBD_FLAG_READ_ONLY

                sock.sendall(struct.pack('>Q', self.size))
                sock.sendall(struct.pack('>H', export_flags))
                sock.sendall(b'\x00' * 124)
                return

            elif opt_type == NBD_OPT_GO or opt_type == NBD_OPT_INFO:
                export_name = ''
                requested_info = []

                if len(opt_data) >= 4:
                    name_len = struct.unpack('>I', opt_data[:4])[0]
                    if len(opt_data) >= 4 + name_len:
                        export_name = opt_data[4:4 + name_len].decode('utf-8', errors='replace')
                        remaining = opt_data[4 + name_len:]
                        if len(remaining) >= 2:
                            nrinfos = struct.unpack('>H', remaining[:2])[0]
                            for i in range(nrinfos):
                                if len(remaining) >= 4 + i * 2:
                                    info_type = struct.unpack('>H', remaining[2 + i * 2:4 + i * 2])[0]
                                    requested_info.append(info_type)

                export_flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH
                if self.read_only:
                    export_flags |= NBD_FLAG_READ_ONLY

                # Send NBD_INFO_EXPORT
                info_data = struct.pack('>HQH', NBD_INFO_EXPORT, self.size, export_flags)
                self._send_option_reply(sock, opt_type, NBD_REP_INFO, info_data)

                # Send NBD_INFO_BLOCK_SIZE
                min_block = 1
                preferred_block = self.mapper.cluster_size
                max_payload = 32 * 1024 * 1024
                block_info = struct.pack('>HIII', NBD_INFO_BLOCK_SIZE,
                                         min_block, preferred_block, max_payload)
                self._send_option_reply(sock, opt_type, NBD_REP_INFO, block_info)

                # Send NBD_INFO_NAME if requested
                if NBD_INFO_NAME in requested_info:
                    name_bytes = export_name.encode('utf-8') if export_name else b''
                    name_info = struct.pack('>H', NBD_INFO_NAME) + name_bytes
                    self._send_option_reply(sock, opt_type, NBD_REP_INFO, name_info)

                self._send_option_reply(sock, opt_type, NBD_REP_ACK, b'')

                if opt_type == NBD_OPT_GO:
                    return

            elif opt_type == NBD_OPT_ABORT:
                self._send_option_reply(sock, opt_type, NBD_REP_ACK, b'')
                raise ConnectionAbortedError("Client aborted")

            elif opt_type == NBD_OPT_LIST:
                export_name = b''
                reply_data = struct.pack('>I', len(export_name)) + export_name
                self._send_option_reply(sock, opt_type, NBD_REP_SERVER, reply_data)
                self._send_option_reply(sock, opt_type, NBD_REP_ACK, b'')

            else:
                self._send_option_reply(sock, opt_type, NBD_REP_ERR_UNSUP, b'')

    def _recv_exact(self, sock: socket.socket, length: int) -> bytes:
        """Receive exactly `length` bytes."""
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
        """Handle a single NBD request."""
        try:
            header = self._recv_exact(sock, 28)
        except (ConnectionError, socket.timeout):
            return False

        magic, flags, cmd, handle, offset, length = struct.unpack('>IHHQQI', header)

        if magic != NBD_REQUEST_MAGIC:
            log(f"Bad magic: {magic:x}")
            return False

        error = 0
        data = b''

        try:
            if cmd == NBD_CMD_READ:
                data = self.mapper.read(offset, length)

            elif cmd == NBD_CMD_WRITE:
                try:
                    write_data = self._recv_exact(sock, length)
                except Exception:
                    return False

                if self.read_only:
                    error = 1  # EPERM
                else:
                    try:
                        self.mapper.write(offset, write_data)
                    except Exception as e:
                        log(f"Write error: {e}")
                        error = 5  # EIO

            elif cmd == NBD_CMD_DISC:
                return False

            elif cmd == NBD_CMD_FLUSH:
                self.mapper.flush()

            elif cmd == NBD_CMD_TRIM:
                pass

            else:
                error = 22  # EINVAL

        except Exception as e:
            log(f"Command error: {e}")
            error = 5  # EIO

        # Send reply
        try:
            reply = struct.pack('>IIQ', NBD_REPLY_MAGIC, error, handle)
            sock.sendall(reply)
            if data:
                sock.sendall(data)
        except Exception:
            return False

        return True


def main():
    """Run the NBD server standalone (for testing)."""
    import argparse
    from .cluster_mapper import ClusterMapper

    parser = argparse.ArgumentParser(description='NTFS-ext4 bridge NBD server')
    parser.add_argument('image', help='Path to NTFS image file')
    parser.add_argument('source', help='Path to ext4 source directory')
    parser.add_argument('-p', '--port', type=int, default=10809)
    parser.add_argument('-H', '--host', default='0.0.0.0')
    parser.add_argument('-r', '--read-only', action='store_true')

    args = parser.parse_args()

    mapper = ClusterMapper(args.image, args.source)
    server = NBDServer(
        mapper=mapper,
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
