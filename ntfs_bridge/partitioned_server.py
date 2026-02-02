"""NBD server that presents NTFS with MBR partition table."""

import struct
import os
import sys
import socket
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.template_synth import TemplateSynthesizer

def log(msg):
    print(f"[Partitioned] {msg}", flush=True)


# NBD Protocol constants
NBD_REQUEST_MAGIC = 0x25609513
NBD_REPLY_MAGIC = 0x67446698
NBD_CMD_READ = 0
NBD_CMD_WRITE = 1
NBD_CMD_DISC = 2
NBD_CMD_FLUSH = 3
NBD_FLAG_HAS_FLAGS = 0x0001
NBD_FLAG_SEND_FLUSH = 0x0004
NBD_OPT_GO = 7
NBD_OPT_INFO = 6
NBD_REP_ACK = 1
NBD_REP_INFO = 3
NBD_INFO_EXPORT = 0
NBD_INFO_BLOCK_SIZE = 3


class PartitionedSynthesizer:
    """Wraps TemplateSynthesizer to add MBR partition table."""

    PARTITION_OFFSET = 1048576  # 1MB in bytes
    MBR_SIZE = 512

    def __init__(self, template_path: str, source_dir: str):
        self.source_dir = source_dir
        self.inner = TemplateSynthesizer(template_path, source_dir)
        self.cluster_size = self.inner.cluster_size

        self.ntfs_size = self.inner.get_size()
        self._total_size = self.PARTITION_OFFSET + self.ntfs_size

        self.mbr = self._create_mbr()

        self.cluster_map = self.inner.cluster_map
        self.mft_record_to_source = self.inner.mft_record_to_source
        self.source_to_clusters = self.inner.source_to_clusters
        self.mft_offset = self.inner.mft_offset

    def _create_mbr(self) -> bytes:
        mbr = bytearray(512)
        mbr[0:2] = b'\xeb\xfe'

        part1_offset = 446
        start_sector = self.PARTITION_OFFSET // 512
        total_sectors = self.ntfs_size // 512

        part_entry = bytearray(16)
        part_entry[0] = 0x80
        part_entry[1:4] = bytes([254, 255, 255])
        part_entry[4] = 0x07
        part_entry[5:8] = bytes([254, 255, 255])
        struct.pack_into('<I', part_entry, 8, start_sector)
        struct.pack_into('<I', part_entry, 12, total_sectors)

        mbr[part1_offset:part1_offset+16] = part_entry
        mbr[510:512] = b'\x55\xaa'

        return bytes(mbr)

    def read(self, offset: int, length: int) -> bytes:
        result = bytearray(length)
        pos = 0

        while pos < length:
            current_offset = offset + pos
            remaining = length - pos

            if current_offset < self.MBR_SIZE:
                chunk_len = min(remaining, self.MBR_SIZE - current_offset)
                result[pos:pos+chunk_len] = self.mbr[current_offset:current_offset+chunk_len]
                pos += chunk_len
            elif current_offset < self.PARTITION_OFFSET:
                chunk_len = min(remaining, self.PARTITION_OFFSET - current_offset)
                pos += chunk_len
            else:
                ntfs_offset = current_offset - self.PARTITION_OFFSET
                if ntfs_offset < self.ntfs_size:
                    chunk_len = min(remaining, self.ntfs_size - ntfs_offset)
                    data = self.inner.read(ntfs_offset, chunk_len)
                    result[pos:pos+len(data)] = data
                    pos += chunk_len
                else:
                    pos += remaining

        return bytes(result)

    def get_size(self) -> int:
        return self._total_size

    def is_mft_region(self, offset: int, length: int) -> bool:
        if offset < self.PARTITION_OFFSET:
            return False
        return self.inner.is_mft_region(offset - self.PARTITION_OFFSET, length)

    def handle_mft_write(self, offset: int, data: bytes):
        if offset >= self.PARTITION_OFFSET:
            self.inner.handle_mft_write(offset - self.PARTITION_OFFSET, data)


class PartitionedNBDServer:
    """NBD server for partitioned synthesizer."""

    def __init__(self, synth: PartitionedSynthesizer, port: int = 10809):
        self.synth = synth
        self.port = port
        self.size = synth.get_size()
        self.read_only = False
        self.write_overlay = {}
        self.overlay_lock = threading.Lock()

    def run(self):
        log(f"NBD server listening on 0.0.0.0:{self.port}")
        log(f"  Volume size: {self.size} bytes ({self.size // 1048576} MB)")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(5)

        while True:
            client, addr = server.accept()
            log(f"Client connected from {addr}")
            t = threading.Thread(target=self._handle_client, args=(client, addr))
            t.daemon = True
            t.start()

    def _handle_client(self, sock, addr):
        try:
            if not self._do_handshake(sock):
                return

            while self._handle_request(sock):
                pass
        except Exception as e:
            log(f"Client error: {e}")
        finally:
            sock.close()
            log(f"Client {addr} disconnected")

    def _do_handshake(self, sock):
        # Send initial magic
        sock.sendall(b'NBDMAGIC')
        sock.sendall(struct.pack('>Q', 0x49484156454F5054))  # IHAVEOPT
        sock.sendall(struct.pack('>H', 0x0001))  # Handshake flags

        # Receive client flags
        sock.recv(4)

        # Handle options
        while True:
            opt_hdr = sock.recv(16)
            if len(opt_hdr) < 16:
                return False

            opt_magic, opt_type, opt_len = struct.unpack('>QII', opt_hdr)
            opt_data = sock.recv(opt_len) if opt_len > 0 else b''

            if opt_type == NBD_OPT_GO or opt_type == NBD_OPT_INFO:
                # Send block size info
                block_info = struct.pack('>HIII', NBD_INFO_BLOCK_SIZE, 1, 4096, 32*1024*1024)
                self._send_opt_reply(sock, opt_type, NBD_REP_INFO, block_info)

                # Send export info
                export_flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH
                export_info = struct.pack('>HQH', NBD_INFO_EXPORT, self.size, export_flags)
                self._send_opt_reply(sock, opt_type, NBD_REP_INFO, export_info)

                # Send ACK
                self._send_opt_reply(sock, opt_type, NBD_REP_ACK, b'')

                if opt_type == NBD_OPT_GO:
                    return True
            else:
                # Unknown option, send unsupported error
                self._send_opt_reply(sock, opt_type, 0x80000001, b'')

        return False

    def _send_opt_reply(self, sock, opt, reply_type, data):
        hdr = struct.pack('>QIII', 0x0003e889045565a9, opt, reply_type, len(data))
        sock.sendall(hdr + data)

    def _handle_request(self, sock):
        hdr = sock.recv(28)
        if len(hdr) < 28:
            return False

        magic, flags, cmd, handle, offset, length = struct.unpack('>IHHQQI', hdr)

        if magic != NBD_REQUEST_MAGIC:
            return False

        if cmd == NBD_CMD_READ:
            data = self._do_read(offset, length)
            # Log important reads
            if offset == 0:
                log(f"READ MBR offset=0 len={length}")
            elif offset == self.synth.PARTITION_OFFSET:
                log(f"READ NTFS boot sector offset={offset} len={length}")
            elif self.synth.is_mft_region(offset, length):
                log(f"READ MFT offset={offset} len={length}")
            reply = struct.pack('>IIQ', NBD_REPLY_MAGIC, 0, handle)
            sock.sendall(reply + data)

        elif cmd == NBD_CMD_WRITE:
            data = b''
            while len(data) < length:
                chunk = sock.recv(length - len(data))
                if not chunk:
                    return False
                data += chunk
            self._do_write(offset, data)
            reply = struct.pack('>IIQ', NBD_REPLY_MAGIC, 0, handle)
            sock.sendall(reply)

        elif cmd == NBD_CMD_DISC:
            return False

        elif cmd == NBD_CMD_FLUSH:
            reply = struct.pack('>IIQ', NBD_REPLY_MAGIC, 0, handle)
            sock.sendall(reply)

        return True

    def _do_read(self, offset: int, length: int) -> bytes:
        # Start with synthesizer data
        result = bytearray(self.synth.read(offset, length))

        # Apply any overlapping overlay data
        with self.overlay_lock:
            for ovl_offset, ovl_data in self.write_overlay.items():
                ovl_end = ovl_offset + len(ovl_data)
                read_end = offset + length

                # Check if overlay overlaps with read range
                if ovl_offset < read_end and ovl_end > offset:
                    # Calculate overlap
                    start_in_result = max(0, ovl_offset - offset)
                    start_in_ovl = max(0, offset - ovl_offset)
                    copy_len = min(ovl_end, read_end) - max(ovl_offset, offset)

                    result[start_in_result:start_in_result + copy_len] = ovl_data[start_in_ovl:start_in_ovl + copy_len]

        return bytes(result)

    def _do_write(self, offset: int, data: bytes):
        log(f"WRITE offset={offset} len={len(data)}")

        # Check if in partition area and mapped
        if offset >= self.synth.PARTITION_OFFSET:
            ntfs_offset = offset - self.synth.PARTITION_OFFSET
            cluster = ntfs_offset // self.synth.cluster_size

            if cluster in self.synth.cluster_map:
                source_path, file_offset = self.synth.cluster_map[cluster]
                write_offset = file_offset + (ntfs_offset % self.synth.cluster_size)
                try:
                    with open(source_path, 'r+b') as f:
                        f.seek(write_offset)
                        f.write(data)
                    log(f"  -> Written to {os.path.basename(source_path)} @ {write_offset}")
                    return
                except Exception as e:
                    log(f"  -> Write error: {e}")

            # Check MFT region
            if self.synth.is_mft_region(offset, len(data)):
                log(f"  -> MFT region write")
                self.synth.handle_mft_write(offset, data)
                # After MFT update, sync any overlay data to newly-mapped clusters
                self._sync_overlay_to_source()

        # Store in overlay
        with self.overlay_lock:
            self.write_overlay[offset] = data
        log(f"  -> Stored in overlay")

    def _sync_overlay_to_source(self):
        """Sync overlay data to source files for newly-mapped clusters."""
        with self.overlay_lock:
            synced_offsets = []
            for ovl_offset, ovl_data in self.write_overlay.items():
                if ovl_offset < self.synth.PARTITION_OFFSET:
                    continue
                ntfs_offset = ovl_offset - self.synth.PARTITION_OFFSET
                cluster = ntfs_offset // self.synth.cluster_size

                if cluster in self.synth.cluster_map:
                    source_path, file_offset = self.synth.cluster_map[cluster]
                    write_offset = file_offset + (ntfs_offset % self.synth.cluster_size)
                    try:
                        with open(source_path, 'r+b') as f:
                            f.seek(write_offset)
                            f.write(ovl_data)
                        log(f"  SYNC: overlay -> {os.path.basename(source_path)} @ {write_offset} ({len(ovl_data)} bytes)")
                        synced_offsets.append(ovl_offset)
                    except Exception as e:
                        log(f"  SYNC error: {e}")

            # Remove synced entries from overlay
            for off in synced_offsets:
                del self.write_overlay[off]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Partitioned NBD server')
    parser.add_argument('template', help='NTFS template file')
    parser.add_argument('source', help='Source directory')
    parser.add_argument('-p', '--port', type=int, default=10809, help='Port')
    args = parser.parse_args()

    log("Creating partitioned synthesizer...")
    synth = PartitionedSynthesizer(args.template, args.source)
    log(f"  NTFS size: {synth.ntfs_size}")
    log(f"  Total size: {synth.get_size()} ({synth.get_size() // 1048576} MB)")

    server = PartitionedNBDServer(synth, port=args.port)
    server.run()


if __name__ == '__main__':
    main()
