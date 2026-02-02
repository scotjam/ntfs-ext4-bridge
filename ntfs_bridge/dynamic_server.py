"""Dynamic NBD server - auto-generates NTFS template from source directory."""

import os
import sys
import struct
import socket
import threading
import subprocess
import tempfile
import shutil

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.template_synth import TemplateSynthesizer


def log(msg):
    print(f"[DynamicServer] {msg}", flush=True)


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


def create_template(source_dir: str, template_path: str, size_mb: int = 100):
    """Create NTFS template using ntfs-3g."""
    source_dir = os.path.abspath(source_dir)

    # Calculate size
    total_size = 0
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            try:
                total_size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    min_size = max(size_mb, (total_size * 3) // (1024 * 1024) + 20)

    log(f"Creating {min_size}MB template from {source_dir}")

    # Create NTFS
    subprocess.run(['truncate', '-s', f'{min_size}M', template_path], check=True)
    subprocess.run(['mkfs.ntfs', '-F', '-q', '-L', 'Dynamic', template_path],
                   check=True, capture_output=True)

    # Mount and copy structure
    mount_point = tempfile.mkdtemp(prefix='ntfs_dyn_')
    try:
        subprocess.run(['mount', '-o', 'loop', template_path, mount_point], check=True)

        for root, dirs, files in os.walk(source_dir):
            rel_path = os.path.relpath(root, source_dir)
            target_base = mount_point if rel_path == '.' else os.path.join(mount_point, rel_path)

            for d in dirs:
                os.makedirs(os.path.join(target_base, d), exist_ok=True)

            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(target_base, f)
                try:
                    shutil.copy2(src, dst)
                    log(f"  + {os.path.join(rel_path, f) if rel_path != '.' else f}")
                except (OSError, IOError) as e:
                    log(f"  ! {f}: {e}")

        subprocess.run(['sync'], check=True)
    finally:
        subprocess.run(['umount', mount_point], check=True)
        os.rmdir(mount_point)

    log(f"Template ready: {template_path}")


class DynamicPartitionedSynthesizer:
    """Wraps TemplateSynthesizer with MBR partition table."""

    PARTITION_OFFSET = 1048576  # 1MB

    def __init__(self, source_dir: str, size_mb: int = 100):
        self.source_dir = os.path.abspath(source_dir)

        # Create template in temp location
        self.template_path = tempfile.mktemp(suffix='.raw', prefix='ntfs_template_')
        create_template(source_dir, self.template_path, size_mb)

        # Initialize synthesizer
        self.inner = TemplateSynthesizer(self.template_path, source_dir)
        self.cluster_size = self.inner.cluster_size
        self.ntfs_size = self.inner.get_size()
        self._total_size = self.PARTITION_OFFSET + self.ntfs_size

        self.mbr = self._create_mbr()

        # Expose for compatibility
        self.cluster_map = self.inner.cluster_map
        self.mft_record_to_source = self.inner.mft_record_to_source
        self.mft_offset = self.inner.mft_offset

    def _create_mbr(self) -> bytes:
        mbr = bytearray(512)
        mbr[0:2] = b'\xeb\xfe'

        part_offset = 446
        start_sector = self.PARTITION_OFFSET // 512
        total_sectors = self.ntfs_size // 512

        part = bytearray(16)
        part[0] = 0x80
        part[1:4] = bytes([254, 255, 255])
        part[4] = 0x07
        part[5:8] = bytes([254, 255, 255])
        struct.pack_into('<I', part, 8, start_sector)
        struct.pack_into('<I', part, 12, total_sectors)

        mbr[part_offset:part_offset + 16] = part
        mbr[510:512] = b'\x55\xaa'

        return bytes(mbr)

    def read(self, offset: int, length: int) -> bytes:
        result = bytearray(length)
        pos = 0

        while pos < length:
            current = offset + pos
            remaining = length - pos

            if current < 512:
                chunk = min(remaining, 512 - current)
                result[pos:pos + chunk] = self.mbr[current:current + chunk]
            elif current < self.PARTITION_OFFSET:
                chunk = min(remaining, self.PARTITION_OFFSET - current)
                # Gap is zeros
            else:
                ntfs_off = current - self.PARTITION_OFFSET
                if ntfs_off < self.ntfs_size:
                    chunk = min(remaining, self.ntfs_size - ntfs_off)
                    data = self.inner.read(ntfs_off, chunk)
                    result[pos:pos + len(data)] = data
                else:
                    chunk = remaining

            pos += chunk

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

    def cleanup(self):
        if os.path.exists(self.template_path):
            os.remove(self.template_path)


class DynamicNBDServer:
    """NBD server with dynamic template generation."""

    def __init__(self, synth: DynamicPartitionedSynthesizer, port: int = 10809):
        self.synth = synth
        self.port = port
        self.size = synth.get_size()
        self.write_overlay = {}
        self.overlay_lock = threading.Lock()

    def run(self):
        log(f"NBD server on port {self.port}")
        log(f"  Volume: {self.size // 1048576} MB")
        log(f"  Source: {self.synth.source_dir}")
        log(f"  Files tracked: {len(self.synth.mft_record_to_source)}")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(5)

        try:
            while True:
                client, addr = server.accept()
                log(f"Client: {addr}")
                t = threading.Thread(target=self._handle_client, args=(client, addr))
                t.daemon = True
                t.start()
        finally:
            self.synth.cleanup()

    def _handle_client(self, sock, addr):
        try:
            if not self._handshake(sock):
                return
            while self._handle_request(sock):
                pass
        except Exception as e:
            log(f"Error: {e}")
        finally:
            sock.close()
            log(f"Disconnected: {addr}")

    def _handshake(self, sock):
        sock.sendall(b'NBDMAGIC')
        sock.sendall(struct.pack('>Q', 0x49484156454F5054))
        sock.sendall(struct.pack('>H', 0x0001))
        sock.recv(4)

        while True:
            hdr = sock.recv(16)
            if len(hdr) < 16:
                return False

            _, opt_type, opt_len = struct.unpack('>QII', hdr)
            if opt_len > 0:
                sock.recv(opt_len)

            if opt_type in (NBD_OPT_GO, NBD_OPT_INFO):
                # Block size info
                data = struct.pack('>HIII', NBD_INFO_BLOCK_SIZE, 1, 4096, 32*1024*1024)
                self._opt_reply(sock, opt_type, NBD_REP_INFO, data)

                # Export info
                flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH
                data = struct.pack('>HQH', NBD_INFO_EXPORT, self.size, flags)
                self._opt_reply(sock, opt_type, NBD_REP_INFO, data)

                self._opt_reply(sock, opt_type, NBD_REP_ACK, b'')

                if opt_type == NBD_OPT_GO:
                    return True
            else:
                self._opt_reply(sock, opt_type, 0x80000001, b'')

    def _opt_reply(self, sock, opt, reply_type, data):
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
            data = self._read(offset, length)
            sock.sendall(struct.pack('>IIQ', NBD_REPLY_MAGIC, 0, handle) + data)

        elif cmd == NBD_CMD_WRITE:
            data = b''
            while len(data) < length:
                chunk = sock.recv(length - len(data))
                if not chunk:
                    return False
                data += chunk
            self._write(offset, data)
            sock.sendall(struct.pack('>IIQ', NBD_REPLY_MAGIC, 0, handle))

        elif cmd == NBD_CMD_DISC:
            return False

        elif cmd == NBD_CMD_FLUSH:
            sock.sendall(struct.pack('>IIQ', NBD_REPLY_MAGIC, 0, handle))

        return True

    def _read(self, offset: int, length: int) -> bytes:
        result = bytearray(self.synth.read(offset, length))

        with self.overlay_lock:
            for ovl_off, ovl_data in self.write_overlay.items():
                ovl_end = ovl_off + len(ovl_data)
                read_end = offset + length

                if ovl_off < read_end and ovl_end > offset:
                    start_res = max(0, ovl_off - offset)
                    start_ovl = max(0, offset - ovl_off)
                    copy_len = min(ovl_end, read_end) - max(ovl_off, offset)
                    result[start_res:start_res + copy_len] = ovl_data[start_ovl:start_ovl + copy_len]

        return bytes(result)

    def _write(self, offset: int, data: bytes):
        # Try to write to mapped cluster
        if offset >= self.synth.PARTITION_OFFSET:
            ntfs_off = offset - self.synth.PARTITION_OFFSET
            cluster = ntfs_off // self.synth.cluster_size

            if cluster in self.synth.cluster_map:
                path, file_off = self.synth.cluster_map[cluster]
                write_off = file_off + (ntfs_off % self.synth.cluster_size)
                try:
                    with open(path, 'r+b') as f:
                        f.seek(write_off)
                        f.write(data)
                    log(f"WRITE -> {os.path.basename(path)} @ {write_off}")
                    return
                except Exception as e:
                    log(f"Write error: {e}")

            # MFT tracking
            if self.synth.is_mft_region(offset, len(data)):
                self.synth.handle_mft_write(offset, data)
                self._sync_overlay()

        # Store in overlay
        with self.overlay_lock:
            self.write_overlay[offset] = data

    def _sync_overlay(self):
        """Sync overlay to source after MFT update."""
        with self.overlay_lock:
            synced = []
            for ovl_off, ovl_data in self.write_overlay.items():
                if ovl_off < self.synth.PARTITION_OFFSET:
                    continue
                ntfs_off = ovl_off - self.synth.PARTITION_OFFSET
                cluster = ntfs_off // self.synth.cluster_size

                if cluster in self.synth.cluster_map:
                    path, file_off = self.synth.cluster_map[cluster]
                    write_off = file_off + (ntfs_off % self.synth.cluster_size)
                    try:
                        with open(path, 'r+b') as f:
                            f.seek(write_off)
                            f.write(ovl_data)
                        log(f"SYNC -> {os.path.basename(path)}")
                        synced.append(ovl_off)
                    except:
                        pass

            for off in synced:
                del self.write_overlay[off]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dynamic NTFS NBD Server')
    parser.add_argument('source', help='Source directory')
    parser.add_argument('-p', '--port', type=int, default=10809)
    parser.add_argument('-s', '--size', type=int, default=100, help='Volume size MB')
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("Error: Must run as root (for mount)")
        sys.exit(1)

    log("Initializing dynamic NTFS synthesizer...")
    synth = DynamicPartitionedSynthesizer(args.source, args.size)

    server = DynamicNBDServer(synth, args.port)
    server.run()


if __name__ == '__main__':
    main()
