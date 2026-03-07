"""
Windows Drive Verification Test for NTFS-Ext4 Bridge

Tests that the NBD-backed drive appears identical to a qcow2 drive to Windows
and that file operations are correctly reflected on the underlying ext4 filesystem.

Run on Windows with: python test_drive_windows.py E:
Where E: is the drive letter of the NBD-backed volume.

Requires: paramiko (pip install paramiko) for SSH verification
"""

import os
import sys
import ctypes
import struct
import time
import argparse
import tempfile
import uuid
from ctypes import wintypes

# Try to import paramiko for SSH verification
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    print("WARNING: paramiko not installed. SSH verification disabled.")
    print("Install with: pip install paramiko")


# Windows API constants
DRIVE_UNKNOWN = 0
DRIVE_NO_ROOT_DIR = 1
DRIVE_REMOVABLE = 2
DRIVE_FIXED = 3
DRIVE_REMOTE = 4
DRIVE_CDROM = 5
DRIVE_RAMDISK = 6

DRIVE_TYPE_NAMES = {
    DRIVE_UNKNOWN: "UNKNOWN",
    DRIVE_NO_ROOT_DIR: "NO_ROOT_DIR",
    DRIVE_REMOVABLE: "REMOVABLE",
    DRIVE_FIXED: "FIXED",
    DRIVE_REMOTE: "REMOTE (Network)",
    DRIVE_CDROM: "CDROM",
    DRIVE_RAMDISK: "RAMDISK",
}

# IOCTL codes
IOCTL_DISK_GET_DRIVE_GEOMETRY = 0x70000
IOCTL_STORAGE_GET_DEVICE_NUMBER = 0x2D1080
IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS = 0x560000

# Load Windows DLLs
kernel32 = ctypes.windll.kernel32
kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
kernel32.GetDriveTypeW.restype = wintypes.UINT

kernel32.GetVolumeNameForVolumeMountPointW.argtypes = [
    wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD
]
kernel32.GetVolumeNameForVolumeMountPointW.restype = wintypes.BOOL

kernel32.GetVolumeInformationW.argtypes = [
    wintypes.LPCWSTR,  # lpRootPathName
    wintypes.LPWSTR,   # lpVolumeNameBuffer
    wintypes.DWORD,    # nVolumeNameSize
    ctypes.POINTER(wintypes.DWORD),  # lpVolumeSerialNumber
    ctypes.POINTER(wintypes.DWORD),  # lpMaximumComponentLength
    ctypes.POINTER(wintypes.DWORD),  # lpFileSystemFlags
    wintypes.LPWSTR,   # lpFileSystemNameBuffer
    wintypes.DWORD,    # nFileSystemNameSize
]
kernel32.GetVolumeInformationW.restype = wintypes.BOOL

kernel32.GetDiskFreeSpaceExW.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
]
kernel32.GetDiskFreeSpaceExW.restype = wintypes.BOOL

kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
]
kernel32.CreateFileW.restype = wintypes.HANDLE

kernel32.DeviceIoControl.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID
]
kernel32.DeviceIoControl.restype = wintypes.BOOL

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
FILE_SHARE_WRITE = 0x2
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


class DISK_GEOMETRY(ctypes.Structure):
    _fields_ = [
        ("Cylinders", ctypes.c_longlong),
        ("MediaType", wintypes.DWORD),
        ("TracksPerCylinder", wintypes.DWORD),
        ("SectorsPerTrack", wintypes.DWORD),
        ("BytesPerSector", wintypes.DWORD),
    ]


class DriveVerifier:
    """Verifies drive properties match expectations for Windows compatibility."""

    def __init__(self, drive_letter: str, ssh_host: str = None, ssh_user: str = "root",
                 ext4_path: str = None):
        self.drive_letter = drive_letter.rstrip(":\\") + ":\\"
        self.drive_path = drive_letter.rstrip(":\\") + ":"
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ext4_path = ext4_path
        self.ssh_client = None
        self.results = {"passed": 0, "failed": 0, "warnings": 0}

    def connect_ssh(self):
        """Connect to Linux server for ext4 verification."""
        if not HAS_PARAMIKO or not self.ssh_host:
            return False

        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(self.ssh_host, username=self.ssh_user)
            return True
        except Exception as e:
            print(f"  SSH connection failed: {e}")
            return False

    def ssh_exec(self, cmd: str) -> tuple:
        """Execute command on Linux server."""
        if not self.ssh_client:
            return None, None
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        return stdout.read().decode('utf-8', errors='replace'), stderr.read().decode('utf-8', errors='replace')

    def close_ssh(self):
        """Close SSH connection."""
        if self.ssh_client:
            self.ssh_client.close()

    def log_result(self, test_name: str, passed: bool, message: str, warning: bool = False):
        """Log test result."""
        if warning:
            status = "WARN"
            self.results["warnings"] += 1
        elif passed:
            status = "PASS"
            self.results["passed"] += 1
        else:
            status = "FAIL"
            self.results["failed"] += 1

        print(f"  [{status}] {test_name}: {message}")

    def test_drive_type(self) -> bool:
        """Test that drive type is DRIVE_FIXED (required for Windows compatibility)."""
        print("\n=== Drive Type Test ===")

        drive_type = kernel32.GetDriveTypeW(self.drive_letter)
        type_name = DRIVE_TYPE_NAMES.get(drive_type, f"UNKNOWN({drive_type})")

        is_fixed = drive_type == DRIVE_FIXED
        self.log_result(
            "GetDriveType",
            is_fixed,
            f"{type_name} (expected: FIXED)"
        )

        if drive_type == DRIVE_REMOTE:
            self.log_result(
                "Network Drive Check",
                False,
                "Drive appears as network drive, not fixed local drive (not ideal)"
            )

        return is_fixed

    def test_volume_guid(self) -> str:
        """Test that volume has a valid GUID."""
        print("\n=== Volume GUID Test ===")

        volume_name = ctypes.create_unicode_buffer(50)
        result = kernel32.GetVolumeNameForVolumeMountPointW(
            self.drive_letter, volume_name, 50
        )

        if result:
            guid = volume_name.value
            # Expected format: \\?\Volume{GUID}\
            has_valid_format = guid.startswith("\\\\?\\Volume{") and guid.endswith("}\\")
            self.log_result(
                "Volume GUID",
                has_valid_format,
                guid
            )

            # Extract just the GUID part
            if "{" in guid and "}" in guid:
                cust_guid = "v" + guid.split("{")[1].split("}")[0].replace("-", "")
                self.log_result(
                    "GUID Format",
                    True,
                    cust_guid
                )

            return guid
        else:
            error = ctypes.get_last_error()
            self.log_result("Volume GUID", False, f"Failed to get GUID (error {error})")
            return None

    def test_volume_info(self) -> dict:
        """Test volume information."""
        print("\n=== Volume Information Test ===")

        volume_name = ctypes.create_unicode_buffer(261)
        serial_number = wintypes.DWORD()
        max_component_length = wintypes.DWORD()
        fs_flags = wintypes.DWORD()
        fs_name = ctypes.create_unicode_buffer(261)

        result = kernel32.GetVolumeInformationW(
            self.drive_letter,
            volume_name, 261,
            ctypes.byref(serial_number),
            ctypes.byref(max_component_length),
            ctypes.byref(fs_flags),
            fs_name, 261
        )

        if result:
            info = {
                "volume_name": volume_name.value,
                "serial_number": f"{serial_number.value:08X}",
                "max_component_length": max_component_length.value,
                "fs_flags": fs_flags.value,
                "fs_name": fs_name.value,
            }

            self.log_result("Volume Name", True, info["volume_name"] or "(empty)")
            self.log_result("Serial Number", True, info["serial_number"])
            self.log_result(
                "Filesystem",
                info["fs_name"] == "NTFS",
                f"{info['fs_name']} (expected: NTFS)"
            )
            self.log_result("Max Component Length", True, str(info["max_component_length"]))

            return info
        else:
            error = ctypes.get_last_error()
            self.log_result("Volume Info", False, f"Failed (error {error})")
            return None

    def test_disk_space(self) -> dict:
        """Test disk space information."""
        print("\n=== Disk Space Test ===")

        free_bytes_available = ctypes.c_ulonglong()
        total_bytes = ctypes.c_ulonglong()
        total_free_bytes = ctypes.c_ulonglong()

        result = kernel32.GetDiskFreeSpaceExW(
            self.drive_letter,
            ctypes.byref(free_bytes_available),
            ctypes.byref(total_bytes),
            ctypes.byref(total_free_bytes)
        )

        if result:
            info = {
                "free_bytes": free_bytes_available.value,
                "total_bytes": total_bytes.value,
                "used_bytes": total_bytes.value - total_free_bytes.value,
            }

            def fmt_size(b):
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if b < 1024:
                        return f"{b:.2f} {unit}"
                    b /= 1024
                return f"{b:.2f} PB"

            self.log_result("Total Size", True, fmt_size(info["total_bytes"]))
            self.log_result("Free Space", True, fmt_size(info["free_bytes"]))
            self.log_result("Used Space", True, fmt_size(info["used_bytes"]))

            return info
        else:
            error = ctypes.get_last_error()
            self.log_result("Disk Space", False, f"Failed (error {error})")
            return None

    def test_disk_geometry(self) -> dict:
        """Test disk geometry via IOCTL."""
        print("\n=== Disk Geometry Test ===")

        # Open volume handle
        volume_path = f"\\\\.\\{self.drive_path}"
        handle = kernel32.CreateFileW(
            volume_path,
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            0,
            None
        )

        if handle == INVALID_HANDLE_VALUE:
            error = ctypes.get_last_error()
            self.log_result("Open Volume", False, f"Failed to open {volume_path} (error {error})")
            return None

        try:
            geometry = DISK_GEOMETRY()
            bytes_returned = wintypes.DWORD()

            result = kernel32.DeviceIoControl(
                handle,
                IOCTL_DISK_GET_DRIVE_GEOMETRY,
                None, 0,
                ctypes.byref(geometry), ctypes.sizeof(geometry),
                ctypes.byref(bytes_returned),
                None
            )

            if result:
                info = {
                    "cylinders": geometry.Cylinders,
                    "media_type": geometry.MediaType,
                    "tracks_per_cylinder": geometry.TracksPerCylinder,
                    "sectors_per_track": geometry.SectorsPerTrack,
                    "bytes_per_sector": geometry.BytesPerSector,
                }

                self.log_result("Cylinders", True, str(info["cylinders"]))
                self.log_result("Tracks/Cylinder", True, str(info["tracks_per_cylinder"]))
                self.log_result("Sectors/Track", True, str(info["sectors_per_track"]))
                self.log_result("Bytes/Sector", info["bytes_per_sector"] == 512,
                              f"{info['bytes_per_sector']} (expected: 512)")

                return info
            else:
                error = ctypes.get_last_error()
                self.log_result("Disk Geometry", False, f"IOCTL failed (error {error})", warning=True)
                return None
        finally:
            kernel32.CloseHandle(handle)

    def test_file_create_write(self, filename: str, content: bytes) -> bool:
        """Test file creation and writing."""
        filepath = os.path.join(self.drive_letter, filename)
        try:
            with open(filepath, 'wb') as f:
                f.write(content)

            # Verify on Windows
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    read_content = f.read()
                if read_content == content:
                    self.log_result(f"Create/Write '{filename}'", True, f"{len(content)} bytes")
                    return True
                else:
                    self.log_result(f"Create/Write '{filename}'", False, "Content mismatch")
                    return False
            else:
                self.log_result(f"Create/Write '{filename}'", False, "File not found after creation")
                return False
        except Exception as e:
            self.log_result(f"Create/Write '{filename}'", False, str(e))
            return False

    def test_file_read(self, filename: str, expected_content: bytes) -> bool:
        """Test file reading."""
        filepath = os.path.join(self.drive_letter, filename)
        try:
            with open(filepath, 'rb') as f:
                content = f.read()

            if content == expected_content:
                self.log_result(f"Read '{filename}'", True, f"{len(content)} bytes")
                return True
            else:
                self.log_result(f"Read '{filename}'", False,
                              f"Content mismatch (got {len(content)}, expected {len(expected_content)})")
                return False
        except Exception as e:
            self.log_result(f"Read '{filename}'", False, str(e))
            return False

    def test_file_rename(self, old_name: str, new_name: str) -> bool:
        """Test file renaming."""
        old_path = os.path.join(self.drive_letter, old_name)
        new_path = os.path.join(self.drive_letter, new_name)
        try:
            os.rename(old_path, new_path)

            if os.path.exists(new_path) and not os.path.exists(old_path):
                self.log_result(f"Rename '{old_name}' -> '{new_name}'", True, "OK")
                return True
            else:
                self.log_result(f"Rename '{old_name}' -> '{new_name}'", False, "Rename verification failed")
                return False
        except Exception as e:
            self.log_result(f"Rename '{old_name}' -> '{new_name}'", False, str(e))
            return False

    def test_file_delete(self, filename: str) -> bool:
        """Test file deletion."""
        filepath = os.path.join(self.drive_letter, filename)
        try:
            os.remove(filepath)

            if not os.path.exists(filepath):
                self.log_result(f"Delete '{filename}'", True, "OK")
                return True
            else:
                self.log_result(f"Delete '{filename}'", False, "File still exists")
                return False
        except Exception as e:
            self.log_result(f"Delete '{filename}'", False, str(e))
            return False

    def test_folder_create(self, foldername: str) -> bool:
        """Test folder creation."""
        folderpath = os.path.join(self.drive_letter, foldername)
        try:
            os.makedirs(folderpath, exist_ok=True)

            if os.path.isdir(folderpath):
                self.log_result(f"Create folder '{foldername}'", True, "OK")
                return True
            else:
                self.log_result(f"Create folder '{foldername}'", False, "Folder not found")
                return False
        except Exception as e:
            self.log_result(f"Create folder '{foldername}'", False, str(e))
            return False

    def test_folder_rename(self, old_name: str, new_name: str) -> bool:
        """Test folder renaming."""
        old_path = os.path.join(self.drive_letter, old_name)
        new_path = os.path.join(self.drive_letter, new_name)
        try:
            os.rename(old_path, new_path)

            if os.path.isdir(new_path) and not os.path.exists(old_path):
                self.log_result(f"Rename folder '{old_name}' -> '{new_name}'", True, "OK")
                return True
            else:
                self.log_result(f"Rename folder '{old_name}' -> '{new_name}'", False, "Verification failed")
                return False
        except Exception as e:
            self.log_result(f"Rename folder '{old_name}' -> '{new_name}'", False, str(e))
            return False

    def test_folder_delete(self, foldername: str) -> bool:
        """Test folder deletion."""
        folderpath = os.path.join(self.drive_letter, foldername)
        try:
            os.rmdir(folderpath)

            if not os.path.exists(folderpath):
                self.log_result(f"Delete folder '{foldername}'", True, "OK")
                return True
            else:
                self.log_result(f"Delete folder '{foldername}'", False, "Folder still exists")
                return False
        except Exception as e:
            self.log_result(f"Delete folder '{foldername}'", False, str(e))
            return False

    def verify_ext4(self, path: str, expected_content: bytes = None, should_exist: bool = True) -> bool:
        """Verify file/folder state on ext4 backend via SSH."""
        if not self.ssh_client or not self.ext4_path:
            return True  # Skip if SSH not available

        ext4_full_path = os.path.join(self.ext4_path, path).replace("\\", "/")

        # Check existence
        stdout, _ = self.ssh_exec(f"test -e '{ext4_full_path}' && echo 'exists' || echo 'missing'")
        exists = stdout.strip() == 'exists'

        if should_exist != exists:
            self.log_result(
                f"ext4 sync '{path}'",
                False,
                f"Expected {'exists' if should_exist else 'missing'}, got {'exists' if exists else 'missing'}"
            )
            return False

        # If file exists and we have expected content, verify it
        if should_exist and expected_content is not None:
            stdout, _ = self.ssh_exec(f"cat '{ext4_full_path}' | base64")
            import base64
            try:
                actual_content = base64.b64decode(stdout.strip())
                if actual_content == expected_content:
                    self.log_result(f"ext4 content '{path}'", True, f"{len(actual_content)} bytes match")
                    return True
                else:
                    self.log_result(f"ext4 content '{path}'", False, "Content mismatch")
                    return False
            except Exception as e:
                self.log_result(f"ext4 content '{path}'", False, str(e))
                return False

        self.log_result(f"ext4 sync '{path}'", True, "exists" if exists else "deleted")
        return True

    def run_all_tests(self):
        """Run all verification tests."""
        print(f"\n{'='*60}")
        print(f"NTFS-Ext4 Bridge Drive Verification")
        print(f"Testing drive: {self.drive_letter}")
        print(f"{'='*60}")

        # Connect SSH if available
        if self.ssh_host:
            print(f"\nConnecting to {self.ssh_host} for ext4 verification...")
            if self.connect_ssh():
                print("  SSH connected successfully")
            else:
                print("  SSH connection failed - ext4 verification disabled")

        # Drive property tests
        self.test_drive_type()
        self.test_volume_guid()
        self.test_volume_info()
        self.test_disk_space()
        self.test_disk_geometry()

        # File operation tests
        print("\n=== File Operations Test ===")
        test_id = str(uuid.uuid4())[:8]
        test_filename = f"_bridge_test_{test_id}.txt"
        test_filename_renamed = f"_bridge_test_{test_id}_renamed.txt"
        test_content = f"NTFS-Ext4 Bridge Test Content {test_id}\n".encode('utf-8')
        # Pad to >700 bytes to ensure non-resident storage
        test_content += b"X" * 800

        if self.test_file_create_write(test_filename, test_content):
            time.sleep(0.5)  # Give time for sync
            self.verify_ext4(test_filename, test_content)

            self.test_file_read(test_filename, test_content)

            if self.test_file_rename(test_filename, test_filename_renamed):
                time.sleep(0.5)
                self.verify_ext4(test_filename, should_exist=False)
                self.verify_ext4(test_filename_renamed, test_content)

                if self.test_file_delete(test_filename_renamed):
                    time.sleep(0.5)
                    self.verify_ext4(test_filename_renamed, should_exist=False)

        # Folder operation tests
        print("\n=== Folder Operations Test ===")
        test_folder = f"_bridge_test_folder_{test_id}"
        test_folder_renamed = f"_bridge_test_folder_{test_id}_renamed"
        test_file_in_folder = f"{test_folder}/nested_file.txt"

        if self.test_folder_create(test_folder):
            time.sleep(0.5)
            self.verify_ext4(test_folder)

            # Create a file inside the folder
            nested_content = b"Nested file content\n" + b"Y" * 800
            if self.test_file_create_write(test_file_in_folder, nested_content):
                time.sleep(0.5)
                self.verify_ext4(test_file_in_folder, nested_content)

                # Delete the nested file first
                self.test_file_delete(test_file_in_folder)
                time.sleep(0.5)

            if self.test_folder_rename(test_folder, test_folder_renamed):
                time.sleep(0.5)
                self.verify_ext4(test_folder, should_exist=False)
                self.verify_ext4(test_folder_renamed)

                if self.test_folder_delete(test_folder_renamed):
                    time.sleep(0.5)
                    self.verify_ext4(test_folder_renamed, should_exist=False)

        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Passed:   {self.results['passed']}")
        print(f"  Failed:   {self.results['failed']}")
        print(f"  Warnings: {self.results['warnings']}")

        if self.results['failed'] == 0:
            print("\n  *** ALL TESTS PASSED ***")
            print("  Drive looks good")
        else:
            print("\n  *** SOME TESTS FAILED ***")
            print("  Drive may not appear as a proper local drive")

        self.close_ssh()
        return self.results['failed'] == 0


def main():
    parser = argparse.ArgumentParser(
        description='Verify NTFS-Ext4 bridge drive properties'
    )
    parser.add_argument('drive', help='Drive letter to test (e.g., E: or E)')
    parser.add_argument('--ssh-host', '-H', default='192.168.1.12',
                        help='Linux server hostname for ext4 verification')
    parser.add_argument('--ssh-user', '-u', default='root',
                        help='SSH username')
    parser.add_argument('--ext4-path', '-p', default='/srv/ntfs-bridge-test',
                        help='Path to ext4 source directory on Linux server')
    parser.add_argument('--no-ssh', action='store_true',
                        help='Skip SSH verification')

    args = parser.parse_args()

    verifier = DriveVerifier(
        args.drive,
        ssh_host=None if args.no_ssh else args.ssh_host,
        ssh_user=args.ssh_user,
        ext4_path=args.ext4_path,
    )

    success = verifier.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
