"""Auto-generate NTFS template from source directory using ntfs-3g."""

import os
import subprocess
import tempfile
import shutil


def create_template_from_source(source_dir: str, output_path: str, size_mb: int = 100):
    """
    Create an NTFS template that mirrors the source directory structure.

    Uses ntfs-3g to create proper NTFS metadata, then our synthesizer
    redirects data reads to the actual source files.
    """
    source_dir = os.path.abspath(source_dir)

    # Calculate needed size (at least 2x source size for NTFS overhead)
    total_source_size = 0
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            try:
                total_source_size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    min_size_mb = max(size_mb, (total_source_size * 2) // (1024 * 1024) + 10)

    print(f"Creating {min_size_mb}MB NTFS template for {source_dir}")

    # Create empty NTFS image
    subprocess.run(['truncate', '-s', f'{min_size_mb}M', output_path], check=True)
    subprocess.run(['mkfs.ntfs', '-F', '-q', '-L', 'AutoTemplate', output_path], check=True)

    # Mount and create file structure
    mount_point = tempfile.mkdtemp(prefix='ntfs_template_')

    try:
        subprocess.run(['mount', '-o', 'loop', output_path, mount_point], check=True)

        # Walk source directory and create matching structure in NTFS
        for root, dirs, files in os.walk(source_dir):
            rel_path = os.path.relpath(root, source_dir)

            # Create directories
            for d in dirs:
                if rel_path == '.':
                    target_dir = os.path.join(mount_point, d)
                else:
                    target_dir = os.path.join(mount_point, rel_path, d)
                os.makedirs(target_dir, exist_ok=True)

            # Create files with same size (content will be from ext4)
            for f in files:
                src_file = os.path.join(root, f)
                if rel_path == '.':
                    dst_file = os.path.join(mount_point, f)
                else:
                    dst_file = os.path.join(mount_point, rel_path, f)

                try:
                    size = os.path.getsize(src_file)
                    # Copy the actual file - ntfs-3g will create proper MFT entry
                    shutil.copy2(src_file, dst_file)
                    print(f"  Created: {f} ({size} bytes)")
                except (OSError, IOError) as e:
                    print(f"  Warning: Could not create {f}: {e}")

        # Sync to ensure everything is written
        subprocess.run(['sync'], check=True)

    finally:
        subprocess.run(['umount', mount_point], check=True)
        os.rmdir(mount_point)

    print(f"Template created: {output_path}")
    return output_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python auto_template.py <source_dir> <output.raw> [size_mb]")
        sys.exit(1)

    source = sys.argv[1]
    output = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    create_template_from_source(source, output, size)
