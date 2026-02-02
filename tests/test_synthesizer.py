"""Test the template-based NTFS synthesizer."""

import os
import sys
import struct
import argparse

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntfs_bridge.template_synth import TemplateSynthesizer, MFT_RECORD_SIZE

SECTOR_SIZE = 512


def validate_boot_sector(data: bytes) -> bool:
    """Validate boot sector structure."""
    print("\n=== Boot Sector Validation ===")

    if len(data) < SECTOR_SIZE:
        print(f"  ERROR: Boot sector too small ({len(data)} bytes)")
        return False

    if data[0:3] != b'\xEB\x52\x90':
        print(f"  WARNING: Non-standard jump instruction: {data[0:3].hex()}")

    oem_id = data[3:11]
    print(f"  OEM ID: {oem_id}")
    if oem_id != b'NTFS    ':
        print("  ERROR: Invalid OEM ID")
        return False

    bytes_per_sector = struct.unpack('<H', data[0x0B:0x0D])[0]
    print(f"  Bytes per sector: {bytes_per_sector}")

    sectors_per_cluster = data[0x0D]
    print(f"  Sectors per cluster: {sectors_per_cluster}")

    total_sectors = struct.unpack('<Q', data[0x28:0x30])[0]
    print(f"  Total sectors: {total_sectors}")

    mft_cluster = struct.unpack('<Q', data[0x30:0x38])[0]
    print(f"  MFT start cluster: {mft_cluster}")

    if data[0x1FE:0x200] != b'\x55\xAA':
        print("  ERROR: Invalid boot signature")
        return False
    print("  Boot signature: OK (55 AA)")

    return True


def validate_mft_record(data: bytes, record_num: int, verbose: bool = True) -> dict:
    """Validate MFT record structure and return info."""
    result = {'valid': False, 'in_use': False, 'is_dir': False, 'filename': None}

    if len(data) < MFT_RECORD_SIZE:
        return result

    signature = data[0:4]
    if signature == b'\x00\x00\x00\x00':
        return result

    if signature != b'FILE':
        return result

    result['valid'] = True

    flags = struct.unpack('<H', data[22:24])[0]
    result['in_use'] = bool(flags & 0x01)
    result['is_dir'] = bool(flags & 0x02)

    if verbose:
        print(f"  Record {record_num}: flags=0x{flags:04x} in_use={result['in_use']} is_dir={result['is_dir']}")

    return result


def dump_hex(data: bytes, offset: int = 0, length: int = 256):
    """Dump data in hex format."""
    for i in range(0, min(length, len(data)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"  {offset+i:08x}: {hex_part:<48} {ascii_part}")


def find_test_setup(use_comprehensive: bool = False):
    """Find test setup (template and source directory)."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_test = os.path.join(os.path.dirname(script_dir), "local-test")

    if use_comprehensive:
        template_path = os.path.join(local_test, "comprehensive-template.raw")
        source_dir = os.path.join(local_test, "comprehensive-source")
    else:
        template_path = os.path.join(local_test, "template.raw")
        source_dir = os.path.join(local_test, "source")

    if not os.path.exists(template_path):
        print(f"Template not found at: {template_path}")
        return None

    if not os.path.exists(source_dir):
        print(f"Source directory not found at: {source_dir}")
        return None

    return template_path, source_dir


def test_cluster_mapping(synth: TemplateSynthesizer) -> tuple:
    """Test that cluster mappings work correctly. Returns (passed, total, failed_files)."""
    print("\n=== Cluster Mapping Test ===")

    if not synth.cluster_map:
        print("  No cluster mappings found (all files may be resident)")
        return (0, 0, [])

    print(f"  Found {len(synth.cluster_map)} mapped clusters")

    # Group by source file
    file_clusters = {}
    for cluster, (source_path, file_offset) in synth.cluster_map.items():
        if source_path not in file_clusters:
            file_clusters[source_path] = []
        file_clusters[source_path].append((cluster, file_offset))

    passed = 0
    total = 0
    failed_files = []

    for source_path, clusters in sorted(file_clusters.items()):
        filename = os.path.basename(source_path)
        rel_path = os.path.relpath(source_path, synth.source_dir)
        file_size = os.path.getsize(source_path) if os.path.exists(source_path) else 0
        num_clusters = len(clusters)

        print(f"\n  {rel_path}: {file_size} bytes, {num_clusters} clusters")

        file_passed = True
        for cluster, file_offset in sorted(clusters):
            total += 1

            # Read cluster from synthesizer
            byte_offset = cluster * synth.cluster_size
            synth_data = synth.read(byte_offset, synth.cluster_size)

            # Read corresponding data from source file
            try:
                with open(source_path, 'rb') as f:
                    f.seek(file_offset)
                    source_data = f.read(synth.cluster_size)
                    if len(source_data) < synth.cluster_size:
                        source_data += b'\x00' * (synth.cluster_size - len(source_data))
            except FileNotFoundError:
                print(f"    ERROR: Source file not found: {source_path}")
                file_passed = False
                continue

            if synth_data == source_data:
                passed += 1
            else:
                print(f"    MISMATCH at cluster {cluster} (file offset {file_offset})")
                print(f"      Synth: {synth_data[:40]!r}...")
                print(f"      Source: {source_data[:40]!r}...")
                file_passed = False

        if file_passed:
            print(f"    OK: All {num_clusters} clusters match")
        else:
            failed_files.append(rel_path)

    return (passed, total, failed_files)


def test_mft_tracking(synth: TemplateSynthesizer) -> dict:
    """Test MFT tracking data structures. Returns summary dict."""
    print("\n=== MFT Tracking Test ===")

    print(f"  MFT offset: {synth.mft_offset}")
    print(f"  Tracked MFT records: {len(synth.mft_record_to_source)}")

    summary = {
        'total_tracked': len(synth.mft_record_to_source),
        'resident_files': [],
        'non_resident_files': [],
        'missing_source': []
    }

    for record_num, source_path in sorted(synth.mft_record_to_source.items()):
        rel_path = os.path.relpath(source_path, synth.source_dir)
        clusters = synth.source_to_clusters.get(source_path, set())

        if not os.path.exists(source_path):
            summary['missing_source'].append(rel_path)
            print(f"    Record {record_num}: {rel_path} - MISSING SOURCE FILE")
        elif len(clusters) == 0:
            summary['resident_files'].append(rel_path)
            print(f"    Record {record_num}: {rel_path} - resident (no clusters)")
        else:
            summary['non_resident_files'].append(rel_path)
            print(f"    Record {record_num}: {rel_path} - {len(clusters)} clusters")

    return summary


def test_file_content_integrity(synth: TemplateSynthesizer) -> tuple:
    """Test that file content can be fully reconstructed. Returns (passed, failed_files)."""
    print("\n=== File Content Integrity Test ===")

    passed = []
    failed = []

    for record_num, source_path in sorted(synth.mft_record_to_source.items()):
        rel_path = os.path.relpath(source_path, synth.source_dir)

        if not os.path.exists(source_path):
            failed.append((rel_path, "Source file missing"))
            continue

        file_size = os.path.getsize(source_path)
        clusters = synth.source_to_clusters.get(source_path, set())

        if len(clusters) == 0:
            # Resident file - content is in MFT, not mapped clusters
            # We can't easily verify these without re-parsing MFT
            print(f"  {rel_path}: {file_size} bytes - resident (skipped)")
            continue

        # Read entire file content from synthesizer via mapped clusters
        sorted_clusters = sorted(clusters)
        reconstructed = bytearray()

        for cluster in sorted_clusters:
            if cluster in synth.cluster_map:
                byte_offset = cluster * synth.cluster_size
                data = synth.read(byte_offset, synth.cluster_size)
                reconstructed.extend(data)

        # Trim to actual file size
        reconstructed = reconstructed[:file_size]

        # Compare with source file
        with open(source_path, 'rb') as f:
            original = f.read()

        if reconstructed == original:
            passed.append(rel_path)
            print(f"  {rel_path}: {file_size} bytes - OK (content matches)")
        else:
            failed.append((rel_path, f"Content mismatch at byte {next((i for i in range(min(len(reconstructed), len(original))) if reconstructed[i] != original[i]), 'unknown')}"))
            print(f"  {rel_path}: {file_size} bytes - FAILED")
            print(f"    Original size: {len(original)}, Reconstructed size: {len(reconstructed)}")

    return (passed, failed)


def test_large_file_spanning(synth: TemplateSynthesizer):
    """Test that large files spanning multiple clusters are handled correctly."""
    print("\n=== Large File Spanning Test ===")

    # Find files with multiple clusters
    multi_cluster_files = []
    for source_path, clusters in synth.source_to_clusters.items():
        if len(clusters) > 1:
            rel_path = os.path.relpath(source_path, synth.source_dir)
            multi_cluster_files.append((rel_path, source_path, clusters))

    if not multi_cluster_files:
        print("  No multi-cluster files found")
        return True

    print(f"  Found {len(multi_cluster_files)} files spanning multiple clusters")

    all_passed = True
    for rel_path, source_path, clusters in multi_cluster_files:
        file_size = os.path.getsize(source_path)
        num_clusters = len(clusters)
        expected_clusters = (file_size + synth.cluster_size - 1) // synth.cluster_size

        print(f"\n  {rel_path}:")
        print(f"    Size: {file_size} bytes")
        print(f"    Clusters: {num_clusters} (expected ~{expected_clusters})")
        print(f"    Cluster numbers: {sorted(clusters)}")

        # Check cluster continuity
        sorted_clusters = sorted(clusters)
        gaps = []
        for i in range(1, len(sorted_clusters)):
            if sorted_clusters[i] != sorted_clusters[i-1] + 1:
                gaps.append((sorted_clusters[i-1], sorted_clusters[i]))

        if gaps:
            print(f"    Cluster gaps: {gaps} (fragmented file)")
        else:
            print(f"    Clusters are contiguous")

        # Verify content
        reconstructed = bytearray()
        for cluster in sorted_clusters:
            byte_offset = cluster * synth.cluster_size
            data = synth.read(byte_offset, synth.cluster_size)
            reconstructed.extend(data)
        reconstructed = reconstructed[:file_size]

        with open(source_path, 'rb') as f:
            original = f.read()

        if reconstructed == original:
            print(f"    Content: OK")
        else:
            print(f"    Content: MISMATCH!")
            all_passed = False

    return all_passed


def test_subdirectory_files(synth: TemplateSynthesizer):
    """Test that files in subdirectories are properly tracked."""
    print("\n=== Subdirectory Files Test ===")

    # Find files in subdirectories
    subdir_files = []
    for source_path in synth.mft_record_to_source.values():
        rel_path = os.path.relpath(source_path, synth.source_dir)
        if os.sep in rel_path or '/' in rel_path:
            subdir_files.append((rel_path, source_path))

    if not subdir_files:
        print("  No files in subdirectories found in tracking")
        return True

    print(f"  Found {len(subdir_files)} files in subdirectories:")

    all_exist = True
    for rel_path, source_path in subdir_files:
        exists = os.path.exists(source_path)
        clusters = synth.source_to_clusters.get(source_path, set())
        status = "OK" if exists else "MISSING"
        print(f"    {rel_path}: {len(clusters)} clusters - {status}")
        if not exists:
            all_exist = False

    return all_exist


def test_resident_vs_nonresident(synth: TemplateSynthesizer, source_dir: str) -> dict:
    """Analyze which files are resident vs non-resident and verify coverage."""
    print("\n=== Resident vs Non-Resident Analysis ===")

    # Get all source files
    all_source_files = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            rel = os.path.relpath(path, source_dir)
            all_source_files.append((rel, path, size))

    tracked_files = set(synth.mft_record_to_source.values())

    resident = []
    non_resident = []
    not_tracked = []

    for rel, path, size in all_source_files:
        if path in tracked_files:
            clusters = synth.source_to_clusters.get(path, set())
            if len(clusters) > 0:
                non_resident.append((rel, size, len(clusters)))
            else:
                resident.append((rel, size))
        else:
            # File not in MFT tracking - likely resident and matched by filename only
            not_tracked.append((rel, size))

    print(f"\n  Non-resident files (data in clusters): {len(non_resident)}")
    for rel, size, num_clusters in non_resident:
        print(f"    {rel}: {size} bytes, {num_clusters} clusters")

    print(f"\n  Resident files (data in MFT): {len(resident)}")
    for rel, size in resident:
        print(f"    {rel}: {size} bytes")

    print(f"\n  Not tracked (small/resident, not matched): {len(not_tracked)}")
    for rel, size in not_tracked:
        expected = "expected (small)" if size < 700 else "UNEXPECTED"
        print(f"    {rel}: {size} bytes - {expected}")

    # Check if untracked files are all small (expected resident)
    unexpected_untracked = [f for f in not_tracked if f[1] >= 700]

    return {
        'total_source': len(all_source_files),
        'non_resident': non_resident,
        'resident': resident,
        'not_tracked': not_tracked,
        'unexpected_untracked': unexpected_untracked
    }


def main():
    parser = argparse.ArgumentParser(description='Test template-based NTFS synthesizer')
    parser.add_argument('--comprehensive', '-c', action='store_true',
                        help='Use comprehensive test setup with folders and varied file sizes')
    parser.add_argument('--template', '-t', type=str, help='Path to template file')
    parser.add_argument('--source', '-s', type=str, help='Path to source directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=== Template-Based NTFS Synthesizer Test ===\n")

    # Find test setup
    if args.template and args.source:
        template_path = args.template
        source_dir = args.source
    else:
        test_setup = find_test_setup(use_comprehensive=args.comprehensive)
        if test_setup is None:
            print("\nNo test setup found.")
            print("Please ensure you have the template and source directory.")
            return 1
        template_path, source_dir = test_setup

    print(f"Template: {template_path}")
    print(f"Source directory: {source_dir}")

    # List source files with sizes
    print("\nSource files:")
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            rel = os.path.relpath(path, source_dir)
            category = "small" if size < 700 else "medium" if size < 4096 else "large"
            print(f"  {rel}: {size} bytes ({category})")
            total_files += 1
            total_size += size
    print(f"\nTotal: {total_files} files, {total_size} bytes")

    # Create synthesizer
    print("\nInitializing synthesizer...")
    synth = TemplateSynthesizer(template_path, source_dir)

    # Test boot sector
    print("\nReading boot sector...")
    boot_sector = synth.read(0, SECTOR_SIZE)
    if not validate_boot_sector(boot_sector):
        print("\nBoot sector validation FAILED")
        return 1

    # Test MFT tracking
    mft_summary = test_mft_tracking(synth)

    # Test cluster mapping
    cluster_passed, cluster_total, cluster_failed = test_cluster_mapping(synth)

    # Test file content integrity
    content_passed, content_failed = test_file_content_integrity(synth)

    # Test large file spanning
    large_file_ok = test_large_file_spanning(synth)

    # Test subdirectory files
    subdir_ok = test_subdirectory_files(synth)

    # Test resident vs non-resident
    resident_analysis = test_resident_vs_nonresident(synth, source_dir)

    # Summary
    print("\n" + "=" * 60)
    print("=== TEST SUMMARY ===")
    print("=" * 60)

    print(f"\nVolume Info:")
    print(f"  Total size: {synth.get_size()} bytes ({synth.get_size() / (1024*1024):.2f} MB)")
    print(f"  Cluster size: {synth.cluster_size} bytes")

    print(f"\nMFT Tracking:")
    print(f"  Total tracked files: {mft_summary['total_tracked']}")
    print(f"  Resident files (data in MFT): {len(mft_summary['resident_files'])}")
    print(f"  Non-resident files (data in clusters): {len(mft_summary['non_resident_files'])}")
    if mft_summary['missing_source']:
        print(f"  Missing source files: {mft_summary['missing_source']}")

    print(f"\nCluster Mapping:")
    print(f"  Total mapped clusters: {len(synth.cluster_map)}")
    print(f"  Cluster tests: {cluster_passed}/{cluster_total} passed")
    if cluster_failed:
        print(f"  Failed files: {cluster_failed}")

    print(f"\nContent Integrity:")
    print(f"  Files verified: {len(content_passed)}")
    if content_failed:
        print(f"  Failed files: {[f[0] for f in content_failed]}")

    print(f"\nLarge file spanning: {'PASSED' if large_file_ok else 'FAILED'}")
    print(f"Subdirectory files: {'PASSED' if subdir_ok else 'FAILED'}")

    print(f"\nFile Coverage:")
    print(f"  Total source files: {resident_analysis['total_source']}")
    print(f"  Non-resident (tracked with clusters): {len(resident_analysis['non_resident'])}")
    print(f"  Not tracked (small/resident): {len(resident_analysis['not_tracked'])}")
    if resident_analysis['unexpected_untracked']:
        print(f"  UNEXPECTED untracked files (>=700 bytes): {[f[0] for f in resident_analysis['unexpected_untracked']]}")
    else:
        print(f"  All untracked files are small (<700 bytes) - OK")

    # Overall result
    all_passed = (
        cluster_total == 0 or cluster_passed == cluster_total
    ) and (
        len(content_failed) == 0
    ) and large_file_ok and subdir_ok and (
        len(resident_analysis['unexpected_untracked']) == 0
    )

    print("\n" + "=" * 60)
    if all_passed:
        print("=== ALL TESTS PASSED ===")
        return 0
    else:
        print("=== SOME TESTS FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
