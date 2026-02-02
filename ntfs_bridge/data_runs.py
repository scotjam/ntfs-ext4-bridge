"""NTFS data run encoding/decoding."""

from typing import List, Tuple


def encode_data_runs(runs: List[Tuple[int, int]]) -> bytes:
    """
    Encode a list of (length, start_lcn) tuples into NTFS data runs format.

    Each run is encoded as:
    - 1 byte header: high nibble = LCN offset bytes, low nibble = length bytes
    - N bytes for run length
    - M bytes for LCN offset (signed, relative to previous LCN)

    Args:
        runs: List of (cluster_count, starting_cluster) tuples

    Returns:
        Encoded data runs bytes, ending with 0x00
    """
    if not runs:
        return b'\x00'

    encoded = bytearray()
    prev_lcn = 0

    for length, lcn in runs:
        # Calculate LCN offset from previous
        lcn_offset = lcn - prev_lcn

        # Determine bytes needed for length (unsigned)
        len_bytes = _min_bytes_unsigned(length)

        # Determine bytes needed for LCN offset (signed)
        off_bytes = _min_bytes_signed(lcn_offset)

        # Header byte
        header = (off_bytes << 4) | len_bytes
        encoded.append(header)

        # Length (little-endian, unsigned)
        encoded.extend(length.to_bytes(len_bytes, 'little', signed=False))

        # LCN offset (little-endian, signed)
        encoded.extend(lcn_offset.to_bytes(off_bytes, 'little', signed=True))

        prev_lcn = lcn

    # Terminator
    encoded.append(0x00)

    return bytes(encoded)


def decode_data_runs(data: bytes) -> List[Tuple[int, int]]:
    """
    Decode NTFS data runs format into a list of (length, start_lcn) tuples.

    Args:
        data: Encoded data runs bytes

    Returns:
        List of (cluster_count, starting_cluster) tuples
    """
    runs = []
    pos = 0
    prev_lcn = 0

    while pos < len(data):
        header = data[pos]
        if header == 0:
            break

        len_bytes = header & 0x0F
        off_bytes = (header >> 4) & 0x0F

        pos += 1

        # Read length (unsigned)
        length = int.from_bytes(data[pos:pos + len_bytes], 'little', signed=False)
        pos += len_bytes

        # Read LCN offset (signed)
        lcn_offset = int.from_bytes(data[pos:pos + off_bytes], 'little', signed=True)
        pos += off_bytes

        lcn = prev_lcn + lcn_offset
        runs.append((length, lcn))
        prev_lcn = lcn

    return runs


def _min_bytes_unsigned(value: int) -> int:
    """Calculate minimum bytes needed to represent unsigned value."""
    if value == 0:
        return 1
    return (value.bit_length() + 7) // 8


def _min_bytes_signed(value: int) -> int:
    """Calculate minimum bytes needed to represent signed value."""
    if value == 0:
        return 1
    if value > 0:
        # Need extra bit for sign
        return (value.bit_length() + 8) // 8
    else:
        # For negative, Python's bit_length doesn't include sign
        return ((-value - 1).bit_length() + 8) // 8


def compress_cluster_list(clusters: List[int]) -> List[Tuple[int, int]]:
    """
    Compress a list of cluster numbers into runs.

    Consecutive clusters are combined into single runs.

    Args:
        clusters: List of cluster numbers (may not be consecutive)

    Returns:
        List of (length, start_cluster) tuples
    """
    if not clusters:
        return []

    runs = []
    run_start = clusters[0]
    run_length = 1

    for i in range(1, len(clusters)):
        if clusters[i] == clusters[i - 1] + 1:
            # Consecutive cluster
            run_length += 1
        else:
            # Gap - save current run and start new
            runs.append((run_length, run_start))
            run_start = clusters[i]
            run_length = 1

    # Don't forget last run
    runs.append((run_length, run_start))

    return runs


def make_data_runs(start_cluster: int, cluster_count: int) -> bytes:
    """
    Create data runs for a contiguous allocation.

    Args:
        start_cluster: First cluster number
        cluster_count: Number of clusters

    Returns:
        Encoded data runs bytes
    """
    if cluster_count == 0:
        return b'\x00'

    runs = [(cluster_count, start_cluster)]
    return encode_data_runs(runs)
