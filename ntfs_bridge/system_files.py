"""Generate NTFS system file data."""

import struct
from .constants import CLUSTER_SIZE


def generate_upcase_table() -> bytes:
    """
    Generate the $UpCase table (128KB).

    Maps each Unicode codepoint (0-65535) to its uppercase equivalent.
    """
    table = bytearray(65536 * 2)  # 128KB

    for i in range(65536):
        # Default: character maps to itself
        upper = i

        # ASCII lowercase a-z (97-122) -> A-Z (65-90)
        if 97 <= i <= 122:
            upper = i - 32

        # Extended Latin lowercase mappings
        # Latin Extended-A
        elif 0x00E0 <= i <= 0x00F6:  # à-ö -> À-Ö
            upper = i - 32
        elif 0x00F8 <= i <= 0x00FE:  # ø-þ -> Ø-Þ
            upper = i - 32

        # More comprehensive would include full Unicode case mappings,
        # but for basic functionality this covers ASCII and common Latin

        struct.pack_into('<H', table, i * 2, upper)

    return bytes(table)


def generate_bitmap(total_clusters: int, allocated_clusters: list) -> bytes:
    """
    Generate cluster bitmap.

    Each bit represents one cluster (1 = allocated, 0 = free).
    """
    # Calculate bitmap size (1 bit per cluster, rounded up to bytes)
    bitmap_bytes = (total_clusters + 7) // 8
    # Round up to cluster size
    bitmap_size = ((bitmap_bytes + CLUSTER_SIZE - 1) // CLUSTER_SIZE) * CLUSTER_SIZE

    bitmap = bytearray(bitmap_size)

    # Mark allocated clusters
    for cluster in allocated_clusters:
        if cluster < total_clusters:
            byte_idx = cluster // 8
            bit_idx = cluster % 8
            bitmap[byte_idx] |= (1 << bit_idx)

    return bytes(bitmap)


def generate_attrdef() -> bytes:
    """
    Generate minimal $AttrDef (attribute definitions).

    Each entry is 160 bytes.
    """
    entries = []

    # Format: name (128 bytes UTF-16), type (4), display_rule (4),
    #         collation (4), flags (4), min_size (8), max_size (8)

    def make_entry(name: str, attr_type: int, flags: int = 0,
                   min_size: int = 0, max_size: int = 0xFFFFFFFFFFFFFFFF):
        entry = bytearray(160)
        name_utf16 = name.encode('utf-16-le')[:128]
        entry[0:len(name_utf16)] = name_utf16
        struct.pack_into('<I', entry, 128, attr_type)  # Type
        struct.pack_into('<I', entry, 132, 0)  # Display rule
        struct.pack_into('<I', entry, 136, 0)  # Collation
        struct.pack_into('<I', entry, 140, flags)  # Flags
        struct.pack_into('<Q', entry, 144, min_size)  # Min size
        struct.pack_into('<Q', entry, 152, max_size)  # Max size
        return bytes(entry)

    # Standard attributes
    entries.append(make_entry("$STANDARD_INFORMATION", 0x10, 0x40, 48, 72))
    entries.append(make_entry("$ATTRIBUTE_LIST", 0x20, 0x80))
    entries.append(make_entry("$FILE_NAME", 0x30, 0x42, 68, 578))
    entries.append(make_entry("$OBJECT_ID", 0x40, 0x40, 0, 256))
    entries.append(make_entry("$SECURITY_DESCRIPTOR", 0x50, 0x80))
    entries.append(make_entry("$VOLUME_NAME", 0x60, 0x40, 2, 256))
    entries.append(make_entry("$VOLUME_INFORMATION", 0x70, 0x40, 12, 12))
    entries.append(make_entry("$DATA", 0x80, 0x00))
    entries.append(make_entry("$INDEX_ROOT", 0x90, 0x40))
    entries.append(make_entry("$INDEX_ALLOCATION", 0xA0, 0x80))
    entries.append(make_entry("$BITMAP", 0xB0, 0x80))
    entries.append(make_entry("$REPARSE_POINT", 0xC0, 0x40, 0, 16384))
    entries.append(make_entry("$EA_INFORMATION", 0xD0, 0x40, 8, 8))
    entries.append(make_entry("$EA", 0xE0, 0x00))

    # End marker (all zeros)
    entries.append(bytes(160))

    result = b''.join(entries)
    # Pad to cluster boundary
    if len(result) % CLUSTER_SIZE != 0:
        padding = CLUSTER_SIZE - (len(result) % CLUSTER_SIZE)
        result += b'\x00' * padding

    return result
