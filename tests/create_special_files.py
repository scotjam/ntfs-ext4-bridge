#!/usr/bin/env python3
"""Create test files with special NTFS filenames."""

import os
import sys

def create_test_files(base_dir):
    """Create test files with various special filenames."""
    source_dir = os.path.join(base_dir, "source")
    os.makedirs(source_dir, exist_ok=True)

    files = []

    # Max NTFS filename is 255 characters
    long_name = "A" * 251 + ".txt"  # 255 total
    files.append((long_name, "Max length filename content - 255 chars"))

    # Slightly shorter max length
    long_name2 = "B" * 200 + ".txt"
    files.append((long_name2, "Long filename content - 204 chars"))

    # Unicode filenames - various languages
    files.append(("中文文件.txt", "Chinese content - 中文内容"))
    files.append(("日本語ファイル.txt", "Japanese content - 日本語の内容"))
    files.append(("한국어파일.txt", "Korean content - 한국어 내용"))
    files.append(("Русский_файл.txt", "Russian content - Русское содержимое"))
    files.append(("ملف_عربي.txt", "Arabic content - محتوى عربي"))
    files.append(("Ελληνικό_αρχείο.txt", "Greek content - Ελληνικό περιεχόμενο"))
    files.append(("עברית_קובץ.txt", "Hebrew content - תוכן עברי"))
    files.append(("ไทย_ไฟล์.txt", "Thai content - เนื้อหาภาษาไทย"))

    # Accented Latin characters
    files.append(("café_naïve_résumé.txt", "Accented content"))
    files.append(("Ångström_Øresund_Müller.txt", "Nordic and German content"))
    files.append(("España_Françoise_Zürich.txt", "European accents"))

    # Special allowed characters
    files.append(("file with spaces.txt", "Spaces in filename"))
    files.append(("file   multiple   spaces.txt", "Multiple consecutive spaces"))
    files.append(("file.with" + "." * 10 + "dots.txt", "Many dots"))
    files.append(("file_with-dashes_and-underscores.txt", "Dashes and underscores"))
    files.append(("file (parentheses).txt", "Parentheses"))
    files.append(("file [brackets].txt", "Square brackets"))
    files.append(("file {braces}.txt", "Curly braces"))
    files.append(("file & ampersand.txt", "Ampersand"))
    files.append(("file @ at sign.txt", "At sign"))
    files.append(("file # hash.txt", "Hash sign"))
    files.append(("file $ dollar.txt", "Dollar sign"))
    files.append(("file % percent.txt", "Percent sign"))
    files.append(("file ! exclamation.txt", "Exclamation"))
    files.append(("file ~ tilde.txt", "Tilde"))
    files.append(("file ' apostrophe.txt", "Apostrophe"))
    files.append(("file ` backtick.txt", "Backtick"))
    files.append(("file ; semicolon.txt", "Semicolon"))
    files.append(("file , comma.txt", "Comma"))
    files.append(("file = equals.txt", "Equals sign"))
    files.append(("file + plus.txt", "Plus sign"))
    files.append(("file ^ caret.txt", "Caret"))

    # Files starting with special characters
    files.append((".hidden_file.txt", "Hidden file (starts with dot)"))
    files.append(("..double_dot.txt", "Starts with double dot"))
    files.append((" leading_space.txt", "Leading space"))
    files.append(("trailing_space .txt", "Trailing space before extension"))
    files.append(("~tempfile.txt", "Starts with tilde"))
    files.append(("_underscore_start.txt", "Starts with underscore"))
    files.append(("-dash_start.txt", "Starts with dash"))

    # Mixed Unicode and special characters
    files.append(("日本語 & English (混合).txt", "Mixed Japanese and English"))
    files.append(("Ελληνικά [test] αρχείο.txt", "Greek with brackets"))
    files.append(("Данные (копия) #1.txt", "Russian with special chars"))

    # Numbers and mixed
    files.append(("123456.txt", "Numeric filename"))
    files.append(("file_v1.2.3_final(2).txt", "Version-like filename"))

    # Larger files (to ensure non-resident storage)
    files.append(("大きなファイル_large.txt", "L" * 5000))  # Japanese + large
    files.append(("Größe_Datei_groß.txt", "M" * 5000))  # German + large
    files.append(("Большой_файл.txt", "N" * 5000))  # Russian + large
    files.append(("大文件_Chinese_large.txt", "O" * 5000))  # Chinese + large

    created = []
    failed = []

    for filename, content in files:
        filepath = os.path.join(source_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            size = os.path.getsize(filepath)
            created.append((filename, size))
            print(f"  OK: {filename} ({size} bytes)")
        except Exception as e:
            failed.append((filename, str(e)))
            print(f"  FAILED: {filename} - {e}")

    print(f"\n=== Summary ===")
    print(f"Created: {len(created)} files")
    print(f"Failed: {len(failed)} files")

    if failed:
        print("\nFailed files:")
        for name, err in failed:
            print(f"  {name}: {err}")

    return created, failed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default to WSL path or Windows path
        if os.path.exists("/home"):
            base_dir = os.path.expanduser("~/ntfs-bridge-special")
        else:
            base_dir = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "ntfs-bridge-special")

    print(f"Creating test files in: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)
    create_test_files(base_dir)
