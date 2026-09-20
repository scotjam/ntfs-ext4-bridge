#!/usr/bin/env python3
"""Summarise a record-only catalogue: what did the guest try to do to ext4?

    python3 tools/review_ext4_attempts.py <image>.ext4-attempts.jsonl [--limit N]

Prints counts by operation and by top-level path, then the attempts most
worth a human's eye first: deletes, renames, zero-filled writes, and writes
that extend past the file's current size. Every line of the catalogue is one
attempt; nothing here is de-duplicated, so a count of 40 means 40 attempts.
"""
import argparse
import collections
import json
import sys


def load(path):
    header = None
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("header"):
                header = r
                continue
            recs.append(r)
    return header, recs


def suspicious(r):
    """Reasons a reviewer should look at this one first."""
    why = []
    op = r.get("op")
    if op in ("delete", "file rename", "directory rename", "materialize"):
        why.append(op)
    if op == "data_write":
        if r.get("all_zero"):
            why.append("zero-filled")
        size = r.get("size_on_ext4")
        if size is not None and any(e > size for _, e in r.get("spans", [])):
            why.append("past EOF (%d)" % size)
    if r.get("reason") == "stale image":
        why.append("stale-image")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()

    header, recs = load(a.catalogue)
    print("catalogue : %s" % a.catalogue)
    if header:
        print("mode      : %s   (pid %s)" % (header.get("mode"), header.get("pid")))
    print("attempts  : %d" % len(recs))
    if not recs:
        print("\nNothing was attempted against pre-existing ext4.")
        return 0

    by_op = collections.Counter(r.get("op") for r in recs)
    print("\nby operation:")
    for op, n in by_op.most_common():
        extra = ""
        if op == "data_write":
            total = sum(r.get("bytes", 0) for r in recs if r.get("op") == op)
            zeros = sum(1 for r in recs if r.get("op") == op and r.get("all_zero"))
            extra = "  (%.1f MiB, %d zero-filled)" % (total / 2 ** 20, zeros)
        print("   %-18s %6d%s" % (op, n, extra))

    by_top = collections.Counter((r.get("path") or "").split("/")[0] for r in recs)
    print("\nby top-level path:")
    for top, n in by_top.most_common(20):
        print("   %-40s %6d" % (top[:40], n))

    flagged = [(suspicious(r), r) for r in recs]
    flagged = [(w, r) for w, r in flagged if w]
    print("\nworth a look first: %d" % len(flagged))
    for why, r in flagged[:a.limit]:
        line = "   %-14s %-52s %s" % (r.get("op"), (r.get("path") or "")[:52], ", ".join(why))
        if r.get("op") == "data_write":
            line += "  spans=%s" % r.get("spans", [])[:3]
            if r.get("preview_hex"):
                line += "  first=%s" % r["preview_hex"][:24]
        print(line)
    if len(flagged) > a.limit:
        print("   ... and %d more (use --limit)" % (len(flagged) - a.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
