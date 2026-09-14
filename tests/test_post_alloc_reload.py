"""Any external rewrite of the image after ClusterMapper is live must be
followed by a hot-cache reload, and populate must prune before it adds.

Pins the two structural facts the fixes rely on, so a refactor cannot
quietly reorder them. The behaviours themselves are exercised by
test_hot_cache_reload.py and test_populate_prune.py.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = os.path.join(os.path.dirname(HERE), 'ntfs_bridge', 'bridge.py')


def body_of(src, name):
    start = src.index(f"    def {name}(")
    nxt = re.search(r"\n    def ", src[start + 1:])
    return src[start:start + 1 + nxt.start()] if nxt else src[start:]


def main():
    src = open(BRIDGE, encoding='utf-8').read()
    setup = body_of(src, 'setup')

    # 1. Every ntfsfix that runs after the mapper exists is followed by reload().
    mapper_at = setup.index('self.mapper = ClusterMapper(')
    after = setup[mapper_at:]
    fixes = [m.start() for m in re.finditer(r"\['ntfsfix', self\.image_path\]", after)]
    assert fixes, "expected a post-alloc ntfsfix in setup(); did it move?"
    for pos in fixes:
        window = after[pos:pos + 2500]
        assert 'self.mapper.image.reload()' in window, (
            "an ntfsfix runs after ClusterMapper is live without a following "
            "self.mapper.image.reload(): the hot cache would serve pre-repair "
            "metadata and stop() would flush it back over the repair")
    print("  ok: post-mapper ntfsfix is followed by a hot-cache reload")

    # 2. No other subprocess touches the image after the mapper is live.
    others = re.findall(r"subprocess\.run\(\s*\[([^\]]*self\.image_path[^\]]*)\]", after)
    assert all('ntfsfix' in o for o in others), (
        f"a subprocess other than ntfsfix rewrites the image after the mapper "
        f"is live and would need the same reload: {others}")
    print("  ok: ntfsfix is the only post-mapper external writer")

    # 3. Populate prunes stale entries before it creates anything.
    pop = body_of(src, '_populate_image')
    prune_at = pop.index('self._prune_stale_entries(tmp_mount)')
    create_at = pop.index('Populating NTFS image from ext4 source')
    assert prune_at < create_at, "prune must run before populate adds entries"
    print("  ok: populate prunes before it adds")

    print("PASS test_post_alloc_reload")


if __name__ == '__main__':
    main()
