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

    # The pre-allocation block moved out of setup() into a helper when the
    # two-way branch landed. The ORDER is what matters and it has to be proven
    # in two places now, not assumed: the helper must be CALLED after the
    # mapper exists, and INSIDE the helper ntfsfix must be followed by reload.
    # Concatenating the bodies would let both pass even if the call were
    # moved above the mapper, so check the call site explicitly.
    mapper_at = setup.index('self.mapper = ClusterMapper(')
    helper = '_allocate_new_sparse_files'
    call_at = setup.find('self.%s()' % helper)
    assert call_at != -1, "expected setup() to call %s()" % helper
    assert call_at > mapper_at, (
        "%s() is called BEFORE the mapper exists; its ntfsfix would then "
        "run with no hot cache to reload" % helper)
    after = setup[mapper_at:] + body_of(src, helper)
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
