"""Verify original research and extension artifacts without writing outside this pass."""
from ..common import preservation_snapshot, sha256
from .shared import BASE, OUT, dump, read, verify_execution


def main():
    verify_execution()
    before = read(OUT / 'logs/frozen_extension_before.json')
    current = {}
    for path in sorted(BASE.rglob('*')):
        relative = path.relative_to(BASE)
        if not path.is_file() or relative.parts[0] in ('interpretation_v1', 'runtime') or '__pycache__' in relative.parts:
            continue
        current[str(relative)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if current != before:
        raise RuntimeError('An original extension file was added, removed, or changed')
    comp_before = read(BASE / 'logs/preservation_before.json')
    comp_now = preservation_snapshot()
    if comp_before != comp_now:
        raise RuntimeError('Protected COMP895 artifacts changed')
    dump('logs/preservation.json', dict(status='PASS', extension_files=len(before),
        extension_bytes=sum(r['bytes'] for r in before.values()),
        comp895_files=len(comp_now), comp895_bytes=sum(r['bytes'] for r in comp_now.values()),
        extension_snapshot_sha256=sha256(OUT / 'logs/frozen_extension_before.json'),
        comp895_snapshot_sha256=sha256(BASE / 'logs/preservation_before.json'),
        exclusions=['new interpretation_v1 subtree', 'runtime dependencies', '__pycache__']))
    print(f'PASS: {len(before)} original extension files and {len(comp_now)} COMP895 files unchanged.')


if __name__ == '__main__':
    main()
