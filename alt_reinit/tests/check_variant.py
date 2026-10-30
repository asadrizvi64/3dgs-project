"""Static checks of the FasterGSReinit variant (part of the R0 flags-off checks, alt_reinit/PLAN.md).

    python alt_reinit/tests/check_variant.py trainer <stock Trainer.py at the base commit> <variant Trainer.py>
    python alt_reinit/tests/check_variant.py tree <variant checkout> [--base 1144604]

trainer: rename the stock Trainer.py (FasterGS -> FasterGSReinit, as make_reinit_variant.sh does), apply the two shared
         clock / save-state patches, and compare with the variant's Trainer.py after removing every
         '# >>> fgs-reinit' ... '# <<< fgs-reinit' block. They must be identical: the track's patch only inserts marked
         blocks. Also checks that every marked block is one the patch defines (init, callbacks, cbclock).
tree:    in the variant's git checkout, the rename commit must change file contents only by the rename, and the
         commits after it may touch only Trainer.py and add reinit_core.py / reinit_trainer.py.
Prints PASS/FAIL lines; exit status 1 on any failure.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VARIANT = 'FasterGSReinit'
SHARED_PATCHES = ('cluster/patches/fastergs_sparse_from.py', 'cluster/patches/fastergs_save_state.py')
BLOCK = re.compile(r'^[ \t]*# >>> fgs-reinit (\S+)\n.*?^[ \t]*# <<< fgs-reinit \1\n', re.S | re.M)
EXPECTED_BLOCKS = ['init', 'callbacks', 'cbclock']
ALLOWED_AFTER_RENAME = {'Trainer.py', 'reinit_core.py', 'reinit_trainer.py'}


def rename(text: str, variant: str = VARIANT) -> str:
    return text.replace('FasterGS', variant).replace('fastergs', variant.lower())


def unrename(text: str, variant: str = VARIANT) -> str:
    return text.replace(variant, 'FasterGS').replace(variant.lower(), 'fastergs')


def strip_blocks(text: str) -> tuple[str, list[str]]:
    names = BLOCK.findall(text)
    return BLOCK.sub('', text), names


def check_trainer(stock: Path, variant: Path) -> list[tuple[str, bool]]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        exp = Path(tmp) / 'Trainer.py'
        exp.write_text(rename(stock.read_text()))
        for p in SHARED_PATCHES:
            subprocess.run([sys.executable, str(REPO / p), str(exp)], check=True, capture_output=True)
        expected = exp.read_text()
    actual = variant.read_text()
    stripped, names = strip_blocks(actual)
    results.append((f'marked blocks are {EXPECTED_BLOCKS}', sorted(names) == sorted(EXPECTED_BLOCKS)))
    results.append(('no stray fgs-reinit markers outside the blocks', 'fgs-reinit' not in stripped))
    same = stripped == expected
    if not same:
        import difflib
        diff = ''.join(difflib.unified_diff(expected.splitlines(True), stripped.splitlines(True), 'expected', 'variant', n=1))
        print(diff[:4000])
    results.append(('variant Trainer.py minus marked blocks == renamed stock + clock + save-state patches', same))
    # every inserted block must be gated: init only parses FGS_REINIT when set; callbacks are inactive unless
    # _fgs_reinit_active / _fgs_audit_final_active; cbclock installs only with FGS_CBCLOCK
    blocks = dict((m.group(1), m.group(0)) for m in BLOCK.finditer(actual))
    results.append(('init block parses FGS_REINIT only when set', "if _fgs_spec else None" in blocks.get('init', '')))
    cb = blocks.get('callbacks', '')
    results.append(('callbacks are gated by _fgs_reinit_active / _fgs_audit_final_active',
                    cb.count("active='_fgs_reinit_active'") == 1 and cb.count("active='_fgs_audit_final_active'") == 1))
    results.append(('cbclock installs only with FGS_CBCLOCK', "environ.get('FGS_CBCLOCK')" in blocks.get('cbclock', '')))
    return results


def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout


def check_tree(repo: Path, base: str) -> list[tuple[str, bool]]:
    results = []
    log = git(repo, 'log', '--format=%H %s', f'{base}..HEAD').splitlines()
    rename_commits = [line.split()[0] for line in log if 'Rename FasterGS ->' in line]
    results.append(('exactly one rename commit after the base', len(rename_commits) == 1))
    if len(rename_commits) != 1:
        return results
    rc_ = rename_commits[0]
    results.append(('the rename commit is the first commit after the base',
                    git(repo, 'rev-parse', f'{rc_}^').strip() == git(repo, 'rev-parse', base).strip()))
    base_files = set(git(repo, 'ls-tree', '-r', '--name-only', base).split())
    ren_files = set(git(repo, 'ls-tree', '-r', '--name-only', rc_).split())
    results.append(('rename commit has the same files, renamed', {rename(f) for f in base_files} == ren_files))
    bad = []
    for f in sorted(base_files):
        a = subprocess.run(['git', '-C', str(repo), 'show', f'{base}:{f}'], capture_output=True).stdout
        b = subprocess.run(['git', '-C', str(repo), 'show', f'{rc_}:{rename(f)}'], capture_output=True).stdout
        try:
            if unrename(b.decode()) != a.decode():
                bad.append(f)
        except UnicodeDecodeError:
            if a != b:
                bad.append(f)
    results.append((f'rename commit changes contents only by the rename ({len(base_files)} files)', not bad))
    if bad:
        print('files differing beyond the rename:', bad[:20])
    changed = set(git(repo, 'diff', '--name-only', rc_, 'HEAD').split())
    results.append((f'after the rename only {sorted(ALLOWED_AFTER_RENAME)} change (got {sorted(changed)})',
                    changed <= ALLOWED_AFTER_RENAME))
    results.append(('tracked files unchanged since the last commit', git(repo, 'status', '--porcelain', '--untracked-files=no').strip() == ''))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('trainer')
    t.add_argument('stock')
    t.add_argument('variant')
    tr = sub.add_parser('tree')
    tr.add_argument('repo')
    tr.add_argument('--base', default='1144604')
    a = ap.parse_args()
    results = check_trainer(Path(a.stock), Path(a.variant)) if a.cmd == 'trainer' else check_tree(Path(a.repo), a.base)
    for name, ok in results:
        print(('PASS ' if ok else 'FAIL ') + name)
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == '__main__':
    main()
