"""Nsight Compute blend-kernel comparison (build_plus.sbatch STAGES=ncublend): A+B+C vs FasterGSPlusCull.

    python phase2/analyze_ncublend.py phase2/logs/run_<job> [--md out.md]

Reads u_<scene>_<arm>_<step>.raw.csv (ncu --page raw --csv; 2 forward blends + 2 macro backwards per file) and prints,
per scene and kernel, the mean over the 2 launches of: duration, instructions executed, issue slots busy, IPC,
registers, theoretical / achieved occupancy, warp cycles per issued instruction, the 5 largest stall reasons
(warps per issued instruction), L1 and DRAM throughput, and the culling arm's change. ncu locks the SM clock (1.6 GHz on
H100) and serializes kernels, so durations are ~20% above training values; compare arms, not absolute times.
CHECK fails if a file lacks 2 launches of each kernel.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

M = {
    'dur_us': 'gpu__time_duration.sum',
    'inst': 'smsp__inst_executed.sum',
    'issue_pct': 'sm__inst_issued.avg.pct_of_peak_sustained_active',
    'ipc': 'sm__inst_executed.avg.per_cycle_active',
    'regs': 'launch__registers_per_thread',
    'occ_theo': 'sm__maximum_warps_per_active_cycle_pct',
    'occ_ach': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'cpi': 'smsp__average_warp_latency_per_inst_issued.ratio',
    'l1_pct': 'l1tex__throughput.avg.pct_of_peak_sustained_active',
    'dram_pct': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'thr_active': 'smsp__thread_inst_executed_per_inst_executed.ratio',
}
STALL = re.compile(r'smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$')


def num(v):
    try:
        return float(v.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def load(path):
    rows = list(csv.reader(open(path, newline='')))
    head, units, data = rows[0], rows[1], rows[2:]
    ki = head.index('Kernel Name')
    out = {'fwd': [], 'bwd': []}
    for r in data:
        k = 'fwd' if 'blend_cu' in r[ki] else 'bwd' if 'blend_backward_macro_cu' in r[ki] else None
        if not k:
            continue
        d = {}
        for name, col in M.items():
            if col in head:
                v = num(r[head.index(col)])
                if v is not None and name == 'dur_us':
                    v *= {'ns': 1e-3, 'nsecond': 1e-3, 'us': 1.0, 'usecond': 1.0, 'ms': 1e3, 'msecond': 1e3}.get(units[head.index(col)], 1.0)
                d[name] = v
        d['stalls'] = {STALL.match(c).group(1): num(r[i]) or 0.0 for i, c in enumerate(head) if STALL.match(c)}
        d['kernel'] = r[ki].split('(')[0]
        out[k].append(d)
    return out


def mean(ds, key):
    vs = [d[key] for d in ds if d.get(key) is not None]
    return sum(vs) / len(vs) if vs else float('nan')


def main(argv):
    md = None
    if '--md' in argv:
        i = argv.index('--md'); md = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    run = Path(argv[0])
    files = sorted(run.glob('u_*_*_*.raw.csv'))
    by = {}
    for f in files:
        _, scene, arm, step = f.name[:-len('.raw.csv')].split('_')
        by[(scene, arm)] = load(f)
    lines, ok = [], True
    for (scene, arm), d in sorted(by.items()):
        for k in ('fwd', 'bwd'):
            if len(d[k]) != 2:
                ok = False; lines.append(f'CHECK FAILED {scene} {arm} {k}: {len(d[k])} launches')
    for scene in sorted({s for s, _ in by}):
        for k, title in (('fwd', 'forward blend_cu'), ('bwd', 'backward macro')):
            a, c = by.get((scene, 'abc'), {}).get(k, []), by.get((scene, 'abce'), {}).get(k, [])
            if not a or not c:
                continue
            lines.append(f'### {scene}: {title}  ({a[0]["kernel"]} vs {c[0]["kernel"]})\n')
            lines.append('| metric | A+B+C | culling | change |')
            lines.append('|---|---|---|---|')
            for name, label, fmt in (('dur_us', 'duration (us, ncu clock)', '{:.0f}'), ('inst', 'warp instructions executed (M)', '{:.1f}'),
                                     ('issue_pct', 'issue slots busy (%)', '{:.1f}'), ('ipc', 'IPC (per SM)', '{:.2f}'),
                                     ('cpi', 'warp cycles per issued instruction', '{:.2f}'), ('thr_active', 'active threads per instruction', '{:.1f}'),
                                     ('regs', 'registers per thread', '{:.0f}'), ('occ_theo', 'theoretical occupancy (%)', '{:.1f}'),
                                     ('occ_ach', 'achieved occupancy (%)', '{:.1f}'), ('l1_pct', 'L1/TEX throughput (%)', '{:.1f}'),
                                     ('dram_pct', 'DRAM throughput (%)', '{:.1f}')):
                va, vc = mean(a, name), mean(c, name)
                if name == 'inst':
                    va, vc = va / 1e6, vc / 1e6
                ch = f'{100 * (vc / va - 1):+.1f}%' if va else 'n/a'
                lines.append(f'| {label} | {fmt.format(va)} | {fmt.format(vc)} | {ch} |')
            sa = {s: sum(d['stalls'][s] for d in a) / len(a) for s in a[0]['stalls']}
            sc = {s: sum(d['stalls'][s] for d in c) / len(c) for s in c[0]['stalls']}
            top = sorted(sa, key=lambda s: -max(sa[s], sc.get(s, 0)))[:6]
            lines.append('| **stalls** (warps per issued instruction) | | | |')
            for s in top:
                lines.append(f'| stall {s} | {sa[s]:.2f} | {sc.get(s, 0):.2f} | {sc.get(s, 0) - sa[s]:+.2f} |')
            lines.append('')
    lines.append('CHECK ' + ('PASSED: 2 launches of each kernel per file' if ok else 'FAILED'))
    text = '\n'.join(lines)
    print(text)
    if md:
        Path(md).write_text(text + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
