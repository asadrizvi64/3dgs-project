"""Nsight Compute on the preprocess kernels (build_plus.sbatch STAGES=ncupre, step 5g): forward preprocess_cu and
preprocess_backward_cu of the stack (BASE, arm abcd) at step NCU_STEP, 2 steps per scene.

    python phase2/analyze_ncupre.py phase2/logs/run_<job> [--md out.md]

Reads q_<scene>_<arm>_<step>.raw.csv (ncu --page raw --csv) and prints, per scene and kernel, the mean over the 2 launches
of duration, DRAM bytes and bandwidth, speed-of-light percentages, occupancy, registers, load efficiency (sectors per
request; 4 is fully coalesced for 4-byte loads, 32 is fully scattered) and the 5 largest stall reasons. Then, from
<scene>_<arm>.pre_sass.csv (one preprocess backward launch, ncu --page source --print-source sass), the instruction and
stall-sample shares by opcode class. ncu locks the SM clock and serializes kernels: compare kernels, not absolute times.
CHECK fails if a file lacks 2 launches of each kernel.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

M = {
    'dur_us': 'gpu__time_duration.sum',
    'grid': 'launch__grid_size',
    'block': 'launch__block_size',
    'regs': 'launch__registers_per_thread',
    'inst': 'smsp__inst_executed.sum',
    'dram_rd': 'dram__bytes_read.sum',
    'dram_wr': 'dram__bytes_write.sum',
    'dram_pct': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'mem_pct': 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
    'sm_pct': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'l2_pct': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'issue_pct': 'sm__inst_issued.avg.pct_of_peak_sustained_active',
    'occ_theo': 'sm__maximum_warps_per_active_cycle_pct',
    'occ_ach': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'ld_sec': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'ld_req': 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'st_sec': 'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
    'st_req': 'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
    'thr_active': 'smsp__thread_inst_executed_per_inst_executed.ratio',
}
STALL = re.compile(r'smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$')
SCALE = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9, 'ns': 1e-3, 'nsecond': 1e-3, 'us': 1.0, 'usecond': 1.0,
         'ms': 1e3, 'msecond': 1e3}


def num(v):
    try:
        return float(v.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def kind(name):
    n = name.split('(')[0]
    return 'bwd' if 'preprocess_backward' in n else 'fwd' if 'preprocess' in n else None


def load(path):
    rows = list(csv.reader(open(path, newline='')))
    head, units, data = rows[0], rows[1], rows[2:]
    ki = head.index('Kernel Name')
    out = {'fwd': [], 'bwd': []}
    for r in data:
        k = kind(r[ki])
        if not k:
            continue
        d = {}
        for key, col in M.items():
            if col in head:
                v = num(r[head.index(col)])
                if v is not None:
                    v *= SCALE.get(units[head.index(col)], 1.0)
                d[key] = v
        d['stalls'] = {STALL.match(c).group(1): num(r[i]) or 0.0 for i, c in enumerate(head) if STALL.match(c)}
        d['kernel'] = r[ki].split('(')[0]
        out[k].append(d)
    return out


def mean(ds, key):
    vs = [d[key] for d in ds if d.get(key) is not None]
    return sum(vs) / len(vs) if vs else float('nan')


def opclass(op):
    if op.startswith(('LDG', 'LD.')) or op == 'LD':
        return 'global load'
    if op.startswith(('STG', 'ST.')) or op == 'ST':
        return 'global store'
    if op.startswith(('RED', 'ATOM', 'ATOMG')):
        return 'global atomic'
    if op.startswith(('LDS', 'STS', 'LDC', 'ULDC')):
        return 'shared/const load'
    if op.startswith('MUFU'):
        return 'MUFU (exp, rcp, sqrt)'
    if op.startswith(('FFMA', 'FMUL', 'FADD', 'FMNMX', 'FSET', 'FSETP', 'FSEL', 'FCHK')):
        return 'fp32 arithmetic'
    if op.startswith(('BRA', 'EXIT', 'BSYNC', 'BSSY', 'CALL', 'RET', 'WARPSYNC', 'BAR')):
        return 'control'
    return 'other (int, move, predicate)'


def sass(path):
    rows = list(csv.reader(open(path, newline='')))
    hi = next(i for i, r in enumerate(rows) if 'Address' in r and 'Source' in r)
    h = rows[hi]
    data = [r for r in rows[hi + 1:] if len(r) == len(h) and r[h.index('Address')].startswith('0x')]
    n = lambda v: int(float(v.replace(',', '') or 0))
    seen, ins = set(), []
    for r in data:
        a = r[h.index('Address')]
        if a in seen:
            break
        seen.add(a)
        src = r[h.index('Source')].strip()
        op = re.sub(r'^@!?U?P\w+\s+', '', src).split()[0] if src else '?'
        ins.append((op, n(r[h.index('Instructions Executed')]), n(r[h.index('Warp Stall Sampling (All Samples)')])))
    ic, sc = Counter(), Counter()
    for op, i, s in ins:
        ic[opclass(op)] += i
        sc[opclass(op)] += s
    top = Counter()
    for op, i, s in ins:
        top[op.split('.')[0]] += s
    return ic, sc, top


def main(argv):
    md = None
    if '--md' in argv:
        i = argv.index('--md'); md = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    run = Path(argv[0])
    lines, ok = [], True
    for f in sorted(run.glob('q_*_*_*.raw.csv')):
        _, scene, arm, step = f.name[:-len('.raw.csv')].split('_')
        by = load(f)
        for k in ('fwd', 'bwd'):
            if len(by[k]) != 2:
                ok = False
                lines.append(f'CHECK FAILED {f.name}: {len(by[k])} {k} launches (want 2)')
        lines.append(f'### {scene} ({arm}, step {step})')
        lines.append('')
        lines.append('| metric | forward preprocess | preprocess backward |')
        lines.append('|---|---|---|')
        f_, b_ = by['fwd'], by['bwd']
        rows = [
            ('duration (us)', 'dur_us', '%.1f'), ('threads launched', None, None), ('registers / thread', 'regs', '%.0f'),
            ('instructions (M)', 'inst', None), ('DRAM read (MB)', 'dram_rd', None), ('DRAM write (MB)', 'dram_wr', None),
            ('DRAM bandwidth (GB/s)', None, None), ('DRAM throughput (% peak)', 'dram_pct', '%.1f'),
            ('memory SOL (%)', 'mem_pct', '%.1f'), ('L2 SOL (%)', 'l2_pct', '%.1f'), ('SM SOL (%)', 'sm_pct', '%.1f'),
            ('issue slots busy (%)', 'issue_pct', '%.1f'), ('occupancy theo / achieved (%)', None, None),
            ('global load sectors / request', None, None), ('global store sectors / request', None, None),
            ('active threads / warp instr.', 'thr_active', '%.1f'),
        ]
        for label, key, fmt in rows:
            cells = []
            for ds in (f_, b_):
                if label == 'threads launched':
                    c = f'{mean(ds, "grid") * mean(ds, "block") / 1e6:.2f} M'
                elif label == 'instructions (M)':
                    c = f'{mean(ds, "inst") / 1e6:.1f}'
                elif label.startswith('DRAM read') or label.startswith('DRAM write'):
                    c = f'{mean(ds, key) / 1e6:.0f}'
                elif label == 'DRAM bandwidth (GB/s)':
                    c = f'{(mean(ds, "dram_rd") + mean(ds, "dram_wr")) / (mean(ds, "dur_us") * 1e-6) / 1e9:.0f}'
                elif label.startswith('occupancy'):
                    c = f'{mean(ds, "occ_theo"):.1f} / {mean(ds, "occ_ach"):.1f}'
                elif label.startswith('global load'):
                    c = f'{mean(ds, "ld_sec") / mean(ds, "ld_req"):.1f}'
                elif label.startswith('global store'):
                    c = f'{mean(ds, "st_sec") / mean(ds, "st_req"):.1f}'
                else:
                    c = fmt % mean(ds, key)
                cells.append(c)
            lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
        lines.append('')
        for k, ds in (('forward preprocess', f_), ('preprocess backward', b_)):
            st = Counter()
            for d in ds:
                for s, v in d['stalls'].items():
                    st[s] += v / len(ds)
            tot = sum(st.values())
            lines.append(f'- {k} stalls (warps per issued instr., total {tot:.1f}): '
                         + ', '.join(f'{s} {v:.2f}' for s, v in st.most_common(5)))
        sp = run / f'{scene}_{arm}.pre_sass.csv'
        if sp.exists() and sp.stat().st_size > 0:
            ic, sc, top = sass(sp)
            ti, ts = sum(ic.values()) or 1, sum(sc.values()) or 1
            lines.append('')
            lines.append(f'Preprocess backward by instruction class (one launch, `{sp.name}`):')
            lines.append('')
            lines.append('| class | instructions (M) | share | stall samples share |')
            lines.append('|---|---|---|---|')
            for c, v in ic.most_common():
                lines.append(f'| {c} | {v / 1e6:.2f} | {100 * v / ti:.1f}% | {100 * sc[c] / ts:.1f}% |')
            lines.append('')
            lines.append('- most stall samples by opcode: ' + ', '.join(f'{o} {100 * v / ts:.1f}%' for o, v in top.most_common(6)))
        lines.append('')
    lines.append('CHECK ' + ('OK' if ok else 'FAILED'))
    text = '\n'.join(lines)
    print(text)
    if md:
        Path(md).write_text(text + '\n', encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
