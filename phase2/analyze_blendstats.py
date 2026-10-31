"""Blend-kernel waste from the counters of phase2/blend_stats.patch (build_plus.sbatch STAGES=blendstats).

    python phase2/analyze_blendstats.py phase2/logs/run_<job>/p_*_abcstats_s0.stats [--md out.md]

Each BLENDSTATS_FWD / _BWD line sums 50 training steps from `start`. Per window it reports:
  forward (blend_cu, one warp per 8x4 subtile, one lane per pixel)
    bbox->ellipse  share of splat-subtile pairs that pass the bounding-box test but fail the exact ellipse test
                   (the work step 2 would remove)
    evals/px/step  pixel evaluations per image pixel per step;  contrib  share of evaluations with alpha >= 1/255
    bad evals      share of evaluations spent on ellipse-rejected pairs
    lane use       evaluations / (32 x warp loop iterations): lanes idle because their pixel finished early
    idle chunks    share of 32-splat chunks a warp skips because all its pixels finished but the tile goes on
  backward (macro kernel, one warp per bucket of 32 splats, 6 pixel rows of 32 lanes)
    active         buckets past the per-tile cut-off / launched
    rows/step      warp row executions per splat step;  bad rows  share whose evaluating lanes all lie in
                   ellipse-rejected subtiles (removed by step 2)
    bad evals, contrib  as above;  term idle  lane slots in executed rows whose pixel already finished
    idle steps     splat steps with no row executed
missed: contributions (alpha >= 1/255) inside ellipse-rejected pairs. If not 0, the ellipse test is not conservative
at the cut-off and step 2 needs a margin to keep images bit-identical. CHECK fails only on missing windows.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def parse(path):
    out = {'FWD': {}, 'BWD': {}}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'BLENDSTATS_(FWD|BWD) start=(\d+) calls=(\d+) (.*)', line)
        if not m:
            continue
        vals = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', m.group(4))}
        vals['calls'] = int(m.group(3))
        out[m.group(1)][int(m.group(2))] = vals
    return out


def pct(a, b):
    return f'{100.0 * a / b:.1f}%' if b else 'n/a'


def main(argv):
    md_out = None
    if '--md' in argv:
        i = argv.index('--md')
        md_out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    lines, ok = [], True
    for path in argv:
        name = Path(path).stem
        d = parse(path)
        lines.append(f'### {name}\n')
        lines.append('| step | bbox->ellipse | evals/px/step | contrib | bad evals | lane use | idle chunks | missed |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for st, f in sorted(d['FWD'].items()):
            px = f['pixels'] / f['calls'] if f['calls'] else 0
            lines.append(f"| {st} | {pct(f['bbox_pairs'] - f['ellipse_pairs'], f['bbox_pairs'])} | "
                         f"{f['evals'] / f['calls'] / px if px else 0:.1f} | {pct(f['contrib'], f['evals'])} | "
                         f"{pct(f['evals_bad'], f['evals'])} | {pct(f['evals'], 32 * f['warp_iters'])} | "
                         f"{pct(f['idle_chunks'], f['chunks'] + f['idle_chunks'])} | {f['contrib_bad']} of {f['contrib']:.2e} |")
        lines.append('')
        lines.append('| step | active buckets | rows/step | bad rows | bad evals | contrib | term idle | idle steps | missed |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for st, b in sorted(d['BWD'].items()):
            lines.append(f"| {st} | {pct(b['active'], b['launched'])} | {b['rows'] / b['steps'] if b['steps'] else 0:.2f} | "
                         f"{pct(b['rows_bad'], b['rows'])} | {pct(b['evals_bad'], b['evals'])} | {pct(b['contrib'], b['evals'])} | "
                         f"{pct(b['term_idle'], 32 * b['rows'])} | {pct(b['steps_idle'], b['steps'])} | {b['contrib_bad']} of {b['contrib']:.2e} |")
        lines.append('')
        if not d['FWD'] or not d['BWD']:
            ok = False
            lines.append(f'CHECK FAILED {name}: missing windows (fwd {len(d["FWD"])}, bwd {len(d["BWD"])})\n')
    lines.append('CHECK ' + ('PASSED: all windows present' if ok else 'FAILED'))
    text = '\n'.join(lines)
    print(text)
    if md_out:
        Path(md_out).write_text(text + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
