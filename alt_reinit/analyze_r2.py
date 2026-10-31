"""R2 analysis (alternative track R, alt_reinit/PLAN.md, amendment 2026-10-31): time-to-quality of the four arms.

    python3 alt_reinit/analyze_r2.py alt_reinit/logs/gs-alt-r2-garden-<job>.out alt_reinit/logs/gs-alt-r2-room-<job>.out

Reads the ar2_<scene>_<arm>_s<seed> runs in the job logs (SEGCLOCK, CBCLOCK, EVAL, REINIT lines) and, next to each log,
run_<job>/runs/<name>/fgs_eval/metrics.jsonl (per-image values of every evaluation).

- **Track clock at step s:** the SEGCLOCK cumulative time at s, plus every CBCLOCK call at iterations < s (the method's
  callbacks between steps, the reinit included). Evaluations are outside it.
- **Target, per scene:** the `ref` arm's 3-seed means at 30k: PSNR >=, SSIM >=, LPIPS <=. No allowance.
- **Candidate step of an arm:** the first evaluation at which the arm's 3-seed mean curve meets all three. Its time is
  the 3-seed mean track clock at that step. `ref` has one by 30k by construction.
- **Near-tie:** the margin on any metric at the candidate (or the deficit at 30k, if there is no candidate) is smaller
  than the reference's seed standard deviation at 30k. The result is then inconclusive until confirmed.
- **Images:** an image fails in a run when its PSNR is more than 3 dB below that image's median over the ref runs at 30k.
  A new failure is one on an image that never fails in the ref runs. Images more than 1 dB below are listed. Each arm
  run is checked at the arm's candidate step, or at 30k when there is none.
- **Outcome per arm:**
  - **improves time-to-quality:** a candidate, earlier in track time than ref's, and no new image failure;
  - **keeps quality but does not improve time:** a candidate, not earlier, and no new image failure;
  - **reduces quality:** no candidate by 30k, or a new image failure.

  Any candidate is pending confirmation (R4: a run with a single evaluation at that step).

Exit status 1 when a run is missing, failed, or lacks evaluations.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as stt
import sys
from collections import defaultdict

ARMS = ('ref', 'stop', 'same', 'target')
ARM_LABEL = {'ref': 'ref', 'stop': 'stop', 'same': 'reinit-same', 'target': 'reinit-target'}
METRICS = ('psnr', 'ssim', 'lpips')
BETTER = {'psnr': 1, 'ssim': 1, 'lpips': -1}
NAME = re.compile(r'^ar2_([a-z]+)_([a-z]+)_s(\d+)$')


def parse_log(path: str) -> list[dict]:
    text = open(path, encoding='utf-8', errors='replace').read()
    hd = re.search(r'^node=(\S+) job=(\S+) gpu=.*? repo=(\S+)', text, re.M)
    job = hd.group(2) if hd else 'local'
    arch = os.path.join(os.path.dirname(path), f'run_{job}', 'runs')
    runs, cur = [], None
    for line in text.splitlines():
        m = re.match(r'^##### (\S+): method=(\S+) scene=(\S+) seed=(\d+) iters=(\d+) reinit=(\S+)', line)
        if m:
            cur = dict(name=m.group(1), method=m.group(2), scene=m.group(3), seed=int(m.group(4)), iters=int(m.group(5)),
                       spec=m.group(6), job=job, repo=hd.group(3) if hd else '', seg={}, segN={}, cb=[], ev={}, reinit={},
                       failed=False, archive=os.path.join(arch, m.group(1)))
            runs.append(cur)
            continue
        if line.startswith('##########'):
            cur = None
            continue
        if cur is None:
            continue
        if m := re.match(r'^SEGCLOCK step=(\d+) seg_s=\S+ cum_s=(\S+) .*?N=(\d+)', line):
            cur['seg'][int(m.group(1))] = float(m.group(2))
            cur['segN'][int(m.group(1))] = int(m.group(3))
        elif m := re.match(r'^CBCLOCK name=(\S+) iteration=(\d+) s=(\S+)', line):
            cur['cb'].append((m.group(1), int(m.group(2)), float(m.group(3))))
        elif m := re.match(r'^EVAL step=(\d+) psnr=(\S+) ssim=(\S+) lpips=(\S+) n_images=(\d+) N=(\d+) t_eval=(\S+) check=(\S+)', line):
            cur['ev'][int(m.group(1))] = dict(psnr=float(m.group(2)), ssim=float(m.group(3)), lpips=float(m.group(4)),
                                              n=int(m.group(5)), N=int(m.group(6)), t_eval=float(m.group(7)), check=m.group(8))
        elif m := re.match(r'^REINIT iteration=(\d+) .*?t_total=(\S+)', line):
            cur['reinit']['t_total'] = float(m.group(2))
        elif m := re.match(r'^REINIT_COUNTS iteration=\d+ n_before=(\d+) .*?n_covered=(\d+) n_survivors=(\d+) n_added=(\d+) n_after=(\d+)', line):
            cur['reinit'].update(n_before=int(m.group(1)), n_removed=int(m.group(2)), n_survivors=int(m.group(3)),
                                 n_added=int(m.group(4)), n_after=int(m.group(5)))
        elif line.startswith(f'FAILED {cur["name"]}'):
            cur['failed'] = True
        elif m := re.match(r'^RESULT \S+ PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+)', line):
            cur['final'] = dict(psnr=float(m.group(1)), ssim=float(m.group(2)), lpips=float(m.group(3)), N=int(m.group(4)))
    for r in runs:
        r['per_image'] = {}
        f = os.path.join(r['archive'], 'fgs_eval', 'metrics.jsonl')
        if os.path.exists(f):
            for line in open(f):
                j = json.loads(line)
                r['per_image'][j['step']] = j['per_image']
    return runs


def clock(run: dict, step: int) -> float | None:
    if step not in run['seg']:
        return None
    return run['seg'][step] + sum(s for _, it, s in run['cb'] if it < step)


def mean_sd(xs):
    xs = list(xs)
    return (stt.mean(xs), stt.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float('nan'), float('nan'))


def meets(m: dict, tgt: dict) -> bool:
    return m['psnr'] >= tgt['psnr'] and m['ssim'] >= tgt['ssim'] and m['lpips'] <= tgt['lpips']


def margins(m: dict, tgt: dict) -> dict:
    return {k: BETTER[k] * (m[k] - tgt[k]) for k in METRICS}


def analyze(runs: list[dict]) -> tuple[str, list[str], list[tuple]]:
    problems, confirm = [], []
    by = defaultdict(lambda: defaultdict(dict))   # scene -> arm -> seed -> run
    for r in runs:
        m = NAME.match(r['name'])
        if not m:
            continue
        scene, arm, seed = m.group(1), m.group(2), int(m.group(3))
        if arm not in ARMS:
            problems.append(f'{r["name"]}: unknown arm')
            continue
        by[scene][arm][seed] = r
        if r['failed']:
            problems.append(f'{r["name"]}: FAILED')
        if r['iters'] != 30000 or 30000 not in r['ev']:
            problems.append(f'{r["name"]}: no evaluation at 30000')
        if any(e['check'] != 'pass' for e in r['ev'].values()):
            problems.append(f'{r["name"]}: an EVAL check failed')
    out = ['# Alternative track R, R2 analysis (k = 5000)', '',
           f'Jobs: {", ".join(sorted({r["job"] for r in runs}))}; repo {", ".join(sorted({r["repo"] for r in runs}))}.',
           'Track clock = SEGCLOCK (training steps) + CBCLOCK (callbacks between steps, reinit included); evaluations excluded.',
           'Target = ref 3-seed means at 30k; candidate = first evaluation where the arm\'s 3-seed means meet all three. '
           'Any candidate is pending R4 confirmation.', '']
    for scene in sorted(by):
        arms = by[scene]
        seeds = sorted(set().union(*[set(a) for a in arms.values()]))
        for arm in ARMS:
            if set(arms.get(arm, {})) != set(seeds):
                problems.append(f'{scene} {arm}: seeds {sorted(arms.get(arm, {}))}, expected {seeds}')
        ref = arms.get('ref', {})
        if not ref or any(30000 not in r['ev'] for r in ref.values()):
            out += [f'## {scene}', '', 'ref runs incomplete: no target.', '']
            continue
        tgt = {k: stt.mean(r['ev'][30000][k] for r in ref.values()) for k in METRICS}
        tsd = {k: mean_sd(r['ev'][30000][k] for r in ref.values())[1] for k in METRICS}
        # per-image reference: median over ref runs at 30k
        ref_img = [r['per_image'].get(30000, {}).get('psnr') for r in ref.values()]
        img_med = [stt.median(v) for v in zip(*ref_img)] if all(ref_img) else None
        ref_fail = set()
        if img_med:
            for v in ref_img:
                ref_fail |= {i for i, (p, md) in enumerate(zip(v, img_med)) if p < md - 3}
        out += [f'## {scene}', '',
                f'Target (ref, {len(ref)} seeds, at 30k): PSNR {tgt["psnr"]:.4f} (sd {tsd["psnr"]:.4f}), '
                f'SSIM {tgt["ssim"]:.5f} (sd {tsd["ssim"]:.5f}), LPIPS {tgt["lpips"]:.5f} (sd {tsd["lpips"]:.5f}). '
                f'Images failing (> 3 dB below the ref median) in a ref run: {sorted(ref_fail) or "none"}.', '']
        rows, res = [], {}
        for arm in ARMS:
            rs = arms.get(arm, {})
            if not rs:
                continue
            steps = sorted(set.intersection(*[set(r['ev']) for r in rs.values()]))
            curve = {s: {k: stt.mean(r['ev'][s][k] for r in rs.values()) for k in METRICS} for s in steps}
            cand = next((s for s in steps if meets(curve[s], tgt)), None)
            at = cand if cand is not None else (30000 if 30000 in curve else (steps[-1] if steps else None))
            t_c = [clock(r, cand) for r in rs.values()] if cand else []
            t30 = [clock(r, 30000) for r in rs.values()]
            mg = margins(curve[at], tgt) if at else {}
            near = bool(mg) and any(abs(mg[k]) < tsd[k] for k in METRICS)
            # images, at the candidate (or 30k)
            new_fail, low = set(), set()
            if img_med and at:
                for r in rs.values():
                    v = r['per_image'].get(at, {}).get('psnr')
                    if not v:
                        problems.append(f'{r["name"]}: no per-image values at {at}')
                        continue
                    for i, (p, md) in enumerate(zip(v, img_med)):
                        if p < md - 3 and i not in ref_fail:
                            new_fail.add(i)
                        if p < md - 1:
                            low.add(i)
            per_seed = []
            for sd, r in sorted(rs.items()):
                c = next((s for s in sorted(r['ev']) if meets(r['ev'][s], tgt)), None)
                per_seed.append((sd, c, clock(r, c) if c else None))
            res[arm] = dict(cand=cand, t=mean_sd([x for x in t_c if x is not None])[0] if cand else None, near=near,
                            new_fail=new_fail)
            fin = {k: mean_sd(r['ev'][30000][k] for r in rs.values() if 30000 in r['ev']) for k in METRICS}
            n30 = mean_sd(r['ev'][30000]['N'] for r in rs.values() if 30000 in r['ev'])[0]
            rows.append((arm, cand, res[arm]['t'], mean_sd([x for x in t30 if x is not None])[0], fin, n30, mg, near,
                         sorted(new_fail), sorted(low), per_seed, rs))
        t_ref = res.get('ref', {}).get('t')
        out += ['| arm | candidate step | track s at candidate | track s to 30k | 30k PSNR | 30k SSIM | 30k LPIPS | N at 30k | margins at candidate (or 30k) | outcome |',
                '|---|---|---|---|---|---|---|---|---|---|']
        for arm, cand, t, t30, fin, n30, mg, near, nf, low, per_seed, rs in rows:
            if arm == 'ref':
                outcome = 'reference'
            elif cand is None or nf:
                outcome = 'reduces quality' + (' (new image failures)' if nf and cand else '')
            elif t_ref is not None and t < t_ref:
                outcome = 'improves time-to-quality (pending R4)'
            else:
                outcome = 'keeps quality, does not improve time (pending R4)'
            if near and arm != 'ref':   # ref meets its own 30k mean with zero margin by construction
                outcome += '; near-tie: inconclusive'
            res[arm]['outcome'] = outcome
            mtxt = ', '.join(f'{k} {mg[k]:+.4f}' for k in METRICS) if mg else '-'
            t_txt = f'{t:.1f}' if t is not None else '-'
            out.append(f'| {ARM_LABEL[arm]} | {cand or "none by 30k"} | {t_txt} | {t30:.1f} | '
                       f'{fin["psnr"][0]:.3f} ± {fin["psnr"][1]:.3f} | {fin["ssim"][0]:.4f} ± {fin["ssim"][1]:.4f} | '
                       f'{fin["lpips"][0]:.4f} ± {fin["lpips"][1]:.4f} | {n30:,.0f} | {mtxt} | {outcome} |')
            if cand:
                confirm.append((scene, arm, cand))
        out += ['', 'Per seed (exploratory only): first evaluation where that seed alone meets the target, and its track time:', '']
        for arm, cand, t, t30, fin, n30, mg, near, nf, low, per_seed, rs in rows:
            out.append(f'- {ARM_LABEL[arm]}: ' + '; '.join(f's{sd}: {c or "none"}' + (f' at {tc:.1f} s' if tc else '')
                                                         for sd, c, tc in per_seed))
        out += ['', 'Images (index in the test set) more than 1 dB below the ref median, at the candidate step or 30k; new '
                'failures (> 3 dB, never failing in ref) in bold:', '']
        for arm, cand, t, t30, fin, n30, mg, near, nf, low, per_seed, rs in rows:
            out.append(f'- {ARM_LABEL[arm]}: ' + (', '.join(f'**{i}**' if i in nf else str(i) for i in low) or 'none'))
        out += ['', 'Reinitialization (per seed): cost, rows before / removed / added / after:', '']
        for arm in ('same', 'target'):
            for sd, r in sorted(arms.get(arm, {}).items()):
                ri = r['reinit']
                if ri:
                    out.append(f'- {ARM_LABEL[arm]} s{sd}: {ri.get("t_total", float("nan")):.2f} s; {ri.get("n_before")} / '
                               f'{ri.get("n_removed")} / {ri.get("n_added")} / {ri.get("n_after")}')
        tev = [e['t_eval'] for a in arms.values() for r in a.values() for e in r['ev'].values()]
        if tev:
            out += ['', f'Evaluator: {len(tev)} calls, median {stt.median(tev):.2f} s, max {max(tev):.2f} s (outside the clock).']
        # PLAN.md readings
        out += ['', 'Pairwise at 30k (3-seed mean differences, PLAN.md R2 reading):', '']
        fin_of = {a[0]: a[4] for a in rows}
        for a, b in (('stop', 'ref'), ('same', 'stop'), ('target', 'same'), ('target', 'ref')):
            if a in fin_of and b in fin_of:
                d = {k: fin_of[a][k][0] - fin_of[b][k][0] for k in METRICS}
                out.append(f'- {ARM_LABEL[a]} - {ARM_LABEL[b]}: PSNR {d["psnr"]:+.3f}, SSIM {d["ssim"]:+.4f}, LPIPS {d["lpips"]:+.4f}')
        out.append('')
    if confirm:
        out += ['## R4 confirmation needed', '', 'Rerun 3 seeds each with `FGS_EVAL=at=<step>` (no other intermediate evaluation):', '']
        out += [f'- {s} {ARM_LABEL[a]} at step {c}' for s, a, c in confirm]
        out.append('')
    if problems:
        out += ['## Problems', ''] + [f'- {p}' for p in problems] + ['']
    return '\n'.join(out), problems, confirm


NAME4 = re.compile(r'^ar4_([a-z]+)_([a-z]+)_s(\d+)$')


def confirmation(runs: list[dict]) -> tuple[str, list[str]]:
    """R4: ar4_<scene>_<arm>_s<seed> runs, each with one evaluation (FGS_EVAL=at=step). The target is the ref
    confirmation runs' mean at their own step; an arm's candidate is confirmed when its runs' mean at its step meets all
    three metrics against it. Failures to reproduce are reported with the metric that failed."""
    by, problems = defaultdict(lambda: defaultdict(dict)), []
    for r in runs:
        m = NAME4.match(r['name'])
        if not m:
            continue
        by[m.group(1)][m.group(2)][int(m.group(3))] = r
        if r['failed'] or len(r['ev']) != 1 or any(e['check'] != 'pass' for e in r['ev'].values()):
            problems.append(f'{r["name"]}: failed, or not exactly one passing evaluation ({sorted(r["ev"])})')
    if not by:
        return '', problems
    out = ['## R4 confirmation (single evaluation per run, no other intermediate evaluation)', '']
    for scene in sorted(by):
        arms = by[scene]
        ref = arms.get('ref')
        if not ref:
            out += [f'### {scene}', '', 'no ref confirmation runs: no target', '']
            continue
        rstep = {next(iter(r['ev'])) for r in ref.values()}
        if len(rstep) != 1:
            problems.append(f'{scene} ref confirmation runs disagree on the step: {rstep}')
            continue
        rstep = rstep.pop()
        tgt = {k: stt.mean(r['ev'][rstep][k] for r in ref.values()) for k in METRICS}
        t_ref = stt.mean(clock(r, rstep) for r in ref.values())
        out += [f'### {scene}', '', f'ref confirmation ({len(ref)} seeds) at {rstep}: PSNR {tgt["psnr"]:.4f}, SSIM '
                f'{tgt["ssim"]:.5f}, LPIPS {tgt["lpips"]:.5f}, track {t_ref:.1f} s', '',
                '| arm | step | seeds | PSNR | SSIM | LPIPS | track s | verdict |', '|---|---|---|---|---|---|---|---|']
        for arm in ARMS:
            if arm == 'ref' or arm not in arms:
                continue
            rs = arms[arm]
            st = {next(iter(r['ev'])) for r in rs.values()}
            if len(st) != 1:
                problems.append(f'{scene} {arm}: confirmation runs disagree on the step: {st}')
                continue
            st = st.pop()
            mm = {k: stt.mean(r['ev'][st][k] for r in rs.values()) for k in METRICS}
            t = stt.mean(clock(r, st) for r in rs.values())
            failed = [k for k in METRICS if BETTER[k] * (mm[k] - tgt[k]) < 0]
            if failed:
                verdict = 'not reproduced: ' + ', '.join(failed) + ' below target'
            elif t < t_ref:
                verdict = 'confirmed: meets the target, earlier than ref'
            else:
                verdict = 'confirmed quality, not earlier than ref'
            out.append(f'| {ARM_LABEL[arm]} | {st} | {len(rs)} | {mm["psnr"]:.4f} | {mm["ssim"]:.5f} | {mm["lpips"]:.5f} | '
                       f'{t:.1f} | {verdict} |')
        out.append('')
    return '\n'.join(out), problems


def main() -> None:
    runs = []
    for p in sys.argv[1:]:
        runs += parse_log(p)
    text, problems, _ = analyze(runs) if any(NAME.match(r['name']) for r in runs) else ('', [], [])
    ctext, cproblems = confirmation(runs)
    problems += cproblems
    print(text + ('\n' + ctext if ctext else ''))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
