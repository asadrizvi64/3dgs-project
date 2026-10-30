"""R1 audit analysis (alt_reinit/PLAN.md): validates every audit run of an alt_reinit/build_reinit.sbatch job and writes
the per-scene tables and the R1 gates as markdown.

    python alt_reinit/analyze_audit.py alt_reinit/logs/<job-name>-<job>.out [more logs] [--stages phase2/stages/psnr_vs_time.csv]

Reads the job log (run headers, SEGCLOCK / CBCLOCK / REINIT* / RESULT / FAILED lines) and the archived audit files in
alt_reinit/logs/run_<job>/runs/<run>/reinit_audit/k_*/ (stats.json, agreement_final.json, masks/*.jpg).

Clock: warm-up to k = SEGCLOCK cum_s at step k (CUDA events of the training steps before k) + the CBCLOCK time of the
method's between-step callbacks at iterations <= k (densify, Morton ordering, ...; the audit's own reinit call is a dry
run and is excluded). Reinit cost = the timed parts of the REINIT line. The reference estimate is stock's time to
stock's own 30k PSNR in the seed-0 stage runs (phase2/stages, SEGCLOCK only); R2 replaces it with this track's runs.
Exit status 1 when a run fails validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATUS = ('kept', 'empty', 'low_alpha', 'mixture', 'geometry')


def parse_log(path: str):
    text = open(path, encoding='utf-8', errors='replace').read()
    hd = re.search(r'^node=(\S+) job=(\S+) gpu=.*? repo=(\S+)', text, re.M)
    if not hd:
        sys.exit(f'{path}: no job header line')
    node, job, repo = hd.groups()
    runs, cur = [], None
    for line in text.splitlines():
        m = re.match(r'##### (\S+): method=(\S+) scene=(\S+) seed=(\d+) iters=(\d+) reinit=(\S+) cbclock=(\d)', line)
        if m:
            cur = dict(name=m.group(1), method=m.group(2), scene=m.group(3), seed=int(m.group(4)), iters=int(m.group(5)),
                       reinit=None if m.group(6) == '-' else m.group(6), cbclock=m.group(7) == '1', lines=[])
            runs.append(cur)
            continue
        if line.startswith('#####'):
            cur = None
        if cur is not None:
            cur['lines'].append(line)
    return dict(node=node, job=job, repo=repo, runs=runs, archive=Path(path).parent / f'run_{job}' / 'runs')


def clock(run):
    """SEGCLOCK cum by step, CBCLOCK entries (name, iteration, s)."""
    seg = {int(m.group(1)): float(m.group(2)) for m in
           (re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+)', ln) for ln in run['lines']) if m}
    cb = [(m.group(1), int(m.group(2)), float(m.group(3))) for m in
          (re.match(r'CBCLOCK name=(\S+) iteration=(\d+) s=(\S+)', ln) for ln in run['lines']) if m]
    return seg, cb


def kv(line: str) -> dict:
    return dict(re.findall(r'(\w+)=(\S+)', line))


def reinit_lines(run):
    out = {}
    for ln in run['lines']:
        m = re.match(r'(REINIT\w*) (?:iteration|k)=(\d+)', ln)
        if m:
            out.setdefault(int(m.group(2)), []).append((m.group(1), ln))
    return out


def reference_estimates(stages_csv: Path) -> dict:
    """Stock's training time to its own 30k PSNR, first sampled stage at or above it (seed-0 stage runs)."""
    est = {}
    if not stages_csv.exists():
        return est
    rows = [r for r in csv.DictReader(open(stages_csv)) if r['arm'] == 'stock']
    for scene in {r['scene'] for r in rows}:
        rs = sorted((int(r['iteration']), float(r['train_time_s']), float(r['test_psnr'])) for r in rows if r['scene'] == scene)
        final = rs[-1][2]
        est[scene] = next(t for it, t, p in rs if p >= final)
    return est


def pct(a, b):
    return f'{100 * a / b:.1f}%' if b else 'n/a'


def fmt(x, nd=3):
    return '' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def analyze(logs, stages_csv: Path):
    est = reference_estimates(stages_csv)
    lines, fails, gates = [], [], []
    for path in logs:
        J = parse_log(path)
        lines.append(f'# Alternative track R, R1 audit analysis: job {J["job"]} (node {J["node"]}, repo {J["repo"]})\n')
        audits = [r for r in J['runs'] if r['reinit'] and 'mode=audit' in r['reinit']]
        others = [r for r in J['runs'] if r not in audits]
        lines.append(f'Runs in the log: {len(J["runs"])} ({len(audits)} audit). Other runs: ' +
                     (', '.join(r['name'] for r in others) or 'none') + '\n')
        lines.append('## Validation\n')
        for r in audits:
            R = reinit_lines(r)
            ks = sorted(k for k, ls in R.items() if any(t == 'REINIT' for t, _ in ls))
            problems = []
            if any(ln.startswith(f'FAILED {r["name"]}') for ln in r['lines']):
                problems.append('run FAILED')
            if not any(ln.startswith(f'RESULT {r["name"]}') for ln in r['lines']):
                problems.append('no final metrics')
            spec = kv(r['reinit'].replace(',', ' '))
            it = [int(x) for x in spec.get('iters', '0').split(':')]
            want = list(range(it[0], (it[1] if len(it) > 1 else it[0]) + 1, (it[2] if len(it) > 2 else (it[1] - it[0] if len(it) > 1 else 1)) or 1))
            if ks != want:
                problems.append(f'REINIT lines at {ks}, expected {want}')
            for k in ks:
                chk = [ln for t, ln in R[k] if t == 'REINIT_CHECK']
                if not chk or 'result=pass' not in chk[0]:
                    problems.append(f'k={k}: replacement check not passed')
                if not (J['archive'] / r['name'] / 'reinit_audit' / f'k_{k:05d}' / 'stats.json').exists():
                    problems.append(f'k={k}: stats.json missing')
                if not any(t == 'REINIT_AGREEMENT' for t, _ in R[k]):
                    problems.append(f'k={k}: no final agreement')
            lines.append(f'- {r["name"]}: ' + ('valid' if not problems else 'INVALID: ' + '; '.join(problems)))
            if problems:
                fails.append(r['name'])
        lines.append('')
        for r in audits:
            if r['name'] in fails and not (J['archive'] / r['name']).exists():
                continue
            lines += scene_report(J, r, est, gates)
    lines.append('## R1 gates (alt_reinit/PLAN.md)\n')
    lines.append('G1: warm-up + reinit cost leaves a positive budget against the reference estimate. '
                 'G2: kept candidates are more photo-consistent than geometry-rejected ones (consistent share and '
                 'median, mean over neighbours). G3: rejection masks inspected (manual).\n')
    lines.append('| scene | k | G1 budget left s | G1 | G2 kept / rejected consistent | G2 | G3 |')
    lines.append('|---|---|---|---|---|---|---|')
    for g in gates:
        lines.append(f'| {g["scene"]} | {g["k"]} | {fmt(g["budget"], 1)} | {g["g1"]} | {g["g2txt"]} | {g["g2"]} | inspect {g["masks"]} images |')
    lines.append('')
    for scene in sorted({g['scene'] for g in gates}):
        ok = [g['k'] for g in gates if g['scene'] == scene and g['g1'] == 'PASS' and g['g2'] == 'PASS']
        lines.append(f'- {scene}: earliest k passing G1 and G2: {min(ok) if ok else "none"} (G3 pending manual inspection)')
    return '\n'.join(lines) + '\n', fails


def scene_report(J, r, est, gates):
    L = [f'## {r["scene"]}: {r["name"]} ({r["iters"]} iterations, seed {r["seed"]})\n',
         f'Settings: `{r["reinit"]}`\n']
    seg, cb = clock(r)
    R = reinit_lines(r)
    adir = J['archive'] / r['name'] / 'reinit_audit'
    ref = est.get(r['scene'])
    rows = []
    for k in sorted(R):
        st_path = adir / f'k_{k:05d}' / 'stats.json'
        S = json.loads(st_path.read_text()) if st_path.exists() else None
        rl = dict((t, ln) for t, ln in R[k])
        rein = kv(rl.get('REINIT', ''))
        cum_k = seg.get(k)
        cb_k = sum(s for n, it, s in cb if it <= k and n != 'reinit_gaussians')
        warm = None if cum_k is None else cum_k + cb_k
        t_re = float(rein['t_total']) if 't_total' in rein else None
        budget = None if (ref is None or warm is None or t_re is None) else ref - warm - t_re
        rows.append(dict(k=k, S=S, rl=rl, rein=rein, warm=warm, cb_k=cb_k, t_re=t_re, budget=budget))
    L.append('### Cost and budget\n')
    L.append(f'Reference estimate (stock seed-0 stage run, time to its own 30k PSNR): {fmt(ref, 1)} s.\n')
    L.append('| k | warm-up s (steps + callbacks) | reinit s | timed target | stride | depth | candidates | neighbours | filter | dedupe | select | knn | rebuild | morton | budget left s |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x in rows:
        p = x['rein']
        part = lambda n: fmt(float(p[f't_{n}']), 2) if f't_{n}' in p else ''  # noqa: E731
        L.append(f'| {x["k"]} | {fmt(x["warm"], 1)} ({fmt(x["cb_k"], 2)} callbacks) | {fmt(x["t_re"], 2)} | {p.get("timed_target", "")} | '
                 f'{p.get("stride", "")} | ' + ' | '.join(part(n) for n in ('depth', 'candidates', 'neighbours', 'filter', 'dedupe', 'select', 'knn', 'rebuild', 'morton')) +
                 f' | {fmt(x["budget"], 1)} |')
    L.append('')
    L.append('### Counts and targets\n')
    L.append('| k | N before | grid candidates | after filter | after dedupe | removed (covered) | survivors | added | N after | targets |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for x in rows:
        c = kv(x['rl'].get('REINIT_COUNTS', ''))
        tg = [ln for t, ln in R[x['k']] if t == 'REINIT_TARGET']
        tt = '; '.join(f'{kv(t)["target"]}: {"reachable" if " reachable " in t else "UNREACHABLE"} (max {kv(t)["max_reachable"]})' for t in tg)
        L.append(f'| {x["k"]} | {c.get("n_before", "")} | {c.get("n_candidates_grid", "")} | {c.get("n_filtered", "")} | {c.get("n_deduped", "")} | '
                 f'{c.get("n_covered", "")} | {c.get("n_survivors", "")} | {c.get("n_added", "")} | {c.get("n_after", "")} | {tt} |')
    L.append('')
    L.append('### Status of the grid candidates\n')
    L.append('| k | kept | empty | low alpha | mixture | geometry | neighbour outcomes (agree / out of view / target invalid / depth / reprojection) |')
    L.append('|---|---|---|---|---|---|---|')
    for x in rows:
        S = x['S']
        if not S:
            continue
        st = S['status']
        tot = sum(st.values())
        no = S['neighbour_outcomes']
        L.append(f'| {x["k"]} | ' + ' | '.join(pct(st[s], tot) for s in STATUS) + ' | ' +
                 ' / '.join(pct(no[n], sum(no.values())) for n in ('agree', 'out_of_view', 'target_invalid', 'depth_mismatch', 'reproj_mismatch')) + ' |')
    L.append('')
    for x in rows:
        S = x['S']
        if not S:
            continue
        k = x['k']
        L.append(f'### k = {k}: where candidates are rejected\n')
        L.append('By depth (scene units; share of the candidates in each bin):\n')
        L.append('| depth bin | candidates | kept | mixture | geometry | low alpha |')
        L.append('|---|---|---|---|---|---|')
        edges = S['by_depth']['edges']
        for i, row in enumerate(S['by_depth']['counts']):
            n = sum(row)
            if not n:
                continue
            lo = '0' if i == 0 else f'{edges[i - 1]:g}'
            hi = 'inf' if i >= len(edges) else f'{edges[i]:g}'
            L.append(f'| {lo}-{hi} | {n} | {pct(row[0], n)} | {pct(row[3], n)} | {pct(row[4], n)} | {pct(row[2], n)} |')
        L.append('')
        L.append('By image region (kept share; rows top to bottom):\n')
        reg = S['by_region']['counts']
        for rr in range(3):
            L.append('| ' + ' | '.join(pct(reg[3 * rr + cc][0], sum(reg[3 * rr + cc])) for cc in range(3)) + ' |')
            if rr == 0:
                L.append('|---|---|---|')
        L.append('')
        P = S.get('photo', {})
        L.append('Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view\'s '
                 'neighbours; the same statistics for every class):\n')
        L.append('| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |')
        L.append('|---|---|---|---|---|---|---|---|')
        for cls in ('kept', 'geometry', 'mixture', 'low_alpha'):
            if cls not in P:
                continue
            m, b = P[cls]['mean_over_neighbours'], P[cls]['best_neighbour']
            ver = m['n'] - m['unverifiable']
            L.append(f'| {cls} | {m["n"]} | {ver} | {pct(m["consistent"], ver)} | {pct(m["ambiguous"], ver)} | {pct(m["inconsistent"], ver)} | '
                     f'{fmt(m["median"])} | {fmt(b["median"])} |')
        L.append('')
        F = S.get('sfm', {})
        if 'error' in F:
            L.append(f'COLMAP check: ERROR {F["error"]}\n')
        else:
            rp = F.get('reprojection_px', {})
            L.append(f'COLMAP points (independent of the model; sparse, biased to texture): {F.get("n_observations")} observations '
                     f'in the diagnostic views; transform deviation {F.get("transform_max_deviation", 0):.1e}; reprojection '
                     f'median {fmt(rp.get("q50"))} px, q90 {fmt(rp.get("q90"))} px (checks the camera conventions).\n')
            L.append('| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |')
            L.append('|---|---|---|---|---|')
            for cls in STATUS:
                q = F.get(f'depth_rel_error_{cls}', {})
                if q.get('n'):
                    L.append(f'| {cls} | {q["n"]} | {fmt(q.get("q50"), 4)} | {pct(q.get("within_1pct", 0), 1)} | {pct(q.get("within_5pct", 0), 1)} |')
            L.append('')
        C = S.get('coverage')
        if C:
            L.append(f'Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes {pct(C["holes"], C["pixels_covered_before"])} '
                     f'of the previously covered pixels, newly covered {pct(C["added"], C["pixels_covered_before"])}.\n')
        agr = adir / f'k_{k:05d}' / 'agreement_final.json'
        if agr.exists():
            A = json.loads(agr.read_text())['by_status']
            L.append('Agreement with the final model\'s depth (agreement with another reconstruction, not accuracy):\n')
            L.append('| audited status | pixels | rel. difference median | within 1% | within 5% |')
            L.append('|---|---|---|---|---|')
            for cls in STATUS:
                q = A.get(cls, {})
                if q.get('n'):
                    L.append(f'| {cls} | {q["n"]} | {fmt(q.get("q50"), 4)} | {pct(q.get("within_1pct", 0), 1)} | {pct(q.get("within_5pct", 0), 1)} |')
            L.append('')
        nb = S['neighbours']
        L.append(f'Neighbours: {nb["views_without"]} views without any; overlap median {fmt(nb["overlap"].get("q50"))}, '
                 f'triangulation angle median {fmt(nb["angle_deg"].get("q50"), 1)} deg (q10 {fmt(nb["angle_deg"].get("q10"), 1)}, '
                 f'q90 {fmt(nb["angle_deg"].get("q90"), 1)}).\n')
        masks = sorted((adir / f'k_{k:05d}' / 'masks').glob('*.jpg'))
        L.append('Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | '
                 'relative depth std, white = 0.2+): ' + (', '.join(f'`{os.path.relpath(m, REPO)}`' for m in masks) or 'none') + '\n')
        g2, g2txt = 'n/a', ''
        if 'kept' in P and 'geometry' in P:
            mk, mg = P['kept']['mean_over_neighbours'], P['geometry']['mean_over_neighbours']
            vk, vg = mk['n'] - mk['unverifiable'], mg['n'] - mg['unverifiable']
            if vk >= 100 and vg >= 100:
                ck, cg = mk['consistent'] / vk, mg['consistent'] / vg
                g2 = 'PASS' if (ck > cg and (mk['median'] or 0) >= (mg['median'] or 0)) else 'FAIL'
                g2txt = f'{100 * ck:.0f}% / {100 * cg:.0f}% (medians {fmt(mk["median"], 2)} / {fmt(mg["median"], 2)})'
            else:
                g2txt = f'too few verifiable samples ({vk} / {vg})'
        g1 = 'n/a' if x['budget'] is None else ('PASS' if x['budget'] > 0 else 'FAIL')
        gates.append(dict(scene=r['scene'], k=k, budget=x['budget'], g1=g1, g2=g2, g2txt=g2txt, masks=len(masks)))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--stages', default=str(REPO / 'phase2' / 'stages' / 'psnr_vs_time.csv'))
    a = ap.parse_args()
    text, fails = analyze(a.logs, Path(a.stages))
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
    if fails:
        print(f'VALIDATION FAILED: {", ".join(fails)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
