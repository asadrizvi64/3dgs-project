"""Side-by-side renders for the Part 2 article, from job 4577691 (stage x1rrender): standard Faster-GS (ctl) and the
MCMC switch, each resumed again from Experiment 1R's saved seed-11 state at step 5000.

    python phase3/article/make_renders.py <raw dir> <job log dir>      -> phase3/article/fig7_renders_<scene>.jpg

<raw dir>/<scene>/<arm>/test_<step>/rgb/NNNNN.png and <raw dir>/<scene>/gt/NNNNN.png (fetched from the run directories);
<job log dir> = phase2/logs/run_4577691 (the .seg clock files and runs/*/test_*/metrics_8bit.txt).

Choice of what to show (fixed here, before looking at any image):
- time: the first evaluation at which the MCMC run is at or above the target (ctl's final PSNR - 0.05 dB), and the ctl
  evaluation whose training time is closest to it, so both panels show the same training-time budget;
- views: the test view with the median per-view PSNR gain at those evaluations (typical), and the one with the largest
  gain (best case), both labelled as such;
- crop: the window (one third of the width and height) where MCMC's error against the photo is lower than ctl's by the
  largest amount, found from the pixel errors.
"""
import os
import re
import statistics as st
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW, LOGS = sys.argv[1], sys.argv[2]
OUT = 'phase3/article'
SEED, DELTA, END = 11, 0.05, 30000
SURF, INK, MUTED = (252, 252, 251), (11, 11, 11), (110, 109, 105)
COL = {'ctl': (235, 104, 52), 'mcmc': (42, 120, 214)}


def font(size, bold=False):
    for f in (['DejaVuSans-Bold.ttf', 'arialbd.ttf'] if bold else ['DejaVuSans.ttf', 'arial.ttf']):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    return ImageFont.truetype(os.path.join(d, 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), size)


def metrics(path):
    t = open(path).read()
    head, _, imgs = t.partition('Image')
    mean = float(re.search(r'^PSNR\t(\S+)', head, re.M).group(1))
    per = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^(\d+)\t([\d.]+)', imgs, re.M)}
    return mean, per


def run(scene, arm):
    name = f'xr_{scene}_{arm}_s{SEED}'
    clk = {}
    for l in open(os.path.join(LOGS, f'{name}.seg')):
        m = re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+)', l)
        if m:
            clk[int(m.group(1))] = float(m.group(2))
        m = re.match(r'FGS_RESUME step=(\d+) N=\d+ cum_s=(\S+)', l)
        if m:
            clk[int(m.group(1))] = float(m.group(2))
    ev = {}
    for s in range(5000, END + 1, 1000):
        ev[s] = metrics(os.path.join(LOGS, 'runs', name, f'test_{s}', 'metrics_8bit.txt'))
    return clk, ev


def choose(scene):
    (cc, ce), (mc, me) = run(scene, 'ctl'), run(scene, 'mcmc')
    tgt = ce[END][0] - DELTA
    sm = next(s for s in sorted(me) if me[s][0] >= tgt)
    sc = min(ce, key=lambda s: abs(cc[s] - mc[sm]))
    gains = {v: me[sm][1][v] - ce[sc][1][v] for v in me[sm][1]}
    order = sorted(gains, key=gains.get)
    typical, best = order[len(order) // 2], order[-1]
    return dict(tgt=tgt, sm=sm, sc=sc, tm=mc[sm], tc=cc[sc], pm=me[sm][0], pc=ce[sc][0], gains=gains,
                views=[('typical view (median gain)', typical), ('view with the largest gain', best)], me=me, ce=ce)


def img(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255


def best_window(gt, a, b, frac=1 / 3):
    """Top-left of the window where b's error is lower than a's by the most (a = ctl, b = mcmc)."""
    d = np.abs(a - gt).mean(2) - np.abs(b - gt).mean(2)
    H, W = d.shape
    h, w = int(H * frac), int(W * frac)
    I = np.pad(d.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    S = I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]
    y, x = np.unravel_index(np.argmax(S), S.shape)
    return x, y, w, h


def to_pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def figure(scene, c):
    th, gap, pad = 300, 14, 28
    f_t, f_b, f_s = font(22, True), font(18, True), font(16)
    rows = []
    for label, v in c['views']:
        gt = img(os.path.join(RAW, scene, 'gt', f'{v:05d}.png'))
        a = img(os.path.join(RAW, scene, 'ctl', f'test_{c["sc"]}', 'rgb', f'{v:05d}.png'))
        b = img(os.path.join(RAW, scene, 'mcmc', f'test_{c["sm"]}', 'rgb', f'{v:05d}.png'))
        x, y, w, h = best_window(gt, a, b)
        full = to_pil(gt)
        dr = ImageDraw.Draw(full)
        for k in range(4):
            dr.rectangle([x - k, y - k, x + w + k, y + h + k], outline=(227, 73, 72))
        crops = [to_pil(t[y:y + h, x:x + w]) for t in (gt, a, b)]
        rows.append((label, v, full, crops, c['ce'][c['sc']][1][v], c['me'][c['sm']][1][v]))
    fw = int(rows[0][2].width * th / rows[0][2].height)
    cw = int(rows[0][3][0].width * th / rows[0][3][0].height)
    W = pad * 2 + fw + 3 * (cw + gap)
    head = 118
    H = head + len(rows) * (th + 64) + 40
    canvas = Image.new('RGB', (W, H), SURF)
    d = ImageDraw.Draw(canvas)
    d.text((pad, 18), f'{scene.capitalize()}: the same training time, about {c["tm"]:.0f} seconds', font=f_t, fill=INK)
    d.text((pad, 50), f'Standard Faster-GS at step {c["sc"]:,} ({c["tc"]:.0f} s, test PSNR {c["pc"]:.2f} dB)   vs   '
                      f'MCMC switch at step {c["sm"]:,} ({c["tm"]:.0f} s, {c["pm"]:.2f} dB: first evaluation at the target)',
           font=f_s, fill=MUTED)
    titles = ['Real photo (red box = zoomed area)', 'Real photo, zoomed', 'Standard Faster-GS', 'Switch to MCMC']
    xs = [pad, pad + fw + gap, pad + fw + gap + cw + gap, pad + fw + gap + 2 * (cw + gap)]
    for i, t in enumerate(titles):
        d.text((xs[i], 86), t, font=f_b, fill=COL['ctl'] if i == 2 else COL['mcmc'] if i == 3 else INK)
    yy = head
    for label, v, full, crops, pc, pm in rows:
        canvas.paste(full.resize((fw, th), Image.LANCZOS), (xs[0], yy))
        for i, cimg in enumerate(crops):
            canvas.paste(cimg.resize((cw, th), Image.LANCZOS), (xs[i + 1], yy))
        d.text((xs[0], yy + th + 8), f'Test view {v}: {label}', font=f_s, fill=INK)
        d.text((xs[2], yy + th + 8), f'{pc:.2f} dB on this view', font=f_s, fill=INK)
        d.text((xs[3], yy + th + 8), f'{pm:.2f} dB on this view', font=f_s, fill=INK)
        yy += th + 64
    d.text((pad, H - 32), f'Rerun of the Switch Test, seed {SEED} from its saved state at step 5,000 (job 4577691). '
                          'Views and zoom area chosen by a fixed rule (phase3/article/make_renders.py).',
           font=font(14), fill=MUTED)
    p = os.path.join(OUT, f'fig7_renders_{scene}.jpg')
    canvas.save(p, quality=92)
    return p


if __name__ == '__main__':
    for scene in ('garden', 'room'):
        c = choose(scene)
        print(scene, 'target', round(c['tgt'], 3), 'mcmc step', c['sm'], round(c['tm'], 1), 's', c['pm'], '| ctl step', c['sc'],
              round(c['tc'], 1), 's', c['pc'], '| views', c['views'], 'median gain', round(st.median(c['gains'].values()), 3))
        if len(sys.argv) > 3 and sys.argv[3] == '--choose-only':
            continue
        print(figure(scene, c))
