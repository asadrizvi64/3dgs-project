"""Comparison sheets from the stage-render runs (phase2/build_plus.sbatch STAGES=stages): renders during training.

    python phase2/stage_sheets.py <scene> <stage archive: ~/3dgs-archive/stages/<scene>> <seg dir: phase2/logs/run_<job>> <out dir>

For each scene it writes (JPEG, cells downscaled to CELL_W pixels wide):
  <scene>_view<i>_by_iteration.jpg   rows = iterations (ITERS), columns = stock | A+B | A+B+C | ground truth
  <scene>_view<i>_by_time.jpg        rows = training-time budgets (fractions of stock's total training time),
                                     columns = the latest render each arm had reached within that budget
Each cell is labelled with the iteration, the training time (CUDA-event clock, from the run's SEGCLOCK lines) and
that image's PSNR at that stage. Two test views per scene: the one whose stock PSNR at 30k is the median, and the
one with the largest stock gain from 7k to 30k (where training progress is most visible).
"""
import os
import re
import statistics as st
import sys

from PIL import Image, ImageDraw, ImageFont

scene, arc, segdir, out = sys.argv[1:5]
ARMS = [('stock', 'stock Faster-GS'), ('ab', 'A+B'), ('abc', 'A+B+C')]
ITERS = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000]
FRACS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
CELL_W = 420
os.makedirs(out, exist_ok=True)
try:
    font = ImageFont.load_default(size=15)
except TypeError:
    font = ImageFont.load_default()


def run_dir(arm):
    return os.path.join(arc, f'r_{scene}_{arm}_s0')


def clock(arm):
    """iteration -> cumulative training seconds (SEGCLOCK every 1000 iterations)."""
    t = open(os.path.join(segdir, f'r_{scene}_{arm}_s0.seg')).read()
    return {int(a): float(b) for a, b in re.findall(r'SEGCLOCK step=(\d+) \S+ cum_s=([\d.]+)', t)}


def per_image(arm, it):
    t = open(os.path.join(run_dir(arm), f'test_{it}', 'metrics_8bit.txt')).read().split('Image', 1)[-1]
    return {int(a): float(b) for a, b in re.findall(r'^(\d+)\t([\d.]+)', t, re.M)}


def render(arm, it, view):
    d = os.path.join(run_dir(arm), f'test_{it}', 'rgb')
    names = sorted(os.listdir(d))
    return Image.open(os.path.join(d, names[view])).convert('RGB')


def gt(view):
    d = os.path.join(run_dir('stock'), 'test_30000', 'rgb_gt')
    names = sorted(n for n in os.listdir(d) if n.endswith('.png'))
    return Image.open(os.path.join(d, names[view])).convert('RGB')


def cell(img, label):
    h = round(img.height * CELL_W / img.width)
    im = img.resize((CELL_W, h), Image.LANCZOS)
    canvas = Image.new('RGB', (CELL_W, h + 22), 'white')
    canvas.paste(im, (0, 22))
    ImageDraw.Draw(canvas).text((4, 3), label, fill='black', font=font)
    return canvas


def sheet(rows, title, path):
    w = max(len(r) for r in rows) * CELL_W
    hs = [max(c.height for c in r) for r in rows]
    page = Image.new('RGB', (w, sum(hs) + 30), 'white')
    ImageDraw.Draw(page).text((6, 6), title, fill='black', font=font)
    y = 30
    for r, h in zip(rows, hs):
        for j, c in enumerate(r):
            page.paste(c, (j * CELL_W, y))
        y += h
    page.save(path, quality=88)
    print('wrote', path, page.size)


clk = {a: clock(a) for a, _ in ARMS}
stages = {a: sorted(int(n.split('_')[1]) for n in os.listdir(run_dir(a)) if re.match(r'test_\d+$', n)) for a, _ in ARMS}
s30 = per_image('stock', 30000)
s7 = per_image('stock', 7000)
med = sorted(s30, key=s30.get)[len(s30) // 2]
gain = max(s30, key=lambda i: s30[i] - s7[i])
views = [med] if gain == med else [med, gain]

for vi, view in enumerate(views):
    g = gt(view)
    rows = [[cell(g, f'ground truth, test view {view}')]]
    for it in ITERS:
        row = []
        for arm, label in ARMS:
            p = per_image(arm, it)[view]
            row.append(cell(render(arm, it, view), f'{label}  it {it // 1000}k  {clk[arm].get(it, float("nan")):.0f} s  {p:.2f} dB'))
        rows.append(row)
    sheet(rows, f'{scene}, test view {view}: renders by iteration (seed 0; time = training clock; PSNR of this view)',
          os.path.join(out, f'{scene}_view{vi}_by_iteration.jpg'))

    total = clk['stock'][30000]
    rows = [[cell(g, f'ground truth, test view {view}')]]
    for f in FRACS:
        budget = f * total
        row = []
        for arm, label in ARMS:
            done = [it for it in stages[arm] if clk[arm].get(it, 1e9) <= budget + 1e-6]
            if not done:
                row.append(cell(Image.new('RGB', g.size, (235, 235, 235)), f'{label}: no stage within {budget:.0f} s'))
                continue
            it = done[-1]
            p = per_image(arm, it)[view]
            row.append(cell(render(arm, it, view), f'{label}  {clk[arm][it]:.0f} s  it {it / 1000:g}k  {p:.2f} dB'))
        rows.append(row)
    sheet(rows, f'{scene}, test view {view}: equal training time ({", ".join(f"{f:g}" for f in FRACS)} x stock total {total:.0f} s); '
                'each cell = latest stage reached within the budget',
          os.path.join(out, f'{scene}_view{vi}_by_time.jpg'))
