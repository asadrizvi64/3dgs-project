"""Time _C.backward for each blend backward mode (0 = stock taming, else macro bits) on the synthetic parity scenes.

Both modes share the forward buffers and the preprocess backward, so the difference is the blend backward.
Prints BENCH lines (median of --reps, CUDA events):
    python bench/phase1/kernel_bench.py --reps 50
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'phase1' / 'tests'))
from test_parity import SCENES, _C, make_inputs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--modes', default='0,1,3,5,7,9,15', help='blend backward modes; 0 = stock (the reference)')
    args = ap.parse_args()
    for scene in SCENES:
        params, settings, extra = make_inputs(scene, 0, False)
        means, scales, rotations, opacities, sh0, sh_rest = params
        image, pb, tb, ib, bb, n_inst, n_buckets, sel = _C.forward(*params, *settings)
        grad_image = torch.randn(image.shape, generator=extra['generator'], device='cuda')
        res = {}
        for mode in (int(m) for m in args.modes.split(',')):
            ms = []
            for i in range(args.reps + 3):
                di = torch.zeros(3, scene.n, device='cuda')
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                _C.backward(di, grad_image, image, means, scales, rotations, opacities, sh_rest, pb, tb, ib, bb,
                            *settings, n_inst, n_buckets, sel, mode)
                b.record()
                torch.cuda.synchronize()
                if i >= 3:
                    ms.append(a.elapsed_time(b))
            res[mode] = statistics.median(ms)
        line = ' '.join(f'm{m}={t:.4f}ms({res[0] / t:.2f}x)' for m, t in res.items())
        print(f'BENCH scene={scene.name} n={scene.n} {scene.width}x{scene.height} instances={n_inst} buckets={n_buckets} {line}',
              flush=True)


if __name__ == '__main__':
    main()
