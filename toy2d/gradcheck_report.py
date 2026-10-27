"""Prints the G0 gradient table: CUDA fp32 vs PyTorch fp64 reference, per scene, backend and parameter.

    python gradcheck_report.py            # markdown table to stdout
    python gradcheck_report.py --json out.json
"""
import argparse
import json

import torch

from gs2d import rasterize, render_reference
from tests.scenes import PARAM_NAMES, SCENES, make_scene, rel_err, upstream_grad

G0_TOL = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write rows to this file")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "run this on a GPU node"

    rows = []
    for scene in SCENES:
        leaves, bins, bg = make_scene(scene)
        dout = upstream_grad(bins.H, bins.W)
        ref = render_reference(*leaves, bins, bg)
        ref_grads = torch.autograd.grad(ref, leaves, dout)
        for backend in ("pixel", "splat"):
            x = [t.detach().float().cuda().requires_grad_(True) for t in leaves]
            out = rasterize(*x, bins.to("cuda"), bg.float().cuda(), backend=backend)
            grads = torch.autograd.grad(out, x, dout.float().cuda())
            row = {"scene": scene, "backend": backend, "forward": rel_err(out, ref)}
            for name, a, b in zip(PARAM_NAMES, grads, ref_grads):
                row[name] = rel_err(a, b)
            rows.append(row)

    cols = ["forward", *PARAM_NAMES]
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}  CUDA {torch.version.cuda}\n")
    print("| scene | backend | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    worst = 0.0
    for r in rows:
        worst = max(worst, *(r[c] for c in cols))
        print(f"| {r['scene']} | {r['backend']} | " + " | ".join(f"{r[c]:.1e}" for c in cols) + " |")
    print(f"\nWorst relative error: {worst:.2e}  ->  G0 accuracy {'PASS' if worst < G0_TOL else 'FAIL'} (< {G0_TOL:g})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
