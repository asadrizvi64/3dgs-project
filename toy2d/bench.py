"""Times the forward and both backward kernels with CUDA events.

    python bench.py                                   # default sweep
    python bench.py --sizes 1024 --counts 50000 --json results/bench.json

For Nsight Compute, profile one iteration of each kernel:
    ncu --set full -k regex:"backward|forward" -c 3 -o profiles/toy2d \
        python bench.py --sizes 1024 --counts 50000 --warmup 0 --iters 1
"""
import argparse
import json
import platform
import statistics
import subprocess

import torch

from gs2d import Gaussians2D, bin_gaussians
from gs2d.params import radii_from_cov
from gs2d.raster import ext


def env_info():
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except Exception:
        driver = "unknown"
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "cc": f"{p.major}.{p.minor}", "driver": driver, "cuda": torch.version.cuda,
            "torch": torch.__version__, "host": platform.node()}


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def bench_one(size, n, warmup, iters, seed=0):
    H = W = size
    g = Gaussians2D(n, H, W, seed=seed).cuda()
    with torch.no_grad():
        means, conic, color, opacity, cov = (t.contiguous() for t in g.activated())
        bins = bin_gaussians(means, radii_from_cov(cov), g.depth, H, W)
    bg = torch.zeros(3, device="cuda")
    dout = torch.randn(3, H, W, device="cuda")
    e = ext()
    common = (means, conic, color, opacity, bg, bins.ranges, bins.point_list)

    fwd = lambda save: e.forward(*common, bins.bucket_offsets, bins.num_buckets, H, W, save)
    out, final_T, n_contrib, _ = fwd(False)
    out_s, final_T_s, n_contrib_s, bstate = fwd(True)

    res = {
        "size": size, "n": n, "pairs": int(bins.point_list.numel()), "buckets": bins.num_buckets,
        "forward_ms": timed(lambda: fwd(False), warmup, iters),
        "forward_save_buckets_ms": timed(lambda: fwd(True), warmup, iters),
        "backward_pixel_ms": timed(lambda: e.backward_pixel(*common, final_T, n_contrib, dout), warmup, iters),
        "backward_splat_ms": timed(
            lambda: e.backward_splat(*common, bins.bucket_offsets, bins.bucket_to_tile, out_s, final_T_s,
                                     n_contrib_s, bstate, dout), warmup, iters),
    }
    res["splat_speedup"] = res["backward_pixel_ms"] / res["backward_splat_ms"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--counts", type=int, nargs="+", default=[10_000, 50_000, 200_000])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json")
    args = ap.parse_args()

    info = env_info()
    print(info)
    rows = []
    for s in args.sizes:
        for n in args.counts:
            r = bench_one(s, n, args.warmup, args.iters)
            rows.append(r)
            print(f"{s}x{s} N={n:>7} pairs={r['pairs']:>9}  fwd {r['forward_ms']:.3f} ms  "
                  f"fwd+save {r['forward_save_buckets_ms']:.3f}  bwd pixel {r['backward_pixel_ms']:.3f}  "
                  f"bwd splat {r['backward_splat_ms']:.3f}  ({r['splat_speedup']:.2f}x)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"env": info, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
