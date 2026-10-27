"""Time the Phase 0 kernels on a larger 2D scene: forward, backward (A) and backward (B).

Prints BENCH lines:
  scene stats (tile-list lengths, buckets, contributors per pixel),
  CPU-emulation op counts (atomics per mode, checkpoint bytes),
  GPU wall time per extension call (CUDA events, median of --reps),
  and GPU correctness on this scene (A vs B, GPU vs CPU emulation; norm-based rel error).

With --ncu it instead runs each kernel exactly once inside cudaProfilerStart/Stop, for
`ncu --profile-from-start off`.

  python bench/phase0/backward_bench.py --height 1080 --width 1920 --n 50000
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase0"))

from splat2d.ops import BUCKET, NUM_PARAMS, TileBins, bin_tiles, load_extension

GROUPS = ("means", "chol", "opacity_logit", "colors")


def make_scene(args: argparse.Namespace) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(args.seed)

    def u(lo: float, hi: float, *shape: int) -> Tensor:
        return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=torch.float64)

    n, h, w = args.n, args.height, args.width
    means = torch.stack([u(-0.02 * w, 1.02 * w, n), u(-0.02 * h, 1.02 * h, n)], dim=-1)
    l0 = 1.0 / u(args.sigma[0], args.sigma[1], n)
    l2 = 1.0 / u(args.sigma[0], args.sigma[1], n)
    l1 = u(-0.5, 0.5, n) * torch.minimum(l0, l2)
    o = u(args.opacity[0], args.opacity[1], n)
    colors = u(0.0, 1.0, n, 3)
    chol = torch.stack([l0, l1, l2], dim=-1)
    return means, chol, torch.log(o / (1.0 - o)), colors  # index order == depth order


def tables(b: TileBins, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        b.tile_offsets.to(device),
        b.tile_ids.to(device),
        b.bucket_offsets.to(device),
        b.bucket_tile.to(device),
    )


def rel(a: Tensor, b: Tensor) -> float:
    d = torch.linalg.vector_norm((a.double().cpu() - b.double().cpu()).flatten())
    return float(d) / max(float(torch.linalg.vector_norm(b.double().cpu().flatten())), 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--sigma", type=float, nargs=2, default=(2.0, 10.0))
    ap.add_argument("--opacity", type=float, nargs=2, default=(0.05, 0.6))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ncu", action="store_true", help="one launch per kernel in a profiler range")
    ap.add_argument("--no-counters", action="store_true", help="skip the CPU emulation")
    args = ap.parse_args()
    tag = f"{args.width}x{args.height}_n{args.n}_s{args.sigma[0]:g}-{args.sigma[1]:g}"
    h, w = args.height, args.width

    params64 = make_scene(args)
    bins = bin_tiles(*params64[:3], h, w)
    counts = (bins.tile_offsets[1:] - bins.tile_offsets[:-1]).double()
    print(
        f"BENCH {tag} scene: tiles={bins.tiles_x * bins.tiles_y} list_entries={int(counts.sum())} "
        f"per_tile mean={counts.mean():.1f} max={int(counts.max())} buckets={bins.num_buckets}"
    )

    dev = torch.device("cuda")
    ext = load_extension(with_cuda=True)
    m32, c32, o32, col32 = (t.float().to(dev).contiguous() for t in params64)
    p32 = (m32, c32, o32, col32)
    tb = tables(bins, dev)
    g_img = torch.randn(h, w, 3, generator=torch.Generator().manual_seed(1)).to(dev)

    def fwd() -> list[Tensor]:
        return ext.forward(*p32, *tb, h, w)

    out = fwd()
    image, t_final, n_contrib, last_contrib, ckpt = out

    def bwd_a() -> list[Tensor]:
        return ext.backward_pixel(g_img, *p32, *tb, t_final, last_contrib, h, w)

    def bwd_b() -> list[Tensor]:
        return ext.backward_splat(g_img, *p32, *tb, image, ckpt, h, w)

    if args.ncu:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        fwd()
        bwd_a()
        bwd_b()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f"BENCH {tag} ncu pass done")
        return

    nc = n_contrib.double()
    print(
        f"BENCH {tag} contributors/pixel mean={nc.mean():.1f} max={int(nc.max())} "
        f"pixel_atomics={int(nc.sum()) * NUM_PARAMS} "
        f"ckpt_MB={ckpt.numel() * ckpt.element_size() / 1e6:.1f}"
    )

    # Correctness on this scene (fp32 on the GPU).
    ga, gb = bwd_a(), bwd_b()
    print(
        f"BENCH {tag} gpu A-vs-B rel: "
        + " ".join(f"{k}={rel(b, a):.1e}" for k, a, b in zip(GROUPS, ga, gb, strict=True))
    )

    if not args.no_counters:
        cpu_ext = load_extension(with_cuda=False)
        mc, cc, oc, colc = (t.float().contiguous() for t in params64)
        pc = (mc, cc, oc, colc)
        tc = tables(bins, torch.device("cpu"))
        gc = g_img.cpu()
        cpu_ext.reset_counters()
        img_c, tf_c, _nc_c, lc_c, ck_c = cpu_ext.forward(*pc, *tc, h, w)
        fwd_c = cpu_ext.get_counters()
        cpu_ext.reset_counters()
        ga_c = cpu_ext.backward_pixel(gc, *pc, *tc, tf_c, lc_c, h, w)
        a_c = cpu_ext.get_counters()
        cpu_ext.reset_counters()
        gb_c = cpu_ext.backward_splat(gc, *pc, *tc, img_c, ck_c, h, w)
        b_c = cpu_ext.get_counters()
        print(
            f"BENCH {tag} counters: pixel_atomics={a_c['atomic_adds']} "
            f"splat_atomics={b_c['atomic_adds']} "
            f"ratio={a_c['atomic_adds'] / max(b_c['atomic_adds'], 1):.1f}x "
            f"checkpoint_bytes={fwd_c['checkpoint_bytes']} evals fwd={fwd_c['splat_pixel_evals']} "
            f"A={a_c['splat_pixel_evals']} B={b_c['splat_pixel_evals']}"
        )
        print(
            f"BENCH {tag} gpu-vs-cpu rel: image={rel(image, img_c):.1e} "
            f"A={max(rel(x, y) for x, y in zip(ga, ga_c, strict=True)):.1e} "
            f"B={max(rel(x, y) for x, y in zip(gb, gb_c, strict=True)):.1e}"
        )

    def time_ms(call: Callable[[], list[Tensor]]) -> float:
        for _ in range(args.warmup):
            call()
        times: list[float] = []
        for _ in range(args.reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            call()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return statistics.median(times)

    t_f, t_a, t_b = time_ms(fwd), time_ms(bwd_a), time_ms(bwd_b)
    print(
        f"BENCH {tag} time_ms (median of {args.reps}, incl. output allocation): "
        f"forward={t_f:.3f} backward_pixel={t_a:.3f} backward_splat={t_b:.3f} "
        f"splat_speedup={t_a / t_b:.2f}x"
    )
    print(f"BENCH {tag} buckets_per_tile_max={int(counts.max()) // BUCKET + 1}")


if __name__ == "__main__":
    main()
