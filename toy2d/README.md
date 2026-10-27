# Phase 0 toy: 2D Gaussian image fitting in CUDA

Fit one image with 2D Gaussians using a 3DGS-style tile rasterizer, with two backward passes:

| Backward | Parallelism | Atomics | Source |
| --- | --- | --- | --- |
| `pixel` | thread per pixel, splats back to front, T recovered by dividing by (1 − α) | one per (pixel, splat) pair | original 3DGS |
| `splat` | warp per (tile, 32-splat bucket), lane per splat, pixels handed lane to lane | one per (splat, bucket) | Taming 3DGS |

## Layout

```
gs2d/params.py       learnable Gaussians -> means, conic, color, opacity
gs2d/binning.py      tile binning + bucket bookkeeping (PyTorch, shared by every backend)
gs2d/reference.py    dense PyTorch rasterizer; ground truth via autograd
gs2d/raster.py       autograd wrapper; JIT-builds csrc/rasterize.cu
csrc/rasterize.cu    forward, backward_pixel, backward_splat kernels
tests/kernel_emulator.py  line-by-line Python mirror of the kernels (no GPU needed)
tests/test_cpu.py    binning, reference gradcheck, emulator vs autograd
tests/test_cuda.py   CUDA fp32 vs fp64 reference (skips without a GPU)
gradcheck_report.py  G0 error table
bench.py             kernel timings (CUDA events, median)
fit.py               image fitting demo
jobs/                cluster setup + SLURM job
```

## Run

On the laptop (CPU only), this checks the math and the kernel logic, not the CUDA code:

```bash
python -m pytest -q tests
```

On the cluster:

```bash
bash jobs/setup_env.sh     # once; edit the module lines first
sbatch jobs/phase0.sbatch  # build, tests, gradient table, bench, fit, ncu
```

Results land in `results/` tagged with the commit hash.

## Gate G0

- [x] CPU tests pass (reference gradcheck, emulator matches autograd to 1e-9). **Done 27 Oct 2026.**
- [x] CUDA builds and `tests/test_cuda.py` passes on the cluster (25/25, H100, job 4473561, commit cfda5b4)
- [x] `gradcheck_report.py`: worst fp32 relative error 7.4e-6 < 1e-5, both backwards ([table](results/gradcheck-cfda5b4-4473561.md))
- [x] `bench.py`: `backward_splat` faster than `backward_pixel` at 1024² with 50k Gaussians (0.49 vs 0.87 ms, 1.76×)
- [x] Nsight Compute profile of both backwards saved (`profiles/bwd-cfda5b4-4473561.ncu-rep` on the cluster)
- [ ] Three sentences on why the per-splat version wins (draft below: check against the profile and rewrite in your words)

### Results (H100, 27 Oct 2026)

| Image | Gaussians | (tile, splat) pairs | bwd pixel (ms) | bwd splat (ms) | splat speedup |
| --- | --- | --- | --- | --- | --- |
| 512² | 10k | 44k | 0.304 | 0.167 | 1.82× |
| 512² | 50k | 112k | 0.241 | 0.257 | 0.94× |
| 512² | 200k | 338k | 0.284 | 0.497 | 0.57× |
| 1024² | 10k | 105k | 1.109 | 0.363 | 3.05× |
| 1024² | 50k | 200k | 0.866 | 0.492 | 1.76× |
| 1024² | 200k | 455k | 0.799 | 0.826 | 0.97× |

Nsight Compute, 1024² / 50k:

| Kernel | Time | L2 throughput | SM throughput | Active warps | Active threads per warp instr. |
| --- | --- | --- | --- | --- | --- |
| backward_pixel | 909 µs | 78.8% | 36.8% | 56.4% | 23.2 / 32 |
| backward_splat | 574 µs | 7.3% | 68.1% | 41.1% | 13.9 / 32 |

Fit demo: 512² synthetic image, 20k Gaussians, 1000 iterations in 3.9 s, 45.6 dB PSNR.

### Why the per-splat backward wins (draft)

1. The per-pixel backward issues one atomicAdd for every (pixel, splat) pair, so it saturates the L2
   atomic units (79% L2 throughput) while the SMs sit at 37%: it is bound by atomic traffic, not math.
2. The per-splat backward keeps each splat's gradient in registers across the tile's 256 pixels and
   commits once per bucket, cutting L2 throughput to 7% and turning the kernel compute-bound (68% SM).
3. Its cost is fixed work per bucket (256 + 31 warp steps, only 14 of 32 lanes useful on average), so it
   loses when Gaussians are tiny and cover few pixels (≤ 1× at 200k Gaussians) and wins by 1.8–3× when
   footprints are large, which is the regime real 3DGS scenes are in.

## Rendering rules

Kept identical in `reference.py`, `kernel_emulator.py` and `rasterize.cu`:

- A Gaussian reaches only pixels in tiles inside its 3σ tile rect.
- power = −½(A dx² + C dy²) − B dx dy; skip if power > 0.
- α = min(0.99, o·exp(power)); skip if α < 1/255; the clamp has zero gradient.
- Front to back; a pixel stops before the splat that would push T below 1e-4.
- out = Σ cᵢ αᵢ Tᵢ + T_final · bg.
