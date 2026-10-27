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
- [ ] CUDA builds and `tests/test_cuda.py` passes on the cluster
- [ ] `gradcheck_report.py`: worst fp32 relative error < 1e-5, both backwards
- [ ] `bench.py`: `backward_splat` faster than `backward_pixel` at 1024² with 50k+ Gaussians
- [ ] Nsight Compute profile of both backwards saved
- [ ] Three sentences on why the per-splat version wins (below)

### Why the per-splat backward wins

_Fill in from the Nsight Compute profile: compare atomic throughput / L2 traffic,
warp stall reasons (look at `lg_throttle` and atomic stalls), and achieved occupancy._

1.
2.
3.

## Rendering rules

Kept identical in `reference.py`, `kernel_emulator.py` and `rasterize.cu`:

- A Gaussian reaches only pixels in tiles inside its 3σ tile rect.
- power = −½(A dx² + C dy²) − B dx dy; skip if power > 0.
- α = min(0.99, o·exp(power)); skip if α < 1/255; the clamp has zero gradient.
- Front to back; a pixel stops before the splat that would push T below 1e-4.
- out = Σ cᵢ αᵢ Tᵢ + T_final · bg.
