# Phase 1: macro blend backward in Faster-GS — first results (2026-10-27)

**Change:** `FasterGSMacro` is Faster-GS @1144604 plus `phase1/fastergs_macro.patch`, built by `phase1/make_variant.sh`.
- The patch replaces only `blend_backward_cu` (Taming-3DGS layout: a lane owns one splat, and pixels are passed lane to lane) with `blend_backward_macro_cu` (Phase 0 backward-B layout: a lane owns 6 pixels of the 16×12 tile and keeps T and ρ = G·(image − C) per pixel; the warp reduces per splat, and splats no pixel uses are skipped by ballot).
- It reuses Faster-GS's forward and its per-bucket (C, T) checkpoints unchanged. The forward, preprocess, preprocess backward and Adam are untouched.
- `FGS_BLEND_BWD=macro|taming` picks the kernel at run time.

H100 SXM 94 GB, ZIH Capella, torch 2.11.0+cu128, CUDA 12.8. Every run had SM at 1980 MHz with no throttling.

## Training: garden, 3 seeds, base and macro alternating on one node (job 4477625, node c35)

| variant | ms/step | train time (s) | PSNR (dB) | SSIM | LPIPS (VGG) | final Gaussians |
|---|---|---|---|---|---|---|
| base: FasterGS @1144604, stock Adam | 8.427 ± 0.029 | 252.8 ± 0.9 | 27.475 ± 0.027 | 0.8667 | 0.1206 | 4.148 M |
| **macro: FasterGSMacro** | **7.553 ± 0.006** | **226.6 ± 0.2** | 27.497 ± 0.048 | 0.8668 | 0.1205 | 4.158 M |
| difference | **−0.87 ms (−10.4%)** | **−26.2 s (−10.4%)** | +0.02 (within noise) | +0.0002 | −0.0001 | +0.25% |

Train time is the sum of `training_iteration` (ms/it × 30k), the same measure as in `experiment_log.csv`. The base matches the earlier Faster-GS runs (252.9 ± 1.0 s, 27.48 dB). Per-seed rows are in `experiment_log.csv`.

## Kernel timing

**On real garden views** (`BWDTIME`, job 4477625): the median of 20 runs of `_C.backward` on the same forward, every 1000 steps. Both variants share `preprocess_backward`, so the difference is the blend backward alone.

| step | Gaussians | stock ms | macro ms | saved ms | ratio |
|---|---|---|---|---|---|
| 1000 | 0.27 M | 0.751 | 0.491 | 0.26 | 1.53× |
| 5000 | 3.08 M | 3.150 | 2.372 | 0.78 | 1.33× |
| 10000 | 4.00 M | 3.719 | 2.810 | 0.91 | 1.32× |
| 20000 | 4.18 M | 3.746 | 2.872 | 0.87 | 1.30× |
| 30000 | 4.18 M | 3.746 | 2.875 | 0.87 | 1.30× |

The saving of 0.87 ms per step matches the training ms/step difference exactly.

At step 5000, the stock blend backward took 1.91 ms in nsys (`results/phase1/`). If preprocess backward and allocation stay the same, the macro blend backward is about 1.1 ms there, **about 1.7× faster than stock for that kernel**. This is an estimate; there is no nsys capture of the macro run yet. STAGE timings in the macro gradcheck run were bwd 3.00 ms at step 5000 and 3.67 ms at step 20000, against 3.69 and 4.54 ms in the base profiling run 4475452 (a different job).

**On synthetic scenes** (`bench/phase1/kernel_bench.py`, job 4477611): full `_C.backward`, median of 50 runs.

| scene | Gaussians | image | buckets | stock ms | macro ms | ratio |
|---|---|---|---|---|---|---|
| tiny | 1k | 160×120 | 100 | 0.096 | 0.077 | 1.24× |
| partial_tiles | 20k | 333×250 | 2.6k | 0.142 | 0.117 | 1.21× |
| garden_res | 200k | 1297×840 | 40k | 1.088 | 0.792 | 1.37× |
| small_splats | 500k | 1297×840 | 24k | 1.021 | 0.789 | 1.29× |
| heavy_overlap | 50k | 640×480 | 224k | 0.300 | 0.230 | 1.31× |

## Correctness

The metric is the same as phase0: ‖macro − stock‖₂ / ‖stock‖₂ in fp64, per output tensor.

1. **Synthetic parity, `phase1/tests/test_parity.py`: 20/20 pass on H100** (jobs 4477601 and 4477708). This covers 5 scenes (including partial tiles, 500k small splats, and heavy overlap with early termination), with densification info off / on / with the abs gradient, proper antialiasing, a determinism check and an empty scene. The worst rel error over all six parameter gradients and the densification rows is **2.3e-6**, against a limit of 1e-5.

2. **Real garden, every 1000 steps of a full 30k run** (`GRADCHECK`, job 4477708), also run against stock-vs-stock (`GRADFLOOR`: a second run of the stock kernel on the same forward):

   | gradient | macro vs stock (max over 30 checks) | stock vs stock (max) |
   |---|---|---|
   | means | 2.1e-4 (step 23000); otherwise ≤ 1.3e-5 | 2.5e-5 |
   | scales | 1.4e-2 (step 23000); typically 1e-4 to 1e-3 | 3.0e-3 |
   | rotations | 1.8e-3 | 1.1e-3 |
   | opacities | 4.8e-7 | 2.2e-7 |
   | sh0, sh_rest | 2.5e-7 | 1.8e-7 |
   | densification (grad-norm row) | 8.1e-7 | — |

   **Scales and rotations miss the 1e-5 target on real scenes, and the stock kernel misses it against itself too.** Their gradients come from the conic gradient through `preprocess_backward_cu`, which multiplies by 1/det(cov2d)². That amplifies the last-bit differences from atomic summation order for near-degenerate Gaussians. Macro vs stock is usually within 1–3× of the stock run-to-run noise. At a few steps (15000, 23000) it is about 10× higher: one outlier tensor of 4.1M Gaussians, dominated by a handful of them. Every gradient that does not pass through that factor (opacities, colours, SH, densification) agrees to ≤ 1e-6 on real views. Training quality matches (table above).

## v2: profile and three kernel options (2026-10-27, jobs 4477993, 4478014, 4478042)

The macro kernel is now one template with three switches. They are set as bits of `FGS_BLEND_BWD` (an int): 1 = macro, 2 = subtile skip, 4 = transposing reduction, 8 = 8 warps per block.

- **Subtile skip (2):** for each splat, skip the pixels whose 8×4 subtile its screen bounds miss. This is the same culling `blend_cu` applies, so the replay now sees exactly the forward's splat sequence. When a 2-row band misses, the whole warp skips it.
- **Transposing reduction (4):** a transposing butterfly reduces the 8 mean2d/conic/colour sums in 9 shuffles and the opacity and |mean2d| sums in 5–6, instead of 5 shuffles per value. The atomics are then issued by up to 11 lanes in parallel.
- **8 warps per block (8)** instead of 4.

All 92 parity tests pass for modes 1, 3, 5, 7 and 15 (max rel err < 2.5e-6).

**Nsight Compute, one launch per mode, synthetic scenes with the abs gradient on** (µs, registers, achieved occupancy, SM throughput, instructions):

| scene | stock | macro v1 (1) | + subtile skip (3) | + transpose reduce (5) | 3 + transpose (7) | 8 warps (9) |
|---|---|---|---|---|---|---|
| garden_res (200k) | 912 µs | **602** (1.52×) | 607 | 694 | 685 | 607 |
| small_splats (500k) | 618 | 381 (1.62×) | **365** (1.69×) | 448 | 415 | 393 |
| heavy_overlap (50k) | 215 | **155** (1.38×) | 156 | 176 | 176 | 195 |

- **The macro kernel is issue-bound,** not shuffle- or memory-bound. At garden_res: 73% SM throughput; most stalls are `not_selected`/`wait`, so warps are eligible and the scheduler is busy. It executes 459M instructions to stock's 747M, which is where the 1.5× comes from.
- **Transposing reduction is slower.** The instruction count barely changes, and the divergent per-lane atomics add branch stalls (1.0 vs 0.4 per instruction) and 7 registers. Shuffles were not the bottleneck.
- **8 warps per block** is neutral, and worse on heavy overlap, where occupancy drops.
- **Subtile skip** cuts instructions by 9% on small splats, and is neutral elsewhere.

**Real garden, subtile skip vs v1:**
- Full rasterizer backward ratio to stock at 4.16M Gaussians: 1.33× vs 1.30× (BWDTIME, job 4478014).
- Gradient parity is the same as v1. Scales and rotations stay within the stock run-to-run floor (max 6.1e-3, against a floor of up to 2.7e-3 in the same run). Everything else is ≤ 1e-6.

**Training, garden, 3 seeds, base and mode 3 alternating on one node (job 4478042, node c65):**

| variant | ms/step | train time (s) | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|---|---|
| base: FasterGS | 8.437 ± 0.006 | 253.1 ± 0.2 | 27.514 ± 0.041 | 0.8671 | 0.1202 |
| **macro + subtile skip (mode 3)** | **7.537 ± 0.023** | **226.1 ± 0.7** | 27.521 ± 0.035 | 0.8672 | 0.1203 |
| difference | −0.90 ms (**−10.7%**) | −27.0 s | +0.01 | +0.0001 | +0.0002 |

Against v1's −10.4% (job 4477625), the difference is within run-to-run noise. **Mode 3 is now the default** (`FGS_BLEND_BWD=macro`): it is never slower than v1 and it matches the forward's culling exactly. `macro1` and the numeric modes remain available.

## Not done yet / next

- **The kernel is issue-bound,** so further speed has to come from fewer instructions per (splat, pixel) pair, for example:
  - computing the exponent incrementally down a lane's column, since dy changes by a constant 2 px per row band;
  - computing `exp2` directly on a pre-scaled conic;
  - a warp-uniform early exit when no lane's pixel can still use the bucket.
- An nsys capture of a full training step to place the blend backward against the rest of the step. The 1.1 ms in-training estimate is consistent with ncu's 1.5× on the synthetic garden_res scene, but a direct measurement would confirm it.
- The other scenes (bicycle, room, bonsai), and stacking with the bf16 SH-rest moments (−6.7%, no quality cost) from `results/optim/`.
