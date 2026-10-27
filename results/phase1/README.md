# Phase 1 baseline measurements: Faster-GS `main` on an H100 (2026-10-27)

Faster-GS @c60ea26 in NeRFICG @623ad16, torch 2.11.0+cu128, CUDA 12.8, driver 580.159.04, ZIH Capella.
Raw rows are in `experiment_log.csv`. Job scripts are `cluster/profile_fastergs.sbatch`, `cluster/repeat_fastergs.sbatch`, and `cluster/patches/fastergs_profiling.py` (hooks are off unless `FGS_PROF=1`).

## Hardware: H100 SXM, 94 GB (job 4475452)

| | |
|---|---|
| Card | NVIDIA H100, device 0x2339, 700 W power limit, so SXM form factor; 94 GB (95,319 MB usable) |
| SMs / L2 | 132 SMs, 60 MB L2 |
| Memory | bus 6016 bit, 1593 MHz, so 2396 GB/s theoretical. ncu reports a peak of 2447 GB/s. **This is not the 3.35 TB/s of the 80 GB H100 SXM.** |
| Measured bandwidth | 2178 GB/s for a 2 GiB device-to-device copy (read+write), 2282 GB/s for a read-only `sum` |
| Clocks during training | SM 1980 MHz steady in every run (min = mean), memory 1593 MHz, 400–430 W mean, max 63 °C, **no throttle events** in any run |

Use **about 2.2 TB/s achievable** in the Phase 3 bandwidth model.

## Stage timing, garden, seed 0 (CUDA events, job 4475452)

`fwd` = learning-rate update, view sampling, render, and loss. `opt` = `optimizer.step()`, `zero_grad()`, and `post_optimizer_step()`.

| step | Gaussians | fwd ms | bwd ms | opt ms | total ms | opt share |
|---|---|---|---|---|---|---|
| 500 | 0.14 M | 1.219 | 1.101 | 0.114 | 2.43 | 5% |
| 5000 | 3.12 M | 1.613 | 3.688 | 2.414 | 7.72 | 31% |
| 20000 | 4.18 M | 1.760 | 4.542 | 3.234 | 9.54 | 34% |

## Time per step against Gaussian count (every 500 steps, `garden_main_4475452_stage_curve.txt`)

- **ms/step ≈ 1.60 + 1.88 × N (N in millions of Gaussians)**, fitted on steps ≥ 2000 (57 points).
- After densification ends (4.18M Gaussians, steps 15k–30k), a step takes 9.39 ms (range 9.32–9.51).
- These hook timings include extra host syncs. The clean runs average 8.36–8.47 ms per step over the whole 30k.

## Nsight Systems: steps 5000–5049, 3.1M Gaussians (job 4475452)

The kernels add up to 6.72 ms of GPU time per step, while the stages total 7.72 ms. The remaining ~1 ms per step is time the GPU sits idle between kernels (launch and host overhead).

| % | ms/step | calls/step | kernel |
|---|---|---|---|
| 35.8 | 2.404 | 6 | `adam_step_cu` (one per parameter group) |
| 28.6 | 1.921 | 1 | `blend_backward_cu` |
| 13.0 | 0.871 | 1 | `preprocess_backward_cu` |
| 6.7 | 0.451 | 1 | `blend_cu` (forward) |
| 5.5 | 0.368 | 14 | `FillFunctor<float>` (gradient zeroing) |
| 3.1 | 0.210 | 1 | `preprocess_cu` (forward) |
| 1.4 | 0.093 | 1 | `fusedssim_backwardCUDA` |
| 0.9 | 0.064 | 1 | `fusedssimCUDA` |
| 0.9 | 0.060 | 1 | `create_instances_cu` |
| 0.9 | 0.059 | 4 | cub radix sort (onesweep) |

## Nsight Compute: DRAM traffic for one step (step 5000, real clocks, job 4475452)

**7.70 GB of DRAM traffic per step** (4.39 GB read + 3.31 GB written) in 6.54 ms of kernel time, averaging 1176 GB/s.

| % time | ms | launches | read MB | write MB | GB/s | % of peak | kernel |
|---|---|---|---|---|---|---|---|
| 35.4 | 2.315 | 6 | 2950 | 2065 | 2166 | 89 | `adam_step_cu` |
| 29.1 | 1.906 | 1 | 367 | 34 | 210 | 9 | `blend_backward_cu` |
| 12.0 | 0.783 | 1 | 323 | 271 | 759 | 31 | `preprocess_backward_cu` |
| 7.1 | 0.462 | 1 | 76 | 227 | 657 | 27 | `blend_cu` |
| 5.1 | 0.332 | 14 | 0 | 601 | 1812 | 74 | `FillFunctor` (zero_grad) |
| 2.9 | 0.190 | 1 | 306 | 60 | 1922 | 79 | `preprocess_cu` |
| 1.4 | 0.091 | 1 | 78 | 11 | 976 | 40 | `fusedssim_backwardCUDA` |
| 1.0 | 0.067 | 4 | 31 | 0 | 468 | 19 | cub onesweep |
| 0.9 | 0.061 | 1 | 26 | 29 | 905 | 37 | `fusedssimCUDA` |
| 0.9 | 0.058 | 1 | 39 | 8 | 826 | 34 | `create_instances_cu` |

**Adam check:** 59 floats per Gaussian × 4 B × 7 passes (reads parameter, gradient, m, v; writes parameter, m, v) × 3.12M Gaussians = 5.16 GB predicted, against 5.02 GB measured. Adam runs at 89% of peak bandwidth, so it is purely bandwidth-bound. Zeroing the gradients adds another 0.6 GB.

## Scene baselines: Faster-GS `main`, 3 seeds each (mean ± sd)

| scene | resolution | train time (s) | PSNR (dB) | SSIM | Gaussians | job / node |
|---|---|---|---|---|---|---|
| garden | 1297×840 | 252.9 ± 1.0 | 27.48 ± 0.04 | 0.867 | 4.16 M | 4475239 / c59 |
| bicycle | 1237×822 | 251.0 ± 0.6 | 25.33 ± 0.05 | 0.768 | 4.87 M | 4475455 / c72 |
| room | 1557×1038 | 150.1 ± 1.2 | 31.64 ± 0.08 | 0.921 | 1.30 M | 4475454 / c16 |
| bonsai | 1559×1039 | 131.3 ± 0.8 | 31.97 ± 0.27 | 0.941 | 1.06 M | 4475453 / c24 |

LPIPS is computed with VGG, so it is not comparable to gsplat's (AlexNet). gsplat on garden (5.06M Gaussians, 27.69 dB) is kept as context only.
