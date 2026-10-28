# Time to quality from the 15k state: dense vs block sampling (garden, 3 seeds)

**Result:** pixel-block sampling does not save time. Error-guided blocks give more PSNR per image's worth of pixels than dense training does, but each sampled step costs far too much to make up for taking 4× as many steps. Measured on the clock and on the projected cost model (scan included), every block arm reaches a given PSNR later than dense training with selective Adam.

The arm that reaches the targets fastest is **dense images + selective (visible-only) Adam** (`dense:sparse`). It reaches the dense baseline's final PSNR in about half the time on every seed and every clock, and it ends +0.05 dB higher. This is a lead for the real trainer, not yet a result there; see the caveats.

## Setup

- **Data and seeds:** Faster-GS (nerficg, lean-Adam commit 1144604), garden at 1/4 resolution, seeds 0/1/2, Capella H100 (jobs 4479587/88/89, all `JOB OK`). For each seed, a stock run saves the full parameter and Adam state at step 15,000, after densification ends at 14,900.
- **Harness:** [`cluster/analysis/time_to_quality.py`](../../cluster/analysis/time_to_quality.py), which uses the stock Faster-GS loss (0.8·L1 + 0.2·(1−SSIM)).
  - Each arm trains from the 15k state for 15,000 *image-equivalents*: 15,000 full images, or 60,000 steps at 25% of the pixels.
  - Test PSNR is measured on 8-bit images every 1,500 image-equivalents.
- **Arms** (`policy:adam:lr`):
  - `dense`: full image, one random view per step.
  - `err_block` / `rand_block`: 25% of 64×64-px blocks, drawn without replacement, weighted by block L1 error or uniformly.
  - `adam=sparse`: only rows with a nonzero gradient are updated (selective Adam).
  - `lr=frac`: all learning rates ×0.25 for sampled arms, so that one image's worth of pixels moves parameters about as far as one dense step. `lr=none` keeps stock LRs.
- **Clocks:**
  - `train_s`: measured wall time of the harness's training steps.
  - `proj_s`: cost model calibrated on each seed's own checkpoint with CUDA events (`logs_s*/calib_s*.json`). It is forward + backward, with the backward share scaled by touched pixels and active rows, plus dense or selective Adam (selective Adam fitted as c0 + c1·active fraction).
  - `proj_scan_s`: `proj_s` plus the measured 2.1 ms visibility/mask scan that selective Adam needs. This is the honest projection.

Calibration, mean of 3 seeds: forward 1.57 ms, backward 4.23 ms, scan 2.11 ms, dense Adam 3.31 ms, selective Adam 0.85 + 2.62·a ms, N ≈ 4.17 M.

## PSNR at equal image-equivalents (mean of 3 seeds; start 27.121)

| arm | 1,500 | 6,000 | 15,000 | ms/step measured | ms/step projected (+scan) |
|---|---|---|---|---|---|
| dense : dense Adam | 27.162 | 27.339 | 27.421 | 11.1 | 9.10 |
| dense : selective Adam | 27.223 | 27.391 | 27.471 | 9.5 | 9.54 |
| err_block : dense Adam | 27.347 | 27.421 | 27.492 | 11.8 | 6.33 |
| rand_block : dense Adam | 27.317 | 27.393 | 27.474 | 11.8 | 6.23 |
| err_block : selective Adam | 27.383 | 27.447 | 27.498 | 9.7 | 6.30 |
| rand_block : selective Adam | 27.357 | 27.422 | 27.473 | 9.7 | 6.17 |
| err_block : selective, stock LR | 27.267 | 27.440 | **27.549** | 9.6 | 6.27 |

Stock 30k Faster-GS on the same seeds: 27.472 / 27.397 / 27.486 (mean 27.452). Seed std of the final values is 0.01–0.02 dB.

- Error-guided blocks beat random blocks by +0.02–0.03 dB at every point, with both Adam variants and on all seeds.
- The bottleneck is cost per step. A sampled step costs 6.3 ms projected, including the scan, and 9.7 ms measured, against 9.1 ms for a dense step. Spending one image's worth of pixels therefore costs about 25 ms projected and 39 ms measured, against 9.1 ms dense. To break even at 4× the steps, a sampled step would need to cost under about 2.3 ms.
- The earlier 3.2 ms estimate (`results/tiles`) assumed an L1-only loss and no scan cost. With the real loss, and the scan counted, it is 6.3 ms.

## Time to reach a target PSNR (seconds after the 15k state; seeds 0/1/2)

Target A is each seed's own `dense:dense` final PSNR (harness baseline).

| arm | measured | projected | projected + scan |
|---|---|---|---|
| dense : dense Adam | 166 / 167 / 101 | 136 / 138 / 82 | 136 / 138 / 82 |
| **dense : selective Adam** | **82 / 55 / 79** | **64 / 44 / 62** | **82 / 56 / 79** |
| err_block : dense Adam | 273 / 268 / 316 | 146 / 144 / 169 | same |
| rand_block : dense Adam | 448 / 401 / 334 | 233 / 213 / 177 | same |
| err_block : selective Adam | 179 / 208 / 104 | 77 / 92 / 45 | 115 / 137 / 67 |
| rand_block : selective Adam | 216 / 307 / 136 | 89 / 131 / 56 | 135 / 199 / 86 |
| err_block : selective, stock LR | 216 / 213 / 165 | 93 / 94 / 71 | 139 / 141 / 107 |

Target B is each seed's stock 30k PSNR. `dense:dense` never reaches it in the harness. `dense:sparse` reaches it in 128/56/87 s measured and 127/56/87 s projected with scan. `err_block:sparse` takes 372/209/316 measured and 239/138/204 with scan.

**Whole training:** the first 15k steps are shared and cost about 125 s. This is the stock 30k run's 250 s minus about 127 s for steps 15k–30k at the stock 8.4 ms/it.

- With selective Adam after step 15k, the whole run to dense-baseline quality takes about 125 + 72 ≈ 197 s instead of 125 + 145 ≈ 270 s on the harness's measured clock. That is about **1.35×**, or about 1.3× on the projected clock.
- Block sampling, even the best arm (err_block + selective Adam), comes to about 125 + 164 s measured and 125 + 106 s projected with scan. It does not beat the selective-Adam dense arm on any clock.

## Caveats

1. **Projections are provisional.** `proj_*` assumes a rasterizer whose backward pass skips untouched pixels and Gaussians. The stock kernels render and backpropagate the full image, so the measured sampled step (9.7 ms) is no cheaper than a dense one.
2. **Step count / LR confound.** Sampled arms take 4× the steps at ¼ LR, so part of their per-image gain may come from smaller, more frequent steps rather than from choosing pixels. The stock-LR block arm (4× steps, stock LR) is the best on PSNR (27.549) but the worst early, which suggests the harness's post-15k schedule is under-stepped. The control is missing: dense images, 4× steps, ¼ LR.
3. **The harness is not the trainer.**
   - The harness `dense:dense` arm ends 0.03 dB below the stock 30k runs, and its steps carry Python overhead (11.1 ms measured vs 8.4 ms/it in the trainer). It also samples views with replacement and adds per-step statistics.
   - The selective-Adam speed-up has to be confirmed inside the Faster-GS trainer: dense Adam up to 15k, selective Adam after, full 30k timing, 3 seeds.
4. **Selective Adam from scratch is different.** It lost about 0.09 dB on one seed in `results/optim`, and Faster-GS reports −0.13 dB for skipping invisible updates. Here it is switched on only after densification and gains +0.05 dB. These are different regimes, so the claim is only for "after 15k".
5. **Scope:** one scene (garden) and one budget (25%, 64-px blocks).

## Files

- `garden_ttq_s{0,1,2}.csv`: per-eval rows with columns policy, adam, lr, frac, seed, n, step, image_equiv, iteration_equiv, train_s, proj_s, proj_scan_s, test_psnr, active_of_n, pix_frac.
- `logs_s*/calib_s*.json`: per-seed cost-model calibration.
- Driver: [`cluster/time_to_quality.sbatch`](../../cluster/time_to_quality.sbatch). Patch that saves state: [`cluster/patches/fastergs_save_state.py`](../../cluster/patches/fastergs_save_state.py).
