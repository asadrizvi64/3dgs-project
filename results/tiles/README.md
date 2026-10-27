# Image tiles vs active Gaussians (garden, 2026-10-27)

**Question:** if a training step uses only some image tiles, how many Gaussians receive a gradient? Those Gaussians are the rows the optimizer and the per-Gaussian backward pass must process.

**Method** (`cluster/analysis/tile_vs_gaussian.py`, job 4478967, H100 node c126):

- Model: stock Faster-GS on garden, seed 0, with checkpoints saved at 5k steps (densifying, 3.11 M Gaussians), 15k (densification just ended, 4.15 M) and 30k (final, 4.14 M). Final test PSNR 27.50.
- Views: 41 training views (every 4th).
- Selection: for each view, choose a fraction of the 16×16 tiles with one of four policies:
  - random single tiles;
  - random 64×64 blocks (4×4 tiles);
  - single tiles with the highest L1 error;
  - blocks with the highest L1 error.
- Measurement: backpropagate an L1 loss restricted to those tiles through the real Faster-GS rasterizer and count the Gaussians whose gradient is nonzero.
- Raw data: `garden_tile_vs_gaussian.csv`. Means over views: `garden_summary.txt`.

## Result 1: one view touches only about 30% of the Gaussians

| checkpoint | Gaussians receiving a gradient from the full image | lean-Adam visibility mask (`n_touched_tiles > 0`) | rows the mask updates without a gradient |
|---|---|---|---|
| 5k | 31.0% of N | 34.5% | +11% |
| 15k | 31.2% | 35.1% | +12% |
| 30k | 29.7% | 35.1% | +18% |

Every Gaussian with a gradient was inside the mask; the mask missed none.

## Result 2: tiles map to Gaussians roughly one-to-one; blocks do better than scattered tiles

The table shows Gaussians receiving a gradient, as a percentage of those active for the full image (30k checkpoint; 5k and 15k are within ±2 points).

| tiles used | random tiles | random 64-px blocks | top-error tiles | top-error blocks | L1 error covered by top-error tiles / blocks |
|---|---|---|---|---|---|
| 5% | 9.8% | 6.2% | 9.2% | 7.2% | 13% / 12% |
| 10% | 18.6% | 12.1% | 17.1% | 14.1% | 22% / 20% |
| 25% | 40.7% | 29.7% | 37.1% | 31.7% | 43% / 40% |
| 50% | 68.1% | 56.2% | 64.7% | 58.9% | 69% / 66% |
| 75% | 87.0% | 79.4% | 86.7% | 83.5% | 88% / 86% |

- **Gaussian sharing is modest.** At 25% of tiles, scattered tiles activate 41% of the Gaussians and contiguous blocks only about 30%. So spatially clustered selection keeps Gaussian work close to proportional to pixel work.
- **Error is not concentrated.** The worst 25% of tiles hold only 40–43% of the L1 error, both mid-training and at the end. Error-guided selection concentrates L1 error about 1.6× per pixel. That is not a bound on learning progress, since error size is not the same as useful gradient.
- **The pattern doesn't change over training.** The 5k, 15k and 30k curves are nearly identical.

## What this means for speed (estimate, not measured)

The estimate starts from step-20k stage times: forward 1.67, backward 4.53 and optimizer 3.23 ms, for a total of 9.43 ms. It assumes:

- blend work scales with pixels;
- per-Gaussian work scales with the number of Gaussians that receive a gradient;
- forward still renders the full image.

Under those assumptions, a step using 25% of tiles as top-error blocks costs about 3.2 ms:

| part | ms |
|---|---|
| full forward | 1.67 |
| blend backward on 25% of pixels | 0.78 |
| per-Gaussian backward on 27% of touched Gaussians | 0.38 |
| sparse Adam on those rows | 0.33 |
| **total** | **about 3.2** |

That step supervises a quarter of an image, so it costs about **12.8 ms per image's worth of pixels, against 9.4 ms for dense training**. Even if the forward pass also ran on only the selected tiles, the cost would be at most about 20% below dense per pixel (7.6 ms).

**Conclusion:** because Gaussian work tracks pixel work almost one-to-one, tile sampling doesn't make supervision cheaper per pixel. It pays off only if training needs fewer pixel visits to reach the same quality. In steps, a 3.2 ms sampled step can afford up to about 2.95× as many steps as dense 9.43 ms steps and still finish sooner. That is the time-to-quality question TurboGS addresses, and it has to be tested by training, not by profiling.

**Caveats on these numbers:**
- The 3.2 ms figure assumes selective (sparse) Adam. With dense Adam the sampled step is about 6.1 ms.
- The activity counts above use an L1-only loss. The training loss also has an SSIM term, which spreads gradients about 5 px beyond each selected block. `cluster/analysis/time_to_quality.py` measures both effects every step with the real loss.

Measured fact that does hold: using contiguous blocks instead of scattered tiles cuts the Gaussians each sparse step touches by about 25% at the same pixel budget.
