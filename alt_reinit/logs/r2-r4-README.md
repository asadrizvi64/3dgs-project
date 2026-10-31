# R2 and R4 results: track R stops

**Result:** the hypothesis is not supported, and the track stops under the pre-registered rule.

| scene | reinit-target | reinit-same | stop |
|---|---|---|---|
| garden | **reduces quality** (SSIM) | **reduces quality** (SSIM) | **reduces quality** (SSIM, LPIPS) |
| room | **keeps quality, no time gain** (crossing not reproduced) | crossing not reproduced; quality a near-tie | **reduces quality** (LPIPS) |

The stop rule after R2 was: stop unless `reinit-target`'s candidate precedes `ref`'s on both scenes. It does not on garden,
and the room candidate did not reproduce in R4.

| | |
|---|---|
| Design | the user's 2026-10-31 decision (PLAN.md, "Amendment"): k = 5000 for both scenes, 3 seeds, the four arms |
| Jobs | R2 garden 4504373, R2 room 4504374, R4 room 4504544 |
| Code | 04de1b6 (R2, R4); the near-tie wording in the analysis is from 0c73752 |
| Analysis | [`r2-analysis.md`](r2-analysis.md), [`r4-room-analysis.md`](r4-room-analysis.md) |

## Garden: every intervention arm loses SSIM

Target (ref, 3 seeds, 30k): PSNR 27.418 ± 0.065, SSIM 0.8668 ± 0.0004, LPIPS 0.1204 ± 0.0004. No arm met it at any
evaluation.

| arm | 30k PSNR | 30k SSIM | 30k LPIPS | N | track s to 30k |
|---|---|---|---|---|---|
| ref | 27.418 | 0.8668 | 0.1204 | 4.17M | 256.0 |
| stop | 27.383 | 0.8618 | 0.1282 | 3.13M | 209.7 |
| reinit-same | 27.407 | 0.8642 | 0.1175 | 3.14M | 225.7 |
| reinit-target | 27.406 | 0.8654 | 0.1133 | 4.17M | 282.6 |

- **PSNR** is within the reference's seed spread for every arm.
- **SSIM** is below by 3–13 reference seed standard deviations in every arm. That is a clear loss, not a near-tie.
- **LPIPS** is better than ref in both reinit arms.
- **reinit-target is also 10% slower** to 30k. It carries 4.17M Gaussians from step 5000 on, while ref reaches that count
  only around 15k.
- **Reading (PLAN.md R2):** reinitialization helps against `stop` (same − stop: +0.025 dB, +0.0024 SSIM, −0.0106 LPIPS).
  But neither count recovers the SSIM that densification beyond 5k gives.

## Room: the crossings did not reproduce

**R2** (ref 3 seeds, 30k: PSNR 31.531 ± 0.209, SSIM 0.9180 ± 0.0039, LPIPS 0.2835 ± 0.0020):

| arm | candidate step | track s at candidate (ref: 150.8 at 30k) |
|---|---|---|
| reinit-target | 24,500 | 119.1 |
| reinit-same | 26,500 | 113.4 |

Both cleared PSNR only by +0.006 / +0.007 dB, within the reference spread, so both were flagged near-tie: inconclusive.
`stop` misses LPIPS (0.2861 vs 0.2835).

**R4** (3 seeds per candidate, one evaluation per run at that step only): the ref confirmation runs reached PSNR 31.672,
SSIM 0.9207, LPIPS 0.2825 at 30k. That is 0.14 dB above the R2 ref runs, within their spread.

| arm | step | PSNR | SSIM | LPIPS | track s | verdict |
|---|---|---|---|---|---|---|
| reinit-target | 24,500 | 31.563 | 0.9226 | 0.2673 | 120.8 | not reproduced: PSNR below |
| reinit-same | 26,500 | 31.491 | 0.9194 | 0.2784 | 115.0 | not reproduced: PSNR and SSIM below |

**At equal step (30k), exploratory, pooling R2 and R4 (6 runs each):**

| arm | PSNR | SSIM | LPIPS |
|---|---|---|---|
| reinit-target | 31.659 | 0.9231 | 0.2649 |
| ref | 31.602 | 0.9194 | 0.2830 |

`reinit-target` is at least as good on all three, clearly so on LPIPS. Its track time to 30k is 146.3 s against ref's
150.8 s (−3%). So on room it keeps (or slightly improves) quality, but its PSNR reaches the reference only at about the
same step. There is no confirmed time-to-quality gain.

## Images

No new image failures in any arm on either scene (> 3 dB below the reference median, on an image that never fails in
ref).
- **garden:** image 22 fails in a ref run and is more than 1 dB low in reinit-target.
- **room:** images more than 1 dB below the reference median are listed per arm in the analysis. Image 34 is low in
  every arm, ref included.

## Cost

- **Reinitialization:** 1.6 s (garden), 2.2 s (room). The removed / added rows are in the analysis.
- **Evaluator:** 0.53 s (garden) and 1.2 s (room) per call, outside the clock. Values equal to NeRFICG's final
  `metrics_8bit.txt` within 2e-5 (r2smoke 4504364).

**GPU time for the whole track: about 4.0 GPU-hours** of the ~8 ceiling.

| stage | jobs | time |
|---|---|---|
| R0 | 4503113, 4503144 | 21 min |
| R1 | 4503153 | 11 min |
| evaluator checks | 4504319, 4504356, 4504364 | 35 min |
| R2 | 4504373, 4504374 | 133 min |
| R4 | 4504544 | 40 min |

## Problems found and fixed on the way

- **R0 (4503113):** pytest was missing from the venv, and `/projects` is read-only on compute nodes. Fixed in 4607ea4.
- **Evaluator memory (4504319, located in 4504356):** NeRFICG sets `cudnn.benchmark = True`. On the first LPIPS-VGG call
  its algorithm search peaked at about 65 GiB, which also contaminated the runs' peak-VRAM numbers. The evaluator now
  computes metrics with benchmark off (04de1b6): +1.4 GiB garden, +3.5 GiB room, below the training peaks. The r2smoke
  4504319 runs are kept in the log but not recorded in `experiment_log.csv`.

## What was not done

- R3 (one factor at a time) and the nine-scene decision: R3 runs only if R2 shows promise, and it did not.
- The cheapest open question is the garden SSIM loss with better LPIPS. It may come from ending opacity resets at 5k
  together with densification. That would be a new hypothesis, not part of this track.
