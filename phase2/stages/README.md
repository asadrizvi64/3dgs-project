# Renders during training: stock vs A+B vs A+B+C (all 9 Mip-NeRF360 scenes, seed 0)

Jobs 4483157–4483165, 2026-10-28, built from [`../build_plus.sbatch`](../build_plus.sbatch) with `STAGES=stages`.

- **Runs:** each scene trained stock Faster-GS, FasterGSPlus A+B and FasterGSPlus A+B+C on seed 0. Every 1,000 iterations the full test set was rendered and evaluated, and the training time (CUDA events) logged. That gives 30 stages per run and 27 runs.
- **Not benchmark values.** The repeated evaluations change these runs: peak memory reads about 67 GiB because of the test renders, and the final numbers are one seed. The benchmark is `../RESULTS.md`, step 3. These runs are in `experiment_log.csv` with role `stage-render`.
- **Opacity resets.** Faster-GS resets opacities every 3,000 iterations while densifying, and test PSNR collapses to about 8–10 dB right after each reset. The 3k, 6k, 9k and 12k stages fall exactly there, so the figures leave them out.

## Files

| file | what |
|---|---|
| [`psnr_vs_time.png`](psnr_vs_time.png) | Test PSNR vs training time, one panel per scene. The dashed line is stock's PSNR at 30k. |
| [`psnr_vs_time.csv`](psnr_vs_time.csv) | Every point: scene, arm, iteration, training time, test PSNR (all 30 stages, reset stages included). |
| `sheets/<scene>_view<i>_by_iteration.jpg` | Rows at 1k, 2k, 4k, 7k, 10k, 15k, 20k, 25k and 30k. Columns: stock, A+B, A+B+C, with ground truth on top. Each cell shows that view's PSNR and the training time. |
| `sheets/<scene>_view<i>_by_time.jpg` | Rows are equal training-time budgets: 5, 10, 20, 35, 50, 75 and 100% of stock's total. Each cell is the latest stage that arm reached within the budget. |
| `sheets/*_crop.jpg` | The same sheets zoomed to the third of the frame where stock's 7k render differs most from the ground truth (red box on the ground-truth image). |

There are two test views per scene: view 0 is the one with the median stock PSNR at 30k, and view 1 the one with the largest stock gain from 7k to 30k.

The full renders, all 30 stages and every test view (35 GB), are on Capella in `/projects/p_scads_mlhpc/syas272h/3dgs-archive/stages/<scene>/r_<scene>_<arm>_s0/test_<iteration>/rgb/`, outside the expiring workspace. Built by [`../stage_curves.py`](../stage_curves.py) and [`../stage_sheets.py`](../stage_sheets.py).

## When each arm first reaches stock's final quality (seed 0, indicative only)

The first stage at or above stock's own 30k test PSNR, on the training clock. Stages are 1,000 iterations apart, and neighbouring stages differ by up to about 0.05 dB from noise, so one noisy point can move a crossing by several thousand iterations.

| scene | stock 30k PSNR | stock | A+B | A+B+C |
|---|---|---|---|---|
| bicycle | 25.320 | 153 s (20k) | 128 s (20k) | 107 s (18k) |
| flowers | 21.630 | 149 s (26k) | 103 s (22k) | 121 s (27k) |
| garden | 27.520 | 209 s (25k) | 150 s (22k) | 160 s (25k) |
| stump | 26.690 | 79 s (14k) | 73 s (14k) | 67 s (14k) |
| treehill | 22.800 | 174 s (29k) | not reached (22.63 at 30k) | not reached (22.72) |
| room | 31.830 | 149 s (30k) | not reached (31.76) | not reached (31.73) |
| counter | 29.160 | 154 s (30k) | not reached (29.11) | 116 s (28k) |
| kitchen | 31.450 | 198 s (30k) | not reached (31.35) | 151 s (28k) |
| bonsai | 32.190 | 133 s (30k) | 97 s (26k) | not reached (31.96) |

- **Outdoor scenes plateau early.** All three arms reach stock's final quality by 14k–27k on bicycle, flowers, garden and stump. This is the case for the "stop early, keep the 30k learning-rate schedule" lever, to be tested with several seeds.
- **Indoor scenes still improve at 30k**, so stopping early would likely cost quality there.
- **Treehill** is the one outdoor scene where neither stack reaches stock's 30k value on this seed. It is also the A+B+C worst scene in the benchmark.
- With one seed, "not reached" by under 0.1 dB is within run-to-run noise. The benchmark's 3-seed paired deltas are the quality result.
