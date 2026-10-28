# Selective Adam after densification, in the real Faster-GS trainer (5 Mip-NeRF360 scenes, 3 seeds each)

## Generalization: bicycle, stump, room, counter (added 2026-10-28)

Same driver and code as garden (`cluster/sparse_switch.sbatch`, commit b5c2186). There was one job per scene, with stock and switch@15k interleaved per seed on that job's node:

| scene | job | node | scale |
|---|---|---|---|
| bicycle | 4480521 | c95 | 1/4 |
| stump | 4480522 | c60 | 1/4 |
| room | 4480523 | c60 | 1/2 |
| counter | 4480524 | c21 | 1/2 |

All 24 runs completed. The SM clock held 1980 MHz in every run. Logs are in `<scene>/`, and job summaries in `logs/gs-switch-<scene>-<job>.out`.

**Per scene: training-step time (CUDA events) and paired quality at 30k (switch − stock, per seed)**

| scene | stock s | switch s | time reduction (per seed) | 2nd half | ΔPSNR (per seed) | mean ΔPSNR | ΔPSNR at 15k (pre-switch) | ΔSSIM | ΔLPIPS |
|---|---|---|---|---|---|---|---|---|---|
| garden (dev) | 255.8 | 230.8 | **9.8%** (9.3 / 9.7 / 10.4) | −17.3% | +0.082 / +0.101 / +0.070 | +0.084 | +0.035 | +0.0009 | −0.0006 |
| bicycle | 252.7 | 218.5 | **13.5%** (13.2 / 13.7 / 13.6) | −22.7% | +0.079 / +0.032 / +0.075 | +0.062 | +0.010 | +0.0006 | +0.0002 |
| stump | 201.3 | 168.6 | **16.3%** (16.7 / 15.7 / 16.5) | −28.4% | +0.077 / +0.116 / +0.086 | +0.093 | +0.038 | +0.0016 | −0.0001 |
| room | 150.1 | 143.1 | **4.7%** (4.0 / 5.1 / 5.0) | −9.2% | +0.149 / +0.018 / −0.024 | +0.048 | −0.027 | +0.0004 | −0.0004 |
| counter | 153.1 | 148.1 | **3.3%** (3.9 / 2.9 / 3.0) | −6.0% | −0.032 / +0.063 / +0.065 | +0.032 | −0.011 | +0.0003 | +0.0002 |

Stock 30k PSNR by scene: bicycle 25.301, stump 26.614, room 31.615, counter 29.079.

- **Claim: faster training with quality within the predefined tolerances on five scenes.**
  - Quality gate (pre-registered in `phase2/PLAN.md`: on every scene, mean ΔPSNR ≥ −0.05 dB and ΔLPIPS ≤ +0.002): **passed on all 5 scenes.**
  - All 15 paired runs were faster.
  - 13 of 15 paired seeds have higher PSNR. The two lower ones (room seed 2 −0.024 dB, counter seed 0 −0.032 dB) are within tolerance, and a few individual SSIM/LPIPS deltas are slightly worse. So this is not "no quality loss anywhere"; it is "within tolerance everywhere".
- **Change after the switch:** the ΔPSNR from 15k to 30k (difference-in-differences) is +0.049 / +0.052 / +0.055 / +0.075 / +0.043 dB for garden / bicycle / stump / room / counter. The 15k gaps are pure run-to-run noise, since the arms are identical up to then. That this gain is positive on all 5 scenes is the strongest evidence so far of a small real quality gain (about +0.05 dB), but it is not yet proven: per-seed noise at 30k is of the same size.
- **Speed gain differs by scene; likely explanation (not yet measured): how much of the step the optimizer is.**
  - The large outdoor scenes (4.2–4.9 M Gaussians at 1/4 resolution) save 10–16% of training (23–28% in the second half).
  - The indoor scenes (1.0–1.3 M Gaussians at 1/2 resolution, about 1557×1038) save only 3–5%.
  - Hypothesis: with fewer Gaussian parameters and more pixels, rendering (especially the backward pass) is a larger share of the step and dense Adam is a smaller one, so skipping Adam rows saves less.
  - **Measured since (Phase 2 step 4a, seed 0, garden vs room):** the optimizer is 31–34% of stock's step on garden but 17–20% on room, where blending dominates (blend backward 38% of kernel time). See `phase2/profile/README.md`. Scene size alone does not establish this; the stage profiling planned in Phase 2 (`FGS_PROF`) must measure backward-render time versus Adam time on an indoor scene before this is stated as the cause.
  - **The mean over the 5 scenes is −9.5%; the median is −9.8%.**
- **Final Gaussian count:** the switch arm ends with 0–4% more Gaussians (room +2–4%; stump −1.2% on one seed). What is verified: the reported counts are taken after Faster-GS's end-of-training opacity prune (`training_cleanup(min_opacity)`).
  - What is not verified: whether the difference comes from that prune. Within each run N stops changing when densification ends at 14.9k, but stock and switch runs are trained independently, so their counts at 15k need not match, and neither the 15k count nor the count just before cleanup was recorded.
  - Hypothesis: rows that selective Adam does not update keep their opacity instead of drifting under momentum, so fewer fall below the prune floor.
  - The v4 clock hook records N at 15k, at the last training step (before cleanup), and after cleanup (`n_gaussians.txt`), which settles this in future runs without a dedicated experiment.
  - **Update (Phase 2 step 3, 81 runs on 9 scenes):**
    - Within each run, the count at 15k equals the count at the last training step.
    - Between arms the counts can already differ at 15k: up to −1.6% on stump. In those runs A is also on from step 0, so this does not isolate B.
    - The end-of-training cleanup then removes 0.4–3.5% of Gaussians in stock and 0.1–0.7% in the selective-Adam arms.
    - Final count differences therefore reflect both the 15k differences and the different cleanup losses. See `phase2/RESULTS.md`.
  - More Gaussians make rendering slightly more expensive after training.
- **For Phase 2:** B passes its gate and goes into the stack. On indoor scenes it adds little, so the stack's per-scene gain there will come mostly from the backward kernel (A).

## Garden (development scene, job 4480440)


**Result:** we ran stock Faster-GS with dense Adam up to iteration 15,000 and the lean-Adam selective path afterwards.

- **Speed:** training-step time drops from **255.8 s to 230.8 s (−9.8%, 1.11×)** for the full 30k-step run. Per seed the drop is 9.25% / 9.70% / 10.36%. The post-switch half is 17.3% faster on average.
- **Quality:** **no observed quality loss in these garden runs.** Paired PSNR is +0.084 dB (+0.082 / +0.101 / +0.070), SSIM +0.0009 and LPIPS −0.0006.
- **Why the PSNR gain is not causal:** the switch runs were already +0.035 dB ahead at 15k, before anything differed. The gap grew afterwards, which supports the hypothesis but does not isolate a quality gain caused by the switch.

Selective Adam from step 0 cost 0.09–0.13 dB (`results/optim`; the Faster-GS paper's own ablation reports −0.13 dB). Deferring it until Gaussian growth has finished kept the speed benefit without an observed quality loss. Garden is the development scene; bicycle, stump, room and counter test whether the 15k policy generalizes.

## Setup

- **Code:** FasterGS at lean-Adam commit 1144604 (nerficg 623ad16), garden, 1/4 resolution, 30k iterations, seeds 0/1/2.
- **Hardware:** one Capella H100 (c95, job 4480440). The six runs were interleaved base/switch per seed. The SM clock held 1980 MHz throughout, with no throttling.
- **Arms:**
  - `swbase`: lean-Adam patch present, flags off. This is the stock `adam_step` kernel.
  - `sw15k`: identical, plus `FGS_SPARSE_FROM=15000` ([patch](../../cluster/patches/fastergs_sparse_from.py)). From iteration 15,000 the FusedAdam updates only Gaussians whose footprint touches at least one tile in the current view (`n_touched_tiles > 0`), using the visibility mask the rasterizer already produces.
  - This is the trainer's cheap, broader rule. It is not the harness's nonzero-gradient rule from `results/ttq`, so this result stands on its own. Visible fraction at step 30k: 0.34–0.39.
- **Clocks:**
  - Training-step time: CUDA events around every `training_iteration`, summed (`logs/*.log.seg`, SEGCLOCK). It agrees with NeRFICG's own timer to within 1 s.
  - NeRFICG timer (`*_timings.txt`): 8.57 ms/it base vs 7.74 ms/it switch.
  - Total job duration is reported separately below.
- **Evaluation:** test-set metrics (8-bit) every 2,500 iterations (`logs/*.log.metrics`). Densification ends at 14,900, so N is fixed during the comparison.

## Results

| seed | arm | train s (CUDA events) | NeRFICG ms/it | PSNR | SSIM | LPIPS | Gaussians |
|---|---|---|---|---|---|---|---|
| 0 | stock | 255.4 | 8.56 | 27.400 | 0.8667 | 0.1209 | 4,143,207 |
| 0 | switch@15k | 231.8 | 7.77 | 27.482 | 0.8676 | 0.1205 | 4,170,530 |
| 1 | stock | 255.9 | 8.58 | 27.410 | 0.8658 | 0.1210 | 4,149,750 |
| 1 | switch@15k | 231.1 | 7.75 | 27.510 | 0.8669 | 0.1201 | 4,188,235 |
| 2 | stock | 256.1 | 8.58 | 27.446 | 0.8671 | 0.1207 | 4,162,763 |
| 2 | switch@15k | 229.6 | 7.70 | 27.516 | 0.8679 | 0.1202 | 4,182,881 |
| **mean** | stock | **255.8** | 8.57 | **27.419** | 0.8665 | 0.1209 | |
| **mean** | switch@15k | **230.8** | 7.74 | **27.503** | 0.8674 | 0.1203 | |

- **Training-step time:** the first 15k iterations take 114.3–115.3 s in both arms, as they should since they are identical. All of the saving is in the second half: 141.0 s becomes 116.6 s.
- **Total job duration per run:** stock 9:50 / 9:47 / 9:57 vs switch 9:21 / 9:24 / about 9:36, a saving of about 25 s per run, the same as the training saving. Most of that duration is the twelve intermediate test evaluations and checkpoint writes these runs did for the curves. A normal run evaluates only at the end.

## How large is the quality gain? (noise check)

The two arms are not bit-identical before the switch. CUDA atomics make same-seed runs differ, and at 15k the switch runs happened to be ahead by 0.04 / 0.05 / 0.02 dB (mean +0.035). Same-seed reruns of stock Faster-GS at 30k also vary by up to 0.11 dB. To account for this we pooled every stock 30k run of this exact code (lean-Adam commit 1144604, flags off) on garden in `experiment_log.csv`:

| seed | stock runs (30k PSNR) | mean | switch@15k | gain |
|---|---|---|---|---|
| 0 | 27.472 (ttq), 27.400 (here), 27.508 (optim) | 27.460 | 27.482 | +0.022 |
| 1 | 27.397 (ttq), 27.410 (here), 27.456 (optim) | 27.421 | 27.510 | +0.089 |
| 2 | 27.486 (ttq), 27.446 (here), 27.500 (optim) | 27.477 | 27.516 | +0.039 |

Against these 9 stock runs the switch averages **+0.05 dB** and is above the per-seed stock mean on every seed. The earlier repeat runs at Faster-GS commit c60ea26, before the lean-Adam patch, are left out of the pool because they are a different build. They are 27.43–27.51 for seeds 0–2, so including them would not change the conclusion. The 9 stock runs come from three jobs on different nodes; only the 3 paired runs here share a node with the switch runs.

## Time to stock quality (same crossing rule for both arms)

The target is each seed's stock final PSNR at 30k. For each arm, the time is the first evaluation at or above the target, linearly interpolated from the previous evaluation, on the CUDA-event training clock.

| seed | target | stock first crossing | switch first crossing |
|---|---|---|---|
| 0 | 27.400 | 27.5k, 219.5 s | 20k, 138.6 s |
| 1 | 27.410 | 22.5k, 179.7 s | 17.5k, 130.7 s |
| 2 | 27.446 | 25k, 206.5 s | 25k, 173.6 s |
| mean | | 201.9 s | 147.6 s (−27%) |

Stock passes its own final PSNR before 30k on every seed because the eval curves are noisy, about ±0.05 dB between neighbouring points. A single noisy evaluation can move a crossing by 2.5k iterations, so this table is indicative only. The verified number is the fixed-schedule saving of 9.8%. An early-stop test would keep the 30k LR schedule and stop at a fixed earlier iteration; shortening the decay horizon would be a separate experimental change.

## Caveats

1. **Scope:** one scene (garden) and one switch point (15k, right after densification ends at 14.9k).
2. **Peak VRAM** in these runs (about 68 GiB in both arms) is dominated by the intermediate test renders, so it is not comparable with earlier training-only figures (about 6.5 GB). Selective Adam does not change memory. Its moments are the same tensors.
3. **Projection vs measurement:** the harness's 1.35× (`results/ttq`) came from reaching quality with fewer image-equivalents. At a fixed 30k iterations, the trainer gets 1.11× from the cheaper optimizer step alone.

## Files

- `logs/{swbase,sw15k}_s{0,1,2}.log.seg`: SEGCLOCK lines (training-step time per 2,500 iterations, cumulative) and the switch line.
- `logs/*.log.metrics`: test PSNR / SSIM / LPIPS every 2,500 iterations.
- `logs/*_timings.txt`: NeRFICG timers.
- `logs/gs-switch-4480440.out`: job summary.
- Full trainer stdout logs (mostly progress bars) remain on the cluster, in `results/sparse_switch_b11a9cb/logs/`.
- Driver: [`cluster/sparse_switch.sbatch`](../../cluster/sparse_switch.sbatch).
