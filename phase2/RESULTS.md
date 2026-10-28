# Phase 2 results

## Step 1: FasterGSPlus build and checks (job 4481219, node c59, 2026-10-28)

FasterGSPlus is Faster-GS @1144604 with lean Adam, plus the Phase 1 macro blend backward, plus the selective-Adam switch and training clock (`cluster/patches/fastergs_sparse_from.py` v3). It was built by `phase1/make_variant.sh` with `VARIANT=FasterGSPlus` (commit 702b5a7). Each gain has its own switch:

| flag | gain | how to turn it on |
|---|---|---|
| A | macro blend backward | env `FGS_BLEND_BWD=macro` (`taming` = stock kernel) |
| B | selective Adam from iteration n | config `TRAINING.OPTIMIZER.SELECTIVE_FROM: n` (env `FGS_SPARSE_FROM` overrides) |
| C | bf16 SH-rest Adam moments | config `TRAINING.OPTIMIZER.SH_REST_BF16_MOMENTS: true` |

Driver: [`build_plus.sbatch`](build_plus.sbatch). Log: [`logs/gs-phase2-build-4481219.out`](logs/gs-phase2-build-4481219.out). Per-run clock files and test logs are in `logs/run_4481219/`.

### Checks (all passed)

| stage | check | result |
|---|---|---|
| build | variant builds and imports; default blend backward is mode 3 | ok |
| unit | Phase 1 gradient parity against FasterGSPlusCudaBackend | 92 passed (max abs diff 1.2e-6) |
| unit | lean-Adam kernel tests against FasterGSPlusCudaBackend | 15/15 PASS |
| switch | all flags on, `SELECTIVE_FROM=600`: switch fires once, at 600; dense before, selective after | PASS |
| switch | all flags off: never switches; dense throughout | PASS |

The lean-Adam test file has 15 checks. The "16/16" in the first draft of `PLAN.md` was a miscount and is now corrected.

### Sanity runs: garden, seed 0, 30k, one node, interleaved

Training time is from CUDA events summed over every `training_iteration`. Test metrics are computed only at 30k.

| run | train s | vs stock | NeRFICG ms/it | PSNR | SSIM | LPIPS | Gaussians | allocated VRAM |
|---|---|---|---|---|---|---|---|---|
| stock FasterGS | 251.4 | | 8.43 | 27.499 | 0.8667 | 0.1206 | 4,152,932 | 6.46 GiB |
| FasterGSPlus, all off | 251.8 | +0.2% | 8.44 | 27.461 | 0.8669 | 0.1208 | 4,148,882 | 6.46 GiB |
| FasterGSPlus, A+B+C | **192.6** | **−23.4% (1.31×)** | 6.47 | 27.525 | 0.8674 | 0.1206 | 4,156,343 | 6.41 GiB |

- **All flags off matches stock.** The time is the same to within 0.2%. PSNR 27.461 lies inside the range of earlier stock seed-0 runs of this code (27.400 / 27.472 / 27.499 / 27.508). One build can therefore serve every arm of the step 2 factorial.
- **A+B+C is 23.4% faster** in this single run. The plan's rough projection was about −27% (188 s from 255.8 s), so the parts add up to most of what was expected. Step 2 measures how much each part contributes and whether they interact.
- **Quality:** A+B+C ends at 27.525 dB, against 27.499 dB stock on the same node. That is one seed only, well inside run-to-run noise (about ±0.05 dB). It shows no sign of a problem, but it is not a quality result; step 2 (3 seeds) and step 3 (9 scenes) are.
- Visible fraction at step 30k with B on: 0.336, as in the earlier switch runs (0.34–0.39).

## Step 2: A × B × C factorial on garden (job 4482364, node c19, 2026-10-28)

The design has 8 arms (each flag on or off) and seeds 0/1/2, giving 24 runs on one node. The arm order was rotated by seed.

- Test metrics were computed only at 30k.
- Training time is from CUDA events summed over every training step.
- The job took 1 h 53 min.
- The analyzer ([`analyze_factorial.py`](analyze_factorial.py)) validated all 24 runs before analysis. Each run had one 30k clock record and finite metrics, and the selective-Adam switch was present exactly in the B-on runs, with the matching final optimizer state.
- Full output: [`logs/factorial-4482364-analysis.md`](logs/factorial-4482364-analysis.md). Per-image test PSNR for all runs: `logs/run_4482364/per_image_psnr.txt`.

### Per arm (mean of 3 seeds; time and ΔPSNR paired per seed against all-off)

| arm | A | B | C | train s | vs all-off | ΔPSNR (s0 / s1 / s2) | SSIM | LPIPS | Gaussians |
|---|---|---|---|---|---|---|---|---|---|
| a0b0c0 | | | | 253.6 | | | 0.8673 | 0.1201 | 4.156 M |
| a0b0c1 | | | on | 234.0 | −7.7% | −0.051 (−0.056 / −0.073 / −0.025) | 0.8667 | 0.1207 | 4.149 M |
| a0b1c0 | | on | | 227.2 | −10.4% | +0.028 (+0.084 / −0.030 / +0.030) | 0.8676 | 0.1200 | 4.172 M |
| a0b1c1 | | on | on | 217.5 | −14.2% | −0.069 (**−0.199** / −0.026 / +0.019) | 0.8671 | 0.1206 | 4.170 M |
| a1b0c0 | on | | | 225.4 | −11.1% | −0.011 (−0.015 / −0.015 / −0.003) | 0.8669 | 0.1204 | 4.160 M |
| a1b0c1 | on | | on | 208.4 | −17.8% | −0.023 (−0.036 / −0.032 / −0.001) | 0.8669 | 0.1205 | 4.161 M |
| a1b1c0 | on | on | | 203.6 | **−19.7%** | +0.022 (+0.041 / −0.025 / +0.049) | 0.8676 | 0.1202 | 4.175 M |
| a1b1c1 | on | on | on | **190.8** | **−24.8%** (−24.5 / −24.7 / −25.0) | −0.018 (−0.032 / −0.060 / +0.037) | 0.8672 | 0.1204 | 4.164 M |

The all-off arm's mean PSNR is 27.520. The time spread across seeds is ±1 s or less in every arm.

### Findings

1. **Full stack: −24.8% training time (1.33×) on every seed, with quality within the pre-registered tolerances on this scene.**
   - ΔPSNR −0.018 dB (tolerance −0.05), ΔLPIPS +0.0003 (tolerance +0.002), ΔSSIM −0.0001.
   - Two of three seeds are lower in PSNR, so this is "within tolerance", not "no loss".
2. **The savings mostly add, except that B and C overlap.**
   - Standalone gains are A −11.1%, B −10.4% and C −7.7%. Multiplying them predicts −26.5%; the measured result is −24.8%.
   - The only clear interaction is B × C: +1.3% in the log-time contrast (+0.8 to +1.6 across seeds). Both B and C cut optimizer memory traffic, so C saves 7.7% alone, 4.2% once B is on, and 6.3% inside A+B.
   - A × B and A × C are within ±0.5%.
3. **C (bf16 SH-rest moments) has a small, consistent quality cost on garden. The earlier single-seed +0.04 dB did not hold.**
   - C alone: −0.051 dB, negative on all 3 seeds; LPIPS +0.0006, worse on all 3.
   - C added to A+B: −0.040 dB, negative on all 3 seeds.
   - Factorial main effect of C: −0.050 dB [−0.108 to −0.011]. Excluding image 22 (below), it is −0.027 dB [−0.039 to −0.010]: still negative on every seed.
4. **A (macro backward) and B (selective Adam from 15k) show no quality cost here.**
   - A alone: −0.011 dB, with main effect +0.015.
   - B alone: +0.028 dB, with main effect +0.012.
   - Both are within seed noise. B's time saving (−10.4%) matches the earlier garden run (−9.8%), and A's (−11.1%) matches Phase 1 (−10.7%).
5. **One rendering artifact.**
   - In run a0b1c1 seed 0 (B+C, no A), test image 22 scores 20.98 dB. In the other 23 runs it scores 27.5–27.9 dB.
   - The render shows a large, faint white haze over the lower right of that view ([comparison with the B-only run of the same seed](logs/run_4482364/img22_b1c1_vs_b1c0_s0.jpg)), most likely a large low-opacity floater near that camera.
   - This one image costs the run 0.29 dB of mean PSNR. That makes "C given B" −0.096 dB, against 0.000 dB without image 22. Every other image matches the B-only run.
   - It is 1 run out of 24 and did not recur in the other B+C seeds or in any full-stack run. One occurrence cannot show whether it is chance or a rare failure mode of B with C, so step 3 should check per-image PSNR for outliers in every run.
6. **Gaussians and VRAM.**
   - B-on arms end with 0.2–0.5% more Gaussians after cleanup, the same direction as before. The counts at 15k and before cleanup were not recorded; clock hook v4 records them in step 3.
   - Allocated VRAM is 6.43–6.47 GiB in all arms, and C saves about 0.04 GiB. This figure is read after the final evaluation, so it is not a training-only peak.

### What this means for step 3

| candidate stack | garden time | garden ΔPSNR | note |
|---|---|---|---|
| A+B | −19.7% | +0.022 | no observed quality cost; just under the 20% median target on garden. B saves more on outdoor scenes and less on indoor ones (3–5%) |
| A+B+C | −24.8% | −0.018 | within tolerance; C adds about 5 points of speed at about −0.04 dB |

Whether C belongs in the stack can be decided across all 9 scenes rather than on garden alone. Stock, A+B and A+B+C, 3 seeds per scene, is 81 runs, about 7–8 GPU-hours.
