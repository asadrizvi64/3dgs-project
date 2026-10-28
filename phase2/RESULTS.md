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
   - Durable copy (outside the cat workspace, which expires 2026-11-26): `~/3dgs-archive/phase2-factorial-4482364/` on Capella. It holds the complete run folders of this run and of its paired B-only run (final checkpoint, renders, config, metrics), the final test renders and ground truth of all 24 runs, and a `manifest.sha256` (1,312 files, 4.4 GB).
   - It is 1 run out of 24 and did not recur in the other B+C seeds or in any full-stack run. One occurrence cannot show whether it is chance or a rare failure mode of B with C, so step 3 should check per-image PSNR for outliers in every run.
6. **Gaussians and VRAM.**
   - B-on arms end with 0.2–0.5% more Gaussians after cleanup, the same direction as before. The counts at 15k and before cleanup were not recorded; clock hook v4 records them in step 3.
   - Allocated VRAM is 6.43–6.47 GiB in all arms, and C saves about 0.04 GiB. This figure is read after the final evaluation, so it is not a training-only peak.

### What this means for step 3 (written before step 3 ran)

| candidate stack | garden time | garden ΔPSNR | note |
|---|---|---|---|
| A+B | −19.7% | +0.022 | no observed quality cost; just under the 20% median target on garden. B saves more on outdoor scenes and less on indoor ones (3–5%) |
| A+B+C | −24.8% | −0.018 | within tolerance; C adds about 5 points of speed at about −0.04 dB |

Whether C belongs in the stack can be decided across all 9 scenes rather than on garden alone. Stock, A+B and A+B+C, 3 seeds per scene, is 81 runs, about 7–8 GPU-hours.

## Step 3: benchmark on all 9 Mip-NeRF360 scenes (jobs 4483034–4483042, 2026-10-28)

The design:
- 3 arms: stock Faster-GS (@1144604, flags off), FasterGSPlus A+B, and FasterGSPlus A+B+C, where B is selective Adam from 15k.
- 3 seeds per arm, on all 9 scenes: outdoor at 1/4 resolution, indoor at 1/2. One job per scene, with that scene's 9 runs on one node and the arm order rotated by seed.
- Clock hook v4, test metrics only at 30k.
- All 81 runs were validated by [`analyze_bench.py`](analyze_bench.py). The checks covered: one clock record, a 15k and a last-step Gaussian count, the training-only VRAM peak, per-image metrics for every run, and the switch present exactly in the A+B / A+B+C runs.
- Every run is in `experiment_log.csv` (81 rows), and its files are in `logs/run_<job>/runs/`. Full analysis output: [`logs/bench-analysis.md`](logs/bench-analysis.md).

**Verdict (pre-registered criterion: median time reduction ≥ 20%, mean ΔPSNR ≥ −0.05 dB, no scene below −0.10 dB, mean ΔLPIPS ≤ +0.002): both stacks pass the benchmark-level criteria** on these nine scenes and this H100 setup.

- **A+B is the conservative configuration:** no scene below −0.02 dB, and the largest per-scene ΔLPIPS is +0.0012.
- **A+B+C is the fastest configuration that passed the agreed gate.** Its worst scene, treehill, is −0.060 dB with ΔLPIPS +0.0024. That LPIPS change is above +0.002 for that scene alone; the registered LPIPS limit applies to the mean across scenes (+0.0002), so the pass stands.
- The report keeps both.

| stack | median time reduction | mean | range | paired runs faster | mean ΔPSNR | worst scene | mean ΔLPIPS |
|---|---|---|---|---|---|---|---|
| A+B | **20.6% (1.26×)** | 19.7% | 17.0–23.7% | 27/27 | +0.119 | counter −0.012 | +0.0000 |
| A+B+C | **23.6% (1.31×)** | 22.8% | 18.4–28.4% | 27/27 | +0.081 | treehill −0.060 | +0.0002 |

### Per scene (training time from CUDA events; deltas paired against stock per seed, mean of 3 seeds)

| scene | stock s | A+B s | A+B time | A+B ΔPSNR | A+B+C s | A+B+C time | A+B+C ΔPSNR | A+B+C ΔLPIPS |
|---|---|---|---|---|---|---|---|---|
| bicycle | 251.3 | 194.8 | −22.5% | +0.065 | 184.5 | −26.6% | +0.040 | +0.0011 |
| flowers | 176.4 | 139.5 | −20.9% | +0.033 | 134.7 | −23.6% | +0.020 | +0.0012 |
| garden | 252.4 | 200.3 | −20.6% | +0.065 | 190.0 | −24.7% | +0.057 | −0.0001 |
| stump | 202.7 | 154.5 | −23.7% | +0.073 | 145.1 | −28.4% | +0.095 | −0.0011 |
| treehill | 179.2 | 141.3 | −21.1% | +0.008 | 135.0 | −24.7% | −0.060 | +0.0024 |
| room | 150.4 | 124.7 | −17.1% | +0.266* | 120.7 | −19.7% | +0.249* | −0.0009 |
| counter | 153.5 | 127.4 | −17.0% | −0.012 | 124.6 | −18.9% | −0.032 | −0.0001 |
| kitchen | 197.7 | 163.8 | −17.1% | +0.431* | 158.3 | −19.9% | +0.190* | −0.0001 |
| bonsai | 131.3 | 108.8 | −17.1% | +0.137* | 107.1 | −18.4% | +0.172* | −0.0005 |

\*Inflated by one-view failures (single test images far below the same image in the other runs), most of them in stock runs (finding 3).

Per-seed values, SSIM, ms/it, Gaussian counts and VRAM per scene are in the analysis file.

### Findings

1. **Speed.**
   - Every one of the 54 paired runs is faster than its stock pair.
   - On outdoor scenes (2.9–4.9 M Gaussians) A+B saves 20.6–23.7% and A+B+C saves 23.6–28.4%.
   - On indoor scenes (1.1–1.6 M Gaussians) A+B saves only about 17% and A+B+C 18.4–19.9%.
   - A+B's median (20.6%) clears the 20% bar by a small margin: it is garden, the fifth of nine sorted values, and all four indoor scenes are below 20%.
2. **Quality: both stacks pass the benchmark-level criteria. Not every scene meets every tolerance individually.**
   - ΔPSNR is above −0.10 dB on every scene: A+B's worst is counter (−0.012 dB), A+B+C's is treehill (−0.060 dB).
   - The LPIPS limit is a mean across scenes. Treehill's A+B+C ΔLPIPS (+0.0024) is above +0.002 on its own.
   - So the claim is "within the pre-registered criteria", not "no loss anywhere": some per-scene and per-seed deltas are negative.
3. **One-view failures.** These are single test images far below the same image in the other runs.
   - Counting test images more than 3 dB below the same image's median in the scene's other runs:
     - **stock: 7 images in 4 of 27 runs**;
     - **A+B: 2 images in 2 runs**;
     - **A+B+C: 1 image in 1 run**.
   - All are on indoor scenes except garden's image 22, the view that failed in the step 2 factorial; here it is 2.5 dB low in stock seed 0, below the 3 dB cut.
   - Worst cases: room stock seed 1, image 17, −14.8 dB; kitchen stock seed 0, images 2 and 4, −11.6 and −10.6 dB.
   - That stock has them shows they can happen without B or C. It does not show that B or C leave their frequency unchanged; these counts are too small to tell either way.
   - They drive the large indoor PSNR gains (room, kitchen, bonsai). **The verdict uses all images.** As a secondary analysis only, removing the affected images from every run of the scene gives: A+B ΔPSNR mean +0.071 (worst counter −0.013); A+B+C +0.034 (worst treehill −0.060). Both would still meet the criteria, and the ΔLPIPS means barely change.
4. **C (bf16 SH-rest moments) on top of A+B: +2 to +6 points of speed for a small, mostly negative quality change.**
   - With one-view failures removed it is negative on 7 of 9 scenes: −0.008 to −0.068 dB, plus kitchen −0.157 with high seed variance.
   - Treehill is the clearest case: −0.068 dB on all 3 seeds, LPIPS +0.0012.
   - This matches step 2 (−0.04 dB on garden). That is why A+B is the conservative configuration and A+B+C the fastest one that passed.
5. **Gaussian counts.**
   - **Within each run** the count is constant from 15k to the last training step, in all 81 runs. N is fixed after densification.
   - **Between arms** the counts already differ at 15k. On stump the mean is 4.309 M for stock against 4.246 M for A+B (−1.4%) and 4.238 M for A+B+C (−1.6%); treehill A+B+C is +1.2%; elsewhere the gap is under 0.5%. The arms differ before the B switch (A is active from step 0, and runs are not bit-identical), so this gap is not attributed to a cause.
   - The end-of-training opacity cleanup then removes 0.4–3.5% of Gaussians in stock (room 3.5%) but only 0.1–0.7% in the selective-Adam arms.
   - **The final count differences therefore reflect both the 15k population differences and the different cleanup losses.**
   - Why selective Adam loses fewer at cleanup is still a hypothesis: rows that are not updated keep their opacity instead of drifting below the floor.
6. **Memory is broadly unchanged.**
   - The training-only peak is the allocated memory read inside the last training step, before the final evaluation.
   - Individual runs peak at 4.80–7.24 GiB, depending on the scene. Garden averages about 6.46 GiB for both stock and A+B.
   - C lowers the peak by up to 0.06 GiB.

## Renders during training (stage renders, 27 runs, seed 0)

Stock, A+B and A+B+C were each trained once more on every scene (seed 0), rendering and evaluating the test set every 1,000 iterations. The figures, comparison sheets (by iteration and at equal training time, full frame and zoomed) and per-stage data are in [`stages/`](stages/README.md); the full renders (35 GB) are in `~/3dgs-archive/stages/` on Capella. These runs are not benchmark values (the repeated evaluations change them, and they are one seed).

- At equal training time, both stacks are at or above stock over most of training on 8 of 9 scenes. On treehill both fall slightly behind stock in the last third (about 0.05–0.15 dB on this seed), consistent with treehill being the weakest benchmark scene.
- Outdoor scenes reach stock's final PSNR well before 30k in all arms (stump at 14k, bicycle at 18–20k); indoor scenes are still improving at 30k. This supports testing an early stop (keeping the 30k schedule) on outdoor scenes first.
- Test PSNR collapses to about 8–10 dB right after each opacity reset (every 3,000 iterations up to 15k); those stages are left out of the figures.
