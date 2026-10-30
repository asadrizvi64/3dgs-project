# Alternative track R: better-positioned Gaussians from depth reinitialization (revised 2026-10-29)

**Status:**
- Hypothesis test only, separate from Phase 3. Combining with the Phase 2 gains (A, B, C) comes after the hypothesis passes (stage R5), never before.
- **R0 and R1 tooling is written and tested on the CPU** (see "Tools"). The CUDA build, the GPU checks and the runs themselves have not run yet.
- R2 is conditional on R1.

**Question.** Does starting training from better-positioned Gaussians save enough later work to pay for its own cost, while keeping rendering quality?

**Hypothesis H.** On stock Faster-GS, a one-shot reinitialization from rendered depth at an early step k, with densification ended at k, reaches the reference's quality target in less total time:

```
T_warm-up(0..k) + T_reinit + T_train(k..target)  <  T_reference(0..target)
```

Both sides use the same quality target, the same candidate-and-confirmation rule, the same clock and the same seeds. If the better start cannot save enough later work to recover its own cost while preserving quality, the track stops before any nine-scene campaign.

## Decisions (2026-10-29)

| decision | choice |
|---|---|
| scenes | garden and room |
| intervention step | the earliest passing value of 2000 or 5000, **per scene**; both audits are recorded |
| quality | **no worsening** of the 3-seed mean PSNR, SSIM or LPIPS (pilot criterion, below); seed variability is reported but never used as an allowance |
| filtering | filtered first; unfiltered later, as the comparison (R3) |
| arms | all four (R2) |
| budget | about 8 GPU-hours as a ceiling; R2 runs only if R1 passes |

## Keeping this off the main track

- **Own files:** everything lives in `alt_reinit/`: plan, patch, driver, job script, analyzer, recorder, tests, logs. Nothing in `phase0/`–`phase2/`, `cluster/patches/` or the FasterGSPlus build is edited.
- **Own build:** variant `FasterGSReinit`, cloned directly from Faster-GS @1144604 (c60ea26 plus lean Adam with its options off; `results/optim` showed this equals stock). No macro patch, no FasterGSPlus code.
- **Shared files are only read:**
  - `cluster/train_fastergs.sbatch` (the runner);
  - `cluster/patches/fastergs_sparse_from.py` v5 (clock) and `fastergs_save_state.py`, both applied to the variant's own copy;
  - the stock FasterGS checkout. The job checks that it carries the v5 clock hook and never patches it.
- **Own recorder:** `alt_reinit/record_runs.py` writes roles `alt-reinit-check`, `-audit`, `-compare`, `-confirm`. `phase2/record_runs.py` could not be reused: it cannot emit these roles, labels short runs `check`, and describes non-FasterGS methods as "phase1 macro".

## What this plan does not claim

- **Agreement between the model's own depth maps does not show the geometry is correct.** Several cameras will agree on an invented wall slightly in front of the real one (`tests/test_core_cpu.py::test_consistent_wrong_surface_passes` builds exactly this case). The filter is a cleaning step, and its statistics are diagnostics. The judge is held-out rendering quality, backed by evidence from the training photos (patch ZNCC) and from COLMAP's feature-matched points.
- **Agreement with the 30k model's depth is agreement with another reconstruction, not accuracy.** It is reported under that name.
- **Expected depth is not a surface point.** Mixing surfaces at 1 m and 3 m gives about 2 m even when the pixel is nearly opaque. The render therefore also gives the weighted depth variance (third colour channel z²), and such mixtures are rejected. A low variance still does not show the depth is correct. Each Gaussian also contributes its *centre* depth over its whole footprint, so a large Gaussian on a slanted surface gives an offset depth with zero variance. R1 measures how much this matters at k.
- **No fixed time budget.** The budget is whatever the measured reference leaves. Projections from Phase 2 ms/step are not used: reinitialized Gaussians can differ in size, overlap and visibility, so their step cost is measured.
- **Novelty is narrow.** Each part is established:
  - EDGS: dense triangulated initialization, no densification;
  - Mini-Splatting2: depth reinitialization at 2k, no multi-view check;
  - MVS-Splatting: initialization from fused MVS depth;
  - MVG-Splatting: multi-view geometric consistency of rendered depth, used to guide densification.

  The possible contribution is a controlled answer to whether self-bootstrapped reinitialization pays for itself inside a fast trainer once every cost is counted.

## Fixed for every arm (the hypothesis test)

- Stock Faster-GS behaviour: stock blend backward (no A), dense Adam throughout (no B), **fp32 Adam moments** (no C; C showed a small, consistent quality cost in Phase 2).
- The same 30k learning-rate schedule in every arm, including runs that stop early.
- The same SH-degree schedule, Morton ordering, final opacity cleanup and NeRFICG timer protocol (timer ON) as the Phase 2 benchmark.
- Garden (1/4 resolution) and room (1/2). Seeds 0, 1, 2; one job per scene, interleaved, arm order rotated by seed.

## Clock

**Track clock = training steps + the method's between-step callbacks.**
- **Steps:** SEGCLOCK, CUDA events around every `training_iteration`.
- **Callbacks:** CBCLOCK, synchronized wall time per call of `densify`, `morton_ordering`, `reset_opacities`, `increase_sh_degree` and, in apply runs, `reinit_gaussians`. It is installed at the first training step, because NeRFICG assigns callback timers when training starts. Morton ordering at iteration 0, identical in every arm, falls before that and is not counted.
- **Excluded:** evaluations, checkpoints, and the audit's dry-run reinit.

The reinit cost includes depth rendering for every training view, candidates, neighbour selection, filtering, deduplication, selection, kNN scales, the optimizer-state rebuild and Morton ordering. That is the `REINIT ... t_total` line, with every part listed.

## Quality target and stopping rule (pilot criterion)

- **Target, per scene,** from this job's own reference runs, the 3-seed means at 30k:
  - PSNR at or above the reference mean;
  - SSIM at or above the reference mean;
  - LPIPS at or below the reference mean.

  **No allowance for worsening.** Seed variability (per-seed values, standard deviation) is reported separately and never widens the target. A near-tie (within the reference's seed spread) is *inconclusive* until confirmed, and passing this criterion is not proof of exact quality equivalence.
- **Candidate stopping point:** the first evaluation at which the 3-seed mean curve meets all three conditions. There is no "next evaluation" requirement, which avoided a conflict at 30k: the reference always has a candidate by 30k.
- **Confirmation (R4):** every candidate, the reference's included, is rerun without intermediate evaluations, stopping there. It counts only if the confirmation runs meet the criterion against the reference's confirmation runs. **Failures to reproduce are counted and reported per arm**, with the metric that failed.
- **Single views, image by image:**
  - a test image is a failure in a run when it is more than 3 dB below that image's median over the reference runs;
  - an arm may not have a failure on any image that never fails in the reference runs, so an unchanged *count* can no longer hide a newly broken view;
  - every image more than 1 dB below its reference median is listed and inspected;
  - camera-motion renders are inspected in R4.
- "Faster on 2 of 3 seeds" is an exploratory speed signal only. It cannot outweigh a quality failure on the third seed.

## Reinitialization, first version

At step k, between `densify` / `morton_ordering` (priority 100 / 99) and the training step (80), in the callback `reinit_gaussians` (priority 95). Settings come from the environment variable `FGS_REINIT`, as the other hooks do, because the shared runner only injects OPTIMIZER keys. Every setting is printed in the `REINIT_CONFIG` line.

1. **Depth for every training view.** One inference render (`rasterize(..., clamp_output=False)`) with one SH band, a black background and colours (z, 1, z²). That gives (Σwz, Σw, Σwz²): expected depth D, alpha A, and the relative depth standard deviation. No kernel change is needed.
2. **Candidate pixels** on a regular grid (default stride 2): A ≥ 0.95 and relative depth std ≤ 0.05. Otherwise the status is empty, low alpha or mixture.
3. **Geometric filter (default on).**
   - **Neighbours:** the 4 views that see the largest share of the view's candidates in-frustum, with median triangulation angle ≥ 3°.
   - **Test per neighbour:** the neighbour's depth is sampled **bilinearly** at the exact projection (all 4 taps valid) and must match within 1%. The point on the neighbour's ray at that depth must reproject into the reference within 1 px.
   - Keep a candidate if 2 of its 4 neighbours agree.
   - Why bilinear: a first version used nearest-pixel depth with a round trip through the neighbour pixel's centre. On a synthetic slanted plane it falsely rejected 12–13% of correct points; bilinear rejects 0.04–0.2%, and 4-tap validity keeps depth edges failing safe.
4. **Deduplication:**
   - a voxel hash with edge equal to the candidate spacing at its depth (z · stride / f, binned by log2);
   - one candidate per voxel, preferring the most agreeing neighbours, ties broken by the track's own seeded generator;
   - across levels, the finest observation wins.
5. **Replacement, with an exact count.**
   - **Removal:** an old Gaussian is removed iff its centre lies within `cover_ring` voxels (default 1, the 3×3×3 block; 0 = the same voxel) of a voxel occupied by a candidate, at that candidate's level.
     - Removal is computed from *all* deduplicated candidates, before any subsampling, so the surviving rows do not depend on which candidates are drawn.
     - Ring 1 because candidates sit at the rendered depth while old centres scatter around the surface by about one footprint: with ring 0 the stand-in scene removed almost nothing.
   - **Addition:** `n_add = target − n_survivors` candidates are drawn uniformly at random (seeded).
   - **Count:** `N_after = N_before − N_removed + N_added` must equal the target, and is asserted in the run.
     - The matched-count arm therefore adds exactly as many points as it removes.
     - If there are too few candidates, the stride is halved once. If the target is still unreachable, or below the survivors, apply runs abort (`REINIT_FAIL`) rather than run at a different count.
   - Each run reports removed rows, surviving original rows and added rows.
   - A centre in a covered voxel does not guarantee that the replacement keeps the old Gaussian's image coverage. The audit therefore renders silhouettes before and after with every opacity forced to 0.99, and reports holes and newly covered pixels.
6. **New rows follow the stock initialization rules** except position and colour:
   - position: the back-projected point;
   - colour: SH DC from the **training-photo pixel colour**, SH rest 0;
   - scale: isotropic, log RMS distance to the 3 nearest neighbours among all rows after the replacement;
   - rotation: identity;
   - opacity: 0.1.

   Ridge-fitted SH, flat scales and other opacities are later one-factor variants. Ridge regression cannot recover directions never observed, and the synthetic conditioning check does not stand in for the real point tracks.
7. **Optimizer state:**
   - rebuilt with the model's `prune` and NeRFICG's `extend_param_groups`, the path densification uses, then Morton ordering;
   - **checked in the run, on the real objects, every time** (untimed): survivors keep their parameters and both Adam moments exactly (tracked through the Morton permutation); new rows hold the constructed values with zero moments; every group's step counter is unchanged; every group has the target count;
   - any failure aborts the run (`REINIT_CHECK ... FAIL`).
8. **If densification continues after k** (not the R2 design, but the smoke runs do it): fresh densification buffers of the new size.

`reinit-same` therefore measures **reinitialization at a fixed count**, not repositioning alone: the replaced rows also get new colour, scale, opacity and zero optimizer moments. Its conclusions are labelled that way.

**Ending densification early** is the stock key `DENSIFICATION_END_ITERATION: k`. In Faster-GS this also ends opacity resets and densification-info accumulation at k. All of that counts as "densification" below.

## Stages

### R0: build and checks (`STAGES="build unit smoke"`, about 15 min)

- **Build:** `make_reinit_variant.sh`.
- **CPU tests (unit stage):**
  - `test_core_cpu.py`: cameras, depth, filter, voxels, the exact-count rule, optimizer bookkeeping with deliberate corruption, photo ZNCC;
  - `test_trainer_cpu.py`: the driver end to end, with a reference compositor and stand-ins for NeRFICG's helpers and pycolmap;
  - `test_analyze_record.py`;
  - `test_check_variant.py`.
- **Static flags-off checks:**
  - **tree:** the rename commit changes contents only by the rename, and only `Trainer.py` and the two driver files change after it;
  - **Trainer:** the variant's `Trainer.py` minus the marked blocks equals stock + the shared hooks, and every marked block is gated by `FGS_REINIT` / `FGS_CBCLOCK`.
- **GPU checks (`test_gpu.py`):**
  - depth from the real kernel: one Gaussian, two overlapping layers (nearly opaque, expected depth about 2, flagged as a mixture), and empty pixels (alpha exactly 0, unclamped output);
  - **flags-off equivalence from identical inputs:** forward image, all six gradients and the densification info within the stock-vs-stock run-to-run floor, and two FusedAdam steps bit-identical between the stock and variant backends.
- **Smoke runs (garden):**
  - stock and variant with flags off over 1200 steps: no hook output, per-step clock within 5%;
  - apply at 2000 with target `match` and `+500000`, to 2300: the in-run check passes, the count equals the target, the callback clock logs the reinit and later densify calls;
  - audit at 2000 with the final agreement;
  - then the analyzer and the recorder must accept the real output.

### R1: reinitialization audit (`STAGES=audit`, garden and room, seed 0, about 20 min)

- **One stock 30k run per scene** with `FGS_REINIT=mode=audit,iters=2000:5000,target=<stock N at 15k>/match`. The 15k counts are garden 4,166,000 and room 1,345,000, from `phase2/logs/bench-analysis.md`.
- **At each k the reinit runs in full on a deep copy** of the model and optimizer, timed exactly as in apply mode, including the in-run check.
  - The first reachable target is timed; every target is checked for feasibility.
  - The live model, optimizer and training RNG are untouched, so training is stock.
- **Untimed diagnostics, written to `reinit_audit/k_*/stats.json`:**
  - status of the grid candidates overall, by depth bin and by 3×3 image region, plus neighbour outcomes;
  - photo evidence: ZNCC in the training photos, the same statistics (mean over neighbours, best neighbour) for kept and every rejected class. Axis-aligned patches assume small baselines, and the neighbour angles are logged;
  - COLMAP sparse points on 16 diagnostic views: reprojection with our cameras (checks the conventions and the dataset's PCA transform), and rendered-depth error by status at the observed pixel;
  - silhouette coverage before and after the replacement;
  - status images of 4 views (`masks/view*.jpg`: photo | status | depth std), and full-resolution depth / status maps (npz, archived).
- **After the last step:** the audited depth against the final model's depth by status, labelled "agreement, not accuracy" (`agreement_final.json`).
- **`analyze_audit.py`** validates each run (all k present, check passed, files present, final agreement written) and writes the tables and gates:
  - **G1:** `T_warm-up(k) + T_reinit` leaves a positive budget against the reference estimate (stock's seed-0 stage-run time to its own 30k PSNR: garden 209 s, room 149 s, SEGCLOCK only, so conservative; R2 replaces it);
  - **G2:** kept candidates are more photo-consistent than geometry-rejected ones (consistent share and median). If not, the filter is not picking out geometry the images support: revise its thresholds once, then decide;
  - **G3:** the masks have been inspected and written up (manual).
- **Choose k per scene:** the earliest k that passes G1–G3.

### R2: controlled training comparison (garden and room, 3 seeds, the chosen k per scene)

| arm | densification and opacity resets | reinit at k | N after k | compared with | measures |
|---|---|---|---|---|---|
| `ref` | stock, to 14.9k | no | grows as stock | — | reference time and quality |
| `stop` | end at k | no | N_k | `ref` | effect of ending densification itself |
| `reinit-same` | end at k | yes, filtered | N_k (matched) | `stop` | reinitialization at a fixed count (new positions, colours, scales, opacities, zero moments) |
| `reinit-target` | end at k | yes, filtered | stock N at 15k | `reinit-same`, `ref` | effect of the count; the hypothesis itself |

That is 24 runs, with everything else matched.

**Before R2: the evaluator.** NeRFICG's intermediate checkpoint-and-render path costs about 54 s per call on room (`results/sparse_switch` timings). Evaluating every 500 steps through it would cost about 50 min per run and break the budget. R2 therefore needs a metrics-only, in-process evaluator (render the test set, PSNR / SSIM / LPIPS-VGG, per-image PSNR, outside the clock). It is written and timed before R2 is scheduled. It is not part of R0/R1.

**How to read R2:**
- `reinit-same ≈ stop` at equal time: reinitialization at this count does not help at this k.
- `reinit-same > stop`, but neither reaches the target: it helps, and the count is the limit; look at `reinit-target`.
- `reinit-target`'s candidate precedes `ref`'s, with clean per-view checks: H is supported on that scene, pending R4.
- `reinit-target` never meets the target by 30k: the reinitialized start is worse than what densification produces. Stop, or do one R3 revision.

### R3: one factor at a time (only if R2 shows promise)

- filtered vs unfiltered (`min_agree` 2 vs 0);
- `cover_ring` 0 vs 1;
- opacity 0.5; flat footprint scales; SH degree 1 by ridge regression;
- the other k.

At most one revision round, each against the R2 winner by the same rule.

### R4: confirmation (only for arms with a candidate)

- Rerun every candidate, the reference's included, without intermediate evaluations, 3 seeds each, and count failures to reproduce.
- Camera-motion inspection: the same smooth path through the training poses, plus the same path raised by 10% of the scene extent. Inspect side by side for floaters, popping and colour shifts.
- Only after R4: decide whether to run all 9 scenes.

### R5: combination with the main track (only after H holds)

Add the Phase 2 gains one at a time, each against the then-current best main-track stack, with the same rule. Success against stock establishes the mechanism; R5 must still show that it improves the strongest existing implementation at matched quality.
1. **A (macro backward):** kernel-only, but the reinitialized Gaussians' size and overlap may change its gain.
2. **B (selective Adam):** its own question. Does ending densification make an earlier switch safe?
3. **C:** last, with its own quality check.

## Stop rules

- **After R1:** stop if the measured cost leaves no positive budget on either scene, or G2 fails after the one threshold revision.
- **After R2:** stop unless `reinit-target`'s candidate precedes `ref`'s on both scenes. A quality failure on any seed blocks a positive claim until it is explained.
- **After R3:** stop if no variant does better.
- **Overall:** about 8 GPU-hours before any nine-scene decision.

## What building R0 turned up (CPU, synthetic scenes)

- **Nearest-pixel neighbour depth was the wrong choice.** It falsely rejected 12–13% of a correct slanted plane, at both 320×208 and 1297×840. Bilinear sampling at the exact projection rejects 0.04–0.2%.
- **"Same voxel" removal was too strict.** The candidates sit at the rendered depth, about one footprint from the old centres, so it removed almost nothing. `cover_ring=1` fixed this in the stand-in scene; R1 reports the real removal counts and coverage holes.
- **Each Gaussian contributes its centre depth over its footprint.** On the stand-in slanted plane with 3×3-pixel splats, back-projected points sit a median 0.05 scene units off the plane, at 3–4 units from the camera. Real Gaussians at k = 2000 may be larger; R1's COLMAP and final-model checks measure it.
- **ZNCC with axis-aligned 7×7 patches needs small baselines.** On the stand-in plane: median 0.92 for neighbours 15° apart (240×160 px), 0.36 at 45° (96×64 px).
- **A reinit before densification ends needs fresh densification buffers,** or the next `densify` call fails on the old sizes.
- **The recorder's de-duplication key would have swallowed the `;` after the job id** (caught by `test_analyze_record.py`).

## Tools (all in `alt_reinit/`)

| file | what | status |
|---|---|---|
| `reinit_core.py` | geometry and bookkeeping: cameras, depth from the (z, 1, z²) render, filter, voxels, the exact-count rule, optimizer checks, ZNCC | CPU-tested |
| `reinit_trainer.py` | the driver in the trainer: apply / audit, diagnostics, final agreement, callback clock | CPU-tested with stand-ins |
| `patches/fastergs_reinit.py` | the Trainer hooks (marked, gated blocks) | applied to Faster-GS c60ea26's `Trainer.py` here; static check passes |
| `make_reinit_variant.sh` | builds `FasterGSReinit` | dry-run (no CUDA build) on a scratch layout and in `test_check_variant.py` |
| `build_reinit.sbatch` | stages `build unit smoke audit` | syntax-checked; smoke checker tested on synthetic logs |
| `analyze_audit.py` | R1 validation, tables, gates | tested on a synthetic job built from a real driver output |
| `record_runs.py` | `experiment_log.csv` rows for this track | tested on successful, failed and shortened runs |
| `tests/` | 40 CPU tests, static checks, 3 GPU checks | GPU checks pending the cluster |
| R2 evaluator, R2–R4 job stages and analysis | | not written (after R1) |

## How to run (from the repo root on Capella)

```bash
STAGES="build unit smoke" sbatch alt_reinit/build_reinit.sbatch
```

```bash
STAGES=audit sbatch --job-name=gs-alt-audit --time=01:30:00 alt_reinit/build_reinit.sbatch
```

```bash
python3 alt_reinit/analyze_audit.py alt_reinit/logs/gs-alt-audit-<job>.out > alt_reinit/logs/audit-<job>-analysis.md
```

```bash
python3 alt_reinit/record_runs.py alt_reinit/logs/gs-alt-audit-<job>.out
```
