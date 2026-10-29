# Alternative track R: better-positioned Gaussians from depth reinitialization (draft, 2026-10-29)

**Status:** hypothesis test only. This track is separate from Phase 3 and must not change it.
Combining with the Phase 2 gains (A, B, C) comes after the hypothesis passes (stage R5), never before.

**Question.** Does starting training from better-positioned Gaussians save enough later work to pay for its own cost, while keeping rendering quality?

**Hypothesis H.** On stock Faster-GS, a one-shot reinitialization from rendered depth at an early step k, with densification ended at k, reaches the reference's quality target in less total time:

```
T_warm-up(0..k) + T_reinit + T_train(k..target)  <  T_reference(0..target)
```

Both sides use the same quality target, the same crossing rule, the same clock and the same seeds. The quality target covers PSNR, SSIM, LPIPS and per-view checks, not PSNR alone.

## Keeping this off the main track

- **Own files:** everything lives in `alt_reinit/` (plan, patches, job script, tools, analysis, logs). Nothing in `phase0/`, `phase1/`, `phase2/`, `cluster/patches/` or the FasterGSPlus build is edited.
- **Own build:** variant `FasterGSReinit`, cloned directly from Faster-GS @1144604 (c60ea26 + lean Adam, options off; `results/optim` showed this equals stock). There is no macro patch and no FasterGSPlus code in it, so A, B and C cannot be switched on by accident.
- **Reused read-only:**
  - `cluster/patches/fastergs_sparse_from.py` v5, for the CUDA-event clock (`FGS_SEGCLOCK`) and `FGS_WALL`, with `SELECTIVE_FROM` unset;
  - `cluster/patches/fastergs_save_state.py`, for warm-up states;
  - `phase2/record_runs.py`, for `experiment_log.csv` rows with role `alt-reinit-<stage>`.
- **Own budget:** at most about 8 GPU-hours before any decision to go beyond garden and room. There is a stop rule after R2 (below).

## Why this test, and what it does not assume

The idea is that training spends many iterations discovering geometry. If better-positioned Gaussians are placed early, fewer iterations may be needed. The closest published support is EDGS (CVPR 2026):
- dense correspondences are triangulated across image pairs, and training starts from them with densification disabled;
- on Mip-NeRF360, its 10k-step model scores 27.54 / 0.834 / 0.154 against 27.49 / 0.816 / 0.215 for its retrained 30k 3DGS baseline;
- its initialization takes about 120 s on an A100, inside a 26-minute vanilla-3DGS regime.

None of this shows that a cheaper initialization from the model's own depth works equally well, or that it pays for itself inside Faster-GS (150–255 s per 30k run here).

**What this plan does not claim:**

- **Agreement between the model's own depth maps does not show the geometry is correct.** An early model can put a wall slightly in front of the real one, and several cameras will agree on that wrong wall. The geometric filter below is a cleaning step, and its statistics are diagnostics. The judge is held-out rendering quality, backed by checks against the training photos themselves.
- **Agreement with the 30k model's depth is agreement with another reconstruction, not accuracy.** It is reported under that name.
- **There is no fixed time budget for the reinitialization.** The budget is whatever the measured reference leaves: `T_reference(0..target) − T_warm-up − T_reinit` must cover the remaining training. Projections from Phase 2 ms/step are not used. Reinitialized Gaussians can differ in size, overlap and visibility, so their step cost is measured, not assumed.
- **Novelty is narrow.** Each part is established:
  - EDGS: dense triangulated initialization, no densification;
  - Mini-Splatting2: depth reinitialization at 2k, without a multi-view check;
  - MVS-Splatting: initialization from fused MVS depth;
  - MVG-Splatting: multi-view geometric consistency of rendered depth, used to guide densification.

  What could be new here is a controlled answer to whether self-bootstrapped reinitialization pays for itself in a fast trainer once every cost is counted.

## Fixed for every arm (the hypothesis test)

- Stock Faster-GS behaviour:
  - stock blend backward (no A);
  - dense Adam throughout (no B);
  - **fp32 Adam moments** (no C; C showed a small, consistent quality cost in Phase 2).
- Same 30k learning-rate schedule (`LEARNING_RATE_MEANS_MAX_STEPS: 30000`) in every arm, including runs that stop early. Shortening the schedule is a later, separate question.
- Same SH-degree schedule, Morton ordering, final opacity cleanup and NeRFICG timer protocol (timer ON) as the Phase 2 benchmark.
- Garden (1/4 resolution) and room (1/2 resolution). Seeds 0, 1, 2. One job per scene, so a scene's runs share a node, interleaved with the arm order rotated by seed.

## Clock and crossing rule (identical for all arms)

**Training clock.** CUDA events around every `training_iteration` (SEGCLOCK). The track's own patch adds CUDA-event timing of every training callback that runs between steps:
- `densify`;
- `reset_opacities`;
- `morton_ordering`;
- the reinitialization, which includes depth rendering, neighbour selection, filtering, deduplication, k-nearest-neighbour (kNN) scales, parameter and optimizer-state rebuild, and Morton ordering.

The warm-up 0..k is on the same clock. Test evaluations are outside it.

**Evaluations.** Full test set every 500 steps from k to 30k (and at k, before the intervention), with PSNR, SSIM, LPIPS (VGG, NeRFICG's metric) and per-image PSNR. The arms are identical up to k. Their quality at k is run-to-run noise and is recorded, so later gaps can be read as differences-in-differences (as in `results/sparse_switch`).

**Quality target**, per scene, from this job's own reference runs:
- PSNR at or above the reference's 3-seed mean 30k PSNR;
- SSIM no lower than the reference mean minus ε_SSIM;
- LPIPS no higher than the reference mean plus ε_LPIPS;
- proposed ε: the reference's own seed-to-seed standard deviation at 30k, measured in this job (decision 3).

**Crossing.** The first evaluation step at which the 3-seed mean curve meets all three conditions, and still meets them at the next evaluation. The time to target is the mean training clock at that step.
- The reference's mean curve meets its own target by 30k at the latest.
- Per-seed crossings (each seed against the same target) are reported as a secondary result.

**Per-view checks at the crossing step, every seed:**
- the number of one-view failures (a test image more than 3 dB below that image's median in the reference runs) must not exceed the reference's count;
- every image more than 1 dB below its reference median is listed and inspected;
- a run's quality failure is never outvoted by the other seeds (see stop rules).

**Confirmation runs (R4).** Arms that cross are rerun without intermediate evaluations, stopping at their crossing step (`NUM_ITERATIONS` = crossing step; the learning-rate horizon stays 30k). They are timed by wall clock (`FGS_WALL` over the whole run) and by the training clock. This checks that the evaluated curve runs did not change the result, and gives the final models for the camera-path inspection.

## Reinitialization, first version (simple on purpose)

At step k, inside the training loop and on the clock:

1. **Depth for every training view.** Faster-GS @c60ea26 has no depth output, but the inference rasterizer (`rasterize(..., clamp_output=False)`) computes `sum(T·α·color) + T·bg`, with `color = max(0.5 + C0·sh0, 0)` when only the DC band is active. So:
   - set `sh0 = (c − 0.5)/C0` per view with `c = (z_view, 1, 0)`, one SH band, background 0;
   - one render then gives `(Σwz, Σw)`, so expected depth is `D = Σwz/Σw` and alpha is `A = Σw`.

   No kernel change is needed. Expected depth mixes foreground and background at edges, so only pixels with `A ≥ 0.95` become candidates.
2. **Candidates:** a regular pixel grid per view, with the stride set so the candidate count covers the target count at the keep rate measured in R1.
3. **Geometric filter (on by default; unfiltered is a later ablation).**
   - **Neighbours:** for each view, the 4 training views that see the largest share of its candidate points in-frustum, excluding views whose triangulation angle at the median depth is below 3°.
   - **Agreement in one neighbour:** a point agrees if the neighbour's depth at the projection matches within 1% (with neighbour alpha at least 0.95) and the round trip reprojects within 1 px.
   - **Keep rule:** keep a point if at least 2 of its 4 neighbours agree. Occlusion makes correct points fail some neighbours, hence 2 of 4.
4. **Deduplicate:** a voxel hash with edge equal to the pixel footprint at the point's depth (z · stride / f, binned by log2 of the footprint). One point per voxel.
5. **Replace only what is covered.** An existing Gaussian is removed if its centre falls in an occupied voxel. Otherwise it is kept with its parameters and optimizer state. This keeps sky and background coverage where nothing was registered.
   - Total N = new + kept. For the matched-count arm, the number of new points is chosen so the total equals N_k. For the target-count arm, so it equals the stock N at 15k (garden 4.166 M, room 1.345 M; `phase2/logs/bench-analysis.md`).
6. **Attributes: stock initialization rules for everything except position and colour.**
   - position: the back-projected point;
   - colour: SH DC from the **training-photo pixel colour**, with SH rest set to 0;
   - scale: log RMS distance to the k nearest neighbours, isotropic (`compute_root_mean_squared_knn_distances`, as in `initialize_from_point_cloud`);
   - rotation: identity;
   - opacity: 0.1.

   Ridge-fitted SH, footprint-based anisotropic scales and higher initial opacity are later one-factor variants. Ridge regression cannot recover viewing directions that were never observed, and the synthetic conditioning check does not stand in for the real point tracks.
7. **Optimizer state:** remove old rows and add new ones with NeRFICG's `Optim.adam_utils.prune_param_groups` / `extend_param_groups`, the path densification uses. New rows start with zero moments, exactly like densified rows (FusedAdam keeps one step counter per parameter group). Then apply `apply_morton_ordering()`.

**Proposed config keys** (track patch, all off by default):
- `TRAINING.REINIT.ITERATION` (−1 = off);
- `TRAINING.REINIT.TARGET_N` (0 = match current N);
- `TRAINING.REINIT.ALPHA_MIN` (0.95);
- `TRAINING.REINIT.FILTER.N_NEIGHBOURS` (4);
- `TRAINING.REINIT.FILTER.MIN_AGREE` (2; 0 = unfiltered);
- `TRAINING.REINIT.FILTER.DEPTH_TOL` (0.01);
- `TRAINING.REINIT.FILTER.REPROJ_TOL_PX` (1.0).

**Ending densification early** is the stock key `DENSIFICATION_END_ITERATION: k`. In Faster-GS this also ends opacity resets at k (`reset_opacities` has `end_iteration='DENSIFICATION_END_ITERATION'`) and stops densification-info accumulation in the backward pass. All of that counts as "densification" in the arm table below.

## Stages

### R0 — build and checks (no quality results)

- **Build `FasterGSReinit`** with the clock patch and the track patch. Flags off must equal stock: a short garden run with flags off and densification ending at 14.9k must show the same per-step clock as stock to within noise.
- **CPU unit tests** (`alt_reinit/tests/`, runnable without a GPU):
  - back-projection and reprojection round trip with real Mip-NeRF360 cameras (error ≤ 1e-3 px);
  - filter on synthetic depth maps: a plane seen by all views is kept; a floater present in one view's depth is rejected;
  - deduplication: one point per voxel;
  - kept or removed bookkeeping.
- **GPU unit tests:**
  - the colour-trick depth matches analytic depth for a single opaque, flat Gaussian facing the camera;
  - after a reinit on a real checkpoint, parameter and optimizer-state shapes agree, and 200 further steps run without NaN.

### R1 — reinitialization audit (garden and room, seed 0; no training comparison)

- **Inputs:** warm-up states from short stock runs at k ∈ {2000, 5000}. k avoids multiples of 3000, where opacity resets fall. Also a 30k state (`FGS_SAVE_STATE`).
- **Run the reinitialization offline** on each state and record:
  - **cost**, per part, on the H100, in seconds;
  - **keep and reject rates** per view, by depth bin and by image region;
  - **rejection masks** for 4 views per scene, inspected and written up: edges, sky, thin structures, textureless surfaces, reflections;
  - **evidence from the training photos**, independent of the model's geometry:
    - (a) 7×7 ZNCC between a point's reference pixel and its reprojection in each agreeing view, classed as consistent (≥ 0.7), inconsistent (< 0.3) or unverifiable (patch too uniform), compared between kept and rejected points;
    - (b) COLMAP sparse points: depth of each observed SfM point against the rendered depth at its pixel. These are triangulated from matched features, so they are independent of the model, but sparse and biased toward texture;
  - **agreement with the 30k model's depth**, reported as such.
- **Gate:**
  - the measured cost leaves a positive budget, i.e. `T_warm-up(k) + T_reinit` is below the reference's time to target (stage-run estimates for now: garden about 209 s, room about 149 s; R2 replaces them);
  - the rejection masks have been inspected;
  - kept points are more photo-consistent than rejected ones. If not, the filter is not picking out geometry the images support, so revise its thresholds once, then decide.
- **Choose k:** the earliest k that passes. Report both.

### R2 — controlled training comparison (garden and room, 3 seeds, the chosen k)

| arm | densification and opacity resets | reinit at k | N after k | compared with | measures |
|---|---|---|---|---|---|
| `ref` | stock, to 14.9k | no | grows as stock | — | reference time and quality |
| `stop` | end at k | no | N_k | `ref` | effect of ending densification itself |
| `reinit-same` | end at k | yes, filtered | N_k (matched) | `stop` | effect of repositioning at a fixed count |
| `reinit-target` | end at k | yes, filtered | stock N at 15k | `reinit-same`, `ref` | effect of the count; the hypothesis itself |

That is 24 runs. Optimizer, learning-rate schedule and everything else are matched. Reported per scene before any summary:
- time to target;
- quality at matched training time (the reference's 25/50/75/100% points);
- curves of PSNR, SSIM and LPIPS against training time;
- N and ms/step after k;
- per-view checks.

**How to read it:**
- `reinit-same ≈ stop` at equal time: repositioning does not help at this k.
- `reinit-same > stop`, but neither reaches the target: repositioning helps, and the count is the limit. Look at `reinit-target`.
- `reinit-target` crosses before `ref` on the mean curve, with clean per-view checks: H is supported on that scene. Go to R3/R4.
- `reinit-target` never reaches the target by 30k: the reinitialized start is worse than what densification produces. Stop, or do one R3 revision.

### R3 — one factor at a time (only if R2 shows promise)

- **Filtered vs unfiltered** (`MIN_AGREE` 2 vs 0): does the filter matter?
- **Attributes, one at a time:** opacity 0.5; footprint-based flat scales with normals from depth gradients; SH degree 1 by ridge regression (base colour not shrunk) from the agreeing views.
- **The other k.**
- At most one revision round. Each variant is compared with the R2 winner, using the same rule.

### R4 — confirmation (only for arms that crossed)

- Rerun without intermediate evaluations, stopping at the crossing step, 3 seeds each: the reference at its crossing, the new arm at its own. Measure wall time and final metrics.
- **Camera-motion inspection:** the same smooth path through the training poses, plus the same path raised by 10% of the scene extent (off-orbit views, where view-dependent colour is least constrained). Rendered for both arms and inspected side by side for floaters, popping and colour shifts. Findings are written up with frames.
- **Only after R4:** decide whether to run all 9 scenes.

### R5 — combination with the main track (only after H holds)

Add the Phase 2 gains one at a time, each against the then-current best main-track stack, with the same crossing rule:
1. **A (macro blend backward):** kernel-only, but the reinitialized Gaussians' size and overlap may change its gain.
2. **B (selective Adam):** tested as its own question. Does ending densification make an earlier switch safe? Its step-0 failure (−0.09 to −0.13 dB, 17% fewer Gaussians) is thought to be an interaction with densification.
3. **C (bf16 SH-rest moments):** last, and only with its own quality check.

## Stop rules

- **After R1:** stop if the measured reinitialization cost leaves no positive budget on either scene.
- **After R2:** stop the track if `reinit-target` does not cross before `ref` on both scenes.
  - "Faster on at least 2 of 3 seeds" is an exploratory speed signal only. It cannot outweigh a quality failure on the third seed.
  - A seed whose quality failure is attributable to the reinitialization blocks a positive claim until it is explained.
- **After R3:** stop if no variant crosses before `ref` on both scenes.
- **Overall:** about 8 GPU-hours before any nine-scene decision. Rough estimate: R1 under 0.5 h; R2 about 3–3.5 h including evaluations every 500 steps; R3 and R4 about 2–3 h.

## Tools to write (all in `alt_reinit/`)

| file | what |
|---|---|
| `build_reinit.sbatch` | stages `build unit audit compare confirm`, in the style of `phase2/build_plus.sbatch` |
| `make_reinit_variant.sh` | clones FasterGS @1144604 to `FasterGSReinit`; applies the clock patch, the track patch and the rename; builds |
| `patches/fastergs_reinit.py` | trainer patch: `REINIT` keys, the reinit callback, CUDA-event timing of the between-step callbacks |
| `reinit_core.py` | depth render (colour trick), neighbour selection, filter, deduplication, attribute construction (shared by the callback and the audit) |
| `tools/reinit_audit.py` | R1 on saved states: cost, keep/reject rates, masks, photo ZNCC, COLMAP agreement, agreement with 30k |
| `analyze_compare.py` | R2–R4 analysis: validation, crossing rule, matched-time quality, per-view checks |
| `tests/` | CPU unit tests of the geometry, plus the GPU checks in R0 |

## Decisions for you

1. **Scenes:** garden (outdoor) and room (indoor) for R1–R4?
2. **k:** chosen by rule in R1, as the earliest passing value of {2000, 5000}? Or fix it now?
3. **Quality target tolerance:** ε_SSIM and ε_LPIPS equal to the reference's own seed-to-seed standard deviation in this job, PSNR at or above the reference mean, and no extra one-view failures? Or stricter or looser?
4. **First reinitialization:** filtered (2 of 4 neighbours) as above, with unfiltered as the R3 ablation? Or the other way round?
5. **R2 arms:** keep both `reinit-same` and `reinit-target` (24 runs), or only `reinit-target` (18 runs, but no way to separate position from count)?
6. **Budget cap:** about 8 GPU-hours before any nine-scene decision?
