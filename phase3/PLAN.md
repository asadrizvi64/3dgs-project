# Phase 3 plan: time to quality through Gaussian population control (draft, 2026-11-08)

Phase 2 made each training step cheaper. Phase 3 asks a different question: **can changing which Gaussians exist,
from mid-training on, reach the same test quality in less training time?** Experiment 1 is a diagnostic. It tests
whether population size and turnover after step 5,000 matter for time to quality at all, before any new method is built.

This plan is fixed before any Experiment 1 run. Anything changed after a run has started is listed under
"Changes after start", with its date and reason.

## Why this direction

- Upstream Faster-GS (090e8ad, 2026-10-31) already has most of our memory-access gains. Our remaining SH-load change is
  2–3% of one kernel, about 0.3% of a training run ([upstream/README.md](../upstream/README.md)). Kernel work has
  reached diminishing returns for us.
- Phase 2 results that point here:
  - Selective Adam is fine from 15k but cost quality when applied during densification (step 2). Training phases
    need different treatment.
  - A garden run grows to about 4–5 M Gaussians. Per-step cost scales with that count.
  - Rare one-view failures (a test image more than 3 dB below normal) happen in stock and optimised runs alike.
    A mean PSNR can hide them, so they are tracked as their own measure.

## Related work to read before claiming anything new

3DGS-MCMC (NeurIPS 2024; relocation and a population cap, already in Faster-GS as `USE_MCMC`), Speedy-Splat
(CVPR 2025; already in Faster-GS as `SPEEDY_SPLAT`), Taming 3DGS (budgeted densification). An outside review also
names FastGS (CVPR 2026) and SAD-GS (SIGGRAPH 2026). Those two have **not been checked yet**, and must be read before
Experiment 2 is designed. Experiment 1 only uses mechanisms that already exist, so it claims nothing new.

## Experiment 1: population intervention at step 5,000

### Base

Upstream main kernels, built as `FasterGSUp`: 1144604 plus `upstream/upstream_main_kernels.patch`. Settings are dense
Adam and fp32 moments, with no phase 2 stack, so results compare against current upstream behaviour. The same clock
hook is used as in phase 2 (train time = SEGCLOCK, CUDA events, evaluation excluded).

### Design

- **Scenes:** garden (1/4 resolution) and room (1/2). **Seeds:** 0, 1, 2.
- **Shared prefix:** each (scene, seed) trains from 0 to 5,000 once with the default config. A full training
  checkpoint (`.train`: model, optimizer state, iteration count) is saved at step 5,000. Every arm continues from that
  same file, so the arms differ only from step 5,000 on.
- **Arms**, each continued to 30,000 with the default learning-rate schedule:

| arm | from step 5,000 | role |
|---|---|---|
| `ctl` | default config unchanged: densify to 14,900, opacity reset every 3,000 | control |
| `stop` | densification ends at 5,000 (`DENSIFICATION_END_ITERATION=5000`). Opacity resets end with it, and hard pruning then runs as it does after the default end. | fewer Gaussians, cheaper steps |
| `mcmc` | `USE_MCMC=true` with `MAX_PRIMITIVES` = the Gaussian count at step 5,000 (population held constant, low-opacity Gaussians relocated). Other MCMC settings follow upstream's yaml comments: opacity and scale regularisation 0.01, no opacity reset, densification end 14,900 as in `ctl`. | replacement at constant size, using the existing method |
| `oracle` | as `mcmc`, but `MAX_PRIMITIVES` = the `ctl` run's final count for the same (scene, seed), reached by MCMC growth | **upper-bound reference only**: it uses a number known only after training, so it is never a method |

- `oracle` runs after `ctl` for the same (scene, seed), in the same job. The other arms are interleaved in a rotated
  order per seed, in one job per scene, as in phase 2.
- **Runs:** 2 scenes × 3 seeds × 4 arms = 24 continuations, plus 6 prefixes.

### Evaluation during training

- Test PSNR, SSIM and LPIPS at every 1,000 steps from 5,000 to 30,000, using the stage-render mechanism from phase 2.
  Each evaluation point is stamped with SEGCLOCK cumulative training time from step 0. Evaluation time is excluded,
  and the prefix time is counted once, identically for every arm.
- Gaussian count at every 1,000 steps, plus training-only peak VRAM.
- The test images are used **only to measure**. No arm makes any decision from them.

### Measures

- **Target quality:** per (scene, seed), the `ctl` run's test PSNR at 30,000 minus 0.05 dB.
- **Time to target (T):** the training time at which an arm's test PSNR first reaches the target, linearly
  interpolated between the two evaluation points around the crossing. An arm that never reaches it gets T = not
  reached, and that counts as a failure for that seed. Two values are reported as sensitivity checks, not decision
  inputs: T for targets of −0.10 and −0.20 dB, and T for "reaches the target and stays at or above it".
- **Final quality at 30,000:** PSNR, SSIM and LPIPS, paired against `ctl` per seed.
- **One-view failures:** the phase 2 definition (a test image more than 3 dB below the median of the same image across
  that scene's other runs), counted per arm.
- Training time to 30,000, Gaussian count over time, peak VRAM.

### Checks before the decision

1. **The checkpoint branch reproduces the uninterrupted run.** `ctl`'s final PSNR and final Gaussian count must lie
   within the seed-to-seed spread of the uninterrupted `FasterGSUp` runs of the same scenes (upbench job 4552203/4).
   If not, branching changed training, Experiment 1 is invalid, and the loader is fixed first.
2. **Each arm did what it says, read from its logs.**
   - `stop`: no densification after 5,000.
   - `mcmc`: Gaussian count constant from 5,000, within the cap.
   - `oracle`: reaches its cap.
   - `ctl`: the default schedule.
3. All 30 runs pass the phase 2 run validation (one clock record per evaluation point, finite metrics, archived
   per-image metrics).

### Decision rule (per non-oracle arm: `stop`, `mcmc`)

An arm **passes** if all of these hold on **both** scenes:

- median paired reduction in T against `ctl` ≥ 10%, with T lower than `ctl` in at least 5 of the 6 (scene, seed) pairs;
- mean final ΔPSNR ≥ −0.10 dB per scene, and mean ΔLPIPS ≤ +0.002;
- no more one-view failures than `ctl` across the 6 pairs.

What happens next:

| outcome | next step |
|---|---|
| a non-oracle arm passes | the mechanism behind it (fewer Gaussians, or turnover at fixed size) becomes Experiment 2's subject: an adaptive policy driven only by training signals (visibility, gradient history, opacity, age, training-view residuals) |
| only `oracle` passes | the population size matters, but it has to be predicted. Experiment 2 is predicting the target count from training signals before step 5,000 |
| nothing passes, `oracle` included | population control after step 5,000 is not the lever on these scenes. Switch to the review's fallback: selective update scheduling (which Gaussians get optimizer updates, and when) |

**What this experiment cannot show:** that any method is new. It is two scenes and three seeds, on one GPU type
(H100). A pass leads to more scenes and to the related-work comparison, not to a claim.

### Engineering needed first (validated by check 1 and check 2 above, before the experiment runs)

- [ ] Continue from a `.train` checkpoint with config overrides (densification end, `USE_MCMC`, `MAX_PRIMITIVES`,
      regularisation weights) through `cluster/train_fastergs.sbatch`. Overrides must take effect from the resumed step.
- [ ] Switch to MCMC mid-run: MCMC initialisation assumptions (opacity parameterisation, noise injection) must not
      require MCMC from step 0. If they do, `mcmc` and `oracle` are redefined, the change is recorded under "Changes
      after start" **before** any arm runs, and the reason is given.
- [ ] Evaluation every 1,000 steps from a resumed run, with SEGCLOCK continuing across the resume.
- [ ] `phase3/analyze_ttq.py`: validation, the T computation (interpolation, never-reached handling), paired tables,
      the decision rule above.

### Cost

6 prefixes (about 1 min each) plus 24 continuations (about 3–4 min each plus about 26 evaluations), around 3 GPU-hours.

## Phase 2 items kept open

- The upstream SH-load comparison (jobs 4552203/4) finishes and is recorded against its own rule
  ([upstream/README.md](../upstream/README.md)). No PR is opened without the user's go-ahead.
- The phase 2 result stands as reported: 1.451× median over Faster-GS c60ea26 (2026-10-10), not over current upstream.

## Changes after start

These are clarifications made on 2026-11-08 while building the code, before any Experiment 1 run (smoke tests excluded):

- **`stop`:** the row above says hard pruning runs after densification ends. In upstream that pruning exists only with
  Speedy-Splat, which is off by default. So `stop` is only `DENSIFICATION_END_ITERATION=5000`: no new Gaussians, no
  opacity resets, then the normal end-of-training cleanup. No pruning is added.
- **`mcmc` / `oracle` settings:** "upstream's yaml comments" also say to use `LEARNING_RATE_OPACITIES` 0.05 with MCMC.
  It is applied, so the full override is `USE_MCMC=true`, `MAX_PRIMITIVES`, both regularisation weights 0.01, and
  opacity learning rate 0.05. Densification end (14,900) and Morton-ordering end (15,000) stay as in `ctl`.
- **Oracle cap:** the `ctl` count at the last training step before the end-of-training cleanup (`FGS_TRAIN_END N`).
- **Evaluation point i:** the state after exactly i training steps, taken before that step's SH-degree, densification,
  ordering and opacity-reset callbacks. Evaluations at multiples of 3,000 therefore show the state before the reset.
- **Branch state:** NeRFICG's `.train` checkpoint pickles the trainer and cannot be used. The branch file (`state_0005000.pt`) holds:
  - the parameters and optimizer state;
  - densification statistics and SH degree;
  - the training-view sampler and the CPU and CUDA random states;
  - the clock's cumulative time.

  Implementation: [cluster/patches/fastergs_resume.py](../cluster/patches/fastergs_resume.py).
