# Phase 2 plan: stack the verified gains into one build and benchmark it against Faster-GS (draft, 2026-10-28)

**Goal:** one Faster-GS build with every gain that has been shown to keep quality. It is benchmarked against stock Faster-GS on all 9 Mip-NeRF360 scenes, 3 seeds each, paired on the same node, and reported per scene before any average.

## What we have (garden, H100, 3 seeds unless noted)

| id | change | where | train time | quality | status |
|---|---|---|---|---|---|
| **A** | Macro blend backward (lane owns pixels, subtile skip; `FGS_BLEND_BWD=macro`) | `FasterGSMacro` variant, branch `claude/phase1-3d-fcf489` (not merged) | **−10.7%** (252.8 → 226.1 s) | +0.01 dB, within noise; synthetic parity ≤ 2.3e-6 | garden only |
| **B** | Selective Adam from 15k (after densification ends) | `cluster/patches/fastergs_sparse_from.py` (env `FGS_SPARSE_FROM`) | **−9.8%** (255.8 → 230.8 s) | no observed loss (+0.05 dB vs 9 stock runs) | garden done; **bicycle / stump / room / counter running** |
| **C** | bf16 SH-rest Adam moments (stochastic rounding) | lean-Adam patch, `SH_REST_BF16_MOMENTS` | −6.7% | +0.04 dB, **1 seed only** | needs seeds and scenes |
| ✗ | Selective Adam from step 0 | lean-Adam patch | −24% | −0.09 to −0.13 dB, 17% fewer Gaussians | rejected (quality) |
| ✗ | Pixel-block sampling | `results/ttq` | slower in time | better per pixel | rejected (cost) |

**The parts touch different stages, so the gains should mostly add.**

- A cuts the blend backward by about 0.87 ms per step, at every step.
- B cuts the optimizer by about 1.6 ms per step in the second half only.
- C cuts optimizer bytes at every step. After the switch it overlaps with B, because B already skips about 65% of rows.

Rough projection from 255.8 s: A −27 s, B −25 s, C about −12 s in the first half and about −4 s in the second. That gives about **188 s (−27%, ≈1.36×)**. This is an estimate only; the factorial run in step 2 measures it.

## Step 0 — gates (no new code)

1. **B generalizes.** The 4-scene jobs (4480521–24) decide this, reported per scene. Pre-registered rule: B goes into the stack if, on every scene, the mean paired ΔPSNR is ≥ −0.05 dB and ΔLPIPS ≤ +0.002. If a scene fails, B is stack-optional and reported separately.
2. **Branches are consolidated.** Phase 1 (A) lives on `claude/phase1-3d-fcf489`. Everything since lives on `phase0-scaffold`, and `master` is far behind. Proposal: merge `claude/phase1-3d-fcf489` into `phase0-scaffold`, which only adds files under `phase1/`, `bench/phase1/` and `results/phase1/`. Then fast-forward `master` to it. **Needs your OK.**

## Step 1 — one build with flags (`FasterGSPlus`)

- Base it on the `FasterGSMacro` variant, which is already Faster-GS @1144604 with lean Adam (the parent of A), via `phase1/make_variant.sh`. Make B a config key instead of an env var: `TRAINING.OPTIMIZER.SELECTIVE_FROM: 15000` (default −1 = off). The same switch code moves from the patch into the variant's Trainer.
- **Flags:**
  - A = `FGS_BLEND_BWD=macro|taming`;
  - B = `SELECTIVE_FROM`;
  - C = `SH_REST_BF16_MOMENTS`.
- With all flags off, the build must equal stock, so the factorial in step 2 runs in one build. Checked by a unit test: the stock and variant gradients match on the same forward, as in Phase 1, and one 30k garden run must fall within the stock range.
- **Checks:**
  - `phase1/tests/test_parity.py` must still pass, 20/20.
  - `cluster/tests/test_lean_adam.py` must still pass, 15/15.
  - A new test confirms that the switch flips at the configured iteration and that rows invisible after it keep their moments.

## Step 2 — factorial on garden (development scene)

There are 8 arms (A × B × C, each on or off) and 3 seeds, run in one job, interleaved per seed. Logging uses the CUDA-event clock (SEGCLOCK), and evaluation happens only at 30k (no intermediate test renders, so peak VRAM is comparable again). That is 24 runs at about 4.5 min each, roughly 2 h on one H100.

It answers three questions:
1. Do the savings add up? The interaction terms come from the 2³ design.
2. Does C hold quality over 3 seeds, alone and stacked?
3. Is the full stack within quality noise?

It also gives a direct stage-timing check (FGS_PROF at steps 5k and 20k) for the stacked build.

## Step 3 — the benchmark

Before it starts: rebuild FasterGSPlus with clock hook v4 (`STAGES="build unit switch"`), and check that the switch runs show the `FGS_N` and `FGS_TRAIN_END` lines.

- **Scenes:** all 9 Mip-NeRF360 scenes. These are bicycle, flowers, garden, stump and treehill at 1/4 resolution, and room, counter, kitchen and bonsai at 1/2. All are already on the cat workspace.
- **Arms and jobs:** stock Faster-GS vs the full stack (and B-less if step 0 made B optional). Each scene gets 3 seeds, interleaved, in one job per scene, so a scene's pairs share a node.
- **Report per scene:**
  - training time (CUDA events and the NeRFICG timer);
  - ms/it;
  - PSNR, SSIM and LPIPS, with paired differences per seed;
  - Gaussians at 15k, at the last step before cleanup, and after cleanup (clock hook v4);
  - training-only peak VRAM, read inside the last training step before the final evaluation (clock hook v4). The step 2 factorial still uses v3, so its VRAM is the trainer's figure read after the final evaluation;
  - total job duration, kept separate from training time.

  After the per-scene report, give the mean over scenes, plus the worst scene.
- **Win criterion (pre-registered):** median training-time reduction ≥ 20% across scenes. Mean ΔPSNR ≥ −0.05 dB, no scene below −0.10 dB, and mean ΔLPIPS ≤ +0.002.
- **Cost:** about 54 runs at about 5 min (bicycle and the indoor scenes take longer), around 5–6 GPU-hours.
- **Optional:** Tanks & Temples (truck, train) and Deep Blending (drjohnson, playroom), the other two datasets in the Faster-GS paper. They are not on the cluster yet; downloading them is a separate decision.

## Step 4 — next levers, measured only after step 3

1. **Early stop with the 30k LR schedule kept.** The switch curves suggest stock quality is reached before 30k. Test by stopping at 25k and 27.5k with the schedule unchanged (shortening the decay horizon would be a separate experiment).
2. **Earlier switch point.** Try 10k or 12.5k, which falls inside densification. Selective Adam during densification cost Gaussians before, so this needs the Gaussian count and quality checked closely.
3. **The densification half is now the larger cost:** 114 s of 231 s. Profile steps 0–15k of the stacked build to find what dominates there. The optimizer is still dense in that phase.
4. **Macro kernel follow-ups:** an nsys capture of a full step (open item in Phase 1), and the forward pass (7%), which is low priority.

## Step 7 — kitchen follow-up (pre-registered 2026-11-07, before any run)

Step 6b (A+B+C+D+S vs stock, 9 scenes x 3 seeds) failed its pre-registered criterion on one scene: kitchen ΔPSNR −0.171 dB
against the −0.10 dB per-scene limit, driven by two one-view failures in the stack's runs. **The step 6b verdict stays FAIL
whatever this follow-up shows.** The follow-up asks one question: is the kitchen deficit reproducible on fresh seeds, or was it
noise from rare one-view failures?

- **Design:** kitchen only (1/2 resolution, as in step 6b), stock Faster-GS vs A+B+C+D+S (`abcds`, build of step 6),
  **10 new seeds, 3–12** (seeds 0–2 are excluded because their results were seen). One job, so all 20 runs share a node;
  arms interleaved per seed with the order rotated by seed, as in step 6b. Same training settings, same timing hook
  (`FGS_WALL=0,15000,30000`).
  `STAGES=bench SCENE=kitchen BENCH_ARMS="stock abcds" SEEDS="3 4 5 6 7 8 9 10 11 12" sbatch --job-name=gs-kfollow-kitchen --time=02:30:00 phase2/build_plus.sbatch`
- **Analysis:** `python phase2/analyze_bench.py phase2/logs/gs-kfollow-kitchen-<job>.out --arms stock abcds --seeds 3 4 5 6 7 8 9 10 11 12`
  (the script reports it as a partial benchmark, which is expected for one scene). All 20 runs must pass its validation.
  A run that fails validation for a technical reason (crash, node failure) is rerun with the same seed; no seeds are added
  or dropped for any other reason.
- **Primary criterion:** mean paired ΔPSNR (A+B+C+D+S − stock) over seeds 3–12, all test images.
  - ≥ −0.10 dB: the step 6b kitchen deficit is **not reproduced**.
  - < −0.10 dB: the deficit is **reproduced**; kitchen counts as a quality regression of the stack.
- **Secondary, pre-specified, not part of the decision:**
  - one-view failures with the step 6b definition (a test image > 3 dB below the median of the same image in the scene's
    other runs, here the other 19): failing images and runs with at least one failure, per arm. An exact McNemar test on the
    seeds where only one arm has a failing run (two-sided, α = 0.05). With 10 seeds this has little power, so a
    non-significant result does not show the rates are equal;
  - ΔPSNR with failing images removed from every run (as in step 6b);
  - ΔSSIM, ΔLPIPS, per-seed ΔPSNR, time and speedup;
  - pooled 13 seeds (0–12, step 6b job plus this one, descriptive only, since seeds 0–2 were seen).
- **Cost:** 20 runs at about 4.3 min, about 1.5 GPU-hours.

## Decisions for you

- Merge the Phase 1 branch and fast-forward `master` (step 0.2)?
- Download Tanks & Temples and Deep Blending for the optional part of step 3, or Mip-NeRF360 only?
- Are the quality tolerances above (−0.05 dB mean, −0.10 dB worst scene, +0.002 LPIPS) the bar you want?
