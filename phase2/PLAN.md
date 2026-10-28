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
  - `cluster/tests/test_lean_adam.py` must still pass, 16/16.
  - A new test confirms that the switch flips at the configured iteration and that rows invisible after it keep their moments.

## Step 2 — factorial on garden (development scene)

There are 8 arms (A × B × C, each on or off) and 3 seeds, run in one job, interleaved per seed. Logging uses the CUDA-event clock (SEGCLOCK), and evaluation happens only at 30k (no intermediate test renders, so peak VRAM is comparable again). That is 24 runs at about 4.5 min each, roughly 2 h on one H100.

It answers three questions:
1. Do the savings add up? The interaction terms come from the 2³ design.
2. Does C hold quality over 3 seeds, alone and stacked?
3. Is the full stack within quality noise?

It also gives a direct stage-timing check (FGS_PROF at steps 5k and 20k) for the stacked build.

## Step 3 — the benchmark

- **Scenes:** all 9 Mip-NeRF360 scenes. These are bicycle, flowers, garden, stump and treehill at 1/4 resolution, and room, counter, kitchen and bonsai at 1/2. All are already on the cat workspace.
- **Arms and jobs:** stock Faster-GS vs the full stack (and B-less if step 0 made B optional). Each scene gets 3 seeds, interleaved, in one job per scene, so a scene's pairs share a node.
- **Report per scene:**
  - training time (CUDA events and the NeRFICG timer);
  - ms/it;
  - PSNR, SSIM and LPIPS, with paired differences per seed;
  - final Gaussians;
  - peak VRAM;
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

## Decisions for you

- Merge the Phase 1 branch and fast-forward `master` (step 0.2)?
- Download Tanks & Temples and Deep Blending for the optional part of step 3, or Mip-NeRF360 only?
- Are the quality tolerances above (−0.05 dB mean, −0.10 dB worst scene, +0.002 LPIPS) the bar you want?
