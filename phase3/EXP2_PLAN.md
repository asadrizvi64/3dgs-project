# Phase 3 Experiment 2: training schedule, resolution × position learning rate (DRAFT, 2026-11-08)

**Status: draft. Not pre-registered yet, and nothing in it has run.** It is written while Experiment 1 is running, and
it does not depend on Experiment 1's results. It is finalised (this status line changed, committed) only after Experiment 1
has been analysed against its own rule and the engineering items below pass their smoke tests. The resolution schedule was
settled on 2026-11-08 after a second review: the fixed schedule, no adaptive scheduler, no further arms. It is launched only after that commit. Anything
changed after the finalising commit goes under "Changes after start", as in [PLAN.md](PLAN.md).

## Question

Experiment 1 asks whether changing the Gaussian population from step 5,000 reaches the control's quality sooner, under
Faster-GS's fixed training schedule. Experiment 2 asks the separate question: **does changing the training schedule
itself, with the population rules untouched, reach the same quality sooner?** Two levers, each tested alone and together:

- **Progressive resolution:** train early steps on downsampled images and switch to full resolution later.
- **Compressed position learning-rate decay:** decay the Gaussian-position learning rate over 15,000 steps instead of
  30,000.

The two experiments stay independent. Combining a schedule lever with a population intervention is a later step, and only
for components that pass their own experiment.

## Motivation, and what is not yet checked

An outside literature review (2026-11-08) reports that resolution scheduling (DashGaussian, CVPR 2025) gives about 1.3–1.45×
on several backbones, and that compressing the position decay (SAD-GS, SIGGRAPH 2026) adds up to 0.9 dB at early checkpoints.
**These are the review's readings, not verified by us.** Several cited papers postdate what the assistant can check from
memory. Before finalising, DashGaussian, SAD-GS, FastGS and ShorterSplatting are read directly, and their reported numbers
are motivation only. None is an expected result on Faster-GS.

Two reasons the published gains may not transfer:

- **Our images are already small.** Garden is trained at 1/4 resolution (about 1297×840) and room at 1/2 (about 1557×1038).
  DashGaussian's gains come mostly from full-resolution Mip-NeRF 360. At our sizes a step's cost is less dominated by pixels,
  so a resolution schedule saves less.
- **Faster-GS is already fast per step.** Per-step savings are a smaller share of total time when fixed per-step costs (the
  optimizer, sorting set-up) are already a large share.

The claim that no paper measures time to a quality floor matched to a control is a **possible gap**, not an established one,
until related work is searched systematically.

## Base

Same build as Experiment 1: `FasterGSUp` (1144604 + `upstream/upstream_main_kernels.patch`), dense Adam, fp32 moments, no
phase 2 stack, the same clock hook. Every run trains **from step 0**; no Experiment 1 checkpoint is reused, because a
resolution schedule changes the early trajectory and the population that grows from it.

## Design

- **Scenes:** garden (1/4) and room (1/2). **Seeds:** 0, 1, 2. **Length:** 30,000 steps for every arm.
- **Arms (2×2):**

| arm | resolution | position learning-rate decay |
|---|---|---|
| `C0` | fixed, full | 30,000 steps (default) |
| `C1` | progressive | 30,000 steps |
| `C2` | fixed, full | 15,000 steps |
| `C3` | progressive | 15,000 steps |

- **Runs:** 2 scenes × 3 seeds × 4 arms = 24, arms interleaved in an order rotated per seed. One job per (scene, seed)
  rather than per scene, so that one job's time limit cannot cut a seed in half (Experiment 1's room job is expected to
  exceed its limit).

### Progressive resolution (`C1`, `C3`)

- **Schedule (chosen 2026-11-08, after review; fixed rather than adaptive):** downsampling factor k = 4 for steps
  0–1,999, 2 for 2,000–5,999 and 1 (full) from 6,000 on. Full resolution comes well before densification ends at 14,900,
  so the last 9,000 densification steps and all later steps see full-resolution images. DashGaussian's adaptive scheduler
  is kept for a later experiment, because it also changes the growth schedule.
- **Mechanism, with camera and photo agreeing exactly:**
  - Output size first: w' = ⌊w / k⌋, h' = ⌊h / k⌋. The training image sizes are not all multiples of 4. A room height of
    1,038, for example, gives 259 rows at k = 4.
  - The photo is cropped to its top-left k·w' × k·h' pixels (at most k − 1 columns and rows dropped at the right and
    bottom edges), then average-pooled by k on the GPU. Each output pixel is the exact mean of a k × k block.
  - The camera copy gets width w', height h', and focal lengths and principal point divided by k. Faster-GS puts pixel
    centres at (i + 0.5) with the origin at the image corner (`kernels_forward.cuh`, `kernels_backward.cuh`), and NeRFICG
    scales intrinsics the same way when it loads downscaled images (`MipNeRF360.py`). Output pixel i' then covers original
    pixels [k·i', k·i' + k), so its centre k·(i' + 0.5) is exactly where the scaled camera puts it. Cropping at the right
    and bottom does not move the origin, so no half-pixel shift is needed.
  - Both happen inside the training clock.
- **Smoke test for the mapping, in three parts.** Rendering at low resolution is not the same operation as rendering at
  full resolution and averaging. Alpha compositing is nonlinear, and the screen-space dilation and the Gaussians' cut-off
  are fixed in pixels. A correct camera can therefore still give a different image, so image equality is not required.
  1. **Geometry (pass/fail):** project a set of 3D points, including the scene's Gaussian means, through the full and the
     k = 2 and k = 4 cameras. Every projected coordinate must satisfy u' = u / k and v' = v / k to within float precision.
  2. **Alignment (pass/fail):** a small synthetic scene of a few sharp, distinctive Gaussians, rendered at full
     resolution and through the k = 2 and k = 4 cameras. The centroid of each blob must sit where check 1 predicts, with no
     systematic offset: mean offset below 0.05 low-resolution pixels in x and y.
  3. **Training (pass/fail):** a short run through both switches (2,000 and 6,000 steps). Camera and photo sizes are equal
     at every step, forward and backward run at each resolution, gradients are finite, and the loss shows no jump larger
     than its normal step-to-step spread at a switch.
  - **Diagnostic only:** a trained garden model rendered through the k = 2 and k = 4 cameras, compared with the
    full-resolution render cropped and average-pooled the same way. The PSNR between them and the best-matching offset
    within ±1 pixel are reported. Neither is a requirement.
- **Test evaluation is always at full resolution.**
- **Densification coupling (decided here, not tuned on results):**
  - Faster-GS accumulates the screen-space mean gradient after converting it from pixels to normalised screen units
    (`0.5 · width`, `0.5 · height` in `kernels_backward.cuh`). The loss is a per-pixel mean. To first order the accumulated
    gradient of a smooth image region is then the same at every resolution, so **the gradient threshold (0.0002), the
    interval (100) and the start and end steps are not rescaled.**
  - First order is not invariance: image frequency content, the loss gradients and visibility all change with resolution.
    So the population is measured rather than assumed to match (Gaussian count every 1,000 steps and at each switch).
  - What does change at low resolution is the missing fine detail: fewer Gaussians cross the threshold early. That is the
    intended effect of a coarse-to-fine schedule, not an error to correct.
  - Reading aid, not a gate: if `C1`'s Gaussian count at 15,000 differs from `C0`'s by more than 25% on a scene, that
    scene's `C1` result is read as "resolution schedule plus a changed population", not resolution alone.
  - Kept as they are: the 2D screen-space dilation and the 3D filter (computed once from the full-resolution focal length).

### Compressed position decay (`C2`, `C3`)

- `LEARNING_RATE_MEANS_MAX_STEPS=15000`, everything else default. This is a whole alternative trajectory, including its
  low-rate second half, not an early-training speed-up alone.
- NeRFICG's `LRDecayPolicy` clamps at the final rate, so from step 15,000 positions train at the final rate
  (1.6e-6 × scene extent) for the remaining 15,000 steps. This is the intended test: whether position updates in the second
  half matter for time to quality.

## Evaluation during training

- Test PSNR, SSIM and LPIPS at full resolution every 1,000 steps from 1,000 to 30,000, stamped with training-only time
  (SEGCLOCK, evaluations excluded). The test images are used only to measure.
- Gaussian count every 1,000 steps.
- **Rendering work, descriptive only:** at steps 1,000, 2,000, 5,000, 6,000, 10,000, 15,000 and 30,000, the number of
  Gaussian–tile instances and the mean and maximum number of Gaussians per tile, averaged over 10 fixed training views at
  that step's training resolution. This separates three possible sources of a speed-up: fewer pixels, a different
  population, or less rasterisation work per Gaussian. It is not a decision input.
- **Memory, measured separately (new instrumentation, see engineering list):** peak allocated memory during training steps
  only, and peak during each test evaluation, with the peak statistics reset around every evaluation.
- Elapsed wall time including evaluations (`FGS_WALL`), as a second clock.

## Measures

- **Target quality:** per (scene, seed), `C0`'s test PSNR at 30,000 minus 0.05 dB. **One target for all four arms** of that
  (scene, seed), so an arm cannot pass by ending at a lower quality.
- **Time to target (T):** as in Experiment 1. First crossing, linear interpolation between evaluation points, "not reached"
  counts as a failure for that seed. Sensitivity checks (not decision inputs): targets −0.10 and −0.20 dB, and "reaches and
  stays".
- **Final quality at 30,000:** PSNR, SSIM, LPIPS paired against `C0`.
- **One-view failures:** the phase 2 definition.
- **Factorial effects**, on log T per (scene, seed), reported descriptively. With 3 seeds no significance test is claimed.
  - resolution effect = ½ [(C1 − C0) + (C3 − C2)]
  - learning-rate effect = ½ [(C2 − C0) + (C3 − C1)]
  - interaction = (C3 − C2) − (C1 − C0)
  - A pair with a "not reached" arm is left out of these, and listed.

## Checks before the decision

1. `C0` reproduces the uninterrupted `FasterGSUp` runs: final PSNR and Gaussian count within the seed spread of upbench jobs
   4552203/4 (and seed-2 reruns 4552280/1).
2. Each arm did what it says, read from its logs: the resolution actually rendered per step range (logged at each switch),
   and the position learning rate at 10,000, 15,000 and 20,000.
3. All runs pass the phase 2 run validation.

## Decision rule (per arm `C1`, `C2`, `C3` against `C0`)

Same as Experiment 1. An arm passes if, on **both** scenes:

- median paired reduction in T ≥ 10%, with T lower in at least 5 of the 6 pairs;
- mean final ΔPSNR ≥ −0.10 dB per scene and mean ΔLPIPS ≤ +0.002;
- no more one-view failures than `C0`.

| outcome | reading and next step |
|---|---|
| `C2` passes | the default position schedule is a weak control. Any Experiment 1 arm that passed is re-checked on the compressed schedule before it is built on |
| `C1` passes | resolution scheduling transfers to Faster-GS at these resolutions. It becomes the schedule for later experiments |
| only `C3` passes | the two levers help only together. The interaction is reported, and `C3` becomes the candidate schedule |
| nothing passes | neither lever shortens time to quality on Faster-GS at these resolutions. That is a result in its own right, reported against the published gains |

**What this cannot show:** that either lever is new (both are published), or that results hold at full resolution or on
other scenes and GPUs. Two scenes, three seeds, one H100 type.

## Engineering needed first (each validated by a smoke test before launch)

- [ ] Resolution schedule hook (env-gated like the other patches): scaled camera copy, area-downsampled photo, a log line at
      each switch. Smoke test: with factor 1 throughout, training is bit-identical to the unpatched run for 1,000 steps.
- [ ] `CFG_SET` for `LEARNING_RATE_MEANS_MAX_STEPS` (the key appears once in the yaml, so the existing mechanism works). Smoke
      test: logged rates at fixed steps match the formula.
- [ ] Peak-memory instrumentation split into training and evaluation, reset around each evaluation.
- [ ] **Find the cause of the 64 GiB evaluation peak** seen in Experiment 1 (next section), and why a room test evaluation
      is slow. Experiment 2 has 30 evaluations per run, so evaluation cost sets the job length. Run after Experiment 1, as
      its own short job on a saved state: one test evaluation, with `torch.cuda.synchronize()` and a peak reset at each
      boundary, recording allocated memory before, the peak during, allocated and reserved memory after, and the peak
      and time of each stage (rendering, post-processing, photo loading, each metric).
- [ ] Tile-instance counters for the rendering-work measure, read from the forward pass's buffer sizes if the backend
      exposes them; otherwise this measure is dropped and the plan says so.
- [ ] `analyze_ttq.py`: arm names, `C0` as the reference arm, the factorial effects.

## Known measurement issue carried over from Experiment 1

Experiment 1's resumed runs report a peak of 69–71 GiB allocated, against 5–6 GiB in the prefix runs. The peak is read with
`torch.cuda.max_memory_allocated()` at the last training step, and never reset. In the resumed runs it therefore includes the
26 test evaluations run during training. In the prefix runs no evaluation has run yet at that point. The extra is about
64 GiB in every resumed run, whether the run holds 0.9 M or 4.2 M Gaussians. That points to a fixed-size allocation in the
evaluation path, not to training memory. The exact allocation is not identified yet. Read-only checks so far; nothing in the
running jobs was changed.

## Cost

24 runs. Training is about 3–4 minutes per run. Evaluations dominate: a room run in Experiment 1 takes about 28 minutes
with 26 evaluations, so 30 evaluations make room the long pole. Estimate: about 10–12 GPU-hours in 6 jobs, before any
evaluation speed-up.
