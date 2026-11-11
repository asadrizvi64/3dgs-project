# Experiment 1A results (2026-11-11)

Pre-registration: [EXP1A_PLAN.md](EXP1A_PLAN.md), frozen at 026f6d1 before launch. The one change after the freeze (the
`--rerun` reference fix, 76366e8) came before any job was submitted, and it is unused here. Full analysis output:
[exp1a-analysis.md](exp1a-analysis.md), from the command fixed in the plan:

```
python phase3/analyze_ttq.py gs-x1a-garden-s30-4561623.out gs-x1a-garden-s31-4561625.out gs-x1a-garden-s32-4561627.out \
    gs-x1a-room-s30-4561624.out gs-x1a-room-s31-4561626.out gs-x1a-room-s32-4561628.out --abl --seeds 30 31 32
```

**Jobs:**
- garden: 4561623, 4561625 and 4561627 (seeds 30–32), 2:11 h, 3:09 h and 2:17 h;
- room: 4561624, 4561626 and 4561628, about 3:35 h each.

All completed, with 10 runs each and no failed run, so no reruns. Repo c94332c, patch v2. The runs are in
`experiment_log.csv`.

## Outcome

| step | result |
|---|---|
| Validation, checks 1a–1d, check 2 (counters), callback records | pass on both scenes |
| 1. `mcmc` replicates Experiment 1R | **PASS**: T lower in 6 of 6 pairs; median ΔT 34.6% garden, 37.0% room |
| 2. Resolution | **PASS**: 6 of 6 pairs used; median \|f_rep\| = 0.06 |
| 3. R, relocation | **needed** |
| 3. Z, noise injection | **needed** |
| 3. G, regularisation | **needed**, sensitive to target selection (not resolved at −0.20 dB) |
| 3. L, opacity learning rate 0.05 | **not needed** (narrowly on garden LPIPS, see below) |
| 4. O, `ctl`'s opacity resets | **explain little** |

On the secondary clock (training step + three instrumented callbacks), every class is the same.

**Pre-registered reading:**
- On garden and room (seeds 30–32, H100), the combined MCMC policy's faster convergence needs relocation, noise
  injection and regularisation, each given the others. It does not need the higher opacity learning rate.
- The control is not slow because of its opacity resets.
- "Needed" is conditional on the rest of this policy, on two scenes and three seeds. It is not a claim that any
  component works alone, or is needed in general.

## Replication

| scene | `ctl` T (s) | `mcmc` T (s) | ΔT | mean ΔPSNR | mean ΔLPIPS |
|---|---|---|---|---|---|
| garden | 171.3, 163.4, 186.0 | 121.4, 106.9, 116.4 | 29.1%, 34.6%, 37.4% | +0.190 dB | −0.0043 |
| room | 129.1, 129.2, 130.7 | 74.4, 81.3, 85.2 | 42.4%, 37.0%, 34.8% | +0.443 dB | −0.0103 |

- This is consistent with Experiment 1R (median 31.4% and 35.5%), on fresh seeds.
- One-view failures, on the control-based reference: `mcmc` 0, `ctl` 1.
- `mrep`, the identical second `mcmc` run, differs from `mcmc` by f between −0.19 and +0.17. Its final PSNR is within
  0.007 dB on garden and 0.054 dB on room.

## Fractions of the advantage lost (primary target, training clock)

f = (T(arm) − T(`mcmc`)) / (T(`ctl`) − T(`mcmc`)). "∞" means the arm never reached the target. For resets, g =
(T(`ctl`) − T(`creset`)) / the same denominator, and "−∞" means `creset` never reached it.

| pair | A (s) | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|
| garden 30 | 49.9 | +0.08 | ∞ | ∞ | ∞ | +0.11 | −∞ |
| garden 31 | 56.5 | −0.02 | +1.36 | ∞ | +1.43 | −0.04 | −0.46 |
| garden 32 | 69.7 | +0.04 | ∞ | ∞ | ∞ | +0.04 | −0.44 |
| room 30 | 54.7 | +0.03 | +0.31 | +0.35 | +0.10 | +0.05 | +0.45 |
| room 31 | 47.9 | +0.17 | ∞ | +0.73 | +0.59 | +0.06 | −∞ |
| room 32 | 45.5 | −0.19 | ∞ | +0.52 | +0.49 | −0.18 | −0.17 |
| **median** | | 0.06 (\|f\|) | ∞ | ∞ | 1.01 | 0.05 | −0.45 |

**Sensitivity:**
- **R and Z:** "needed" at −0.10 dB and for "reaches and stays".
- **G:** at −0.20 dB the median f is about 0.28, which is "not resolved"; it is "needed" at the other targets. Hence
  "sensitive to target selection".
- **L:** "not needed" at every target.
- **O:** g is negative at most targets. Removing resets makes `ctl` slower, if anything.

## What "needed" means here: mostly quality, not only speed

Removing R, Z or G does not just slow the policy down. Those arms end clearly below `mcmc`, and in many pairs they never
reach `ctl`'s quality at all:

| arm (vs `mcmc`) | garden ΔPSNR | garden ΔLPIPS | room ΔPSNR | room ΔLPIPS | pairs never reaching the target |
|---|---|---|---|---|---|
| `mreloc` | −0.338 | +0.0193 | −0.727 | +0.0261 | 4 of 6 |
| `mnoise` | −0.303 | +0.0106 | −0.407 | −0.0051 | 3 of 6 (all garden) |
| `mreg` | −0.272 | +0.0090 | −0.394 | +0.0079 | 2 of 6 (garden) |
| `mlr` | −0.038 | +0.0019 | −0.056 | −0.0045 | 0 of 6 |
| `mrep` | −0.007 | −0.0001 | −0.054 | +0.0000 | 0 of 6 |

(Full-precision values from the RESULT lines. The analysis itself uses the metric files' 3-decimal values, as
pre-registered.)

- So relocation, noise and regularisation are needed for `mcmc` to **reach and exceed** the control's quality. Without
  any one of them, the fixed-size population mostly fails to get there.
- On room, `mnoise` and `mreg` do reach the target, but later: f between 0.10 and 0.73.
- **L, the boundary case.** On garden, `mlr`'s LPIPS is higher than `mcmc`'s by 0.002 in every seed at the files'
  3-decimal precision. That is exactly the limit (≤ +0.002) for "not needed". At full precision it is +0.0019 (per seed
  +0.0021, +0.0018, +0.0019). The class follows the rule at both precisions. Still, the higher opacity learning rate is
  not needed for speed, and leaving it out costs a small perceptual amount on garden.

## Opacity resets

- `creset` (the control without resets after the branch) is not faster. On garden it is slower in every seed, and it
  never reaches the target in seed 30. On room it is faster in one seed, slower in one, and never reaches the target in
  one.
- It ends with more Gaussians (5.13 M against 4.18 M on garden; 1.88 M against 1.34 M on room), and roughly equal
  quality to `ctl`.
- The control's gap to `mcmc` is therefore not a by-product of its periodic resets.
- Limit: this is resets removed from the control, not the role of resets inside MCMC (which uses none).

## The two clocks

Callback time from the branch to 30,000 (densify, Morton ordering, opacity resets):

| arm | garden | room |
|---|---|---|
| `ctl` | 1.72–1.78 s | 0.69–0.70 s |
| `creset` | 1.93–1.99 s | 0.81–0.83 s |
| MCMC arms | 0.31–0.35 s | 0.18–0.20 s |
| `mreloc` | 0.07 s | 0.04 s |

- This is under 1% of the training-step time to 30,000 (about 200–260 s on garden and 110–160 s on room).
- Including it changes A by about 1 s, and no classification.
- **For Experiment 1R:** the callbacks its clock left out were small, and larger in `ctl` than in `mcmc`. Including them
  would have slightly increased, not reduced, `mcmc`'s advantage. 1R's percentages remain training-step figures.
- The secondary clock is still not complete. SH-degree steps and dispatch overhead are outside both clocks.

## Other observations (descriptive)

- **Check 1e** (reported, not a gate), `ctl` − uninterrupted reference:
  - room seed 30: −0.213 dB, 3.2 times the pilot's same-seed SD, the largest in Experiments 1R and 1A;
  - garden: −0.057 to +0.018 dB.

  1a–1d pass for every seed: identical views, identical loaded state, evaluation neutral, configuration as intended.
  Across 1R and 1A (six seeds per scene) the mean `ctl` − reference is −0.022 dB on garden and +0.004 dB on room.
- **One-view failures** cluster on room views 34–36 in every arm except `mcmc`. That includes `mrep` (view 36, −8.1 dB on
  seed 32), an identical rerun of `mcmc`. These views are unstable under GPU non-determinism alone, as the noise pilot
  already suggested for view 35.
- **Noise and relocation interact.** In the switch smoke test, removing noise cut relocations about sixfold. Both are
  "needed" here, and leave-one-out cannot separate how much of each one's effect runs through the other.

## Limits

- Two scenes, three seeds each, one GPU type, one build (`FasterGSUp`, dense Adam), branch at 5,000.
- Leave-one-out tests necessity given the other components. It does not test sufficiency, or interactions beyond pairs.
- T is training-step time to `ctl`'s final PSNR − 0.05 dB. The callback-inclusive clock agrees.
- Experiment 1A was designed after Experiment 1R's result was seen. Its plan was revised twice after review, before
  launch, as recorded in the plan.

## Possible next steps (not decided; each needs its own pre-registration)

- **Sufficiency and interaction:** add relocation, noise or regularisation alone, or in pairs, to the fixed-size
  policy. This would show which minimal combination keeps the quality and the speed.
- **Generality:** more scenes, before any claim beyond garden and room.
- **Opacity learning rate:** drop it from the policy, given the small garden LPIPS cost.
