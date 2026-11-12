# Experiment 3: where MCMC's time-to-quality advantage comes from (DRAFT, 2026-11-12)

"E1" in [LIT_REVIEW.md](LIT_REVIEW.md).

**Status: DRAFT. Not pre-registered, and nothing in it has run.**
- It becomes final (this line changed, committed) only after:
  - the engineering in "Before launch" passes its smoke tests;
  - the open questions are settled by review;
  - the user gives the go-ahead.
- It launches only after that commit. Any change after it goes under "Changes after launch", with date and reason.
- Experiment 2 stays frozen. Nothing here touches the cluster checkout while Track R is active.

## Question

Experiments 1R and 1A showed that switching Faster-GS to the MCMC policy at step 5,000 reaches the control's quality
29–42% sooner on garden and room (12 of 12 pairs, training-step clock). It also ends higher, with about 27% fewer
Gaussians. The literature reports MCMC as both faster and smaller, but never separates the two
([LIT_REVIEW.md](LIT_REVIEW.md), rows 2 and 9, second pass).

**Q1 (generality):** does the advantage hold across all nine Mip-NeRF 360 scenes?

**Q2 (source):** when it holds, does it come mainly from
- **cheaper steps:** fewer Gaussians, so less time per step; or
- **fewer steps:** the relocation, noise and regularisation policy reaching the target in fewer optimisation steps?

**Q3 (supporting):** does it depend on the population size the policy is given, and on switching in mid-training
rather than using MCMC from step 0?

Q3 compares only "switch at 5,000" with "from step 0". The switch-time sweep is E2, a separate plan.

## The decomposition (exact, per pair)

For one arm and one (scene, seed):
- **T′** = training-step time from the branch (step 5,000) to the target, the same target and interpolation as in 1R
  and 1A;
- **S′** = steps from the branch to the target, interpolated in the same way, in step space;
- **c** = T′ / S′, the mean time per step over that stretch.

So T′ = S′ × c exactly, and for `mcmc` against `ctl`:

ln(T′_ctl / T′_mcmc) = ln(S′_ctl / S′_mcmc) + ln(c_ctl / c_mcmc)

- The **step share** is the first term divided by the left side. The **cost share** is the second term divided by the
  left side. They add up to 1.
- **The cost share is "fewer Gaussians".** Time per step on this build scales with N and its visibility; per-step cost
  is logged, and so is mean N over the stretch.
- **The step share is "a better policy".**

Using post-branch quantities removes the shared prefix (0–5,000), which is identical for every arm of a pair and would
dilute both shares.

## Design

- **Build and protocol, as in 1R and 1A:**
  - build `FasterGSUp`, dense Adam;
  - resume patch v7 (evaluation observational, `FGS_EVAL_CHECK` in every branched run);
  - ablation patch v2 with `FGS_ABL=1` in every run;
  - test metrics every 1,000 steps from 5,000;
  - view traces;
  - an uninterrupted reference `up` per (scene, seed), first for even seeds and last for odd;
  - `cudnn.benchmark` on.
- **Scenes:** all nine Mip-NeRF 360 scenes:
  - outdoor at 1/4: bicycle, flowers, garden, stump, treehill;
  - indoor at 1/2: room, counter, kitchen, bonsai.

  All have been trained on the cluster before (`experiment_log.csv`).
- **Seeds:** fresh, 40, 41, 42 (seeds 0–32 have been used).
- **Arms**, all except `mcmc0` branching at 5,000 from the shared prefix:

| arm | what | final N | question |
|---|---|---|---|
| `ctl` | standard Faster-GS (ADC) | grows to about 1–5 M | reference |
| `mcmc` | MCMC switched on at 5,000, cap = N at 5,000; exactly 1R's arm | N(5k), held | Q1, Q2 |
| `oracle` | MCMC switched on at 5,000, cap = `ctl`'s final N, same seed (runs after `ctl`); 1R's oracle | grows to `ctl`'s count | Q2/Q3: the policy at the control's count |
| `stop` | ADC with densification stopped at 5,000 (no growth, pruning or resets after it); 1R's `stop` | N(5k), held | Q2: the small count without the MCMC policy |
| `mcmc0` | MCMC from step 0 (MCMC initialisation, growth of 5% per densification step to the cap), cap = the same seed's N at 5,000 | N(5k) | Q3: switch at 5,000 against from the start, at the same final count |

- **How the arms answer Q2**, at matched counts:
  - `oracle` against `ctl`: same final count, different policy;
  - `mcmc` against `stop`: same small count, different policy;
  - `mcmc` against `oracle`: same policy, different count.

  These cross-check the decomposition.
- `mcmc0`'s evaluation also starts at 5,000, so its T′ and S′ are measured on the same stretch.
- **Runs:** 9 scenes × 3 seeds × 7 runs (`up`, prefix, 5 arms) = 189. One job per (scene, seed).

## Measures

- **Primary** (training-step clock, target = `ctl`'s final PSNR − 0.05 dB):
  - T′, S′ and c per arm;
  - for `mcmc` against `ctl`: ΔT′ (%), the step share and the cost share.
- **Secondary:**
  - the same on the callback-inclusive clock;
  - **training wall time**: elapsed `FGS_WALL` 0 → 30,000 minus the measured evaluation time (new, see
    "Before launch"). The literature compares on wall time.
- **Quality:** final PSNR, SSIM, LPIPS; one-view failures on the control-based reference (1A's definition: the
  scene's `ctl` and `up` runs, excluding the run itself).
- **Population:** N at every 1,000 steps; mean N over the branch-to-target stretch.
- **Sensitivity:** targets −0.10 and −0.20 dB, and "reaches and stays".

## Decision rule (fixed before launch)

**Step 1, Q1, generality.** `mcmc` generalises if, on at least 7 of the 9 scenes, it meets 1R's per-scene criteria:
- median ΔT′ ≥ 10%;
- mean ΔPSNR ≥ −0.10 dB;
- mean ΔLPIPS ≤ +0.002.

And across all 27 pairs:
- T′ is lower than `ctl`'s in at least 20 (one-sided sign test, p < 0.01);
- no more one-view failures than `ctl`.

Scenes that miss are named, with their numbers.

**Step 2, Q2, decomposition.** Only over pairs where `mcmc`'s T′ is at least 10% below `ctl`'s (n used pairs; at least
15 needed, otherwise "not resolved"):
- **mainly fewer steps (policy):** median step share ≥ 0.5, and S′_mcmc < S′_ctl in at least 0.75·n pairs;
- **mainly cheaper steps (count):** median cost share ≥ 0.5, and c_mcmc < c_ctl in at least 0.75·n pairs;
- **both:** each share's median between 0.25 and 0.75, and both inequalities hold in at least 0.75·n pairs;
- **not resolved:** anything else.

**Cross-checks**, reported with the step-2 class and expected to agree with it:
- policy at the control's count: `oracle` against `ctl`, as ΔT′, ΔS′ and quality;
- policy at the small count: `mcmc` against `stop`;
- count within the policy: `mcmc` against `oracle`.

If the cross-checks contradict the class, for example "mainly cheaper steps" while `oracle` beats `ctl` by as much as
`mcmc` does, the class is reported as **inconsistent**, with both readings.

**Step 3, Q3.** `mcmc0` against `mcmc` at the same final count:
- "switch at 5,000 better": median ΔT′ ≥ 10% in `mcmc`'s favour, and lower in at least 20 of 27 pairs;
- "from the start better": the same, the other way;
- otherwise "no material difference".

Quality is reported alongside.

**Always reported:**
- per-pair and per-scene tables of every quantity;
- both clocks and wall time;
- the sensitivity targets. A class that changes with the target, or with the clock, is labelled sensitive.

**Reading:** the conclusions are limited to:
- Mip-NeRF 360;
- this build (Faster-GS with upstream kernels, dense Adam);
- an H100;
- MCMC's densification window as Faster-GS configures it (until 14,900; see open question 4).

## Checks before the decision

As in 1A, applied to every (scene, seed):
- **Validation:** runs complete; identical metrics at the branch.
- **Check 1a–1d:** exact views for `ctl`; the exact state loaded by every branched arm; evaluation neutral;
  configurations and settings in effect.
- **Check 2, from the counters:**
  - `mcmc`: N held, relocations > 0, noise steps = 25,000;
  - `oracle`: N reaches the cap;
  - `stop`: N constant after 5,100, no resets;
  - `ctl`: resets at 6,000, 9,000, 12,000;
  - `mcmc0`: N reaches and holds the cap, relocations > 0, noise steps = 30,000.
- **Callback records:** valid in every run and prefix.
- **Evaluation time:** one record per evaluation.

Any failure means the decision is not evaluated until the cause is found. A (scene, seed) block that does not
complete is rerun whole.

## Before launch: engineering and smoke tests

| item | what | gate |
|---|---|---|
| Evaluation time | Resume patch v8: with `FGS_EVAL_TIME=1`, `fgs_eval` synchronises at its start and end and prints `FGS_EVALTIME step= s=`. It runs outside the training clock, so that clock is unchanged; the analysis subtracts the sum from the wall time. | Smoke: B gates rerun under v8 (random states, model state, views identical); the per-evaluation time is consistent with `FGS_WALL` |
| `mcmc0` arm | A from-scratch MCMC run, cap from the same seed's prefix log, evaluating from 5,000 | Smoke: N reaches and holds the cap; counters as in check 2; view trace recorded |
| S′, c and shares | `analyze_ttq.py --exp3`: steps to target, time per step, step and cost shares, the cross-checks, the step 1–3 rules, wall time | Fixture tests with known answers: pure step effect, pure cost effect, both, inconsistent, too few used pairs, a scene failing Q1 |
| New scenes | The `exp3` stage, short (branch 2,000, end 3,000), on one new outdoor and one new indoor scene (bicycle, counter) | Validation and checks 1a–1d pass on real output |
| Patch order | v7 → v8 changes the cluster's method directory | Applied only with the queue empty and Track R not running |

## Cost

- **Scaled from 1R and 1A:** about 12 min per run on garden-like scenes and 18 min on room-like ones, mostly the 25
  evaluations.
  - Outdoor: about 1.5 h per (scene, seed) job. Indoor: about 2.2 h.
  - Total: 15 outdoor jobs and 12 indoor jobs, **about 50 GPU-hours.** Bicycle (about 5 M Gaussians) may run longer.
- **Smoke tests:** about 1.5 GPU-hours.
- **Staging, to limit the cost of a design error:**
  - Stage A: bicycle, stump and counter, 3 seeds (about 15 GPU-hours).
  - Stage B: the other six scenes.
  - Both stages run regardless of Stage A's outcome, unless an engineering check fails. The decision uses all nine
    scenes. Stage A's numbers are not used to change the rule.
- **Time limits:** 3:00 for outdoor jobs and 4:00 for indoor jobs.

## How the result will be reported

- Experiment 3 is designed after 1R, 1A and the literature review. Its question comes from them. It uses fresh seeds.
- What is new is the decomposition and its scope (nine scenes, paired, two clocks plus wall time). Relocation, noise and
  regularisation, and MCMC's better quality, are prior work (3DGS-MCMC).
- Steps 1–3 are reported as classified, with the per-scene tables. No claim beyond the stated scope.

## Out of scope

- E2, the switch-time sweep (2k / 5k / 10k / 15k).
- An adaptive controller.
- Other codebases. A one-scene gsplat check is recommended separately, see the open questions.
- Experiment 2 (schedule levers).

## Open questions for review

1. **The simplified policy without the opacity learning-rate change** (1A: not needed for speed; small garden LPIPS
   cost). Add it as a sixth arm (+13% cost), or leave it out? Default: out.
2. **`mcmc0`'s cap:**
   - N at 5,000, the default: the same final count as `mcmc`, isolating "switch against start".
   - `ctl`'s final N would pair it with `oracle` instead. Both would add a seventh arm.
3. **Thresholds:**
   - 7 of 9 scenes for generality;
   - 20 of 27 pairs (sign test);
   - 0.75·n for consistency;
   - 0.5 and the 0.25–0.75 band for the shares;
   - at least 15 used pairs.
4. **MCMC's densification window.** Faster-GS ends relocation at 14,900, while its own comment and 3DGS-MCMC suggest
   about 24,900 when using MCMC. 1R and 1A used 14,900. Keep it, for continuity (default), or test 24,900 as a
   separate arm?
5. **A cross-codebase check first:** run `ctl` and `mcmc` on one scene in gsplat's default and MCMC strategies, timed
   the same way, before the 50 GPU-hours. It would show whether the effect is specific to Faster-GS.
6. **Three seeds per scene** (27 pairs) against two (18 pairs, about 35 GPU-hours).
