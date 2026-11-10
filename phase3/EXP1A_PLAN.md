# Experiment 1A: which parts of the MCMC policy make it faster (pre-registered 2026-11-10)

**Status: FINAL PRE-REGISTRATION, frozen 2026-11-10 at launch, in the commit that adds this line.**
- All engineering gates passed: jobs 4561006, 4561007, 4561028 and 4561041, and 14 analysis test scenarios. The open
  questions were settled by review. The user gave the go-ahead.
- Binding:
  - the design (eight arms, seeds 30–32);
  - the measures and the two clocks;
  - the decision rule (steps 1–4), with the revisions of 2026-11-10;
  - the checks before the decision, including the callback-record requirement;
  - the reporting limits.
- The analysis command, fixed now:
  `python phase3/analyze_ttq.py <the three garden logs> <the three room logs> --abl --seeds 30 31 32`. A (scene, seed)
  block rerun under the timeout rule replaces its original through `--rerun`.
- Any change from here on is listed under "Changes after launch", with its date and reason.

**History:** DRAFT until launch. Revised twice after review on 2026-11-10; see the dated notes throughout.

## Question

Experiment 1R ([EXP1R_RESULTS.md](EXP1R_RESULTS.md)) found that the combined MCMC policy, switched on at step 5,000,
reaches the control's quality sooner on garden and room:
- median T 31% and 36% lower;
- 6 of 6 pairs, at every sensitivity target;
- final PSNR 0.19 and 0.30 dB higher, with about 27% fewer Gaussians.

That arm changed five things at once. **Which of them are needed for the speed-up?**

| # | component | `ctl` | `mcmc` | switched by |
|---|---|---|---|---|
| R | population rule | grow by gradient (clone / split) and prune, every 100 steps until 14,900 | fixed N; every 100 steps until 14,900 (the same callback window), Gaussians below opacity 0.005 are relocated onto living ones, sampled by opacity | `USE_MCMC` |
| Z | noise injection | none | `add_noise` on positions after every optimizer step, scaled by the position learning rate | `USE_MCMC` |
| O | opacity resets | at 6,000, 9,000 and 12,000 | none | `USE_MCMC` |
| G | regularisation | opacity 0, scale 0 | opacity 0.01, scale 0.01 | config |
| L | opacity learning rate | 0.025 | 0.05 | config |

Arm names contain only lower-case letters, to fit the analysis's log pattern. (Read from `Methods/FasterGSUp/Trainer.py` and `Model.py`, 2026-11-10. With `MAX_PRIMITIVES` = N at the branch,
`mcmc_densification` adds no Gaussians, so R is relocation only.)

**Not part of the question:**
- the oracle;
- the population size itself (Experiment 1R's oracle has `ctl`'s size and ends higher still);
- combining any of this with Experiment 2's schedule levers.

## What stays as in Experiment 1R

- **Build, clock and evaluation:**
  - build `FasterGSUp`, dense Adam;
  - resume patch v7, with evaluation observational and `FGS_EVAL_CHECK` in every branched run;
  - test metrics every 1,000 steps from 5,000;
  - `cudnn.benchmark` on, as in 1R.
- **Scenes and branch:** garden 1/4 and room 1/2, with a shared prefix 0 → 5,000 per (scene, seed).
- **Run length:** 30,000 steps.
- **Measures:** T to the target (`ctl`'s final PSNR − 0.05 dB, training-only time, interpolated), the sensitivity
  targets, final ΔPSNR and ΔLPIPS, and one-view failures.
- **Check 1**, exactly as in 1R:
  - 1a–1d are the gates;
  - 1e is reported only;
  - an uninterrupted reference per seed, whose position alternates by seed.

## Design

- **Seeds:** fresh, 30, 31, 32. Seeds 0–23 have all been used before.
- **Arms**, all branching from the same saved state at 5,000:

| arm | what it is | R | Z | O | G | L |
|---|---|---|---|---|---|---|
| `ctl` | standard Faster-GS | grow / prune | – | resets | 0 | 0.025 |
| `mcmc` | 1R's MCMC arm, unchanged | relocate | on | none | 0.01 | 0.05 |
| `mrep` | an identical second `mcmc` run, same seed | relocate | on | none | 0.01 | 0.05 |
| `mreloc` | `mcmc` without relocation: fixed N, dead Gaussians stay | **none** | on | none | 0.01 | 0.05 |
| `mnoise` | `mcmc` without noise injection | relocate | **–** | none | 0.01 | 0.05 |
| `mreg` | `mcmc` without regularisation | relocate | on | none | **0** | 0.05 |
| `mlr` | `mcmc` with `ctl`'s opacity learning rate | relocate | on | none | 0.01 | **0.025** |
| `creset` | `ctl` without opacity resets after the branch | grow / prune | – | **none** | 0 | 0.025 |

- **Leave-one-out arms** (`m*`): each removes one component from the full policy. They test whether a component is
  **needed** given the others. They do not test whether a component is enough on its own.
- **`creset`** asks one specific question. `ctl`'s curve drops at each opacity reset, which could by itself explain
  part of the gap. This arm removes only the resets from `ctl`.
- **`mrep`** measures the same-seed noise of T in this experiment. The noise pilot measured final PSNR, not T.
- **Order:** the arms run in an order rotated by seed. The reference `up` runs first for even seeds and last for odd
  seeds, as in 1R.
- **Runs:** 2 scenes × 3 seeds × (prefix + 8 arms + reference) = 60.

## Measures for the question

Per pair (scene, seed), with T from the primary target:

- **Advantage:** A = T(`ctl`) − T(`mcmc`).
- **Fraction lost when component k is removed:** f_k = (T(`m<k>`) − T(`mcmc`)) / A. An arm that never reaches the target
  counts as f = +∞, which is at least 1.
- **Fraction recovered by removing the resets:** g = (T(`ctl`) − T(`creset`)) / A.
- **Noise reference:** f_rep = (T(`mrep`) − T(`mcmc`)) / A. It is 0 in expectation.
- **Quality:** each `m*` arm's final PSNR and LPIPS minus `mcmc`'s, and `creset`'s minus `ctl`'s.

- **Pairs used for classification:** a pair is used only if A ≥ 10% of T(`ctl`). That is a positive advantage of at
  least the size the replication rule asks for. Per-scene values are always reported alongside.

**Two clocks (revised 2026-11-10):**
- **The training clock (`SEGCLOCK`)** covers only the training step itself. The densify, Morton-ordering and
  opacity-reset callbacks run between steps, outside it. This was true for Experiment 1R too
  ([EXP1R_RESULTS.md](EXP1R_RESULTS.md), disclosed there). The primary T stays on this clock, for comparability with 1R.
- **Secondary clock, "training step + three instrumented callbacks":** the training clock plus the measured time of
  the densify, Morton-ordering and opacity-reset callbacks. It comes from patch v2's `FGS_ABL_CB`: CUDA events,
  cumulative before each evaluation point, with the prefix's included.
  - Every fraction and classification is also computed on this clock.
  - It is not a complete training-time clock. Other work between steps (SH-degree increase, loop and callback
    dispatch overhead) is outside both clocks.
- **Callback records are an instrumentation requirement** (added 2026-11-10). Every arm and every prefix must have
  exactly one valid `FGS_ABL_CB` record: cumulative times non-negative and non-decreasing, ending at the total.
  A missing or inconsistent record is a validation failure, so the decision is not evaluated. The secondary clock is
  never dropped silently.

## Decision rule (fixed before launch)

**Step 1. Replication.** `mcmc` must pass Experiment 1R's decision rule against `ctl` in this experiment:
- median ΔT ≥ 10% on both scenes;
- T lower in ≥ 5 of 6 pairs;
- both quality limits;
- no more one-view failures than `ctl`, on the control-based reference below.

**One-view failures (revised 2026-11-10).**
- Image i of a run is a failure if its final PSNR is more than 3 dB below the median of image i over the scene's `ctl`
  runs and uninterrupted reference runs (all seeds), excluding the run itself.
- That reference does not depend on which ablation arms are present. Experiment 1R's definition (the median over every
  run of the scene) would change with the eight arms.
- The 1R-style counts are printed for comparison only. They are not described as a reproduction of 1R's criterion.

If it does not pass, nothing below is interpreted, and the result is reported as a failure to replicate 1R.

**Step 2. Resolution (revised 2026-11-10).** Classifications are made only if all three of these hold. Otherwise every
component is **not resolved**, and the fractions are reported descriptively with the reason.
- At least 5 pairs are used, each with A ≥ 10% of T(`ctl`). So `mcmc` may replicate with 5 of 6 wins and still be
  decomposed, over those 5 pairs.
- `mrep` reaches the target in every used pair. A replicate that never reaches it fails this step, whatever the median
  of the others.
- The median of |f_rep| over the used pairs is ≤ 0.2.

**Step 3. Each component R, Z, G, L**, over the n used pairs:
- **needed:** median f_k ≥ 0.5, and f_k > 0 in at least n − 1 pairs;
- **not needed:** median f_k ≤ 0.2, and on both scenes the arm's mean final ΔPSNR against `mcmc` ≥ −0.10 dB and mean
  ΔLPIPS ≤ +0.002;
- **which pairs each part uses** (stated 2026-11-10, as implemented):
  - the time-to-quality part (f_k, g and their pair counts) uses only the n used pairs;
  - the quality part uses all three seeds of each scene, including any pair excluded from the time part. That is the
    more conservative choice.
- **not resolved:** anything else.

**Step 4. Opacity resets**, over the same pairs:
- **resets explain a major part of the gap:** median g ≥ 0.5, and g > 0 in at least n − 1 pairs;
- **resets explain little:** median g ≤ 0.2;
- **not resolved:** anything else.

**Always reported:**
- every f_k, g and f_rep per pair;
- the same quantities at the −0.10 and −0.20 dB targets, and for "reaches and stays";
- the quality differences (PSNR and LPIPS);
- one-view failures per arm;
- callback time per arm;
- each arm's N over time.

**How to read the result:**
- The leave-one-out fractions need not add up to 1. Components can interact, and several can each be "needed".
- If an arm's classification at the primary target disagrees with its classification at the sensitivity targets,
  that component is described as sensitive to target selection.
- If it disagrees with its classification on the secondary clock, it is described as sensitive to the clock boundary.
- A "needed" component is needed given the rest of this policy, on these two scenes and three seeds. It is not shown
  to work alone, and it is not called necessary in general.
- `creset` measures what removing resets from the control policy does. It is not the contribution of resets inside
  MCMC.

## Checks before the decision

- **Validation and check 1 (1a–1d), as in 1R.** Check 1d now also covers the new switches:
  - each arm's switch values, printed on its `FGS_ABL` line at the end of training, must be the ones in the table above;
  - each arm's configuration may differ from `ctl`'s only in its own keys.
- **Check 2: each arm did what it says.** Read from new counters, printed once at the end of training:

| arm | requirement |
|---|---|
| `mcmc`, `mrep`, `mnoise`, `mreg`, `mlr` | relocations > 0; N held at N at the branch |
| `mreloc` | relocations = 0; N held |
| `mnoise` | noise steps = 0 |
| every other MCMC arm | noise steps = 25,000 |
| `ctl` | 3 opacity resets after the branch (6,000, 9,000, 12,000) |
| `creset` and all MCMC arms | 0 opacity resets |
| `ctl` and `creset` | N grows after the branch (as `ctl` in 1R) |

## Before launch: engineering and smoke tests

**Trainer patch** (`cluster/patches/fastergs_ablate.py`, new; v7 resume patch unchanged). It adds three switches, read
from the environment. Their defaults reproduce the current behaviour:

| switch | default | effect when 0 |
|---|---|---|
| `FGS_MCMC_RELOCATE` | 1 | MCMC densification skips relocation; with the cap at N it then does nothing |
| `FGS_MCMC_NOISE` | 1 | `post_optimizer_step` gets `inject_noise=False` |
| `FGS_OPACITY_RESET` | 1 | `reset_opacities` does nothing; the extra reset at 500 is untouched, as it is before the branch |

- It also adds counters, printed with `FGS_ABL=1` on one `FGS_ABL` line at the end of training: the switch values,
  relocation events, Gaussians relocated, dead Gaussians left in place, noise steps, and the iterations of the opacity
  resets. The v7 resume patch is left unchanged.
- Counting dead Gaussians costs one device sync per densification step. It is done in every MCMC arm alike.
  *Corrected 2026-11-10, after review:* an earlier version of this draft said this time is counted as training. It is
  not. It falls in the densify callback, outside the training clock (see "Two clocks"). Smoke test `x1atime` checks
  that counting does not change the training clock.
- **v2 (2026-11-10):** with `FGS_ABL=1`, the densify, Morton-ordering and opacity-reset callbacks are timed with CUDA
  events, adding no synchronisation.
  - Output: one `FGS_ABL_CB` line with their total time by callback, and the cumulative time before every 1,000-step
    point.
  - The prefix runs with `FGS_ABL=1` too, so its callback time before the branch is known.
  - Upgrading from v1 means restoring the trainer saved before v1 and applying v2. The patch refuses to run on top of
    v1.
- The regularisation and opacity learning rate need no patch: they are set through `CFG_SET`, as in 1R.

**Smoke tests** (garden, short: branch at 2,000, end at 3,000, plus one long-enough check for the resets):

| test | requirement | type |
|---|---|---|
| Defaults unchanged | With all switches unset, `ctl` and `mcmc` pass 1R's checks 1a–1d and their counters match the table above, on the same saved state | gate |
| Each switch does only its own thing | With one switch at 0, only that counter changes (relocations, noise steps or resets), and 1b–1c still pass | gate |
| Resets | A `ctl` and a `creset` run from 5,000 to 13,000: `ctl` logs resets at 6,000, 9,000 and 12,000, `creset` none | gate |
| Analysis | `analyze_ttq.py` (or a new `analyze_abl.py`) computes f_k, g and f_rep and the classifications. Fixture tests with known answers for each class, a non-replicating `mcmc`, an unresolvable `mrep`, and a "not reached" arm | gate |
| Stage smoke | The new stage end to end on a short garden run, both run orders, through the analysis's checks | gate |
| Counting vs clock (`x1atime`, added 2026-11-10) | `ctl` and `mcmc` from the same state, 5,000 → 8,000, no evaluation, counting on and off, two runs each. Per arm, the mean training time with counting is within 2% of the mean without. Wall time and callback time reported. | gate |
| Stage smoke on v2 | The stage smoke repeated with patch v2. Every arm and both prefixes produce valid `FGS_ABL_CB` records (the validation check above), and the analysis runs end to end on both clocks | gate |

**How to read `x1atime`.** The counting sync runs inside the densify callback, outside the training clock. So the 2% gate
shows only that the training-step clock is unchanged. Any extra cost of counting shows up in wall time and callback
time, which are reported separately and are not gated. It is the same in every MCMC arm.

**Built 2026-11-10:**
- `cluster/patches/fastergs_ablate.py`: tested on a copy of the deployed trainer. It is idempotent and changes only
  the four places above.
- Stages `x1asmoke` and `exp1a` in `phase2/build_plus.sbatch`, and `phase3/check_x1a.py`.
- `analyze_ttq.py --abl` and `phase3/tests/exp1a_fixture.py`. Seven scenarios each give their intended answer:
  - the classes needed, not needed and not resolved;
  - resets explaining little and resets explaining a major part;
  - a non-replicating `mcmc`;
  - an unresolvable `mrep`;
  - an arm that never reaches the target (counted as f = ∞);
  - a wrong noise count, and a wrong switch: both stop the decision.

  The Experiment 1 and 1R analyses give byte-identical output before and after the change.
- **Revised 2026-11-10, after review of 6b67dbe** (patch v2 and analysis): thirteen fixture scenarios, adding:
  - `mcmc` winning 5 of 6 pairs, decomposed over the 5 used pairs;
  - `mrep` never reaching the target, which is not resolved;
  - an arm with worse LPIPS, which is then not classified as "not needed";
  - a one-view failure on the control-based reference, which fails replication;
  - a component whose class changes with the clock, which is flagged;
  - logs without callback timing: *revised the same day after the second review*. This is now a validation failure
    (instrumentation), not a dropped clock;
  - a callback record whose cumulative time decreases, which is also a validation failure (14 scenarios in all).
- `x1asmoke` uses Experiment 1R's saved garden seed-10 state. Gate "defaults unchanged": `ctl` with no switch set sees
  exactly 1R's `ctl` views over 5,000–12,999.

**Smoke-test results (2026-11-10, patch v1, repo 6b67dbe):**
- **Switch smoke, job 4561006** (`x1asmoke`, 25 min): all 74 checks pass.

  | arm | relocations (Gaussians) | noise steps | resets | N at 7,000 / 13,000 |
  |---|---|---|---|---|
  | `ctl` | 0 | 0 | 6,000, 9,000, 12,000 | 4,089,627 |
  | `creset` | 0 | 0 | none | 4,953,550 |
  | `mcmc` | 20 events (680,932) | 2,000 | none | 3,080,861 (held) |
  | `mrep` | 20 (680,180) | 2,000 | none | held |
  | `mreloc` | 0 (307,656 dead left in place) | 2,000 | none | held |
  | `mnoise` | 20 (115,086) | 0 | none | held |
  | `mreg` | 20 (249,469) | 2,000 | none | held |
  | `mlr` | 20 (194,396) | 2,000 | none | held |

  - **Defaults unchanged:** `ctl` with no switch set saw exactly Experiment 1R's `ctl` views over 5,000–12,999. Every
    arm loaded exactly the saved state. Every evaluation restored the random states, model and optimizer state, flag and
    mode.
  - Diagnostics: `mcmc`'s views equal 1R's `mcmc` over 5,000–6,999, and `mrep`'s equal `mcmc`'s. `creset`'s views equal
    `ctl`'s over 5,000–12,999.
  - Descriptive, not a finding: removing noise cuts relocations about sixfold (115 k against 681 k in 2,000 steps).
    Noise and relocation interact.
- **Stage smoke, job 4561007** (`exp1a`, garden, seeds 0 and 1, branch 2,000, end 3,000; 28 min): validation, check 2
  (counters, N held, resets, noise) and checks 1a–1d pass on real output, in both run orders. 1e: `ctl` − reference
  +0.028 and −0.045 dB.
- **Found while deploying v2:** the `exp1a` stage did not set `FGS_ABL=1` on the prefix, so the callback clock would
  have lacked the prefix's callback time. Fixed before any v2 run. The v2 stage smoke (job 4561029) was cancelled after
  10 s and is resubmitted with the fix.

**Counting vs clock, job 4561028** (`x1atime`, patch v2, repo 9b69665, 11 min): **PASS.**
- Garden seed 10, 5,000 → 8,000, from 1R's saved state. Training time is the training-step clock; wall time is
  `FGS_WALL`, which includes callbacks.

  | arm | training, counting off (2 runs) | training, counting on (2 runs) | change | wall, off / on | callbacks with counting on |
  |---|---|---|---|---|---|
  | `ctl` | 22.725, 22.700 s | 22.473, 22.751 s | −0.44% | 23.42, 23.38 / 23.18, 23.44 s | 0.54 s (densify 0.53, Morton 0.016) |
  | `mcmc` | 21.205, 21.255 s | 21.133, 21.255 s | −0.17% | 21.51, 21.57 / 21.45, 21.57 s | 0.12 s (densify 0.10, Morton 0.019) |

- The training-step clock is unchanged by counting, well within 2%.
- Wall time does not increase measurably either. The two conditions overlap.
- Descriptive: over these 3,000 steps the three callbacks cost about 2.4% of `ctl`'s training-step time and 0.6% of
  `mcmc`'s. Over 1R's longer runs the share would differ, since `ctl`'s densification runs until 14,900 with a growing
  N. The secondary clock measures it.

**Stage smoke on v2, job 4561041** (`exp1a`, garden, seeds 0 and 1, branch 2,000, end 3,000; repo 5899801; 28 min):
**PASS.**
- 18 of 18 `FGS_ABL_CB` records, one per arm and per prefix: 8 arms + prefix × 2 seeds. All valid: non-negative,
  non-decreasing, ending at the total.
- Validation, including the callback-record check and check 2, passes. Checks 1a–1d pass.
- The analysis runs end to end on both clocks. Output: [x1a-stagesmoke2-analysis.md](x1a-stagesmoke2-analysis.md),
  with `--descriptive`. The decision lines there mean nothing at 1,000 steps after the branch; they only exercise the
  code path.
- Callback time over the 1,000 steps after the branch: `ctl` and `creset` 0.08–0.09 s; MCMC arms 0.01–0.03 s; `mreloc`
  lowest, at 0.01 s.
- Fix made with it: the callback-time table now shows two decimals instead of one.

**All engineering gates pass (2026-11-10).** The plan is ready to freeze. It is frozen at launch, on the user's
go-ahead, in the commit that changes the status line.

## Cost

Estimates scaled from Experiment 1R (garden about 12 minutes per run, room about 18 minutes, mostly the 25
evaluations):
- **Garden:** one job per seed, 11 runs each (reference, prefix, 8 arms), about 2.2 h.
- **Room:** one job per seed, about 3.4 h.
- **In all:** about 17 GPU-hours, plus about 1 GPU-hour of smoke tests.
- **Time limits:** 3:30 for garden jobs and 4:30 for room jobs.
- **Timeout rule, as in 1R:** a (scene, seed) block that does not complete is rerun whole.

**Evaluation dominates the cost.** On room, elapsed time from 5,000 to 30,000 is about 1,440 s, against 145 s of
training. Saving the test images is about three quarters of each evaluation, but it cannot simply be skipped: NeRFICG
computes the metrics from the saved 8-bit images. Making evaluation cheaper would change the metric path. That is
not done here, so the measurements stay comparable with 1R.

## How the result will be reported

- Experiment 1A is designed after Experiment 1R's result was seen. Its question comes from that result. It uses fresh
  seeds.
- Claims are limited to: which components of this MCMC policy are needed for its speed-up on garden and room (seeds
  30–32, H100), given the other components.
- If `mcmc` does not replicate, that is the headline, and no component is classified.

## Out of scope

- Add-one arms: each component added alone to `ctl`. They would test sufficiency, not necessity (open question 2).
- Other population sizes, and the oracle.
- Experiment 2's schedule levers. [EXP2_PLAN.md](EXP2_PLAN.md) stays frozen as a draft.

## Decisions after review (2026-11-10)

These were made on the review of 6b67dbe, before any Experiment 1A arm ran:
- **Open question 1:** no `mcmc` + resets arm; `creset` stays as it is.
- **Open question 2:** no add-one arms for now.
- **Open question 3:** the 0.5 / 0.2 thresholds are kept, as operational thresholds.
- **Open question 4:** three fresh seeds are acceptable for a diagnostic ablation.
- **Open question 5:** `cudnn.benchmark` stays on.
- No further arms.

## Open questions for review (as first drafted)

1. **The reset component inside MCMC.** `mcmc` + resets would complete the leave-one-out set. It is left out because a
   reset clamps every opacity to 0.01, just above the 0.005 relocation threshold (`Model.reset_opacities`), which would mix two components in one arm. `creset` asks the
   reset question from the `ctl` side instead. Default: not included.
2. **Add-one arms for G and L.** These are cheap: `ctl` with MCMC's regularisation, and `ctl` with opacity learning rate
   0.05, both through `CFG_SET`. They would show whether either alone speeds up `ctl`. Default: not included, to keep
   the question to necessity; about 2.4 GPU-hours if added.
3. **Thresholds.** 0.5 for "needed", 0.2 for "not needed" and for the noise check. Set from what would count as a large
   or small share of the effect, before any run. `mrep` decides whether they can be resolved.
4. **Three seeds.** Same as 1R. With the noise check in step 2, too few seeds shows up as "not resolved", not as a
   false classification.
5. **`cudnn.benchmark` during evaluation.** Turning it off would cut evaluation memory from about 71 to 9.5 GiB, with
   metrics unchanged to about 5e-7 dB (job 4559014). Default: keep it on, for comparability with 1R.

## Changes after launch

- 2026-11-10, after the freeze (026f6d1) and before any job was submitted: with `--rerun`, the analysis now takes the
  uninterrupted reference runs from the rerun logs too, not only the arm and prefix blocks. Without this, a rerun
  block would have been checked against its original reference. Only the rerun path changes, and no result had been
  seen. Experiment 1R's analysis and the fixture outcomes are unchanged.
