# Phase 3 Experiment 1 results (2026-11-08)

Plan: [PLAN.md](PLAN.md). Full analysis output: [exp1-analysis.md](exp1-analysis.md), from

```
python phase3/analyze_ttq.py phase2/logs/gs-x1-garden-4553942.out phase2/logs/gs-x1-room-4553943.out \
    --rerun phase2/logs/gs-x1-room-s1-4554356.out phase2/logs/gs-x1-room-s2-4554357.out \
    --ref phase2/logs/gs-up-garden-4552203.out phase2/logs/gs-up-room-4552204.out \
          phase2/logs/gs-up-garden-s2-4552280.out phase2/logs/gs-up-room-s2-4552281.out --descriptive
```

Jobs: garden 4553942 (all three seeds); room 4553943 (seed 0 used; it hit its time limit during seed 2). Room seeds 1
and 2 come in full from reruns 4554356 and 4554357, by the rule recorded in PLAN.md before those seeds were analysed.
All 38 runs are in `experiment_log.csv`.

## Status: the pre-registered decision is NOT EVALUATED (adjudication running, see the end)

The plan says: if check 1 fails, Experiment 1 is invalid until the cause is found. **Check 1 fails on room.** The decision
is therefore not made. Everything below the check is reported as description only.

### Validation and check 2: pass

All 24 continuations have every clock point and every test evaluation. All four arms of each (scene, seed) give
identical test metrics at the branch step. Each arm did what it says:

- `ctl` grows after the branch, as in the default schedule.
- `stop` grows once, at the branch step itself, then holds the same count from 6,000 to 30,000 in all 6 runs.
- `mcmc` holds its count from 5,000 in all 6 runs.
- `oracle` reaches its cap by 15,000 in all 6 runs.

One change was made to the analysis after seeing results, and it is disclosed here. Its `stop` check required no growth
from step 5,000 on. The smoke test had already shown that densification still runs once at the branch step itself, and
the smoke test was corrected for that. The analysis check was not. It now requires the count to be constant from 6,000 on,
matching the smoke test. With the old check, all 6 `stop` runs were flagged for that single branch-step growth.

### Check 1: garden passes, room fails

| scene | branched `ctl` PSNR | uninterrupted runs | allowed difference (their range) | difference | result |
|---|---|---|---|---|---|
| garden | 27.517 | 27.500 (27.469–27.528) | 0.059 | +0.017 | pass |
| room | 31.610 | 31.711 (31.679–31.736) | 0.057 | −0.101 | **fail** |

Final Gaussian counts match on both scenes (garden 4.156 M vs 4.159 M, room 1.299 M vs 1.297 M).

**What the room gap is made of (diagnostic, chosen after seeing the data; not grounds for a pass):**

- It sits almost entirely on two test views, images 34 and 35. These views fail now and then in every build: image 35 is
  2.1 dB lower in branched `ctl` than in the uninterrupted runs on average, image 34 0.8 dB lower.
- With image 35 left out, the gap is −0.051 dB against an allowed 0.048 dB. With images 34 and 35 left out it is
  −0.030 dB against 0.065 dB.
- The same upbench job also ran three uninterrupted `ups` runs, a build with identical gradients up to the order of
  atomic additions. Their image 35 ranges from 27.4 to 31.1 dB. Taking all six uninterrupted runs as the reference, the
  gap is −0.080 dB against a range of 0.294 dB.
- The smoke test already showed that a resumed run restores the saved state bit for bit (SHA-256) and replays the same
  training views. What no test has covered is whether **evaluating during training** changes the run. The uninterrupted
  reference runs never evaluate before the end, while every branched run evaluates 26 times.

So the gap is consistent with seed noise on two unstable views. It is also consistent with an effect of in-training
evaluation. Three seeds cannot separate the two, and the plan does not allow choosing after the fact.

## Description of the arms (not the decision)

Time to target T is training-only time from step 0 until test PSNR reaches `ctl`'s final value minus 0.05 dB. ΔT is the
reduction against `ctl` in the same seed.

| scene | arm | T per seed (s) | ΔT per seed | final ΔPSNR | final ΔLPIPS | final N | training time to 30k (s) |
|---|---|---|---|---|---|---|---|
| garden | `ctl` | 173 / 181 / 157 | | 27.517 dB | | 4.17 M | 233 |
| garden | `stop` | not reached ×3 | | −0.177 | +0.0083 | 3.14 M | 192 |
| garden | `mcmc` | 115 / 120 / 110 | 33% / 34% / 30% | +0.157 | −0.0043 | 3.09 M | 204 |
| garden | `oracle` | 113 / 122 / 113 | 35% / 33% / 28% | +0.307 | −0.0113 | 4.17 M | 252 |
| room | `ctl` | 122 / 117 / 126 | | 31.610 dB | | 1.35 M | 145 |
| room | `stop` | not reached / 111 / 107 | — / 5% / 16% | −0.153 | +0.0037 | 0.97 M | 122 |
| room | `mcmc` | 77 / 78 / 72 | 37% / 33% / 43% | +0.560 | −0.0113 | 0.95 M | 131 |
| room | `oracle` | 85 / 84 / 83 | 30% / 28% / 35% | +0.617 | −0.0157 | 1.35 M | 152 |

One-view failures at 30k: `ctl` 2, `stop` 4, `mcmc` 0, `oracle` 0, all on room.

Applying the rule anyway (`--descriptive`; **not the pre-registered decision**):

- **`mcmc` would pass every criterion.** It reaches the target 30–43% sooner in all 6 pairs, ends higher in PSNR and
  lower in LPIPS on both scenes, and has no one-view failures.
- **`stop` would fail every criterion.** It is cheaper per step but never catches up on garden, and ends 0.15–0.18 dB
  lower.
- **`oracle` would pass** (reference only). On garden it reaches the target no sooner than `mcmc`, despite 35% more
  Gaussians.

How large the `mcmc` margin is matters for reading the check 1 failure. Room's −0.10 dB `ctl` gap is a fifth of
`mcmc`'s +0.56 dB lead there. If evaluation during training does cost quality, it affects `mcmc` and `ctl` alike,
since both evaluate on the same schedule. That is an argument for why the arms might still be comparable. It is not a
substitute for the pre-registered check.

**Reading of `mcmc`, as noted in the plan:** it changes several things at once: relocation, noise injection,
regularisation and the opacity learning rate. A pass would show that this combined MCMC policy reaches quality sooner,
not that relocation alone does.

**Elapsed time including evaluation:** garden `ctl` takes 806 s from the branch to 30,000 with the 26 evaluations, against
233 s of training. Evaluation dominates the jobs, and in room it is about 90% of the elapsed time.

## Adjudication (pre-registered 2026-11-09, before any of it ran)

The user approved the proposal below on 2026-11-09. It is fixed here, with its implementation, before the jobs are
submitted. Nothing in it changes after results are seen; any change would be disclosed as one.

The plan's literal instruction is "the loader is fixed first". The loader is already verified bit for bit, so there is
nothing to fix there. The open question is whether evaluation during training changes the run.

1. **Evaluation-perturbation test.** Room seed 0, two uninterrupted runs from 0 to 8,000. One evaluates every 1,000
   steps from 5,000, as the branched runs do. The other does not evaluate. Both record the view trace.
   - **Pass:** identical view sequences over steps 0–7,999. View order is drawn from the CUDA generator, which
     densification also draws from, so any random draw made by evaluation would show up there.
   - Gaussian counts every 1,000 steps are printed as a diagnostic only.
2. **Extended check 1 on room.** Six more seeds, 3–8.
   - Each seed gets an uninterrupted `up` run, exactly as in upbench, and a prefix to 5,000 plus a branched `ctl` run,
     exactly as in `exp1`. There are no other arms. Within a seed the order alternates: `up` first on even seeds,
     `ctl` first on odd seeds.
   - Check 1 is then re-run with all 9 seeds per side (seeds 0–2 as already analysed, room seeds 1 and 2 from the
     reruns).
   - **Criterion, unchanged:** the mean difference must be within the uninterrupted runs' range (max − min), for final
     PSNR and for final N.
3. **Outcome:**
   - If both parts pass, the original decision rule is applied to the original 24 continuations, unchanged. It is
     reported with this adjudication disclosed.
   - If either part fails, Experiment 1 stays invalid, the cause is investigated, and the arms are not compared.

**Implementation:**
- Stage `x1adj` in `phase2/build_plus.sbatch`. Part 1 is checked inside the job, which prints a PASS/FAIL line.
- Part 2 is computed by `phase3/adjudicate.py`. It uses the same sources as `analyze_ttq.py`. Run on the original
  three seeds, it reproduces check 1 exactly: 31.610 vs 31.711, allowed 0.057, fail; N passes.

**Jobs:**
- `gs-x1adj-a`: part 1, then seeds 3–5.
- `gs-x1adj-b`: seeds 6–8.
- If a job hits its time limit, every incomplete seed is rerun whole under the same configuration (the rule already
  used for Experiment 1).

**Command, after the jobs:**

```
python phase3/adjudicate.py     --ctl phase2/logs/gs-x1-room-4553943.out phase2/logs/gs-x1-room-s1-4554356.out phase2/logs/gs-x1-room-s2-4554357.out           phase2/logs/gs-x1adj-a-<job>.out phase2/logs/gs-x1adj-b-<job>.out     --ref phase2/logs/gs-up-room-4552204.out phase2/logs/gs-up-room-s2-4552281.out           phase2/logs/gs-x1adj-a-<job>.out phase2/logs/gs-x1adj-b-<job>.out
```

**One procedural difference, disclosed in advance:** the original three `up` runs ran before the resume and evaluation
patch existed. The six new ones run with it applied but inactive, since all of its hooks are switched off unless their
settings are set. Part 1's no-evaluation run uses that same inactive path.

Cost: about 3.5 GPU-hours, mostly evaluating the six branched `ctl` runs (about 27 minutes each on room).

### Supplementary diagnostics and limitations (added 2026-11-09, while jobs 4555954 and 4555955 run, before any of their results)

These diagnostics come from an outside review. They add reporting only. The jobs, their configuration and both pass
criteria above are unchanged. Every new number is computed by `adjudicate.py` from files the jobs already write.

**1. Identical views do not prove an identical optimisation.**
- Part 1 rules out evaluation drawing random numbers. It does not rule out evaluation changing the model state or the
  numerics in some other way.
- Reported alongside, for the two 8,000-step runs: final N; PSNR, SSIM and LPIPS at 8,000; and the per-view PSNR
  differences (the largest one, and views 34 and 35).
- These have no noise reference. Two runs without evaluation also differ, through the order of atomic additions, and
  no such pair exists at 8,000. They are descriptive only.

**2. The nine-seed criterion is more permissive than the three-seed one.**
- The allowed difference is the range of the uninterrupted runs. A range tends to grow with the number of runs, so a
  systematic gap of the size seen at three seeds could pass with nine.
- Reported alongside: the ctl − up difference for each seed, paired by seed (the two share the seed and, up to the
  branch, the view sequence), using full-precision final PSNR; the same difference with views 34 and 35 left out; the
  per-seed differences on views 34 and 35; and for each, the mean, a 95% t interval and the number of negative seeds.
- On the original three seeds: mean −0.105 dB, interval [−0.246, +0.037], all 3 negative. With views 34 and 35 left
  out: −0.030 dB, [−0.136, +0.077], 2 of 3 negative.
- **How this will be read, fixed now:**
  - If the rule passes but the nine-seed interval lies entirely below zero, the report says a systematic gap of about
    that size may remain.
  - Arm differences smaller than that gap are then not claimed. The decision itself still follows the rule.

**3. A precision detail of the frozen check.**
- Check 1, in `analyze_ttq.py` and `adjudicate.py` alike, reads ctl's final PSNR from the test-set summary, which is
  rounded to 0.01 dB. It reads the uninterrupted runs' PSNR at full precision.
- The rounding moves a mean by at most 0.005 dB. The rule keeps the frozen source; the diagnostics use full precision
  on both sides.

**4. How the outcome will be reported.**
- The original three-seed check 1 **failed**. The adjudication was designed after that failure was seen, although its
  criteria were recorded before the new runs.
- If both parts pass, the result is described as "Experiment 1 supported after a documented post-result adjudication".
  It is never described as a passed validation.
- If either part fails, Experiment 1 stays invalid, as stated above.
