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

## Status: Experiment 1 is INVALID under the pre-registered rules (adjudication part 1 failed; see the end)

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
  - *Correction (2026-11-09, adjudication part 1):* the smoke test's runs all evaluated at the same step, so the matching
    views there do not cover runs that evaluate differently. Evaluation does change the later view order; see "Part 1
    result" at the end.

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
     densification also draws from. *(Corrected after part 1 ran: the sampler draws from the CPU generator. See
     "Part 1 result" at the end.)* *(That correction was itself wrong on the GPU nodes: the sampler draws from the CUDA
     generator after all. See the second correction under "Part 1 result".)* A draw by evaluation from that generator would therefore show up there. A draw from
     another generator, or one whose state is restored afterwards, would not. (Rationale reworded 2026-11-09, while the
     jobs ran and before any of their results; the criterion is unchanged.)
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
- A pass in part 1 shows that no perturbation of training-view sampling was detected. It does not show that
  evaluation draws no random numbers at all: it could draw from another generator, or restore the generator's state,
  without changing the view sequence. Nor does it rule out evaluation changing the model state or the numerics in some
  other way.
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

## Part 1 result (job 4555954; recorded 2026-11-09, while the part 2 runs continue)

**Part 1 fails.** Under the pre-registered outcome rule, **Experiment 1 stays invalid and the arms are not compared**,
whatever part 2 shows.

**What the job measured** (room seed 0, 0 → 8,000):
- The view sequences are identical over steps 0–5,167 and differ from step 5,168 on: 2,823 of the remaining 2,832 steps.
- 5,168 = 19 × 272, where 272 is the number of training views. It is the first reshuffle of the view order after the
  first evaluation, at step 5,000. All 19 earlier reshuffles match.
- Diagnostics, descriptive only: at step 8,000 the run with evaluation has 1,194,125 Gaussians against 1,207,452 without,
  and test PSNR 29.844 against 29.886 dB. The counts already differ slightly at step 1,000 (173,090 against 173,380),
  long before any evaluation, from the order of atomic additions.

**Cause, found and confirmed in isolation** (CPU-only test on a login node, same environment):
- NeRFICG's view sampler (`RandomSequentialSampler.shuffle`) reshuffles with `torch.randperm` on the **CPU** generator.
  My pre-registration said CUDA; that was wrong, and it is corrected above.
- Every test evaluation builds new metric objects (`BaseRenderer.compute_image_metrics`). Building
  `LearnedPerceptualImagePatchSimilarity(net_type='vgg')` initialises its network from the CPU generator before the
  pretrained weights are loaded.
- In the test, building it changed the CPU generator state, and the next `randperm(272)` came out different. Building
  the PSNR and SSIM metrics left the state unchanged.

**Second correction, 2026-11-09 (Experiment 1R smoke job 4556213):** the generator named above is wrong for the cluster
runs. On a GPU node NeRFICG sets `torch.set_default_tensor_type(torch.cuda.FloatTensor)` (`src/Framework.py:291`).
`randperm` in the sampler and the LPIPS initialisation therefore both draw from the **CUDA** generator. The login-node
test had no GPU, so it saw the CPU generator. In the smoke job, each evaluation changed the CUDA state, while the CPU
state did not change at all across 3,000 steps. The cause stands with "CPU" read as "CUDA": evaluation and the sampler
share one generator. The text above is kept as written.

**What this means for Experiment 1:**
- Every branched run of every arm evaluated at step 5,000, so from step 5,168 on each saw a different view order than an
  uninterrupted run with the same seed. The branched `ctl` runs are therefore not replicas of the uninterrupted
  reference after that point, and check 1 compared runs with different view orders.
- This is a cause of non-identity. It is **not yet shown** to cause the −0.10 dB room gap: a different view order is
  another random draw, not an obvious bias. Part 2 measures whether a gap remains across nine seeds. It still runs, as
  investigation of the cause, not to rescue the decision.
- Within Experiment 1, the arms evaluated on the same schedule. Their CPU generators were therefore advanced alike by
  evaluation, but no view traces were recorded for those runs, and the arms may draw from that generator differently
  themselves.
- Benchmarks without in-training evaluation (phases 1 and 2 timing, upbench) are unaffected. The phase 2 `stages` runs
  were already labelled as changed by their evaluations.

**Fix for future runs (not applied; the running jobs use this method directory):**
- Save the CPU and CUDA generator states before each in-training evaluation and restore them after.
- Then repeat part 1 to verify that the view sequences match.

## Part 2 result and adjudication outcome (jobs 4555954 and 4555955, 2026-11-09)

Full output: [adjudication-analysis.md](adjudication-analysis.md). All 18 part 2 runs completed. Job 4555954 is marked
FAILED by Slurm only because part 1 failed. The 20 new runs are in `experiment_log.csv`.

**Outcome: part 1 FAIL, part 2 PASS. Experiment 1 stays invalid under the pre-registered rules, and the arms are not
compared.** The descriptive arm table above stays description only.

### Part 2 under the frozen rule: pass

| | branched `ctl` (9 seeds) | uninterrupted `up` (9 seeds) | difference | allowed (range of `up`) | result |
|---|---|---|---|---|---|
| final PSNR | 31.586 dB | 31.666 dB | −0.080 dB | 0.200 dB | pass |
| final N | 1.296 M | 1.291 M | +0.005 M | 0.036 M | pass |

The PSNR gap barely moved: −0.101 dB at three seeds, −0.080 dB at nine. The pass comes from the allowed range, which
grew from 0.057 dB to 0.200 dB. That is the weakness of this criterion noted above.

### Supplementary diagnostics (declared before results)

Paired by seed, branched `ctl` minus `up`, full-precision final PSNR:

| | mean | 95% t interval | seeds negative |
|---|---|---|---|
| all test views | −0.082 dB | [−0.191, +0.027] | 7 of 9 |
| views 34 and 35 left out | −0.032 dB | [−0.123, +0.060] | 5 of 9 |

- **Reading fixed in advance:** the interval includes zero, so the clause "a systematic gap of about that size may
  remain" is **not** triggered. The data neither establish a systematic gap nor rule out one of up to about 0.19 dB.
- Part 1's state at 8,000 (no noise reference; descriptive): 13,327 fewer Gaussians and −0.041 dB PSNR with evaluation.
  The largest per-view difference is +1.63 dB (view 29); views 34 and 35 differ by −0.96 and +1.19 dB.

**Observation, chosen after seeing the data (not a finding):** on view 35, branched `ctl` is lower than `up` in 8 of 9
seeds, by +1.70 to −4.14 dB. If the seeds were independent coin flips, 8 or more of 9 in one direction would have a
two-sided probability of about 0.04. This comparison was not declared in advance, and views 34 and 35 were singled out
after the first three seeds. On view 34, 7 of 9 seeds are lower.

### What is established

1. In-training evaluation changes the training run. Building the LPIPS-VGG metric at each evaluation advances the CPU
   generator that NeRFICG's view sampler uses, so the view order differs from the next reshuffle on. Measured in part 1,
   and confirmed in an isolated test.
2. Across nine room seeds, branched runs that evaluate score 0.08 dB lower on average than uninterrupted runs that do
   not. The 95% interval runs from −0.19 to +0.03 dB, and the gap is mostly on views 34 and 35.
3. Experiment 1 cannot support a decision. Its branched runs differ from the reference by more than the restore path
   (bit for bit) and atomic-order noise.

### Disclosed after results

- A bug in `adjudicate.py` counted part 1's line twice when the same log was passed to both `--ctl` and `--ref`, as the
  command above does. It would have miscounted a part 1 pass as a fail. It did not change this outcome, since part 1
  failed. It was fixed after the first run of the analysis, and `adjudication-analysis.md` is the fixed output.

### Possible next step (not decided; needs a new pre-registration)

An Experiment 1 rerun with the generator states saved before each evaluation and restored after:
- part 1 repeated first, to verify that the view sequences then match;
- check 1 against fresh uninterrupted runs;
- the original arms and decision rule unchanged.

This would be a new experiment, not a repair of this one.

*(2026-11-09: run as Experiment 1R. All checks passed and `mcmc` passed the decision rule; see
[EXP1R_RESULTS.md](EXP1R_RESULTS.md). Experiment 1 itself stays invalid.)*
