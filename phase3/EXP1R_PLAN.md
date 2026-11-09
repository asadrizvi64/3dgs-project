# Experiment 1R: Experiment 1 rerun with observational evaluation (pre-registered 2026-11-09)

**Status: FINAL PRE-REGISTRATION, frozen 2026-11-09 at launch, in the commit that adds this line.**
- All revised validation gates passed (jobs 4559014 and 4559015), and the user gave the go-ahead.
- The binding parts of this plan:
  - what stays as in Experiment 1;
  - sections 1–6 of "What changes", with check 1 in its current version;
  - "How the result will be reported", including the 2026-11-09 additions.
- Any change from here on is listed below "Changes after launch", with its date and reason.
- The analysis command, fixed now: `python phase3/analyze_ttq.py <garden log> <room logs> --exp1r --seeds 10 11 12`.
  Logs of reruns under the timeout rule are added with `--rerun`.

**History:** DRAFT until launch. **Revised after review on 2026-11-09:** check 1 is now exact gates plus a paired
report, and smoke test A has a revised tolerance and is being repeated. Approved to proceed to this revised validation
stage only, not to the experiment. See "Revisions after the smoke test and pilot" near the end. Before that, on
2026-11-09, the plan was returned to review: one smoke gate failed as written, and the noise pilot gave more than 12
seeds for the earlier check 1b on room.
- Nothing in the experiment runs until its smoke tests pass and this file is marked final.
- The smoke tests and the noise pilot below are engineering runs. They are reported here, and they do not test any arm.
- Once the plan is final, any change is listed with its date and reason, as in [PLAN.md](PLAN.md).

## Why

**Experiment 1 is invalid under its own rules ([RESULTS.md](RESULTS.md)). It was invalidated by evaluation-induced
random-number contamination:**
- Every in-training evaluation builds a new LPIPS-VGG metric.
- Building it initialises a network from a global random generator.
- NeRFICG's view sampler reshuffles from that same generator. On the GPU nodes this is the CUDA generator, because
  NeRFICG makes CUDA tensors the default. An earlier version of this draft said CPU; corrected 2026-11-09 from the
  smoke test (see the results at the end).

So a run that evaluates sees a different view order from the first reshuffle after its first evaluation. Every branched
run evaluated; the uninterrupted reference runs did not.

**Unresolved:**
- Across nine room seeds, branched runs scored 0.08 dB lower than uninterrupted ones. The 95% interval runs from −0.19
  to +0.03 dB.
- The view-order change explains why the runs differ. It is not shown to explain that gap.

**Not to be assumed:** Experiment 1's descriptive `mcmc` result (target reached 30–43% sooner) is an observation. It
may not reproduce.

Experiment 1R asks the same question with evaluation made observational, on fresh seeds. It is a new experiment.
Experiment 1 stays permanently documented as invalid.

## What stays exactly as in Experiment 1

- **Build:** `FasterGSUp` (1144604 + `upstream/upstream_main_kernels.patch`), dense Adam, fp32 moments, and the same
  clock hook (training time = SEGCLOCK, CUDA events, evaluation excluded).
- **Scenes and resolutions:** garden 1/4, room 1/2.
- **Branch:** shared prefix 0 → 5,000 per (scene, seed), saved as a resume-patch state file.
- **Arms and their settings:** `ctl`, `stop`, `mcmc` and `oracle`, with every clarification in PLAN.md's "Changes after
  start":
  - the `stop` definition;
  - the MCMC overrides, including opacity learning rate 0.05;
  - the oracle cap from `FGS_TRAIN_END N`;
  - evaluation point i = the state after exactly i steps.
- **Evaluation schedule:** test metrics every 1,000 steps from 5,000 to 30,000, through NeRFICG's unchanged metric code.
- **Measures:**
  - the target: `ctl` final PSNR − 0.05 dB;
  - T, interpolated, with "not reached";
  - the sensitivity targets;
  - final ΔPSNR and ΔLPIPS;
  - one-view failures (phase 2 definition).
- **Decision rule, per `stop` and `mcmc`, on both scenes:**
  - median paired ΔT ≥ 10%, with T lower in ≥ 5 of 6 pairs;
  - mean ΔPSNR ≥ −0.10 dB and mean ΔLPIPS ≤ +0.002;
  - no more one-view failures than `ctl`.
- **Checks 2 and 3:** each arm did what it says (with the `stop` check from step 6,000, as corrected in Experiment 1),
  and run validation.
- **Reading of `mcmc`:** a pass means the combined MCMC policy reaches quality sooner, not that relocation alone does.

## What changes

### 1. Evaluation is observational (resume patch v7, [cluster/patches/fastergs_resume.py](../cluster/patches/fastergs_resume.py))

**What `fgs_eval` saves before rendering the test set and restores afterwards**, in a `finally` block so they come back
even if evaluation raises:
- the torch CPU state, the torch CUDA states (all devices), the Python `random` state and the NumPy state;
- the model's training flag and the dataset's mode;
- `cudnn.benchmark`.

**Other generators:** the code path these runs use contains no explicitly constructed `torch.Generator` objects
(searched 2026-11-09), so restoring the global states covers every generator in the training path. Track R's
`FasterGSReinit` method has its own, and it is not used here. MCMC's sampling
(`torch.multinomial` on GPU tensors) uses the CUDA state.

**Not changed: building LPIPS once and reusing it.** That would rewrite NeRFICG's metric code, which the final evaluation
and every earlier phase also use. Restoring the random state already makes the rebuilt metric harmless to training.
Reuse remains an option for evaluation cost; smoke test A measures what that cost is.

### 2. Fresh seeds: 10, 11, 12 for the arms

Experiment 1's results on seeds 0–2 have been seen. New seeds make this a replication.

### 3. Reference runs in the same jobs

For each seed, one uninterrupted `up` run to 30,000 without evaluation, on the same build and patch. View traces are
recorded for every run.

### 4. Check 1: exact gates plus a paired report (revised 2026-11-09)

Experiment 1's check 1 compared a difference of means with the range of the reference runs. That criterion grows more
lenient as seeds are added. It is dropped. This change follows an outside review and is made before any Experiment 1R
run.

**Current version.** Decided by review on 2026-11-09, after the noise pilot's results were seen and before any Experiment
1R arm ran. It replaces the equivalence version below, which is kept as first written.

Check 1 now asks whether branching preserves the intended training process. It no longer asks whether two
non-deterministic trajectories end at nearly the same PSNR. The pilot showed that two uninterrupted runs, with identical
views, differ by an SD of 0.05–0.07 dB in final PSNR.

| check | requirement, for every seed of every scene | type |
|---|---|---|
| **1a** | The prefix's view trace (steps 0 to 4,999) followed by `ctl`'s (5,000 to 29,999) equals the same-seed reference run's (0 to 29,999), line for line | gate |
| **1b** | Every branched arm loaded exactly the saved state: N, parameter and optimizer SHA-256 equal to the prefix's `FGS_SAVE_STATE` (densification statistics too, except MCMC arms, which drop them) | gate |
| **1c** | In every branched arm, at every in-training evaluation (5,000 to 29,000), `FGS_EVALCHECK` shows the four random states restored, parameters, optimizer state and densification statistics unchanged, and the training flag and dataset mode restored | gate |
| **1d** | Configuration. `ctl`'s training configuration equals the reference's except the run name. `stop`, `mcmc` and `oracle` differ from `ctl` only in their own keys. The settings in effect after loading (MCMC on or off, densification end, opacity learning rate, both regularisation weights, MCMC cap) equal the arm's configuration. The prefix differs from the reference only in name and `NUM_ITERATIONS`, with PPISP and the 3D filter off, so `NUM_ITERATIONS` has no effect before step 5,000. | gate |
| **1e** | Per seed: `ctl` − reference final PSNR (full precision) and final N, also in units of the pilot's same-seed SD; the two largest per-view differences; the run order | reported, not a gate |

- These checks run on the experiment's own runs, not only in the smoke tests. Every branched run evaluates with
  `FGS_EVAL_CHECK=1`, and every run records a view trace.
- **Outcome:** if 1a–1d pass on both scenes, the arms are evaluated by the decision rule. If any of them fails,
  Experiment 1R is invalid until the cause is found.
- **Limitation, to be stated wherever the result is reported:** passing 1a–1d establishes implementation fidelity. It
  does not establish statistical equivalence of final PSNR between branched and uninterrupted training.
- **Not chosen:** widening the ±0.05 dB margin after the pilot failed to support it, and running more than 12 seeds per
  scene to reach the margin.

**Superseded version (2026-11-09, before the pilot; kept as written):**

- **1a (exact):** `ctl`'s view trace over steps 0–29,999 must equal its same-seed reference run's, for every seed.
- **1b (paired equivalence, TOST):** per scene, the 90% t interval of the mean paired difference `ctl` − `up` must lie
  within the margin:
  - final PSNR: ±0.05 dB, the same offset the target uses;
  - final N: ±1% of the mean N.
- **Sample size for 1b, rule fixed now, before the pilot:**
  - n = the smallest number of seeds per scene that gives at least 80% power on both measures and both scenes.
  - Power is computed by simulation at a true difference of 0, using the same-seed SD measured in the noise pilot.
  - Seeds 10–12 count toward n. Additional check-1 seeds (13, 14, …) run only the prefix, `ctl` and `up`.
  - If n would exceed 12 on either scene, check 1 cannot be made conclusive at these margins, and the plan returns to
    review before launch.
- **Outcome of check 1:**
  - Both 1a and 1b pass: the arms are evaluated by the decision rule.
  - 1a fails, or 1b's interval lies outside the margin: Experiment 1R is invalid.
  - 1b's interval crosses the margin: check 1 is inconclusive. That does not count as a pass, and the arms are not
    compared.
  - Always reported: the per-seed differences, their mean and interval, and the per-view differences on the lowest
    views.

### 5. Run order alternates (added 2026-11-09)

- In the pilot, 7 of 8 same-seed differences were negative (the second run lower). That is not evidence of a bias, but
  the pilot cannot be assumed to be centred on zero.
- The reference run therefore runs **first for even seeds and last for odd seeds**:
  - seed 10: `up`, prefix, arms, `oracle`;
  - seed 11: prefix, arms, `oracle`, `up`;
  - seed 12: as seed 10.
- Each job prints the actual order on an `EXP1R_ORDER` line, and 1e reports it per seed.
- This is a scheduling control, not an arm.

### 6. Reading the result against the noise floor (added 2026-11-09)

- The decision rule is unchanged. The same-seed PSNR noise (SD 0.05–0.07 dB) is the size of the 0.05 dB target offset,
  so the first crossing time T can itself move.
- The existing sensitivity targets (−0.10 and −0.20 dB) and "reaches −0.05 dB and stays" are reported for every arm.
- If an arm passes at −0.05 dB but its advantage disappears at the other targets, the result is described as
  sensitive to target selection.
- An advantage that holds across all of them is the stronger result.

## Before launch: smoke tests and the noise pilot

The smoke job runs stages `exp1smoke x1rsmoke`, all on room seed 0 except exp1smoke, which uses garden. The pilot runs
as its own job. Gates must pass; diagnostics are reported only.

| test | what runs | requirement | type |
|---|---|---|---|
| **A. Evaluation correctness** | The saved Experiment 1 room seed-0 state at 5,000 (job 4553943), resumed three times to 7,000, evaluating at 5,000 and 6,000: guard on, guard off (v6 behaviour), and guard on with `cudnn.benchmark` off | Guard on and off give identical per-image metrics and full-precision means at 5,000, from the same state | gate |
| **A. Evaluation memory and time profile** | The same three runs with `FGS_EVAL_PROF`: time, calls and peak allocated memory per stage (rendering, loading photos, saving images, constructing and running each metric), plus memory before and after | Report the cause of the 64 GiB peak. The candidate, from Track R's note on job 4504356: cuDNN's algorithm search for LPIPS-VGG at full resolution, about 65 GiB on the first call. Benchmark-off metric values are compared with benchmark on | diagnostic |
| **B. Random states preserved** | Four uninterrupted runs 0 → 8,000, two evaluating every 1,000 from 5,000 with `FGS_EVAL_CHECK`, two not | Digests of the four random states identical before and after every evaluation | gate |
| **B. Model state preserved** | same | Parameters, optimizer state and densification statistics identical before and after every evaluation (SHA-256) | gate |
| **B. Mode preserved** | same | Training flag and dataset mode restored after every evaluation | gate |
| **B. Camera sampling** | same | All four view traces identical over steps 0–7,999 | gate |
| **B. Gaussian count and metrics** | same | N every 1,000 steps and metrics at 8,000, comparing evaluated with non-evaluated runs, with the same-condition pairs as the noise reference | diagnostic (see note) |
| **Resume path** | `exp1smoke`, unchanged, on the patched code | Every existing check passes | gate |
| **Noise pilot** | Garden and room, seeds 20–23: two identical uninterrupted 30k runs each, with view traces | Report same-seed SDs; n for check 1b by the rule above. Each pair's view traces must be identical | gate on views; n by rule |
| **Analysis** | `analyze_ttq.py` gains checks 1a and 1b, with references from the same job's `b_<scene>_up_s<seed>` runs | Fixture tests with a mismatched trace and with a failing interval | gate |

**Revised validation stage (2026-11-09).** The rows above are kept as first planned; their results are at the end. The
revised plan adds or replaces these:

| test | what runs | requirement | type |
|---|---|---|---|
| **A, repeated** (stage `x1ra`) | The same saved room seed-0 state at 5,000 (job 4553943), resumed to 5,100 and evaluated once at 5,000, unprofiled: guard on ×2, guard off ×2, guard on with `cudnn.benchmark` off ×2 | For each of the 4 guard-on / guard-off pairs: per-image rows identical at their output precision, and full-precision means within 1e-5 dB PSNR, 1e-6 SSIM and 1e-6 LPIPS. Replaces the failed bit-identity criterion, which stays documented. | gate |
| **A, within-condition differences** | same | The same differences within each condition, and with benchmark off | diagnostic |
| **Evaluation memory** | same | Peak allocated memory at the last training step. It is never reset, so it includes the evaluation at 5,000. Compared with `cudnn.benchmark` on (4 runs) and off (2 runs). No profiler and no added synchronisation. | diagnostic, not blocking |
| **B gates** | unchanged; passed in job 4556213 | Also applied inside the experiment, as check 1c | gate |
| **Stage smoke** (`exp1r`, short) | Garden, seeds 0 and 1 (both run orders), branch at 2,000, end at 3,000: reference, prefix, all four arms, all through the new stage | `analyze_ttq.py --exp1r --branch 2000 --end 3000 --seeds 0 1 --checks-only` passes on the real output: validation and 1a–1d | gate |
| **Analysis** | `analyze_ttq.py --exp1r` (checks 1a–1e); `phase3/tests/exp1r_fixture.py` | Clean fixture passes and reaches a decision. Six planted faults are each caught by the intended check and stop the decision (exit 2): a changed view (1a), a wrong parameter digest (1b), an unrestored random state (1c), a configuration difference, a setting in effect that differs from the configuration, and a prefix configuration difference (1d). Experiment 1's analysis gives identical output before and after the change. | gate, passed 2026-11-09 |

**Note on the count and metric diagnostic.** Once the four exact gates in B pass, evaluation can affect training only
through GPU non-determinism, which runs without evaluation have too. A tolerance on two evaluated runs against two
non-evaluated ones would therefore test no further mechanism. It would also fail by chance often. The size of that
noise is measured directly by the pilot, at 30,000 steps and with more pairs. This choice is open to review.

## Jobs and cost

- **Smoke job:** about 1 GPU-hour.
- **Pilot:** about 1.3 GPU-hours.
- **Experiment, with seeds 10–12:**
  - garden: one job, about 3 hours;
  - room: one job per seed, about 2 hours each.
  - About 9–10 GPU-hours in all.
- **Additional check-1 seeds, if the pilot calls for them:** about 20 minutes per garden seed and 33 minutes per room
  seed. Six extra per scene would add about 5.3 GPU-hours. *(Dropped with the revised check 1, 2026-11-09.)*
- **Revised validation stage:** test A repeated, about 0.4 GPU-hours; stage smoke, about 0.5 GPU-hours.
- **Added cost in the experiment:** `FGS_EVAL_CHECK` hashes the full model and optimizer state twice per evaluation. That
  time falls inside the evaluation, so it is excluded from the training clock but adds to elapsed time. The stage smoke
  measures it.
- **Timeout rule, as in Experiment 1:** a (scene, seed) block that does not complete is rerun whole (prefix, arms and
  reference) under the same configuration.

## How the result will be reported

- Experiment 1 stays invalid, with its cause.
- Experiment 1R is a separate experiment, designed after Experiment 1's descriptive results were seen, which is why
  it uses fresh seeds.
- If `mcmc` passes, the claim is limited to this: the combined MCMC policy reached target quality sooner on garden and
  room (seeds 10–12, H100).
- If check 1 fails or is inconclusive with identical view traces, the remaining difference is something other than
  view order. Investigating it comes before any arm comparison.
- Added 2026-11-09:
  - Check 1 establishes implementation fidelity, not statistical equivalence of final PSNR.
  - With three seeds per scene, the result does not establish generalisation beyond these scenes.
  - The revisions to check 1 and smoke test A were made after the pilot's results were seen, and are reported as such.

## Out of scope

- No new arms and no MCMC ablations. Separating relocation, noise injection, regularisation and the opacity learning
  rate is a later experiment, and only if `mcmc` passes here.
- Experiment 2 ([EXP2_PLAN.md](EXP2_PLAN.md)) stays frozen as a draft. Any experiment that evaluates during training
  needs the v7 fix.

## Open questions for review

1. **A dedicated generator for the view sampler.**
   - The alternative is to give `RandomSequentialSampler` its own torch generator, seeded from the run seed. Every arm
     would then see the same view order, whatever else draws random numbers.
   - It would reduce noise in the paired arm comparisons. But it departs from stock NeRFICG behaviour, and it would
     change the reference runs too.
   - In Experiment 1's smoke test, `stop` and `mcmc` saw a different view order from the uninterrupted run after the
     branch. That is not explained yet; the traces will show whether it recurs.
   - Default in this draft: no. The 2026-11-09 review did not change this, so it stays no.
2. **Seeds.** Fresh seeds 10–12 (default) or seeds 0–2 again? Fresh seeds give an independent replication. The same
   seeds would allow a run-by-run comparison with Experiment 1. *Resolved 2026-11-09 by review: fresh seeds 10, 11, 12.*
3. **The margins for 1b**: 0.05 dB and 1% of N. They are set from what matters to the decision, not from the pilot, and
   they are fixed before it runs. *Resolved 2026-11-09: the equivalence test is dropped from the gate (section 4), and
   the margin is not widened.*

## Smoke test and pilot results (2026-11-09)

Jobs: smoke **4556213** (stages `exp1smoke x1rsmoke`, repo 6c37ab0, 37 min, ended FAILED on the x1rsmoke check) and
pilot **4556214** (stage `noisepilot`, 71 min, completed). Logs: `phase2/logs/gs-x1r-smoke-4556213.out`,
`phase2/logs/gs-x1r-pilot-4556214.out`. Pilot analysis: [noise-pilot-analysis.md](noise-pilot-analysis.md).

**Outcome: the plan returns to review before launch.** Two reasons, both under rules fixed before the jobs ran:
1. Smoke gate A (identical full-precision means) failed as written.
2. The pilot gives n > 12 seeds for room under the check 1b rule.

### Gates

| gate | result |
|---|---|
| Resume path (`exp1smoke`, all existing checks) | PASS (37 of 37) |
| A. Identical per-image metrics, guard on vs off, same state at 5,000 | PASS |
| **A. Identical full-precision means, guard on vs off** | **FAIL** |
| B. Four view traces of 8,000 steps, all identical | PASS (no difference anywhere) |
| B. Random states restored after every evaluation (5,000, 6,000, 7,000; both evaluating runs) | PASS |
| B. Parameters, optimizer state and densification statistics unchanged by every evaluation (SHA-256) | PASS |
| B. Training flag and dataset mode restored after every evaluation | PASS |
| Pilot: identical view traces within each same-seed pair | PASS (8 of 8 pairs) |
| Analysis gate (`analyze_ttq.py` checks 1a and 1b, fixture tests) | not built yet |

**The failed gate.** The three resumes of the same saved state gave these full-precision means at 5,000:

| run | PSNR | SSIM | LPIPS |
|---|---|---|---|
| guard on | 29.254138702 | 0.8904180374 | 0.3371558503 |
| guard off | 29.254138751 | 0.8904180435 | 0.3371558266 |
| guard on, benchmark off | 29.254138604 | 0.8904180481 | 0.3371557723 |

- Every pair differs, in about the eighth significant digit. That includes the two guard-on runs, which differ only in
  the cuDNN benchmark setting.
- Up to the end of the first evaluation, guard on and guard off run the same code: the guard acts only after rendering.
- So the most likely cause is run-to-run GPU non-determinism in rendering or the metrics, not the guard. The gate
  assumed that a metric computation from the same state is bit-identical across processes. It is not.
- This is a diagnosis, not a pass. The gate stays failed as written. A replacement needs review. Candidate: guard on and
  off must differ by no more than two guard-on repeats of the same state do. That would need a rerun of test A with a
  repeat.

### What the evaluation does to the random states (correction)

- In every evaluation, rendering changed the **CUDA** state and left the CPU, Python and NumPy states unchanged. The
  restore then brought the CUDA state back.
- The CPU state digest is the same at 5,000, 6,000 and 7,000. Between those points the sampler reshuffled the 272 room
  views at least 7 times, so the reshuffle does not draw from the CPU generator on the GPU nodes.
- The reason: on a GPU node NeRFICG sets `torch.set_default_tensor_type(torch.cuda.FloatTensor)`
  (`src/Framework.py:291`). `torch.randperm` in the sampler and the LPIPS network's initialisation therefore draw from
  the CUDA generator.
- My login-node test had no GPU, so it saw the CPU generator. My 2026-11-09 correction, "CPU, not CUDA", was therefore
  itself wrong for the cluster runs. The original statement, CUDA, was right. The mechanism (evaluation and the sampler
  share one generator) and the v7 fix are unaffected, since v7 restores all four states. Corrected in PLAN.md and
  RESULTS.md, with the earlier text kept.
- "No explicit generators": true for the code path these runs use. Track R's `FasterGSReinit` method creates its own
  generators, but it is not part of Experiment 1 or 1R.

### Diagnostics

**Evaluation memory.**

| run | evaluation | peak allocated at the last training step |
|---|---|---|
| rn_room_ev_s0, rn_room_ev2_s0 | 5,000, 6,000, 7,000; no profiling | 71.15, 71.16 GiB |
| rn_room_ne_s0, rn_room_ne2_s0 | none | 6.47 GiB |
| rs_room_* (all three) | 5,000, 6,000; profiled | 9.62 GiB |
| garden `exp1smoke` runs | at 2,000; no profiling | 67.97 GiB |

- With profiling on, the total peak over each whole evaluation was 9.45–9.62 GiB, with or without the cuDNN benchmark.
  The largest stage was the LPIPS forward pass.
- Without profiling, the same evaluation path still leaves a peak about 64 GiB above training.
- The profiled runs therefore did not show the 64 GiB peak, and the cause is **not identified**. The cuDNN-benchmark
  explanation is neither confirmed nor ruled out: the benchmark-off comparison was made only in profiled runs, where
  the peak did not occur either way.
- One possibility, untested: the profiler's synchronisations and peak resets change what cuDNN's algorithm search
  sees, for example the free memory it sizes its workspace from.
- A test that would separate these: unprofiled evaluating runs, benchmark on vs off, with a peak reset just before and
  a reading just after each evaluation.

**Evaluation time (room, 39 test views, profiled).** About 54 s per evaluation:

| stage | time |
|---|---|
| saving images (78 files) | 40.5 s |
| metric computation | 12.2–12.5 s |
| loading the photos | 10.7 s |
| LPIPS forward | 1.1 s |
| rendering | 0.08 s |

Loading the photos happens inside the metric stage, so the stage times overlap. Saving images is about three quarters of
the cost.

**Benchmark off.** Same metrics to about the eighth significant digit as benchmark on (table above).

**Gaussian count and metrics, room seed 0, 0 → 8,000.**

| run | N at 1,000 | N at 8,000 | test PSNR at 8,000 |
|---|---|---|---|
| rn_room_ev_s0 (evaluates) | 172,953 | 1,203,739 | 30.042 |
| rn_room_ev2_s0 (evaluates) | 173,326 | 1,213,418 | 29.956 |
| rn_room_ne_s0 (no evaluation) | 172,153 | 1,206,482 | 29.954 |
| rn_room_ne2_s0 (no evaluation) | 173,044 | 1,201,176 | 29.906 |

- The counts differ from step 1,000, before any evaluation, in both pairs. That is GPU non-determinism.
- The evaluating pair and the non-evaluating pair overlap in N and in PSNR.

### Noise pilot

- Two identical uninterrupted 30k runs per seed (20–23), on both scenes.
- View traces were identical in every pair, so the pairs differ only through GPU non-determinism.

| scene | d PSNR, up2 − up (dB) | mean | SD | SD of d N | largest per-view d |
|---|---|---|---|---|---|
| garden | −0.012, −0.078, −0.125, −0.030 | −0.061 | 0.050 | 0.39% of N | −1.10 dB (view 1) |
| room | −0.073, −0.150, +0.012, −0.084 | −0.074 | 0.066 | 0.36% of N | −2.85 dB (view 35) |

**By the rule fixed above** (smallest n with ≥ 80% TOST power on both measures and both scenes):
- garden needs 11 seeds;
- room needs more than 12 (58% power at 12).

Check 1b therefore cannot be made conclusive at a ±0.05 dB margin, and **the plan returns to review**. The Gaussian
count is not the limit: power is ≥ 97% from 4 seeds on both scenes.

**Observations, descriptive only:**
- Same-seed, same-view-order PSNR differences reach 0.15 dB. Their mean on room (−0.074 dB) is the size of Experiment 1's
  room `ctl` − `up` gap (−0.082 dB). That gap is therefore well within what GPU non-determinism alone produces between
  two otherwise identical runs.
- 7 of the 8 differences are negative (the second run lower). With 8 pairs, a split at least this uneven happens by
  chance about 7% of the time, so this is not evidence of an order effect. It is noted because a systematic effect would
  bias a paired check.
- On room, view 35 has the largest per-view difference in 3 of 4 pairs. In Experiment 1, view 35 carried the largest
  `ctl` − `up` difference in 8 of 9 seeds. That view is unstable under non-determinism alone.
- An SD from 4 pairs is imprecise. Its 95% interval spans roughly 0.6 to 3.7 times the estimate.

### For review

1. **Check 1b.**
   - Under the fixed rule, check 1b cannot be made conclusive at ±0.05 dB with up to 12 seeds.
   - Options:
     - a wider PSNR margin, disclosed as set after the pilot;
     - more seeds, beyond 12;
     - a check 1 that relies on the exact gates (1a and the B gates), with the paired PSNR difference reported
       descriptively against the pilot's same-seed spread.
   - The choice belongs to review. Nothing here picks one.
2. **Gate A.** Replace "bit-identical means" with a criterion that allows for measured run-to-run non-determinism, and
   rerun test A with a repeat.
3. **The 64 GiB peak.** It is not a gate. It is still unexplained, and the profiler as built does not reproduce it.
4. **The decision rule's measures.** The noise floor matters beyond check 1. Same-seed PSNR noise of 0.05–0.07 dB is
   comparable to the decision rule's −0.10 dB quality allowance, and to the 0.05 dB target offset.

## Revisions after the smoke test and pilot (2026-11-09)

Decided by outside review of commit 0e3477b, after the smoke-test and pilot results above were seen, and before any
Experiment 1R arm ran. The review approved the revised validation stage, not the experiment itself.

| # | change | reason |
|---|---|---|
| 1 | Check 1 becomes exact gates 1a–1d plus the paired report 1e (section 4). The TOST equivalence gate is dropped; the ±0.05 dB margin is not widened; no seeds beyond 12. | Same-seed GPU non-determinism (SD 0.05–0.07 dB) makes a ±0.05 dB equivalence gate unreachable at 12 seeds. Fidelity of the branch is the question check 1 can answer. |
| 2 | Smoke test A: bit-identical means replaced by absolute tolerances of 1e-5 dB PSNR, 1e-6 SSIM and 1e-6 LPIPS. Per-image rows must still match at output precision. | Observed run-to-run differences were at most 1.5e-7 dB PSNR, 1.1e-8 SSIM and 7.8e-8 LPIPS, including between two guard-on runs. The failed criterion stays documented above. |
| 3 | Smoke test A is repeated with two guard-on and two guard-off resumes of the same state (stage `x1ra`). | The tolerance must hold for every guard-on / guard-off pair, with the within-condition spread reported alongside. |
| 4 | Seeds stay 10, 11, 12, and every arm, threshold and target stays as planned. | No change to the scientific design. |
| 5 | The reference run alternates position by seed (section 5). | 7 of 8 pilot differences were negative. |
| 6 | Sensitivity targets are read as in section 6. | The noise floor is the size of the target offset. |
| 7 | The analysis-script tests come before the plan is final (passed 2026-11-09, table above). The new stage gets its own short smoke run. | Required by the review. |

- **The 64 GiB peak does not block Experiment 1R.** All arms use the same evaluation procedure. The memory diagnostic in
  `x1ra` is an unprofiled comparison with `cudnn.benchmark` on and off. Speeding up evaluation (image saving takes
  about three quarters of each evaluation) is a separate engineering task.
- **Experiment 1R launches only if** test A (repeated), the stage smoke and the analysis tests all pass, and the user
  gives the go-ahead after seeing them.

## Revised validation results (2026-11-09)

Jobs:
- **4559014**: stage `x1ra`, repo 8db2f2e, 14 min, completed. Log `phase2/logs/gs-x1ra-4559014.out`.
- **4559015**: stage `exp1r`, a short garden run, 17 min, completed. Log `phase2/logs/gs-x1r-stagesmoke-4559015.out`.

| gate | result |
|---|---|
| A, repeated: the 4 guard-on / guard-off pairs, per-image rows identical | PASS (4 of 4) |
| A, repeated: full-precision means within 1e-5 dB / 1e-6 / 1e-6 | PASS (4 of 4); largest differences PSNR 1.22e-6 dB, SSIM 2.6e-8, LPIPS 1.0e-7 |
| Stage smoke: validation, and checks 1a–1d on real output (garden, seeds 0 and 1, both run orders) | PASS |
| Analysis fixture tests | PASS (earlier, same commit) |

**Test A, within-condition differences:**
- Guard on vs guard on: 6.9e-7 dB PSNR. Guard off vs guard off: 8.3e-7 dB. Benchmark off vs benchmark off: 6.4e-7 dB.
- These are the same size as the guard-on vs guard-off differences. The differences are run-to-run non-determinism, not
  an effect of the guard.
- The largest cross-condition PSNR difference is about 8 times below the tolerance.

**The 64 GiB peak: cause found.**
- With `cudnn.benchmark` on, the peak allocated memory was 71.15 GiB in all four runs. With it off, it was 9.45 GiB in
  both runs.
- Everything else was identical: same state, same single evaluation, no profiler. The extra ~62 GiB is therefore
  cuDNN's algorithm search under `cudnn.benchmark`, presumably for the LPIPS-VGG convolutions, as Track R's note
  suggested.
- Turning the benchmark off changed the metrics by at most about 5e-7 dB, the same size as the run-to-run differences.
- Why the profiled runs in job 4556213 did not show the peak is not established. The profiler's synchronisations
  around each stage are one possibility.
- The evaluation procedure for Experiment 1R is **not** changed: the benchmark stays on, as in every arm and as in
  Experiment 1. Switching it off during evaluation (`FGS_EVAL_CUDNN_BENCH=0`) is an option for later experiments.

**Stage smoke:**

| item | result |
|---|---|
| Run order | Logged as planned: seed 0 reference first, seed 1 reference last |
| Evaluation checks | 8 `FGS_EVALCHECK` lines, one per branched run, all restored |
| 1e, `ctl` − reference at 3,000 steps | −0.012 dB (seed 0) and +0.048 dB (seed 1). Reported only; at 3,000 steps this says nothing about 30,000. |
| Cost of the evaluation check | Garden branched runs took 88–93 s, against 87–90 s without the check in job 4556213. About 1–5 s per evaluation. |

**All gates of the revised validation stage pass.**
- Experiment 1R is ready to launch: garden in one job, room in one job per seed, seeds 10–12, stage `exp1r`.
- It launches only on the user's go-ahead.
- This file becomes the final pre-registration at launch, with the commit it is frozen at.

## Launch (2026-11-09)

Submitted at commit c3ea481, stage `exp1r`. The cluster checkout is not touched while these jobs run.

| job | scene | seeds | time limit |
|---|---|---|---|
| 4559237 | garden | 10, 11, 12 | 4:30 |
| 4559238 | room | 10 | 3:00 |
| 4559239 | room | 11 | 3:00 |
| 4559240 | room | 12 | 3:00 |

**Result (2026-11-09):** all checks pass; `mcmc` passes the decision rule, `stop` fails, the oracle passes. See
[EXP1R_RESULTS.md](EXP1R_RESULTS.md).

## Changes after launch

(None.)
