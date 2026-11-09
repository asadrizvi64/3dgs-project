# Experiment 1R: Experiment 1 rerun with observational evaluation (DRAFT, 2026-11-09)

**Status: DRAFT. Not pre-registered. Returned to review on 2026-11-09:** one smoke gate failed as written, and the noise
pilot gives more than 12 seeds for check 1b on room. See "Smoke test and pilot results" at the end.
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

### 4. Check 1 replaced by an exact check plus a paired equivalence test

Experiment 1's check 1 compared a difference of means with the range of the reference runs. That criterion grows more
lenient as seeds are added. It is dropped. This change follows an outside review and is made before any Experiment 1R
run.

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
  seed. Six extra per scene would add about 5.3 GPU-hours.
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
   - Default in this draft: no.
2. **Seeds.** Fresh seeds 10–12 (default) or seeds 0–2 again? Fresh seeds give an independent replication. The same
   seeds would allow a run-by-run comparison with Experiment 1.
3. **The margins for 1b**: 0.05 dB and 1% of N. They are set from what matters to the decision, not from the pilot, and
   they are fixed before it runs.

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
