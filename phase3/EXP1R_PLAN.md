# Experiment 1R: Experiment 1 rerun with observational evaluation (DRAFT, 2026-11-09)

**Status: DRAFT. Not pre-registered.**
- Nothing in the experiment runs until its smoke tests pass and this file is marked final.
- The smoke tests and the noise pilot below are engineering runs. They are reported here, and they do not test any arm.
- Once the plan is final, any change is listed with its date and reason, as in [PLAN.md](PLAN.md).

## Why

**Experiment 1 is invalid under its own rules ([RESULTS.md](RESULTS.md)). It was invalidated by evaluation-induced
random-number contamination:**
- Every in-training evaluation builds a new LPIPS-VGG metric.
- Building it initialises a network from the CPU random generator.
- NeRFICG's view sampler reshuffles from that same generator.

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

**Other generators:** NeRFICG's source contains no explicitly constructed `torch.Generator` objects (searched
2026-11-09), so restoring the global states covers every generator in the training path. MCMC's sampling
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

## Smoke test and pilot results

(Recorded here when the jobs finish.)
