# Experiment 1R: Experiment 1 rerun with evaluation that leaves training unchanged (DRAFT, 2026-11-09)

**Status: DRAFT. Not pre-registered.** Nothing in it runs until it has been reviewed and marked final. Once it is
final, any change is listed with its date and reason, as in [PLAN.md](PLAN.md).

## Why

Experiment 1 is invalid under its own rules ([RESULTS.md](RESULTS.md)).

**The cause:**
- Every in-training evaluation builds a new LPIPS-VGG metric.
- Building it initialises a network from the CPU random generator.
- NeRFICG's view sampler reshuffles from that same generator.

So from the first reshuffle after the first evaluation, a run that evaluates sees a different view order from one that
doesn't. Every branched run evaluates; the uninterrupted reference runs do not. Check 1 therefore compared runs with
different view orders.

**What is established and what is not:**
- Across nine room seeds, branched runs scored 0.08 dB lower than uninterrupted ones. The 95% interval runs from −0.19
  to +0.03 dB, and the gap sits mostly on views 34 and 35.
- Whether that gap is real is unresolved.
- The descriptive Experiment 1 results (`mcmc` reached quality 30–43% sooner) are observations, not findings.

Experiment 1R asks the same question as Experiment 1, with evaluation made neutral and on fresh seeds. It is a new
experiment. It does not repair Experiment 1, whose status stays invalid.

## What stays exactly as in Experiment 1

- **Build:** `FasterGSUp` (1144604 + `upstream/upstream_main_kernels.patch`), dense Adam, fp32 moments, the same clock
  hook (training time = SEGCLOCK, CUDA events, evaluation excluded).
- **Scenes and resolutions:** garden 1/4, room 1/2.
- **Branch:** shared prefix 0 → 5,000 per (scene, seed), saved with resume patch state files.
- **Arms and their settings:** `ctl`, `stop`, `mcmc` and `oracle`, with every clarification in PLAN.md's "Changes after
  start" (the `stop` definition, the MCMC overrides including opacity learning rate 0.05, the oracle cap from
  `FGS_TRAIN_END N`, and evaluation point i = the state after exactly i steps).
- **Evaluation schedule:** test metrics every 1,000 steps from 5,000 to 30,000.
- **Measures:** the target (`ctl` final PSNR − 0.05 dB), T with interpolation and "not reached", the sensitivity
  targets, final ΔPSNR / ΔLPIPS, and one-view failures (phase 2 definition).
- **Decision rule, per `stop` and `mcmc`, on both scenes:**
  - median paired ΔT ≥ 10%, with T lower in ≥ 5 of 6 pairs;
  - mean ΔPSNR ≥ −0.10 dB and mean ΔLPIPS ≤ +0.002;
  - no more one-view failures than `ctl`.
- **Checks 2 and 3:** each arm did what it says (with the `stop` check from step 6,000, as corrected in Experiment 1),
  and run validation.
- **Reading of `mcmc`:** a pass means the combined MCMC policy reaches quality sooner, not that relocation alone does.

## What changes

1. **Evaluation leaves training unchanged (resume patch v7).**
   - `fgs_eval` saves the torch CPU, torch CUDA, Python and NumPy random states before rendering the test set, and
     restores them afterwards. Nothing else changes.
   - The end-of-training evaluation is not affected, since no training follows it.
2. **Fresh seeds: 10, 11, 12.** Experiment 1's arm results on seeds 0–2 have been seen. New seeds make this a
   replication rather than a second reading of the same runs.
3. **Uninterrupted reference runs in the same jobs.**
   - For each (scene, seed), one uninterrupted `up` run to 30,000 without evaluation, as in upbench.
   - Same build and patch, so the fix is present but evaluation is off.
   - Experiment 1 took its reference from an earlier job, built before the resume patch existed.
4. **View traces for every run:** prefixes, all arms, and reference runs.
5. **Check 1, two parts:**
   - **1a (new, exact):** `ctl`'s view trace over steps 0–29,999 must equal its reference run's, for every (scene,
     seed). With the fix, nothing else in `ctl` draws from the generators differently.
   - **1b (unchanged):** `ctl`'s mean final PSNR and final N within the reference runs' range of their mean, per scene.
   - Both must pass. If either fails, Experiment 1R is invalid and the arms are not compared.
   - Reported alongside, not as criteria: the per-seed `ctl` − `up` differences, their mean and 95% t interval, and the
     per-view differences on the views with the lowest PSNR.

## Before launch: engineering and smoke tests

Each must pass before the experiment is submitted. The results go in this file.

- [ ] **E1. Resume patch v7.**
  - The random-state save and restore around `fgs_eval`.
  - `fgs_eval` prints digests of the four random states before and after each evaluation.
  - The digests must match.
- [ ] **S1. Evaluation neutrality** (a repeat of adjudication part 1 on the patched code).
  - Room seed 0, two uninterrupted runs 0 → 8,000, one evaluating every 1,000 steps from 5,000 and one not.
  - Pass: identical view traces over steps 0–7,999.
  - Diagnostics only: N every 1,000 steps, and test metrics at 8,000.
- [ ] **S2. Resume smoke test (`exp1smoke`)**, rerun unchanged on the patched code. Every existing check must pass.
- [ ] **S3. Analysis.**
  - `analyze_ttq.py` gains check 1a (view traces) and reads the reference from the same job's `b_<scene>_up_s<seed>`
    runs.
  - It is tested on a fixture that contains a mismatched trace.

## Jobs and cost

- **Garden:** one job, 3 seeds × (prefix + 4 arms + reference), about 3 hours. Limit 4 hours.
- **Room:** one job per seed, since room continuations take about 27 minutes each, mostly evaluation. About 2 hours
  each; limit 2 h 45.
- **Timeout rule (as in Experiment 1):** any (scene, seed) block that does not complete is rerun whole (prefix, arms
  and reference) under the same configuration.
- **Total:** about 9–10 GPU-hours, plus about 0.5 for S1 and S2.

## How the result will be reported

- Experiment 1 stays invalid. Experiment 1R is reported as a separate experiment, designed after Experiment 1's
  descriptive results were seen. That is why it uses fresh seeds.
- If `mcmc` passes, the claim is that the combined MCMC policy reached target quality sooner on garden and room
  (seeds 10–12, H100), and nothing broader.
- If check 1 fails again with identical view traces, the remaining difference is something other than view order, and
  investigating it comes before any arm comparison.

## Out of scope

- No new arms and no MCMC ablations. Separating relocation, noise injection, regularisation and the opacity learning
  rate would be a later experiment, and only if `mcmc` passes here.
- Experiment 2 (EXP2_PLAN.md) stays frozen as a draft. Any experiment that evaluates during training would need the
  v7 fix too.

## Open questions for review

1. **A dedicated generator for the view sampler.**
   - The alternative to the fix above is to give `RandomSequentialSampler` its own torch generator, seeded from the run
     seed. Every arm would then see the same view order, whatever else draws random numbers.
   - The Experiment 1 smoke test found that `stop` and `mcmc` saw a different view order from the uninterrupted run
     after the branch. That is not yet explained: MCMC's sampling runs on the GPU, so its draws should not reach the
     CPU generator.
   - Common view orders would reduce the noise in the paired arm comparisons. But they depart from stock NeRFICG
     behaviour, and they would change the reference runs too.
   - Default in this draft: no. Use the minimal fix, and record traces so that the arms' view orders are measured, not
     assumed.
2. **Check 1b's false-fail risk.**
   - With identical view traces, `ctl` and its reference differ only by the order of atomic additions. Even so, on
     unstable views that noise is large: one build's image 35 ranged from 27.4 to 31.1 dB across seeds.
   - The range rule at three seeds can fail by chance. Requiring it on both scenes roughly doubles that risk.
   - Default in this draft: keep 1b unchanged, for continuity with Experiment 1. The alternative is to make 1a the
     validity condition and report 1b and its interval descriptively.
3. **Seeds.** Fresh seeds 10–12 (default) or seeds 0–2 again? Fresh seeds give an independent replication. The same
   seeds would allow a run-by-run comparison with Experiment 1.
