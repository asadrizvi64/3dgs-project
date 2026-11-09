# Experiment 1R results (2026-11-09)

Pre-registration: [EXP1R_PLAN.md](EXP1R_PLAN.md), frozen at c3ea481 before launch. Full analysis output:
[exp1r-analysis.md](exp1r-analysis.md), from the command fixed in the plan:

```
python phase3/analyze_ttq.py phase2/logs/gs-x1r-garden-4559237.out phase2/logs/gs-x1r-room-s10-4559238.out \
    phase2/logs/gs-x1r-room-s11-4559239.out phase2/logs/gs-x1r-room-s12-4559240.out --exp1r --seeds 10 11 12
```

**Jobs:**
- garden: 4559237, seeds 10–12, 3:44 h;
- room: 4559238, 4559239 and 4559240, one per seed, about 1:50 h each.

All four completed with no failed run, so no reruns. The runs are in `experiment_log.csv`.

## Outcome

**`mcmc` passes the pre-registered decision rule on both scenes. `stop` fails. The oracle reference passes.**

Pre-registered reading: the combined MCMC policy reached the target quality sooner than standard Faster-GS on garden
and room, seeds 10–12, on an H100. The policy combines a fixed population with relocation, noise injection, opacity
and scale regularisation, and opacity learning rate 0.05. The result does not say which part is responsible.

## Validity

| check | garden | room |
|---|---|---|
| Validation (runs complete, same metrics at the branch, each arm did what it says) | pass | pass |
| 1a: `ctl`'s training views equal the reference's over 0–29,999 | pass, 3 of 3 seeds | pass, 3 of 3 seeds |
| 1b: every arm loaded exactly the saved state | pass | pass |
| 1c: every evaluation (25 per branched run) restored the random states, model and optimizer state, flag and mode | pass | pass |
| 1d: configurations, and the settings in effect after loading | pass | pass |

Passing 1a–1d establishes implementation fidelity. It does not establish statistical equivalence of final PSNR between
branched and uninterrupted training.

**1e, reported only: `ctl` − reference, final PSNR (dB)**

| scene | seed 10 | seed 11 | seed 12 | mean | pilot same-seed SD |
|---|---|---|---|---|---|
| garden | −0.099 | +0.059 | −0.052 | −0.031 | 0.050 |
| room | +0.117 | +0.181 | −0.092 | +0.068 | 0.066 |

- Final N differs by between −0.49% and +0.91%.
- On room seed 11 the difference is 2.7 times the pilot's same-seed SD, the largest of the six. With identical views,
  identical starting state and configuration, and evaluation shown not to touch training, the remaining sources are GPU
  non-determinism and branching itself. Three seeds cannot separate those two.
- Run order: on both scenes the reference ran first for seeds 10 and 12 and last for seed 11. The differences show no
  pattern by order, but three seeds cannot test one.

## Decision rule

| arm | scene | median ΔT | T lower than `ctl` | mean ΔPSNR (dB) | mean ΔLPIPS | one-view failures |
|---|---|---|---|---|---|---|
| **`mcmc`** | garden | **31.4%** (25.2, 31.4, 33.5) | 3 of 3 | +0.193 | −0.0047 | 0 |
| | room | **35.5%** (31.3, 35.5, 38.9) | 3 of 3 | +0.297 | −0.0103 | 0 |
| `stop` | garden | not reached in any seed | 0 of 3 | −0.143 | +0.0083 | 0 |
| | room | reached in 1 seed (5.8%) | 1 of 3 | −0.093 | +0.0040 | 2 |
| oracle (reference) | garden | 21.2% (17.3, 21.2, 31.5) | 3 of 3 | +0.350 | −0.0120 | 0 |
| | room | 30.2% (30.0, 30.2, 41.9) | 3 of 3 | +0.507 | −0.0153 | 0 |

- `ctl` had 1 one-view failure (garden seed 10, image 22, −3.5 dB).
- `mcmc` meets every criterion: median ΔT ≥ 10% on both scenes, T lower in 6 of 6 pairs, both quality limits, and no
  more one-view failures than `ctl`.
- `stop` misses six of the eight criteria. It only meets room's ΔPSNR limit (−0.093 dB against −0.10).

## Sensitivity (pre-registered, section 6 of the plan)

`mcmc` reached the target before `ctl` in **every pair at every target**:

| target | garden, `ctl` / `mcmc` T (s) | room, `ctl` / `mcmc` T (s) |
|---|---|---|
| −0.05 dB (primary) | 135 / 101, 145 / 99, 172 / 115 | 123 / 85, 140 / 90, 142 / 87 |
| −0.10 dB | 129 / 97, 142 / 97, 166 / 113 | 110 / 83, 130 / 80, 139 / 85 |
| −0.20 dB | 123 / 92, 113 / 87, 132 / 108 | 98 / 80, 110 / 77, 123 / 77 |
| −0.05 dB and stays | 169 / 101, 173 / 99, 225 / 137 | 123 / 85, 140 / 90, 142 / 87 |

By the plan's own reading, the result is **not sensitive to target selection**.

## Descriptive, not part of the decision

- **`mcmc` also ends higher.** Its final PSNR is 0.19 dB (garden) and 0.30 dB (room) above `ctl`, with about 27% fewer
  Gaussians (3.08 M vs 4.17 M; 0.95 M vs 1.35 M). Its training time to 30,000 is 12% and 7% lower. Part of the earlier
  crossing therefore comes from a higher curve, not only a faster one.
- **The oracle reaches more quality at the same size.** It ends with the same Gaussian count as `ctl` but 0.35 and 0.51
  dB higher. That points at the MCMC training policy, not the population size alone, as a source of the quality gain.
  This is a comparison of two arms on three seeds, not a tested claim.
- **Monitoring cost is the same across arms.** Elapsed time from 5,000 to 30,000, including the 25 evaluations, is
  883–1006 s on garden and 1418–1454 s on room. Evaluation dominates room.
- **Peak memory (69–71 GiB) is not a training figure.** It comes from cuDNN's algorithm search during evaluation
  (`x1ra`, job 4559014). Training itself peaks at about 4–7 GiB.
- **The other arms' view orders.** The `stop`, `mcmc` and `oracle` view orders differ from the reference from the first
  reshuffle after the branch: step 5,152 on garden (32 × 161 views) and 5,168 on room (19 × 272), in every seed.
  `ctl`'s never does. The arms use the CUDA generator differently between the branch and that reshuffle (densification,
  none, MCMC sampling), so the view order is part of each arm's random draw. This answers open question 1 of the
  plan: the pattern recurs, and it has an explanation. A dedicated sampler generator was not adopted, and this changes
  no check.
- **Experiment 1, for context only.** Experiment 1 is invalid, and stays so. Its descriptive `mcmc` result (target
  30–43% sooner) is in the same direction and of similar size. Experiment 1R, on fresh seeds with observational
  evaluation, is the result that counts.

## Limits

- Two scenes, three seeds each, one GPU type (H100), one build (`FasterGSUp`, dense Adam). No claim beyond these.
- T is measured against `ctl`'s own final quality, minus 0.05 dB, in training-only time (CUDA events, evaluation
  excluded).
- The pass is for the combined MCMC policy. Separating relocation, noise injection, regularisation and the opacity
  learning rate needs ablations: a new experiment with its own pre-registration (plan, "Out of scope").
- The revisions to check 1 and smoke test A were made after the noise pilot's results were seen, and before any
  Experiment 1R arm ran (plan, "Revisions after the smoke test and pilot").

## Next step (not decided)

By the plan's outcome rule, a non-oracle arm passing points Experiment 2 at that mechanism. The natural first step is
the MCMC ablation the review called the interesting next question. It would test which components of the policy
produce the faster, higher curve. It needs its own pre-registration, and [EXP2_PLAN.md](EXP2_PLAN.md) stays frozen
until then.
