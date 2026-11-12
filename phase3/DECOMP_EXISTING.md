# Fewer steps or cheaper steps? From the completed 1R and 1A runs (2026-11-12)

**Descriptive and post hoc.** No new training, not pre-registered, no decision rule. Script:
[decomp_existing.py](decomp_existing.py). Full per-seed output: [decomp-existing-analysis.md](decomp-existing-analysis.md).
It replaces Experiment 3, which was shelved after review ([EXP3_PLAN.md](EXP3_PLAN.md)).

## Method

The window runs from the branch at step 5,000 to the first crossing of the target. The target is the one used in 1R and
1A: `ctl`'s final PSNR minus 0.05 dB. Over that window:
- T′ = training-step seconds;
- S′ = steps, with the crossing step interpolated the same way as the crossing time;
- c = T′/S′, the mean seconds per step.

The saving then splits exactly:

ln(T′ctl/T′arm) = ln(S′ctl/S′arm) + ln(c_ctl/c_arm)

The **step share** is the first term's share of the total.

This is an accounting identity. It says where the time went, not why. Gaussian count and policy can each change both
factors.

## Results (medians over the seeds where both arms reach the target)

| | garden 1R | garden 1A | room 1R | room 1A |
|---|---|---|---|---|
| `mcmc`: time saved | 37% | 40% | 41% | 43% |
| `mcmc`: steps, ratio to `ctl` | 0.71 | 0.68 | 0.63 | 0.60 |
| `mcmc`: cost per step, ratio to `ctl` | 0.88 | 0.88 | 0.95 | 0.95 |
| `mcmc`: step share | 0.72 (0.71–0.73) | 0.74 (0.71–0.76) | 0.92 (0.89–0.93) | 0.91 (0.89–0.93) |
| mean N in window, `ctl` / `mcmc` (M) | 3.97 / 3.08 | 4.01 / 3.10 | 1.29 / 0.95 | 1.29 / 0.95 |
| `oracle` (MCMC at `ctl`'s count): time saved | 25% | – | 35% | – |
| `oracle`: steps / cost per step, ratio to `ctl` | 0.66 / 1.14 | – | 0.57 / 1.15 | – |

**1. Most of `mcmc`'s saving is fewer steps.**
- **Garden:** about 0.72 of the log saving comes from fewer steps. Room: about 0.92.
- **Consistency:** every one of the 12 pairs falls within 0.71–0.76 on garden and 0.89–0.93 on room.
- **Cost per step:** only 3–13% lower, although `mcmc` holds 22–26% fewer Gaussians. So per-step cost is far from
  proportional to N here.

**2. At the control's count, the saving remains, and comes entirely from fewer steps.**
- **The oracle:** the same MCMC policy, grown to `ctl`'s count; its mean N in the window is within 3% of `ctl`'s.
- **Time saved:** 25% on garden and 35% on room, in all 6 pairs (21–48% per pair).
- **Steps and cost:** it needs fewer steps (median 34% fewer on garden, 43% on room), while each step costs 13–16% *more*.
- **What this rules out:** "MCMC is faster only because it has fewer Gaussians". It is a controlled comparison at
  roughly matched count, not just accounting. It holds for this bundle (relocation, noise, regularisation, opacity
  learning rate, no resets), on two scenes, three seeds each.
- **Not tested:** why the oracle's steps cost more at a similar count. Possible reasons include its N trajectory within
  the window, Gaussian size or overlap, and relocation overhead.

**3. The ablations change the step count, not just the cost.**
- **Removing relocation, noise or regularisation:**
  - per-step cost goes *down* (room: 0.73–0.93 of `ctl`'s, against `mcmc`'s 0.95);
  - the step count goes *up* sharply, and often the target is never reached (garden: 7 of 9 arm-seeds).
- **Room `mreg`:** step share 0.09 (−0.19 to 0.63). What little saving remains there is almost all cheaper steps.
- **`mlr`:** indistinguishable from `mcmc` (step share 0.72 garden, 0.95 room).
- **`creset`:** more Gaussians (about 4.7 M against 4.0 M on garden), and costlier steps (1.07–1.12 of `ctl`'s).

**4. `stop`:**
- **Scope:** standard densification frozen at 5,000.
- **Reaching the target:** only room seed 10 does, and with 13% *more* steps than `ctl`.
- **What it suggests:** a small population without MCMC's policy does not get the step saving.

## What this supports, and what it does not

**Supported, on garden and room only:**
- MCMC's time-to-quality gain in Faster-GS is mainly a convergence-in-steps effect.
- It persists at matched Gaussian count.

This is the "count vs policy" question that the literature review found open. The existing runs answer it for two
scenes, which is enough for a report, not for a general claim.

**Not supported:**
- Generality beyond two scenes.
- Which mechanism makes the steps fewer.
- Anything about MCMC from step 0, or the switch time.
- The cost-per-step figures are means over windows that differ between arms. They are not a controlled rasterisation
  benchmark.
