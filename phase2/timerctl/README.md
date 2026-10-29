# Timer on/off control: does the A+B+C advantage depend on NeRFICG's per-step timer?

Phase 2 step 4b, 2026-10-29: jobs 4485370 (garden, node c136) and 4485371 (room, node c79). Built from [`../build_plus.sbatch`](../build_plus.sbatch) with `STAGES=timerctl`; analysis by [`../analyze_timerctl.py`](../analyze_timerctl.py). The full output is in [`timerctl-analysis.md`](timerctl-analysis.md).

**Design**

- Arms: stock Faster-GS and FasterGSPlus A+B+C.
- NeRFICG's per-step timer either **ON** (the benchmark protocol: device sync after every training step) or **OFF** (`TIMING.ACTIVATE: false`).
- Seeds 0, 1 and 2: 12 runs per scene, all on one node, interleaved, order rotated by seed.
- Clocks:
  - **Wall time without profiling:** the device is synchronized only at iterations 0, 15k and 30k, so this includes everything between steps.
  - **Per-step clock:** the benchmark's summed CUDA events per training step (SEGCLOCK).
- Every run is in `experiment_log.csv` (role `timer-control`).

**Checked first** (by the analyzer; all passed):
- every run has both wall windows, the clock record and its results;
- timer-OFF runs wrote no NeRFICG timing and timer-ON runs did;
- **within each seed, all four runs saw exactly the same sequence of training views** (equal view hashes at 15k and 30k).

The same path was first tested end to end with a 1,200-iteration run on garden in job 4485369, which also passed the unit and switch checks; both readers accepted its real output.

## 1. How much training time does A+B+C save?

Paired by seed against stock under the same timer setting; per seed in brackets.

| scene | timer | wall time stock → A+B+C | wall saving | per-step clock saving |
|---|---|---|---|---|
| garden | ON (benchmark protocol) | 255.8 → 194.6 s | **−23.9%** (−23.2 / −24.9 / −23.6) | −24.2% |
| garden | OFF | 252.7 → 190.3 s | **−24.7%** (−24.7 / −24.7 / −24.6) | −24.9% |
| room | ON (benchmark protocol) | 151.1 → 121.8 s | **−19.4%** (−20.1 / −20.2 / −17.8) | −19.6% |
| room | OFF | 148.2 → 117.8 s | **−20.5%** (−20.8 / −20.4 / −20.3) | −20.6% |

These match the 9-scene benchmark (garden −24.7%, room −19.7%) to within about a point. The wall-time saving and the per-step clock saving agree to within 0.3 points.

## 2. Does the advantage change with the timer off?

**It grows slightly.** Switching the timer off helps both arms, A+B+C a little more:

| scene | stock, timer OFF vs ON (wall) | A+B+C, timer OFF vs ON (wall) |
|---|---|---|
| garden | −1.2% (−1.0 / −1.5 / −1.1) | −2.2% (−2.9 / −1.3 / −2.3) |
| room | −1.9% (−2.3 / −2.3 / −1.2) | −3.3% (−3.1 / −2.5 / −4.2) |

- **The timer is a small cost: 3–4 s per run.** The nsys windows showed about 1 ms of GPU idle time per step, about 30 s per run. Most of it therefore does not come from the timer's per-step sync, and it stays when the timer is off: it lies inside the steps.
- **The benchmark clock captures nearly all of the time.** Wall time exceeds the summed per-step clock by only 2.5–4.2 s with the timer on and about 1–2 s with it off, out of 120–256 s. The time between steps that per-step events leave out is under 2% of a run.
- **The benchmark numbers in steps 1–3 stand as measured** under the timer-on protocol. With the timer off, A+B+C's advantage is about one point larger on both scenes.

## 3. Do the matched views and image quality validate the comparison?

- **Views:** identical within every seed across all four configurations, on both scenes.
- **Garden quality:** the timer setting does not move test PSNR beyond seed noise: stock +0.010 dB (+0.077 / −0.013 / −0.035), A+B+C +0.004 dB (+0.007 / −0.028 / +0.033). A+B+C is +0.04 dB over stock under either setting.
- **Room quality:**
  - One single-view failure: stock, timer OFF, seed 1, test image 13, 12.5 dB below the same image in the other runs. It is the same view that failed in the benchmark's stock seed 1.
  - With images 13 and 16 removed from every room run, stock's timer OFF−ON difference is within noise (−0.07 / +0.08 / −0.09 dB).
  - **Unexplained:** A+B+C on room is higher with the timer off on all three seeds: +0.12 / +0.07 / +0.14 dB with all images, about +0.12 dB with views 13 and 16 removed. Garden shows nothing like it.
  - The timer changes only CPU/GPU overlap, not the computation, and runs are never bit-identical (atomics), so this may be noise that happened to line up. But three same-sign seeds are worth one check before the timer-off setting is used for quality claims: rerun A+B+C on room with the timer on and off for two more seeds, or compare the runs' parameters at 15k.
  - **For timing this does not matter;** the benchmark's quality results all use the timer-on protocol.
  - **Status (2026-10-29): left open by decision.** Not checked further; any later quality claim made with the timer off should run that check first.

## Summary

- A+B+C saves **24% of training time on garden and 19–20% on room**, measured by whole-run wall time as well as by the benchmark's per-step clock.
- **The advantage does not depend on the per-step timer;** without it, it is about one point larger.
- The comparison is matched (identical view sequences) and, on garden, quality-neutral with respect to the timer. On room, one quality shift remains unexplained (A+B+C about +0.12 dB with the timer off); it affects only timer-off quality, not any reported result.
