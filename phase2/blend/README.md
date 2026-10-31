# Blend kernels, step 1: how much of the blend work is wasted?

Phase 2 step 5a, 2026-10-31, job 4503268 (node c45).
- **Counting build:** FasterGSPlusStats, which is A+B+C plus [`../blend_stats.patch`](../blend_stats.patch).
- **Stage:** `STAGES=blendstats` in [`../build_plus.sbatch`](../build_plus.sbatch).
- **Analysis:** [`../analyze_blendstats.py`](../analyze_blendstats.py); the full tables are in [`blendstats-analysis.md`](blendstats-analysis.md).
- **Runs:** one A+B+C run per scene (garden, room), seed 0, 30k steps. Counters were summed over 50-step windows starting at 2k, 5k, 10k, 15k, 20k and 29k.
- **Timing:** the counters add atomics and an ellipse test, so these runs' times are not benchmark values. Quality was normal: garden 27.48 dB, room 31.66 dB. Both runs are in `experiment_log.csv`.

## What was counted

**Forward (`blend_cu`):** one warp per 8×4 subtile, one lane per pixel. The kernel already skips a splat whose bounding box misses the subtile. The counters record:
- how many of the remaining splat-subtile pairs an **exact ellipse test** would also reject. This is the test Faster-GS uses per tile, generalized to the 8×4 rectangle;
- pixel evaluations, and how many of them pass alpha ≥ 1/255;
- idle lanes, whose pixel had already finished;
- idle chunks: warps whose pixels had all finished while the rest of the tile carried on.

**Backward (macro kernel):** one warp per bucket of 32 splats. Each warp step processes 6 rows of 32 pixels, and each row spans 2 pixel rows and both subtile halves. The counters record:
- launched vs active buckets;
- row executions, and the rows whose evaluating lanes all lie in ellipse-rejected subtiles;
- idle lanes, and splat steps with no work.

**Check:** the counting build's images are **bit-identical** to A+B+C's in the 23 equivalence cases it ran. The 2 empty-scene taming cases failed, as expected, because of the stock zero-block bug: this build doesn't carry the guard that the readback patch adds. Training uses the macro kernel and never has an empty view.

## Results at 20k steps (the pattern is the same in every window from 5k on)

| | garden | room |
|---|---|---|
| **forward:** bbox-passing pairs the ellipse test rejects | **18.8%** | **18.6%** |
| forward: pixel evaluations spent on those pairs | 19.1% | 18.8% |
| forward: evaluations that contribute (alpha ≥ 1/255) | 30.3% | 53.3% |
| forward: lane use (pixel finished early) | 93.7% | 93.8% |
| forward: idle chunks (warp done, tile not) | 6.1% | 6.9% |
| **backward:** row executions removable by the ellipse test | **9.9%** | **10.0%** |
| backward: lane evaluations in rejected subtiles | 19.0% | 18.7% |
| backward: active / launched buckets | 92.1% | 94.6% |
| backward: idle lanes in executed rows (pixel finished) | 5.1% | 6.6% |
| backward: splat steps with no work | 2.5% | 0.9% |

- **Most wasted work: bounding-box culling.** About 19% of the pairs the kernels evaluate can be rejected exactly. In the forward pass a pair is a whole warp iteration, so that much of the loop goes away. In the backward pass, a row mixes two subtiles, so only about 10% of row executions go away. The other 9% would need a lane layout in which each 32-lane row is one 8×4 subtile.
- **Finished pixels waste little:** about 6% idle lanes or chunks in the forward pass, under 7% idle lanes and under 3% empty steps in the backward pass. Steps 3 and 4 of the plan (per-subtile cut-off in the backward, per-warp early exit in the forward) would each recover at most a few hundredths of a millisecond. **Both are dropped.**
- **Even with exact culling, many evaluations fall below the cut-off:** only 30% (garden) and 53% (room) contribute. The rest lie inside the 8×4 rectangle's ellipse bound but miss the individual pixel. Finer culling than per subtile would cost more than it saves.
- **The exact ellipse test is not quite conservative:** every window has 0–12 contributions (out of about 4–9 × 10⁹) inside rejected pairs. This is float rounding at the 1/255 cut-off (the test against fast-math `expf`). Step 2 must widen the test's threshold slightly to keep images bit-identical, and the equivalence test will check that.

## Estimated gain of step 2 (exact ellipse culling per subtile)

The A+B+C kernel times at 20k come from step 4a: forward blend 0.55 / 0.60 ms and backward blend 1.30 / 1.16 ms (garden / room). If the removed work scales with kernel time:

| | garden | room |
|---|---|---|
| forward: −18.8% / −18.6% of the blend loop | ≤ 0.10 ms | ≤ 0.11 ms |
| backward, current lane layout: −10% of rows | ≤ 0.13 ms | ≤ 0.12 ms |
| **total, as a share of the step** (6.49 / 4.20 ms) | **≤ 3.5%** | **≤ 5.4%** |
| backward with subtile-aligned rows: −19% instead | ≤ 0.25 ms | ≤ 0.22 ms |
| total with that layout | ≤ 5.4% | ≤ 7.8% |

These are upper bounds. Part of each kernel's time is loading and checkpoint writes, which culling doesn't remove, and the ellipse test itself costs a few instructions per splat-subtile pair.

**Next:** step 2 with the current backward layout: the ellipse test in the forward pass, and per-subtile ellipse bits in the backward. Then an equivalence test (bit-identical images) and 3-seed timing on garden and room. The subtile-aligned backward layout is a follow-up if step 2 lands near its bound.
