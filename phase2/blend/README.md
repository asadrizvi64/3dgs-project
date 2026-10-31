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

# Step 2 (step 5b): exact ellipse culling per subtile — slower, not adopted

2026-10-31. Patch: [`../ellipse_cull.patch`](../ellipse_cull.patch), built as FasterGSPlusCull (A+B+C plus the patch; `FGS_ELLIPSE_CULL=0` turns it off). Build and tests: job 4503656. Timing: jobs 4503657 (garden) and 4503658 (room), stage `rbbench` with `CMP_VARIANT=FasterGSPlusCull`. All 12 runs are in `experiment_log.csv`.

**What the patch changes:**
- **Forward (`blend_cu`):** a splat that passes the bounding-box test is also tested against the warp's 8×4 subtile with the exact ellipse test, before the ballot.
- **Backward (macro kernel):** each staged splat gets one bit per subtile (bounds and ellipse both hit). A row is skipped when neither of its subtiles is hit (warp-uniform), and a lane when its own subtile isn't.
- **Margin:** the test's power threshold is raised by 0.01 (alpha cut-off × e^−0.01, about 1%), far above the rounding misses step 1 counted.
- **Zero-bucket guard** in the taming backward, as in the readback patch.

**Tests:** 92 unit tests and all 25 equivalence cases passed. Images are bit-identical to A+B+C, and gradients are within 10⁻⁵ (the empty-scene cases now pass too, because of the guard).

**Timing** (timer on, the benchmark protocol; seeds 0 / 1 / 2, arms interleaved with the order rotated by seed; within each seed both arms saw the same training views):

| scene | wall time A+B+C → culling | wall change | per-step clock change | test PSNR change |
|---|---|---|---|---|
| garden | 195.73 → 202.51 s | **+3.46%** (+3.18 / +3.64 / +3.56) | +3.47% | +0.008 dB (+0.033 / −0.006 / −0.003) |
| room | 121.79 → 127.99 s | **+5.09%** (+6.18 / +4.67 / +4.40) | +5.15% | −0.231 dB (−0.277 / +0.077 / −0.491) |

- **Culling makes the step about 0.2 ms slower on both scenes**, on every seed. Step 1's estimate was up to 0.23 ms *faster*. The slowdown is about as large as the hoped-for gain.
- **Likely cause (not measured per kernel):** the estimate assumed kernel time scales with the pairs evaluated. But the added test runs for every bbox-passing pair, which is about 30 instructions plus a `logf` and two divisions, in the forward ballot stage and six times per splat in the backward staging. The removed pairs were cheap: the forward loop skips them in a few cycles and the backward drops only about 10% of rows. The kernels are evidently not limited by the evaluations that culling removes.
- **Quality:** garden is unchanged. Room is 0.23 dB lower on average, driven by two seeds (−0.28, −0.49 dB). In step 4c, two builds whose images were also bit-identical differed by at most 0.14 dB on room. Three seeds can't tell whether this is seed noise or something the synthetic equivalence test misses. Since the patch is slower anyway, this was not investigated further.
- **Decision: not adopted.** The subtile-aligned backward layout (the follow-up that depended on this step landing near its bound) is dropped as well. Any further culling would need a much cheaper test, for example one precomputed per splat and tile in `create_instances` instead of per warp in the blend loop.

# Step 5c: Nsight Compute on the blend kernels — why culling was slower

2026-10-31, job 4503800 (stage `ncublend`; the first attempt, 4503798, failed because `/projects` is read-only on compute nodes). Each run trains A+B+C or FasterGSPlusCull normally to step 20,000 (garden and room, seed 0). Nsight Compute then captures the forward blend and the macro backward of 2 steps with `--set full`. Script: [`../analyze_ncublend.py`](../analyze_ncublend.py); full tables, including stall reasons: [`ncublend-analysis.md`](ncublend-analysis.md). Reports: `$DATA/ncu/4503800/`.

**Clock:** ncu locks the SM clock (1.6 GHz) and runs kernels one at a time, so its durations are about 20% above training values. Compare arms, not absolute times.

| per kernel, mean of 2 launches | garden A+B+C → culling | room A+B+C → culling |
|---|---|---|
| forward: duration | 685 → 972 µs (**+42%**) | 750 → 907 µs (**+21%**) |
| forward: instructions executed | +7.4% | +2.3% |
| forward: theoretical occupancy | **75% → 37.5%** | **75% → 37.5%** |
| forward: issue slots busy | 75% → 54% | 80% → 63% |
| backward: duration | 1597 → 1643 µs (+2.9%) | 1423 → 1463 µs (+2.8%) |
| backward: instructions executed | +5.7% | +4.4% |
| backward: issue slots busy | 80% → 83% | 80% → 82% |
| backward: registers / theoretical occupancy | 71 → 64 / 44% → 50% | same |

Converted to the training clock, the per-step totals (forward + backward) come to about +0.27 ms on garden and +0.16 ms on room. That is in line with the +0.2 ms per step measured in step 5b.

**Why the forward pass got slower: half the occupancy, from a compiler artifact.**
- The culling kernel is limited to 4 blocks per SM by **barriers**, where the original allows 32. Registers (40) and shared memory are unchanged.
- In the SASS, the culling variant contains a `WARPSYNC.COLLECTIVE` … `BAR.SYNC R14, R14` fallback: a block barrier whose ID is held in a register. The original uses only `BAR.SYNC 0x0`. The extra branch around the ellipse test appears to keep the compiler from proving that the warp is converged at `__syncthreads_and` / `block.sync`, so it emits this collective path. The occupancy calculation then reserves the full barrier set.
- The variant also spills one register to the stack (`STACK:8`).
- With occupancy halved, the forward kernel can no longer hide its latency: issue slots busy fall from 75–80% to 54–63%.

**Why the forward pass would not be faster even at full occupancy:** it executes 2–7% **more** instructions. SIMT runs the test whenever any lane's splat passes the bounding box, which is nearly every 32-splat chunk. The test costs about 50 instructions per chunk (`logf`, two reciprocals, the edge projection). The evaluations it removes are about 19% of a handful of pairs per chunk, at about 20 instructions each.

**Why the backward pass got slower:** it is issue-bound (80% of issue slots busy; the top stall is `not_selected`, meaning eligible warps are waiting for an issue slot). So its time follows instruction count, and culling adds 4–6% instructions. Six rectangle tests per staged splat plus a per-lane cell check in every row cost more than the roughly 10% of rows they skip, many of which the `j >= lim[r]` check already skipped cheaply. The better occupancy (fewer registers) can't help an issue-bound kernel.

**What limits the A+B+C kernels:**
- **Backward (about 2/3 of blend time):** issue-bound at about 80% of peak issue, IPC 3.2 of 4. Only fewer instructions per pixel-splat evaluation will speed it up. Occupancy (44%, register-limited) and memory (L1 62–69%, DRAM 9–12%) are not the limit.
- **Forward:** 75–80% of issue slots busy at 68% achieved occupancy. The largest stall is `barrier` (2.9–4.8 warps per issued instruction): warps wait at the per-batch `__syncthreads_and` / `block.sync` for the slowest warp in the tile.

**Implications for further blend work:**
- **Culling** only pays off if the per-subtile test is nearly free inside the blend loop. One option is to compute a 6-bit subtile mask once per tile instance in `create_instances`, which already runs the per-tile ellipse test, and store it with the instance; blend kernels would then only read bits. It would also have to keep the forward kernel's barriers at fixed IDs: check the SASS for `BAR.SYNC 0x0` only and the ncu block limit before timing.
- **Backward instruction count** is the main lever: the per-row exponent, alpha and gradient arithmetic, and the reduction. Counting instructions per evaluation from the SASS is the next measurement if blend work continues.

# Step 5d: where the backward kernel's instructions go

2026-10-31, from the A+B+C Nsight Compute reports of job 4503800 (step 20,000, seed 0). Per-instruction SASS counts were exported with `ncu --page source --print-source sass` (`../logs/run_4503800/<scene>.bwd_sass.csv`, one launch each) and grouped into code regions by [`../analyze_bwd_sass.py`](../analyze_bwd_sass.py). Full tables: [`bwd-sass-analysis.md`](bwd-sass-analysis.md). Because the kernel is issue-bound (step 5c), instruction share is a direct proxy for time share.

| region (macro backward, mode 3) | runs | garden share | room share | warp instructions |
|---|---|---|---|---|
| staging: splat staging, checkpoint loads | once per bucket | 4.7% | 4.1% | ~450 per bucket |
| loop head: splat from shared memory, flags | every splat step | 8.6% | 7.3% | 29 per step |
| row checks (6 unrolled rows) | every step | 6.5% | 5.7% | ~22 per step |
| **evaluate:** delta, exponent, `expf`, alpha cut-off | per row where any lane evaluates | **17.9%** | **20.2%** | 17 per row |
| **gradient:** contributing-pixel math | per row where any lane contributes | **22.3%** | **28.2%** | 30 per row |
| ballot of `used` | every step | 0.9% | 0.8% | 3 per step |
| **reduce:** 9 sums × 5 `shfl_down` + adds, 9 global atomics | steps with any contribution (92.5% / 94.5%) | **36.7%** | **31.8%** | **132 per step** |
| loop tail | every step | 2.4% | 2.0% | 8 per step |
| total, one launch | | 1078.7M | 1107.8M | 338 / 398 per step |

- **The per-splat warp reduction is the largest single cost: about a third of the kernel.** Each splat step that any lane contributed to reduces 9 values (mean2d 2, conic 3, colour 3, opacity 1) across 32 lanes. That takes 45 shuffles, 46 adds and 9 atomics, 132 instructions. It runs on 92–95% of steps. One step evaluates only 3.6 (garden) to 4.7 (room) of its 6 rows on average, so the reduction costs more than the rows' own arithmetic.
- **Evaluation and gradient together are 40–48%.** At 17 + 30 instructions per row, they are fairly tight already.
- **Staging is cheap in instructions (4–5%) but takes 15–17% of the stall samples:** it waits on the checkpoint and gradient-image loads.
- **Phase 1's transposing reduction is not the fix.** On synthetic scenes it barely changed the instruction count, and its divergent per-lane atomics added stalls ([results/phase1/macro_backward.md](../../results/phase1/macro_backward.md)).

**Candidate: a deferred reduction.** Instead of reducing each splat across the warp right away:
- each lane writes its 9 partial sums for the step to shared memory (9 stores);
- every K steps, each lane sums one (splat, value) column over the 32 lanes (on average 9 loads + 9 adds per step and lane) and issues the atomic.

That is about 30 instructions per step instead of 132, which would cut the backward kernel's instructions by roughly 25–30%. At the measured 1.2–1.3 ms per step for this kernel, that is up to about 0.3 ms, or 5–7% of the training step. Costs and risks:
- shared memory: K × 32 × 9 floats per warp, 9.2 KB for K = 8, so about 37 KB per 4-warp block. That would lower the block limit from 7 to about 4 per SM. Whether occupancy then matters for this issue-bound kernel has to be measured.
- bank conflicts in the column reads (needs padding);
- one more `__syncwarp` per flush;
- a different summation order, so the gradients are not bit-identical: the same 10⁻⁵ tolerance as the other gradient tests applies.

**Smaller candidates in the row body:**
- an incremental exponent down a lane's column (dy changes by a constant 2 px per row band);
- `exp2` on a pre-scaled conic.
Each would save a few of the 17 evaluate instructions, about 3–5% of the kernel together.

# Step 5e: deferred warp reduction — slower, not adopted

2026-10-31. Patch: [`../deferred_reduce.patch`](../deferred_reduce.patch), built as FasterGSPlusDR (A+B+C plus the patch; mode bit 32, `FGS_DEFERRED_REDUCE=0` turns it off).
- **Jobs:** build and tests 4504225; Nsight Compute 4504226; timing 4504227 (garden) and 4504228 (room). The first build (4504195) failed: 8 steps per flush exceeded the 48 KB static shared-memory limit per block.
- **Design:** each lane stores its 9 per-splat sums (11 with |mean2d|) to shared memory. Every 7 steps (5 with |mean2d|), each lane sums one (splat, value) column over the 32 lanes, reading in rotated order `(i + lane) & 31` to avoid bank conflicts, and issues that column's atomic. 63 or 55 columns fill 2 rounds of 32 lanes. 4-warp blocks only.

**Tests:** 92 unit and 25/25 equivalence cases passed. Images are bit-identical. Gradients are within 1.1 × 10⁻⁶ in the macro cases, the same level as A+B+C against itself (heavy overlap, partial tiles and |mean2d| included).

**Timing** (timer on; seeds 0 / 1 / 2, interleaved with the order rotated by seed; identical training views per seed):

| scene | wall time A+B+C → deferred | wall change | per-step clock | test PSNR change |
|---|---|---|---|---|
| garden | 193.42 → 206.95 s | **+7.00%** (+6.58 / +8.69 / +5.74) | +7.08% | +0.003 dB |
| room | 122.60 → 133.96 s | **+9.26%** (+9.57 / +8.78 / +9.43) | +9.23% | +0.066 dB |

Quality and LPIPS unchanged. Slower on every seed.

**Nsight Compute, macro backward at step 20k** (full tables: [`ncu-dr-analysis.md`](ncu-dr-analysis.md)):

| | garden A+B+C → deferred | room A+B+C → deferred |
|---|---|---|
| duration (ncu clock) | 1603 → 2161 µs (**+35%**) | 1415 → 1895 µs (**+34%**) |
| warp instructions executed | **+1.4%** | **+2.6%** |
| registers per thread | 71 → 84 | 71 → 84 |
| theoretical / achieved occupancy | 44% / 39% → **25% / 23%** | 44% / 38% → **25% / 22%** |
| issue slots busy | 80% → 60% | 80% → 61% |

Executed instructions by opcode (garden, one launch, from `../logs/run_4504226/garden_abcd.bwd_sass.csv`):

| | A+B+C | deferred | change |
|---|---|---|---|
| SHFL | 132.9M | 0 | −132.9M |
| FADD | 187.9M | 84.3M | −103.6M |
| global atomics (REDG, warp instructions) | 26.6M | 1.0M | −25.6M |
| integer address math (IMAD, LOP3, LEA, IADD3, VIADD, SHF) | 92.0M | 295.5M | **+203.5M** |
| LDS / STS | 48.6M | 106.5M | +57.9M |

**Why it is slower:**
1. **The instruction savings were eaten by address arithmetic.** The shuffles, adds and atomics went away as planned (about −260M). But the rotated column read `col[(i + lane) & 31]` needs its own address computation for each of the 32 loads, about 3 integer instructions per load, about +200M. My estimate of about 30 instructions per step counted only the loads and adds. This is a flaw in the implementation, not in the idea.
2. **Occupancy fell from 44% to 25%.** At 47 KB of shared memory, only 4 blocks of 4 warps fit per SM, and the flush code raised registers from 71 to 84. The kernel was issue-bound with about 2.4 eligible warps per scheduler. With fewer warps, issue slots busy fell from 80% to 60%, so even equal instruction counts take a third longer.

**Decision: not adopted in this form.** A revised version would need to fix both causes:
- **Address math:** pad the column to stride 33 and fully unroll the 32 loads, so each is one `LDS` with an immediate offset.
- **Occupancy:** a 3-step window (27 columns, 1 round). That is about 11 KB per block instead of 32 KB, so shared memory no longer limits occupancy; registers must stay near 71.

By the same opcode accounting, that would be roughly −250M + 100M ≈ −14% backward instructions at unchanged occupancy, about 2–3% of the step. This estimate is untested, and smaller than the original 5–7% hope.
