# Step 5g: where the preprocess backward kernel's time goes

2026-11-06. Data comes from Nsight Compute job 4537266 (`STAGES=ncupre`, [`../build_plus.sbatch`](../build_plus.sbatch)):

- The stack A+B+C+D (FasterGSPlusD, arm `abcd`), seed 0, steps 20,000–20,001, garden and room. These are not timing runs.
- Both preprocess kernels were captured with `--set full`: forward `preprocess_cu` and `preprocess_backward_cu`, 2 launches each.
- One backward launch was also exported per SASS instruction (`../logs/run_4537266/<scene>_abcd.pre_sass.csv`).
- Tables come from [`../analyze_ncupre.py`](../analyze_ncupre.py); full output is in [`ncu-pre-analysis.md`](ncu-pre-analysis.md).
- ncu locks the SM clock and runs kernels one at a time. Absolute durations therefore run high; compare kernels and ratios.

Step 4a found that preprocess backward was 20% of garden's kernel time at 20k under A+B+C. It was the largest kernel after Adam and the blend backward.

## Result: limited by scattered memory access, not by DRAM bandwidth or arithmetic

| garden (room) | forward preprocess | preprocess backward |
|---|---|---|
| duration | 271 µs (197) | **1213 µs (346)** |
| DRAM read + write | 517 MB (152) | 882 MB (228) |
| DRAM bandwidth reached | 1908 GB/s (769) | **727 GB/s (661)**, 30% (27%) of peak |
| L2 throughput (% of peak) | 58% (26%) | **79% (75%)** |
| SM throughput / issue slots busy | 38% / 39% | **7% / 4%** |
| global load / store sectors per request | 21.5 / 7.7 | **22.8 / 27.4** |
| top stalls (warps per issued instr.) | long_scoreboard 9.5 | **lg_throttle 59, long_scoreboard 54** |

- **The backward kernel moves less data than DRAM could carry in its run time.** It moves 1.7× the forward kernel's DRAM bytes but runs 4.5× longer. It reaches only 30% of DRAM bandwidth.
- **The L2 cache is the bottleneck, at 79% of its peak.** The cause is the number of L2 requests, not the number of bytes.
- **The kernel issues instructions only 4% of the time.** Warps wait on the load/store queue (lg_throttle) and on outstanding loads (long_scoreboard).
- **Stores are almost fully scattered: 27 of 32 possible sectors per request.**

**Why:**
- Each thread handles one Gaussian. It reads and writes that Gaussian's 15 SH-rest coefficients (`float3[15]`, 180 bytes) with 4-byte scalar accesses.
- Neighbouring lanes are therefore 180 bytes apart. Every warp-wide load or store touches about 32 different sectors.
- The SASS confirms this (garden, one launch):
  - 2.84 M `LDG.E.CONSTANT` and 2.57 M `STG.E` scalar accesses.
  - That is about 57 loads and 54 stores per active warp.
  - Together they account for 43% of stall samples.
  - The `FMUL`s that consume the loaded values take another 24%.
- Arithmetic is 72% of the instructions, but there are so few instructions that compute does not limit the kernel.

**Visibility:**
- Only 47,606 of 129,767 warps (37%) have a lane with a visible Gaussian (`n_touched_tiles > 0`). The other warps exit after one load.
- Active warps run 31 of 32 lanes on average, so visible Gaussians come in long contiguous runs. A warp-cooperative scheme would therefore waste little work on invisible lanes.

## What this suggests (not yet tested)

**Load and store the SH-rest blocks warp-cooperatively.**
- A warp's 32 Gaussians are contiguous, so their SH-rest coefficients form one 5,760-byte block, and their gradients form another.
- The warp can load the block with coalesced 16-byte accesses into shared memory. Each lane then computes from its own row, and the gradient block is written back the same way.
- The block size is a multiple of 16 bytes, so warps' blocks stay 16-byte aligned if the buffer is.
- Lanes with invisible Gaussians in an active warp would write their old gradient values, which are unchanged, or zero if the buffer is known to be zero. This must be checked against how the gradient buffers are initialised.

**Estimated ceiling:**
- At the forward kernel's bandwidth of about 1.9 TB/s, the same 882 MB would take about 0.46 ms instead of 1.21 ms on garden (ncu clocks).
- That ceiling is roughly 0.6–0.7 ms per step on garden and less on room, against steps of about 6.3 ms and 4.0 ms.
- This is an upper bound from bytes and bandwidth, not a measurement.
- Shared memory needed: about 5.6 KB per warp for coefficients plus the same for gradients, unless the gradients are written in place.

# Step 5h: coalesced SH-rest access in preprocess backward — garden −10.8%, room −3.7%

2026-11-06/07.

**Patch:** [`../sh_coalesce.patch`](../sh_coalesce.patch), built as FasterGSPlusDS, arm `abcds`, on top of A+B+C+D.
- Each warp copies its 32 Gaussians' SH-rest block (5,760 bytes) to shared memory with 16-byte loads.
- Each lane runs the SH backward in place on its own row. The function reads each band's coefficients before writing that band's gradients, in the original order of arithmetic.
- The warp then writes the gradient block back with 16-byte stores.
- Zeros are written for:
  - lanes whose Gaussian is invisible;
  - bands above the active SH degree.

  The gradient buffer is created with `torch::zeros` (`rasterization_api.cu`), so these zeros match what the original kernel left there.
- Warps with no visible Gaussian exit before touching memory, as before.
- The old path is kept for these cases:
  - SH degree 0;
  - fewer than 15 SH-rest coefficients;
  - unaligned buffers.

**Tests (build job 4537424):**
- 92 unit tests and 25/25 equivalence cases passed against FasterGSPlusD.
- Largest relative difference: 1.1 × 10⁻⁷, at the level of the reference's own run-to-run noise (9.9 × 10⁻⁸).

**Kernel (ncu job 4537425, step 20k, same protocol as step 5g; tables in [`ncu-pre-ds-analysis.md`](ncu-pre-ds-analysis.md)):**

| preprocess backward | garden A+B+C+D → +S | room A+B+C+D → +S |
|---|---|---|
| duration | 1213 → **341 µs (−72%)** | 346 → **94 µs (−73%)** |
| DRAM bandwidth reached (% of peak) | 30% → **85%** | 27% → **79%** |
| L2 throughput (% of peak) | 79% → 55% | 75% → 55% |
| DRAM read / write | 476 / 406 → 392 / 321 MB | 131 / 97 → 110 / 72 MB |
| global load / store sectors per request | 22.8 / 27.4 → 9.0 / 12.0 | 22.1 / 26.6 → 8.8 / 11.9 |
| stalls per issued instruction | 129 → 32 | 120 → 28 |
| registers / theoretical occupancy | 80 / 37.5% → 72 / 43.8% | same |

The kernel is now limited by DRAM bandwidth. Little time is left to gain in it without moving fewer bytes.

**Timing (jobs 4537426 garden and 4537427 room):**
- Step 5f protocol: timer on, 3 seeds, interleaved, order rotated by seed.
- Wall time is FGS_WALL over 0–30k. SEGCLOCK is the CUDA-event clock.

| scene | wall A+B+C+D → +S | change | per seed (s) | SEGCLOCK | PSNR |
|---|---|---|---|---|---|
| garden | 191.07 → **170.53 s** | **−10.75%** | −20.09 / −20.18 / −21.37 | 187.17 → 166.67 s (−10.95%) | 27.528 → 27.531 |
| room | 120.66 → **116.18 s** | **−3.71%** | −5.76 / −4.88 / −2.79 | 118.17 → 113.70 s (−3.78%) | 31.664 → 31.688 |

- All 6 seeds are faster. Quality is unchanged: SSIM and LPIPS are within seed noise, and the Gaussian counts match.
- On garden, the time saved per step (about 0.68 ms by the NeRFICG timer: 6.28 → 5.60 ms) matches the kernel saving measured under ncu.
- This is the largest single step in phase 2.
