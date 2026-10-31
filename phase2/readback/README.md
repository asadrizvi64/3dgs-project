# Readback patch: does removing the per-step host reads speed up A+B+C?

Phase 2 step 4c, 2026-10-31. Patch: [`../fastergs_readback.patch`](../fastergs_readback.patch), built as the variant FasterGSPlusRB (A+B+C plus the patch). Build and equivalence test: job 4503021. Timing: jobs 4503022 (garden) and 4503023 (room). Every run is in `experiment_log.csv`.

**What the patch changes** (in the rasterizer's forward pass; nothing in the math):
- the two count reads after preprocessing (visible primitives, instances) become one host round trip instead of two;
- the host no longer waits on the memset stream: the main stream waits for it on the GPU instead;
- the bucket count is no longer read back. Buffers are sized for an upper bound (⌊(instances + 31 · tiles) / 32⌋) and both backward kernels read the real count on the GPU and skip the extra blocks;
- a zero-bucket guard in the taming backward. Stock Faster-GS launches zero blocks there when nothing is visible, which leaves a pending CUDA error. This is never reached in training.

**Equivalence test** ([`../tests/test_readback_equiv.py`](../tests/test_readback_equiv.py), 25 cases, all passed in job 4503021):
- forward images bit-identical to A+B+C;
- equal instance counts, and bucket counts within the bound;
- gradients within 10⁻⁵ relative error, with the stock-vs-stock difference printed for comparison;
- 30 calls in a row without host syncs;
- an empty scene.

Two earlier runs of the test failed, both times because of the test: wrong tile size (16×16 instead of 16×12), tied random depths (even stock is not bit-reproducible then), and a 10⁻⁶ gradient limit below the float-atomics noise on the 300k-primitive case (1.1–1.2 × 10⁻⁶).

## Design

- Arms: A+B+C (FasterGSPlus) and A+B+C + readback (FasterGSPlusRB).
- Seeds 0, 1 and 2; one node per scene; arms interleaved, order rotated by seed.
- Timer on (the benchmark protocol).
- Clocks: whole-run wall time (device synced only at 0, 15k and 30k) and the benchmark's per-step clock (`ms_it`, summed CUDA events).
- **Within each seed, both arms saw exactly the same training views** (equal view hashes at 15k and 30k).

## Results

Per seed (s0 / s1 / s2), readback against A+B+C:

| scene | wall time A+B+C → +readback | wall change | per-step clock change | test PSNR change |
|---|---|---|---|---|
| garden | 193.97 → 194.12 s | **+0.08%** (+0.61 / −0.31 / −0.06) | +0.05% | −0.006 dB (−0.020 / −0.005 / +0.006) |
| room | 123.41 → 122.52 s | **−0.72%** (−1.04 / −0.44 / −0.69) | −0.66% | −0.056 dB (+0.046 / −0.075 / −0.140) |

- **Garden: no effect.** The differences go both ways and are smaller than the run-to-run spread.
- **Room: about 0.9 s per run (0.7%) faster,** on all three seeds. That is roughly 0.03 ms per step.
- **Quality:** unchanged within seed noise. The forward pass is bit-identical, so the runs differ only through float-atomics order, as two runs of the same build do.
- **Memory:** peak allocation unchanged (6.4–6.5 GiB garden, 6.6 GiB room).

## What this means

- **The blocking reads are not where the idle time is.** The nsys windows showed about 1 ms of GPU idle per step (step 4a), and step 4b showed it isn't the timer either. Removing all three per-step host reads recovers about 0.03 ms on room and nothing on garden. The GPU queue stays far enough ahead that these syncs barely stall it.
- **Decision: not adopted into the main stack.** The gain is under 1% on one scene and nil on the other, and the patch adds upper-bound buffer sizing and an extra kernel check that every later rasterizer change would have to carry.
- The remaining idle time is more likely in launch-bound stretches inside the step (chains of small kernels) than in host syncs. Given step 4a's other findings, the blend kernels (about half of room's kernel time) are the larger lever.
