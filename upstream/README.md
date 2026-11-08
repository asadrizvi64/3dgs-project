# Upstream contribution: coalesced SH-rest loads in `preprocess_backward_cu`

Candidate pull request to the official Faster-GS repository,
[nerficg-project/faster-gaussian-splatting](https://github.com/nerficg-project/faster-gaussian-splatting) (Apache-2.0).
Plan written 2026-11-08, before any of the jobs below ran.

## Why upstream main and not our base

All phase 1/2 results use 1144604 = upstream **c60ea26** (2026-10-10) plus our lean-Adam hooks (off in the stock arm).
Upstream has since added three kernel commits:

| commit | date | change | overlap with our work |
|---|---|---|---|
| 40f5271 | 2026-10-29 | exp2 in blending (−0.5·log2 e folded into the conic) | none |
| c83e7b8 | 2026-10-29 | SH-rest **gradients** written warp-wide via shared memory (coalesced 16-byte **stores**) | the store half of our S (phase2/sh_coalesce.patch) |
| d23bb78 | 2026-10-29 | gradient buffers `torch::empty`, zeroed in-kernel for invisible Gaussians | S relied on zero-initialised buffers; upstream no longer has them |
| 090e8ad | 2026-10-31 | README only | |

So **our 1.451× (step 6b) is against c60ea26, not against current upstream main**, and must not be quoted as a speedup over
upstream main. Upstream main still reads each Gaussian's 45 SH-rest floats with per-thread scalar loads 180 bytes apart
across the warp; the load half of S is not upstream.

## The change ([`sh_coalesced_loads.patch`](sh_coalesced_loads.patch), against upstream main 090e8ad)

- The warp loads its 32 Gaussians' SH-rest coefficients (contiguous in memory) into the shared buffer that c83e7b8 already
  allocates for the gradients, with 16-byte loads (scalar loads if the tensor is not 16-byte aligned). Skipped when the
  SH degree is 0 or no Gaussian in the warp is visible.
- `convert_sh_to_color_backward` takes the Gaussian's coefficient row directly and computes the gradients **in place**:
  within each SH band all coefficients are read before that band's gradients are written. Arithmetic unchanged.
- c83e7b8's coalesced store and d23bb78's zeroing are kept as they are. No extra shared memory, no new launch parameters.
- 2 files, +40 / −34 lines.

## Builds (`phase2/build_plus.sbatch` stages `upbuild`, `upequiv`, `upbench`, `upncu`)

- `FasterGSUpPure`: c60ea26 + [`upstream_main_kernels.patch`](upstream_main_kernels.patch) (= `git diff c60ea26 090e8ad -- FasterGSCudaBackend`), i.e. upstream main's backend exactly. `FasterGSUpPureS`: the same + the patch. Used for the equivalence test only.
- `FasterGSUp` / `FasterGSUpS`: the same two on our base 1144604 (lean-Adam code present but off: dense Adam, fp32 moments), with the clock hook, so they are timed exactly like our stock arm. No macro patch in any of these.
- `stock` (FasterGS, c60ea26 kernels) is included in the timing runs as context: it shows what upstream's own three commits gained.

## Measurements and decision rule (fixed before running)

1. **Equivalence** (`upstream/tests/test_sh_load_equiv.py`, both build pairs): forward images bit-identical; all gradients
   within 1e-5 relative error; SH-rest gradients bit-identical whenever the reference reproduces itself bit-exactly; all four
   SH degrees, odd primitive counts, mostly-off-screen scene, misaligned coefficient tensor.
2. **Kernel** (`upncu`): Nsight Compute on `preprocess_backward_cu` at step 20000, garden and room, seed 0: duration, DRAM
   bytes, L2 sectors per request, main + S vs main.
3. **End to end** (`upbench`): garden (1/4) and room (1/2), seeds 0–2, 30k, arms stock / up / ups interleaved in one job
   per scene, order rotated by seed; train time = SEGCLOCK (CUDA events); quality PSNR / SSIM / LPIPS; analysed with
   `analyze_bench.py --arms stock up ups --dense-arms stock up ups`.

**Propose the PR** if (1) passes, (2) shows a lower kernel duration on both scenes, and (3) shows main + S faster than
main in at least 5 of the 6 paired runs with no scene's mean ΔPSNR below −0.10 dB. If (1) and (2) hold but (3) does not,
the gain is a kernel-level one that is within end-to-end noise; whether to propose it then is the user's decision, and
the PR text would say so. If (1) fails, no PR.

Whatever the outcome, the PR text reports only measurements against upstream main, labels the GPU (H100 SXM, CUDA 12.8),
and does not cite the phase 2 stack figures as gains over upstream.

## Result (2026-11-08)

Jobs: build and equivalence 4552202; Nsight Compute 4552205; timing 4552203 (garden) and 4552204 (room). Seed 2 of both
scenes was rerun as jobs 4552280 and 4552281, with all three arms of that seed. In the original jobs, two runs
(garden s2 `ups`, room s2 `up`) finished training but failed to record their results. The cause was a `git merge` of
the cluster checkout while they were running, which rewrote `cluster/train_fastergs.sbatch` under the running bash
process. Analysis: [`phase2/logs/upbench-analysis.md`](../phase2/logs/upbench-analysis.md)
(`analyze_bench.py ... --rerun ...`, which lists the replaced seed-2 blocks). All runs are in `experiment_log.csv`.

**Against the rule above: (1) and (2) hold, (3) does not.**

| step | result | |
|---|---|---|
| (1) equivalence | 96/96 tests on both build pairs; largest relative gradient difference 8.3e-8; SH-rest gradients bit-identical | pass |
| (2) kernel, `preprocess_backward_cu` at step 20k | garden 646.3 → 631.6 µs (−2.3%), room 206.6 → 200.6 µs (−2.9%); DRAM throughput 82 → 84% and 76 → 79% | pass |
| (3) end to end, main + S loads vs main | garden 230.0 → 229.7 s (per seed +0.49 / −0.18 / +0.09%), room 145.0 → 145.4 s (−0.91 / +0.38 / −0.22%); faster in **3 of 6** pairs (5 needed) | **fails** |

- **Reading:** the change makes one kernel 2–3% faster. That kernel is about 0.3% of a training step, so the
  end-to-end difference is below run-to-run noise, which is about ±0.5% per pair here.
- **Quality:** unchanged within noise (ΔPSNR vs main: garden −0.025, room −0.050 dB; gradients are identical up to atomic order).
- **Decision:** by the rule, whether to propose this is the user's call, and a PR would describe it as a small kernel
  cleanup with no measurable end-to-end gain. My recommendation is not to open one.
- **Context, upstream's own gain:** upstream main (090e8ad) is 8.8% faster than c60ea26 on garden and 3.8% on room
  (6/6 pairs). Part of the phase 2 stack's gain over c60ea26 is now in upstream main. The 1.451× figure stays a
  comparison against c60ea26 only.
