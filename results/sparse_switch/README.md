# Selective Adam after densification, in the real Faster-GS trainer (garden, 3 seeds)

**Result:** we ran stock Faster-GS with dense Adam up to iteration 15,000 and the lean-Adam selective path afterwards.

- **Speed:** training-step time drops from **255.8 s to 230.8 s (−9.8%, 1.11×)** for the full 30k-step run. The second half alone goes from 141.0 s to 116.6 s (−17%).
- **Quality:** PSNR is **+0.084 dB** higher (paired, all 3 seeds positive: +0.082 / +0.101 / +0.070). SSIM is +0.0009 and LPIPS −0.0006.

This differs from selective Adam from step 0, which cost 0.09–0.13 dB (`results/optim`; the Faster-GS paper's own ablation reports −0.13 dB). So for this scene, deferring selective updates until Gaussian growth has finished keeps the speed benefit without the quality loss.

## Setup

- **Code:** FasterGS at lean-Adam commit 1144604 (nerficg 623ad16), garden, 1/4 resolution, 30k iterations, seeds 0/1/2.
- **Hardware:** one Capella H100 (c95, job 4480440). The six runs were interleaved base/switch per seed. The SM clock held 1980 MHz throughout, with no throttling.
- **Arms:**
  - `swbase`: lean-Adam patch present, flags off. This is the stock `adam_step` kernel.
  - `sw15k`: identical, plus `FGS_SPARSE_FROM=15000` ([patch](../../cluster/patches/fastergs_sparse_from.py)). From iteration 15,000 the FusedAdam updates only Gaussians whose footprint touches at least one tile in the current view (`n_touched_tiles > 0`), using the visibility mask the rasterizer already produces.
  - This is the trainer's cheap, broader rule. It is not the harness's nonzero-gradient rule from `results/ttq`, so this result stands on its own. Visible fraction at step 30k: 0.34–0.39.
- **Clocks:**
  - Training-step time: CUDA events around every `training_iteration`, summed (`logs/*.log.seg`, SEGCLOCK). It agrees with NeRFICG's own timer to within 1 s.
  - NeRFICG timer (`*_timings.txt`): 8.57 ms/it base vs 7.74 ms/it switch.
  - Total job duration is reported separately below.
- **Evaluation:** test-set metrics (8-bit) every 2,500 iterations (`logs/*.log.metrics`). Densification ends at 14,900, so N is fixed during the comparison.

## Results

| seed | arm | train s (CUDA events) | NeRFICG ms/it | PSNR | SSIM | LPIPS | Gaussians |
|---|---|---|---|---|---|---|---|
| 0 | stock | 255.4 | 8.56 | 27.400 | 0.8667 | 0.1209 | 4,143,207 |
| 0 | switch@15k | 231.8 | 7.77 | 27.482 | 0.8676 | 0.1205 | 4,170,530 |
| 1 | stock | 255.9 | 8.58 | 27.410 | 0.8658 | 0.1210 | 4,149,750 |
| 1 | switch@15k | 231.1 | 7.75 | 27.510 | 0.8669 | 0.1201 | 4,188,235 |
| 2 | stock | 256.1 | 8.58 | 27.446 | 0.8671 | 0.1207 | 4,162,763 |
| 2 | switch@15k | 229.6 | 7.70 | 27.516 | 0.8679 | 0.1202 | 4,182,881 |
| **mean** | stock | **255.8** | 8.57 | **27.419** | 0.8665 | 0.1209 | |
| **mean** | switch@15k | **230.8** | 7.74 | **27.503** | 0.8674 | 0.1203 | |

- **Training-step time:** the first 15k iterations take 114.3–115.3 s in both arms, as they should since they are identical. All of the saving is in the second half: 141.0 s becomes 116.6 s.
- **Total job duration per run:** stock 9:50 / 9:47 / 9:57 vs switch 9:21 / 9:24 / about 9:36, a saving of about 25 s per run, the same as the training saving. Most of that duration is the twelve intermediate test evaluations and checkpoint writes these runs did for the curves. A normal run evaluates only at the end.

## How large is the quality gain? (noise check)

The two arms are not bit-identical before the switch. CUDA atomics make same-seed runs differ, and at 15k the switch runs happened to be ahead by 0.04 / 0.05 / 0.02 dB. Same-seed reruns of stock Faster-GS at 30k also vary by up to 0.07 dB. To account for this we pooled every stock 30k run of this code on garden:

| seed | stock runs (30k PSNR) | mean | switch@15k | gain |
|---|---|---|---|---|
| 0 | 27.472 (ttq), 27.400 (here) | 27.436 | 27.482 | +0.046 |
| 1 | 27.397 (ttq), 27.410 (here), 27.456 (optim) | 27.421 | 27.510 | +0.089 |
| 2 | 27.486 (ttq), 27.446 (here), 27.500 (optim) | 27.477 | 27.516 | +0.039 |

Against all 8 stock runs, the switch is about **+0.06 dB**, and it is above the per-seed stock mean on every seed. The safe claim is **no quality loss, probably a small gain** (+0.04 to +0.09 dB). The 8 stock runs come from three jobs on different nodes; only the 3 paired runs here share a node with the switch runs.

## Time to stock quality

The switch run's PSNR curve stays above stock from about 17.5k onward on all seeds (`logs/*.log.metrics`). It first reaches each seed's stock final PSNR after 138.6 / 130.7 / 173.6 s of training, against 255.4 / 255.9 / 256.1 s for stock to finish.

That suggests a shorter schedule would hold quality. It is **not** a measured speed-up:

- the eval curves are noisy (±0.05 dB between neighbouring points);
- the crossing is the first crossing, linearly interpolated;
- a run stopped early would also run a different LR schedule (the means LR decays to iteration 30k).

Testing it needs runs with a shorter `NUM_ITERATIONS`.

## Caveats

1. **Scope:** one scene (garden) and one switch point (15k, right after densification ends at 14.9k).
2. **Peak VRAM** in these runs (about 68 GiB in both arms) is dominated by the intermediate test renders, so it is not comparable with earlier training-only figures (about 6.5 GB). Selective Adam does not change memory. Its moments are the same tensors.
3. **Projection vs measurement:** the harness's 1.35× (`results/ttq`) came from reaching quality with fewer image-equivalents. At a fixed 30k iterations, the trainer gets 1.11× from the cheaper optimizer step alone.

## Files

- `logs/{swbase,sw15k}_s{0,1,2}.log.seg`: SEGCLOCK lines (training-step time per 2,500 iterations, cumulative) and the switch line.
- `logs/*.log.metrics`: test PSNR / SSIM / LPIPS every 2,500 iterations.
- `logs/*_timings.txt`: NeRFICG timers.
- `logs/gs-switch-4480440.out`: job summary.
- Full trainer stdout logs (mostly progress bars) remain on the cluster, in `results/sparse_switch_b11a9cb/logs/`.
- Driver: [`cluster/sparse_switch.sbatch`](../../cluster/sparse_switch.sbatch).
