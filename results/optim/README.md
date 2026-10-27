# Lean Adam on Faster-GS: first optimizer experiment (2026-10-27)

**Change** (`cluster/patches/fastergs_lean_adam.patch`, applied to Faster-GS @c60ea26 as commit 1144604):

- **Sparse Adam:** Gaussians that aren't visible in the current view (`n_touched_tiles == 0` in the forward pass) are skipped. Their parameters and moments stay unchanged that step. The config key is `TRAINING.OPTIMIZER.SPARSE_ADAM`.
- **bf16 moments for the SH-rest coefficients:** `exp_avg` and `exp_avg_sq` for the 45 SH-rest coefficients per Gaussian are stored in bfloat16 and written with stochastic rounding. The config key is `TRAINING.OPTIMIZER.SH_REST_BF16_MOMENTS`.
- Both options default to off. With both off, training runs the original `adam_step` kernel.

**Unit checks** (`cluster/tests/test_lean_adam.py`, 16/16 pass on H100):

- With no Gaussians masked out, the new kernel is bit-identical to the original.
- Masked-out rows stay untouched.
- The bf16 moments track fp32 Adam: the 3000-step parameter update differs by 1.0%, and the per-element `v` by a median 2.7%. The cause is stochastic-rounding noise, not a bug.
- Stochastic rounding avoids the bf16 freeze: the moment converges to 0.975, where round-to-nearest freezes at 0.500.
- Every Gaussian with a gradient is flagged visible.

## Garden, H100 node c24, job 4475665 (all runs interleaved on one GPU)

| variant | seeds | train time (s) | vs base | PSNR (dB) | vs base | SSIM | LPIPS (VGG) | final Gaussians |
|---|---|---|---|---|---|---|---|---|
| base (stock Adam) | 3 | 251.8 ± 1.2 | | 27.488 ± 0.028 | | 0.8670 | 0.1205 | 4.15 M |
| **lean (sparse + bf16)** | 3 | **187.5 ± 0.9** | **−25.5%** | 27.358 ± 0.041 | **−0.13** | 0.8627 | 0.1286 | 3.45 M |
| sparse only | 1 | 191.4 | −24.0% | 27.399 | −0.09 | 0.8629 | 0.1284 | 3.44 M |
| bf16 only | 1 | 234.9 | −6.7% | 27.531 | +0.04 | 0.8670 | 0.1204 | 4.15 M |

The baseline matches the earlier Faster-GS runs (252.9 ± 1.0 s, 27.48 dB), so the patch with options off changes nothing.

## Stage timing, seed 0 (CUDA events, ms)

| step | base fwd / bwd / opt | lean fwd / bwd / opt | Gaussians, base → lean |
|---|---|---|---|
| 5000 | 1.52 / 3.38 / 2.42 | 1.47 / 3.15 / 1.44 | 3.13 M → 2.48 M |
| 20000 | 1.67 / 4.53 / 3.23 | 1.61 / 4.23 / **1.21** | 4.17 M → 3.44 M |

## Findings

1. **bf16 SH-rest moments cost nothing in quality** and save 6.7%, which is close to the predicted saving in optimizer bytes.
2. **Sparse Adam gives most of the speedup but loses about 0.1 dB.** It also leaves training with **17% fewer Gaussians**, so part of its 24% speedup comes from cheaper forward and backward passes, not only from the optimizer. At a fixed count, the optimizer itself drops from 3.23 ms to 1.21 ms per step at step 20k (−63%).
3. **Why sparse Adam ends with fewer Gaussians is not yet known.** The prime suspect is how stock Adam's leftover momentum keeps moving Gaussians that are off-screen, which interacts with densification and opacity-based pruning. That's untested.
4. **Net result:** 25.5% faster than Faster-GS on garden, at −0.13 dB PSNR (about 4.6× the seed sd) and +0.008 LPIPS. **This isn't yet a "same quality" win.** bf16 alone is one.
