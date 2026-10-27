# Phase 1 plan: Phase 0 backward in Faster-GS (draft, waiting for OK)

## What Faster-GS already does (read from Faster-GS @1144604, `rasterization/`)

No separate Phase 1 spec exists. The only Phase 1 material is the baseline profiling in `results/phase1/` and the lean-Adam experiment in `results/optim/`.

Faster-GS's rasterizer already contains most of the Phase 0 design:

| | Phase 0 | Faster-GS `main` |
|---|---|---|
| Forward | 16×16 tile, splats staged in shared memory, (C, T) checkpoint every 32 splats | Same idea: 16×12 tile, splats staged in shared memory, (C, T) checkpoint every 32 splats (`bucket_color_transmittance`). It also culls by bounding box per 8×4 warp subtile. |
| Backward | **B:** one warp per (tile, 32-splat bucket), 4 warps per block. **A lane owns 8 pixels** and loops over the 32 splats, keeping only (T, ρ) per pixel. The warp reduces once per splat; a ballot skips splats that no pixel uses. | **Taming-3DGS:** one warp per bucket, **1 warp per block**. **A lane owns 1 splat**, and the 192 pixels are passed lane to lane with `shfl_up` (223 steps, 9 shuffles per step). Every bucket pays the full 223 steps, even when its splats cover only a few pixels. |

So Phase 0's 8.3× speedup was measured against the per-pixel atomic baseline (A), **not against what Faster-GS actually runs**. What carries over to Phase 1 is only the **backward work layout**: lane-owns-pixels instead of lane-owns-splat. toy2d suggests this matters. The Taming layout fell to 0.57–0.97× of per-pixel A at 200k small splats, while Phase 0 B stayed 9.1× faster than A at 200k. That points the same way but is not a head-to-head comparison.

**Ceiling:** at step 5000 (3.1M Gaussians, H100), `blend_backward_cu` takes 1.91 ms of 6.72 ms GPU time per step (29%). `blend_cu` (forward) takes only 0.46 ms (7%).

| backward speedup | saved per step | train time on garden (now 253 s) |
|---|---|---|
| 2× | ~0.95 ms | ~225 s (−11%) |
| 3× | ~1.27 ms | ~215 s (−15%) |

The optimizer (36%) is a separate track (lean Adam). The two gains should add up.

## Decision 1: scope is the backward only; keep Faster-GS's forward

Porting the Phase 0 forward would replace a kernel that is already the same design (and has subtile culling) and only 7% of the step. The new backward reads Faster-GS's existing forward outputs unchanged: `bucket_color_transmittance`, `tile_n_processed` (last contributor per pixel), `tile_max_n_processed`, `final_transmittances`, and `image`.

## Decision 2: where it plugs in — a Faster-GS variant, `FasterGSMacro`

- Make a clone of Faster-GS @1144604 at `nerficg/src/Methods/FasterGSMacro` with the backend renamed to `FasterGSMacroCudaBackend`. This follows the same pattern as `FasterGSFused`. The user's `FasterGS` checkout and its venv package stay untouched.
- The variant builds **both** backward kernels into one extension. The choice is made at runtime with `FGS_BLEND_BWD=taming|macro` (default `macro` in the variant), passed to `_C.backward` as an int. That gives a same-build A/B comparison and lets one process check gradient parity.
- In this repo: `phase1/fastergs_macro.patch` (against 1144604), `phase1/make_variant.sh` (clone, rename, apply the patch), `phase1/tests/`, and `bench/phase1/*.sbatch`. **Nothing in `cluster/` changes.** Training goes through the existing `cluster/train_fastergs.sbatch` with `METHOD=FasterGSMacro`.

## API (unchanged at the Python level)

`diff_rasterize(...)` and `_Rasterize.forward` stay identical. `_C.backward(...)` gains one trailing argument, `blend_backward_mode: int`. The new kernel uses the same signature as `blend_backward_cu<compute_abs_grad>`:

- Inputs: tile ranges and bucket offsets, instance→primitive indices, mean2d, conic+opacity, color, bg, grad_image, image, final T, n_processed, max_n_processed, bucket→tile, bucket (C, T).
- Outputs: atomics into grad_mean2d, grad_conic, grad_opacity, grad_sh0, and (optionally) grad_mean2d_abs.

`preprocess_backward_cu` is unchanged.

## Kernel: `blend_backward_macro_cu` (Phase 0 B adapted to 16×12 tiles)

- One warp per bucket and 4 warps per block (tune 4 vs 8). The bucket's 32 splats (mean2d, conic, opacity, color, clamp mask) and the tile's G = dL/dimage are staged in shared memory.
- **A lane owns 6 pixels:** `local = r*32 + lane`, giving column `lane % 16` and row `2r + lane / 16`. Each pixel keeps 3 registers:
  - T, from the checkpoint and replayed bit-exactly;
  - ρ = G·(image − C_ckpt), where `image` includes the background. This folds Faster-GS's `grad_alpha_common = −T_final·G·bg` term into ρ, so dL/dα = T·(G·c) − ρ_after/(1−α);
  - `last`, the pixel's n_processed.
- For splat j of the bucket, a pixel takes part only if it is inside the image, j < last, and α ≥ 1/255. These are the same rules Faster-GS uses (no 0.99 clamp, (1−α) floored at 1e-6). Each lane sums 9 values in registers (11 when the abs gradient is on): mean2d (2), conic (3), opacity (1), color (3), and abs mean2d (2). If the warp ballot shows any pixel used the splat, the warp reduces and lane 0 issues one atomic per value. Opacity gets the same post factor as stock (σ' unless proper antialiasing).
- Buckets at or past `tile_max_n_processed` exit early, same as stock.
- Possible v2 after the first timing: skip a lane's pixel rows using the splat's `screen_bounds`, and replace the 9×5 `shfl_down` reduction with a transpose-reduce.

## Correctness checks (GPU, Capella H100; Alpha A100 also works)

Metric, as in phase0: rel err = ‖a − b‖₂ / ‖b‖₂ in fp64, for each output tensor: grad means, scales, rotations, opacities, sh0, sh_rest, plus densification_info (3 rows) and the 2D helper gradients.

1. **Synthetic parity** (`phase1/tests/test_parity.py`, pytest): random 3D scenes from 1k to 500k Gaussians, several cameras, odd image sizes (partial tiles), abs gradient on and off, proper antialiasing on and off. One forward, then both backward kernels on the same buffers. Pass: rel err ≤ 1e-5 for every tensor. Also an edge-case scene with heavy overlap and early termination.
2. **Real-scene parity during training:** with `FGS_GRADCHECK_EVERY=1000` in a garden run, every 1000 steps runs both backwards on that step's forward and logs `GRADCHECK step=… <tensor>=<relerr>`. Pass: ≤ 1e-5 throughout the 30k steps.
3. **Training quality:** garden, 3 seeds, PSNR within about 2 seed-sd of base (27.48 ± 0.04 dB), plus SSIM and LPIPS.

## Benchmark

- **Kernel:** CUDA events around the blend backward only (macro vs taming, same forward, median of 50) at steps 500, 5000 and 20000 of a garden run. Also one nsys capture of steps 5000–5049, comparing kernel tables against `results/phase1/`.
- **Training:** `bench/phase1/compare_macro.sbatch` runs one Capella job on one node: build, run the parity tests (abort on failure), then interleave base (`FasterGS`) and macro (`FasterGSMacro`) on garden with seeds 0, 1, 2. It reports total train time (sum of `training_iteration`, as in `experiment_log.csv`), ms per step, PSNR, SSIM, LPIPS and the final Gaussian count. Results go to `results/phase1/macro_*.md` and rows in `experiment_log.csv`.
- **Win criterion:** lower train time at PSNR within noise. Other scenes (bicycle, room, bonsai) come after garden.

## Done for this session

Agreed plan → variant patch and kernel → parity tests pass on GPU → first garden timing (kernel ms and 3-seed train time vs Faster-GS) → commit on `claude/phase1-3d-fcf489` (ask before pushing).
