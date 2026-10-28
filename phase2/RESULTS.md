# Phase 2 results

## Step 1: FasterGSPlus build and checks (job 4481219, node c59, 2026-10-28)

FasterGSPlus is Faster-GS @1144604 with lean Adam, plus the Phase 1 macro blend backward, plus the selective-Adam switch and training clock (`cluster/patches/fastergs_sparse_from.py` v3). It was built by `phase1/make_variant.sh` with `VARIANT=FasterGSPlus` (commit 702b5a7). Each gain has its own switch:

| flag | gain | how to turn it on |
|---|---|---|
| A | macro blend backward | env `FGS_BLEND_BWD=macro` (`taming` = stock kernel) |
| B | selective Adam from iteration n | config `TRAINING.OPTIMIZER.SELECTIVE_FROM: n` (env `FGS_SPARSE_FROM` overrides) |
| C | bf16 SH-rest Adam moments | config `TRAINING.OPTIMIZER.SH_REST_BF16_MOMENTS: true` |

Driver: [`build_plus.sbatch`](build_plus.sbatch). Log: [`logs/gs-phase2-build-4481219.out`](logs/gs-phase2-build-4481219.out). Per-run clock files and test logs are in `logs/run_4481219/`.

### Checks (all passed)

| stage | check | result |
|---|---|---|
| build | variant builds and imports; default blend backward is mode 3 | ok |
| unit | Phase 1 gradient parity against FasterGSPlusCudaBackend | 92 passed (max abs diff 1.2e-6) |
| unit | lean-Adam kernel tests against FasterGSPlusCudaBackend | 15/15 PASS |
| switch | all flags on, `SELECTIVE_FROM=600`: switch fires once, at 600; dense before, selective after | PASS |
| switch | all flags off: never switches; dense throughout | PASS |

The lean-Adam test file has 15 checks. The "16/16" in the first draft of `PLAN.md` was a miscount and is now corrected.

### Sanity runs: garden, seed 0, 30k, one node, interleaved

Training time is from CUDA events summed over every `training_iteration`. Test metrics are computed only at 30k.

| run | train s | vs stock | NeRFICG ms/it | PSNR | SSIM | LPIPS | Gaussians | allocated VRAM |
|---|---|---|---|---|---|---|---|---|
| stock FasterGS | 251.4 | | 8.43 | 27.499 | 0.8667 | 0.1206 | 4,152,932 | 6.46 GiB |
| FasterGSPlus, all off | 251.8 | +0.2% | 8.44 | 27.461 | 0.8669 | 0.1208 | 4,148,882 | 6.46 GiB |
| FasterGSPlus, A+B+C | **192.6** | **−23.4% (1.31×)** | 6.47 | 27.525 | 0.8674 | 0.1206 | 4,156,343 | 6.41 GiB |

- **All flags off matches stock.** The time is the same to within 0.2%. PSNR 27.461 lies inside the range of earlier stock seed-0 runs of this code (27.400 / 27.472 / 27.499 / 27.508). One build can therefore serve every arm of the step 2 factorial.
- **A+B+C is 23.4% faster** in this single run. The plan's rough projection was about −27% (188 s from 255.8 s), so the parts add up to most of what was expected. Step 2 measures how much each part contributes and whether they interact.
- **Quality:** A+B+C ends at 27.525 dB, against 27.499 dB stock on the same node. That is one seed only, well inside run-to-run noise (about ±0.05 dB). It shows no sign of a problem, but it is not a quality result; step 2 (3 seeds) and step 3 (9 scenes) are.
- Visible fraction at step 30k with B on: 0.336, as in the earlier switch runs (0.34–0.39).

## Next: step 2

An 8-arm factorial (A × B × C on/off) on garden, 3 seeds, interleaved per seed in one job: 24 runs of about 5 min, roughly 2 h on one H100. See [`PLAN.md`](PLAN.md).
