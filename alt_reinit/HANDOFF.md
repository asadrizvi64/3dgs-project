# Handoff: run alternative track R (R0 + R1) on Capella

For the session that has GPU access. Everything below is on branch `claude/confident-pascal-1204h5` of
`asadrizvi64/3dgs-project` (commit 3e7fa95, plus this file), branched from `master` / `phase0-scaffold` at 5ffc3ce.
The full design is in [`PLAN.md`](PLAN.md); this file is the operating summary.

## What this is

**Track R tests one hypothesis, separately from Phase 3.** On stock Faster-GS, does a one-shot reinitialization of the
Gaussians from rendered depth at an early step k (2000 or 5000), with densification ended at k, reach the stock
reference's quality sooner once every cost is counted?

```
T_warm-up(0..k) + T_reinit + T_train(k..target)  <  T_reference(0..target)
```

Quality must not get worse: 3-seed mean PSNR, SSIM and LPIPS, plus checks on every test image.

**State:**
- R0 (build and checks) and R1 (audit) tooling is written and **40 CPU tests pass**.
- **Nothing has run on a GPU yet.** The CUDA build, the 3 GPU checks, the smoke runs and the audit are all pending.
- R2 (the 24-run training comparison) is **not to be started**: it depends on R1 and needs an evaluator that isn't written yet.

## Rules: keep it off the main track

- **Run from a separate clone,** so the Phase 3 checkout is never switched:

  ```bash
  git clone -b claude/confident-pascal-1204h5 "$(git -C ~/3dgs-project remote get-url origin)" ~/3dgs-alt
  ```

  Submit every job from `~/3dgs-alt`. Slurm logs go to `~/3dgs-alt/alt_reinit/logs/`, which must exist (it does).
- **Edit only `alt_reinit/`.** Do not touch `phase0/`–`phase2/`, `cluster/`, `experiment_log.csv` (except through
  `alt_reinit/record_runs.py`), or the FasterGS / FasterGSPlus method directories on the workspace.
- **What the jobs add to the shared workspace:**
  - a new method, `nerficg/src/Methods/FasterGSReinit` (rebuilt by each `build` stage; the script refuses to overwrite a directory it did not make);
  - its extension, `FasterGSReinitCudaBackend`, in `fastergs-venv` (plus the `.built-FasterGSReinit` marker);
  - audit outputs under `/projects/p_scads_mlhpc/syas272h/3dgs-archive/alt_reinit/<job>/`.

  Nothing else is modified.
- **The stock FasterGS checkout must already carry the v5 clock hook**, applied earlier by `phase2/build_plus.sbatch`.
  The job only checks this, prints `FAILED stock FasterGS lacks the v5 clock hook` if missing, and never patches it.
- **Commit results to `claude/confident-pascal-1204h5` only,** not to `phase0-scaffold` / `master`, unless the user says to merge.
- **Budget:** about 8 GPU-hours ceiling for the whole track before any nine-scene decision. R0 + R1 should take well under 1 GPU-hour.

## Step 1: R0 (build, unit, smoke; one H100, about 15 min)

```bash
cd ~/3dgs-alt && STAGES="build unit smoke" sbatch alt_reinit/build_reinit.sbatch
```

The log is `alt_reinit/logs/gs-alt-r0-<job>.out`, with per-run files in `alt_reinit/logs/run_<job>/`.

**What a passing log shows:**
- **build:** `FasterGSReinitCudaBackend ok`.
- **unit:**
  - `CPU tests: 40 passed ...`;
  - `static tree checks PASS: 6`;
  - `static Trainer checks PASS: 6`;
  - GPU `CHECK` lines (depth single layer and two layers, empty corners, forward / 6 gradients / densification info vs stock, FusedAdam bit-identical), then `3 passed`.
- **smoke:** five runs (`ar0_garden_stock_s0`, `ar0_garden_off_s0`, `ar0_garden_applymatch_s0`, `ar0_garden_applyplus_s0`, `ar0_garden_audit_s0`), then only `PASS` lines from the smoke checker, then `PASS analyzer accepted the smoke audit` and `PASS recorder parsed the 5 smoke runs, none failed`.
- **Last line:** `JOB OK`.

**If R0 fails,** fix inside `alt_reinit/` only. Rerun the CPU tests from the clone root, commit, push to the same branch, and resubmit R0:

```bash
python -m pytest -q alt_reinit/tests/
```

**The parts most likely to break on real hardware** (written from the source at Faster-GS c60ea26 / NeRFICG 623ad16, never executed):

| symptom | where | likely cause |
|---|---|---|
| error in the `ar0_garden_audit_s0` run at iteration 2000 | `reinit_trainer.run_reinit`, `copy.deepcopy(live)` | the Gaussians module / FusedAdam not deep-copyable; copy params and optimizer state explicitly instead |
| `sfm` has an `error` key (smoke check `SfM check ran` FAILs) | `reinit_trainer._sfm_check` | pycolmap API differences (`im.camera.camera_id`, `cam_from_world()`, `points2D[i].has_point3D()`); mirror what `Datasets/MipNeRF360.py` in the installed NeRFICG does |
| SfM reprojection median well above 1 px | same | camera convention or the PCA world transform; check `transform_max_deviation` first |
| `TypeError` in `_render_channels` | `R.extract_settings` / `R.rasterize` keyword names | the binding signature of the installed variant |
| slow or out of memory in `t_knn` | `Optim.knn_utils.compute_root_mean_squared_knn_distances` on millions of points | the fused kNN needs `points.cuda()`; time and memory are logged in `REINIT ... t_knn=` |
| `REINIT_CHECK ... FAIL` | `reinit_core.verify` | a real bookkeeping bug: stop and report; do not loosen the check |
| `REINIT_FAIL ... need N new points but only M candidates` in an apply smoke run | candidate count | the stride is already halved once; report the numbers before changing anything |
| flags-off clock more than 5% from stock | smoke check | first rerun to rule out node noise; the static checks show the code paths are identical |

## Step 2: R1 (audit; garden and room, seed 0; about 20–30 min). Only after R0 ends with `JOB OK`

```bash
cd ~/3dgs-alt && STAGES=audit sbatch --job-name=gs-alt-audit --time=01:30:00 alt_reinit/build_reinit.sbatch
```

**What it runs:**
- One stock 30k run per scene with `FGS_REINIT=mode=audit,iters=2000:5000,target=<stock N at 15k>/match` (garden 4,166,000, room 1,345,000) and `FGS_CBCLOCK=1`.
- At k = 2000 and 5000 the full reinitialization runs on a **deep copy** of the model and optimizer: timed, checked, then discarded. Training itself is stock.
- Diagnostics go to `reinit_audit/k_*/stats.json`, `masks/*.jpg` and `agreement_final.json`.

**Then analyze and record:**

```bash
python3 alt_reinit/analyze_audit.py alt_reinit/logs/gs-alt-audit-<job>.out > alt_reinit/logs/audit-<job>-analysis.md
```

```bash
python3 alt_reinit/record_runs.py alt_reinit/logs/gs-alt-r0-<r0job>.out alt_reinit/logs/gs-alt-audit-<job>.out
```

**G3 is manual:** look at every `alt_reinit/logs/run_<job>/runs/ar1_*/reinit_audit/k_*/masks/view*.jpg`. Each image is three panels:
- the photo;
- the status: green kept, red rejected by geometry, orange depth mixture, blue low alpha, black empty;
- the relative depth std (white = 0.2 or more).

Write up where rejections fall (edges, sky, thin structures, textureless surfaces, reflections), with paths, in a short
`alt_reinit/logs/audit-<job>-README.md`.

Commit the logs, analysis, write-up and the new `experiment_log.csv` rows to the branch and push.

## Step 3: decide (rules from PLAN.md; already pre-registered)

**Gates per scene and k** (from `analyze_audit.py`; G3 manual):
- **G1:** warm-up + reinit cost leaves a positive budget against the reference estimate (stock seed-0 stage run: garden about 209 s, room about 149 s to its own 30k PSNR).
- **G2:** kept candidates are more photo-consistent (training-photo ZNCC) than geometry-rejected ones.
- **G3:** the masks have been inspected and written up.

**k per scene** = the earliest of 2000 / 5000 passing G1–G3; record both.

**Stop the track** if no scene has a positive budget, or G2 still fails after **one** threshold revision (`depth_tol`,
`reproj_tol`, `rel_std_max` in `FGS_REINIT`). Report that, and do not continue.

**If R1 passes,** the next step is **not** R2 runs. It is writing and timing a metrics-only, in-process evaluator: test
renders plus PSNR / SSIM / LPIPS-VGG and per-image PSNR, outside the training clock. NeRFICG's checkpoint-and-render path
costs about 54 s per call on room and would break the budget. R2's job stages and analysis are not written yet either.

## What to report back

Per scene and k:
- the reinit cost by part, and the warm-up time;
- the budget left, and the G1 / G2 / G3 results;
- `N_before`, candidates (grid, after filter, after dedupe), removed / surviving / added rows, and which targets are reachable (the `max_reachable` vs stock's 15k count matters for the `reinit-target` arm);
- status shares (kept / mixture / geometry / low alpha / empty) and where rejections fall;
- photo-evidence medians, kept vs rejected;
- COLMAP reprojection median (the camera-convention check) and depth error by status;
- coverage holes after replacement;
- agreement with the final model (agreement, not accuracy).

Plus any fixes made, with commit ids.

## Decisions already made (2026-10-29)

| decision | choice |
|---|---|
| scenes | garden and room |
| k | the earliest passing of 2000 / 5000, per scene; both audited |
| quality | no worsening of the 3-seed mean PSNR, SSIM, LPIPS; variability reported, never an allowance; near-ties inconclusive until confirmed; image-by-image failure check |
| filter | filtered first (2 of 4 neighbours); unfiltered later (R3) |
| R2 arms | all four: `ref`, `stop`, `reinit-same` (reinitialization at fixed count), `reinit-target` (stock 15k count) |
| budget | about 8 GPU-hours ceiling; R2 only if R1 passes |
| combination with A / B / C | only after the hypothesis holds (R5), one at a time |
