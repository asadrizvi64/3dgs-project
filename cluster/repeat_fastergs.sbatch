#!/usr/bin/env bash
# Timing repeats: Faster-GS main vs FasterGSFused, back to back in ONE job (same node, same GPU),
# interleaved so node-level drift hits both variants equally.
#   sbatch --export=ALL,SCENE=garden,SEEDS="0 1 2" cluster/repeat_fastergs.sbatch   (from the repo root)
#SBATCH --job-name=gs-repeat-fastergs
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail

SCENE=${SCENE:-garden}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"FasterGS FasterGSFused"}
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data)

echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader)"
for seed in $SEEDS; do
  for m in $METHODS; do
    echo; echo "########## $m seed=$seed  start $(date +%T)"
    SCENE=$SCENE METHOD=$m SEED=$seed bash "$HERE/train_fastergs.sbatch"
  done
done

echo; echo "########## SUMMARY (method seed ms/it train_s psnr ssim lpips n_gaussians vram_alloc_gib run)"
for seed in $SEEDS; do
  for m in $METHODS; do
    low=$(echo "$m" | tr 'A-Z' 'a-z')
    run=$(ls -dt "$DATA"/nerficg/output/$m/${low}_${SCENE}_*_s${seed}_* | head -1)
    ms=$(grep -A2 "^training_iteration:" "$run/timings.txt" | awk -F': ' '/per iteration/{print $2}')
    met=$(tail -1 "$run"/test_*/metrics_8bit.txt | sed 's/PSNR://; s/SSIM://; s/LPIPS://')
    ng=$(grep -o 'N_Gaussians:[0-9]*' "$run/n_gaussians.txt" | cut -d: -f2)
    vr=$(awk '/allocated:/{print $2; exit}' "$run/vram_stats.txt")
    clk=$(sed 's/^CLOCK //' "$run/gpu_clock_summary.txt" 2>/dev/null | cut -d' ' -f2-7)
    echo "RESULT $m $seed $ms $(awk -v m="$ms" 'BEGIN{printf "%.1f", m*30}') $met $ng $vr $(basename "$run") $clk"
  done
done
