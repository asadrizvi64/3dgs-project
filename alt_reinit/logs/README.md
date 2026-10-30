# alt_reinit job logs

Slurm writes `alt_reinit/build_reinit.sbatch` job logs here (`<job-name>-<job>.out`). Each job's per-run files go to
`run_<job>/`: clock `.seg` files, run logs, and under `runs/<run>/` the small result files (metrics, timings, and for
audit runs `reinit_audit/k_*/stats.json`, `agreement_final.json` and the mask images). The full audit output, with the
depth maps, is archived on Capella under `/projects/p_scads_mlhpc/syas272h/3dgs-archive/alt_reinit/<job>/`.
This directory must exist before `sbatch`: Slurm does not create the output directory.
