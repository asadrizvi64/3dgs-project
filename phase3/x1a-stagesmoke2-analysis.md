## Validation

- garden: all checks pass

## Check 1 (Experiment 1R): exact gates 1a-1d, paired report 1e

- garden: 1a-1d all pass

  1e (reported, not a gate): ctl - reference, per seed

  | seed | order | d PSNR (dB) | d PSNR / pilot SD | d N | d N (% of N) | largest per-view d (dB) |
  |---|---|---|---|---|---|---|
  | 0 | up first | -0.0077 | -0.15 | +5920 | +0.305% | view 2 -0.22, view 5 +0.14 |
  | 1 | up last | -0.0317 | -0.63 | +122 | +0.006% | view 4 -0.52, view 15 -0.52 |

  mean d PSNR -0.0197 dB over 2 seed(s). Noise pilot, same-seed differences on this scene: SD 0.0504 dB. Passing 1a-1d establishes implementation fidelity, not statistical equivalence of final PSNR.


## Time to target (target = ctl PSNR at 30k minus 0.05 dB; training seconds from step 0)

### garden

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 9.5 / 9.5 |  | 24.970 |  |  | 1.945 M | 9.7 | 4.7 | 67.97 |
| mcmc | n/r / n/r | n/r / n/r | 24.790 | -0.180 | +0.0220 | 1.046 M | 9.0 | 3.9 | 67.97 |
| mrep | n/r / n/r | n/r / n/r | 24.785 | -0.185 | +0.0220 | 1.046 M | 9.0 | 3.9 | 67.97 |
| mreloc | 8.9 / 8.9 | +6.0% / +6.4% | 24.940 | -0.030 | +0.0185 | 1.046 M | 9.0 | 3.9 | 67.97 |
| mnoise | n/r / n/r | n/r / n/r | 24.885 | -0.085 | +0.0205 | 1.046 M | 8.9 | 3.8 | 67.97 |
| mreg | n/r / n/r | n/r / n/r | 24.865 | -0.105 | +0.0175 | 1.046 M | 8.7 | 3.6 | 67.97 |
| mlr | n/r / 8.9 | n/r / +6.2% | 24.920 | -0.050 | +0.0180 | 1.046 M | 9.0 | 4.0 | 67.97 |
| creset | 9.5 / 9.3 | +0.4% / +1.8% | 24.980 | +0.010 | +0.0000 | 1.945 M | 9.6 | 4.7 | 67.97 |

- sensitivity, target −0.10 dB: ctl 9/9; mcmc n/r/n/r; mrep n/r/n/r; mreloc 9/9; mnoise 9/9; mreg n/r/9; mlr 9/9; creset 9/9
- sensitivity, target −0.20 dB: ctl 9/9; mcmc n/r/9; mrep n/r/9; mreloc 8/8; mnoise 8/8; mreg 8/8; mlr 9/8; creset 9/9
- sensitivity, reaches −0.05 dB and stays: ctl 9/9; mcmc n/r/n/r; mrep n/r/n/r; mreloc 9/9; mnoise n/r/n/r; mreg n/r/n/r; mlr n/r/9; creset 9/9

## One-view failures at 30k (a test image > 3 dB below the median of the same image in the scene's other runs)


garden ctl: 0; garden mcmc: 0; garden mrep: 0; garden mreloc: 0; garden mnoise: 0; garden mreg: 0; garden mlr: 0; garden creset: 0

## Decision (phase3/EXP1A_PLAN.md)

NOT EVALUATED: validation or check 1 failed (see above). Fix the cause before reading the arms.

**Descriptive only (--descriptive): the rule applied anyway. This is NOT the pre-registered decision.**

### One-view failures (control-based reference: median of the scene's ctl and uninterrupted runs, excluding the run itself)

- none

| arm | garden |
|---|---|
| ctl | 0 |
| mcmc | 0 |
| mrep | 0 |
| mreloc | 0 |
| mnoise | 0 |
| mreg | 0 |
| mlr | 0 |
| creset | 0 |

For comparison only (Experiment 1R's definition, median over all runs present, which depends on the arm set): garden ctl: 0; garden mcmc: 0; garden mrep: 0; garden mreloc: 0; garden mnoise: 0; garden mreg: 0; garden mlr: 0; garden creset: 0

**Step 1, replication: mcmc vs ctl by Experiment 1R's rule: FAIL**
  - misses garden: median T reduction >= 10%
  - misses garden: mean ΔPSNR >= -0.10 dB
  - misses garden: mean ΔLPIPS <= +0.002
  - misses T lower than ctl in >= 5 of 6 pairs (0)
  - meets one-view failures (control-based reference) not above ctl

### Quality of each arm against mcmc (ctl and creset against ctl), mean over seeds

| arm | garden ΔPSNR | garden ΔLPIPS |
|---|---|---|
| mrep (vs mcmc) | -0.005 | +0.0000 |
| mreloc (vs mcmc) | +0.150 | -0.0035 |
| mnoise (vs mcmc) | +0.095 | -0.0015 |
| mreg (vs mcmc) | +0.075 | -0.0045 |
| mlr (vs mcmc) | +0.130 | -0.0040 |
| creset (vs ctl) | +0.010 | +0.0000 |

### Callback time (densify, Morton ordering, opacity resets; outside the training clock), seconds from the branch to the end

| arm | garden |
|---|---|
| ctl | 0.09 / 0.09 |
| mcmc | 0.03 / 0.03 |
| mrep | 0.03 / 0.03 |
| mreloc | 0.01 / 0.01 |
| mnoise | 0.03 / 0.03 |
| mreg | 0.02 / 0.02 |
| mlr | 0.03 / 0.03 |
| creset | 0.08 / 0.09 |

### Fractions at target −0.05 dB (primary), training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |
| garden | 1 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |

### Fractions at target −0.10 dB, training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |
| garden | 1 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |

### Fractions at target −0.20 dB, training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |
| garden | 1 | -0.1 | no | n/a | n/a | n/a | n/a | n/a | n/a |

### Fractions at target −0.05 dB and stays, training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |
| garden | 1 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |

### Fractions at target −0.05 dB (primary), training clock + callbacks

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |
| garden | 1 | n/a | no | n/a | n/a | n/a | n/a | n/a | n/a |

OUTCOME: mcmc does not replicate Experiment 1R here. No component is classified (plan, step 1).
