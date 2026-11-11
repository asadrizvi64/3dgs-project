## Validation

- garden: all checks pass
- room: all checks pass

## Check 1 (Experiment 1R): exact gates 1a-1d, paired report 1e

- garden: 1a-1d all pass

  1e (reported, not a gate): ctl - reference, per seed

  | seed | order | d PSNR (dB) | d PSNR / pilot SD | d N | d N (% of N) | largest per-view d (dB) |
  |---|---|---|---|---|---|---|
  | 30 | up first | +0.0018 | +0.04 | -8112 | -0.194% | view 4 -0.33, view 0 -0.30 |
  | 31 | up last | -0.0573 | -1.14 | +12793 | +0.308% | view 1 -0.40, view 5 -0.24 |
  | 32 | up first | +0.0181 | +0.36 | -11142 | -0.269% | view 19 +1.10, view 9 -0.25 |

  mean d PSNR -0.0125 dB over 3 seed(s). Noise pilot, same-seed differences on this scene: SD 0.0504 dB. Passing 1a-1d establishes implementation fidelity, not statistical equivalence of final PSNR.

- room: 1a-1d all pass

  1e (reported, not a gate): ctl - reference, per seed

  | seed | order | d PSNR (dB) | d PSNR / pilot SD | d N | d N (% of N) | largest per-view d (dB) |
  |---|---|---|---|---|---|---|
  | 30 | up first | -0.2134 | -3.21 | +10250 | +0.797% | view 15 -1.64, view 35 -1.31 |
  | 31 | up last | -0.0289 | -0.44 | -537 | -0.041% | view 34 -2.57, view 35 +1.53 |
  | 32 | up first | +0.0598 | +0.90 | +13098 | +1.026% | view 34 +2.48, view 32 -0.86 |

  mean d PSNR -0.0608 dB over 3 seed(s). Noise pilot, same-seed differences on this scene: SD 0.0664 dB. Passing 1a-1d establishes implementation fidelity, not statistical equivalence of final PSNR.


## Time to target (target = ctl PSNR at 30k minus 0.05 dB; training seconds from step 0)

### garden

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 171.3 / 163.4 / 186.0 |  | 27.530 |  |  | 4.175 M | 232.0 | 1018.3 | 69.34 |
| mcmc | 121.4 / 106.9 / 116.4 | +29.1% / +34.6% / +37.4% | 27.720 | +0.190 | -0.0043 | 3.098 M | 203.8 | 960.6 | 69.32 |
| mrep | 125.4 / 105.9 / 118.9 | +26.8% / +35.2% / +36.1% | 27.713 | +0.183 | -0.0043 | 3.098 M | 203.9 | 959.5 | 69.32 |
| mreloc | n/r / 184.0 / n/r | n/r / -12.6% / n/r | 27.383 | -0.147 | +0.0150 | 3.098 M | 183.2 | 939.5 | 69.32 |
| mnoise | n/r / n/r / n/r | n/r / n/r / n/r | 27.417 | -0.113 | +0.0063 | 3.098 M | 198.1 | 966.3 | 69.32 |
| mreg | n/r / 187.8 / n/r | n/r / -14.9% / n/r | 27.450 | -0.080 | +0.0047 | 3.098 M | 188.0 | 958.2 | 69.32 |
| mlr | 126.8 / 104.8 / 119.0 | +26.0% / +35.9% / +36.0% | 27.680 | +0.150 | -0.0023 | 3.098 M | 202.8 | 979.4 | 69.32 |
| creset | n/r / 189.2 / 216.8 | n/r / -15.8% / -16.5% | 27.487 | -0.043 | -0.0010 | 5.127 M | 259.0 | 1082.8 | 69.34 |

- sensitivity, target −0.10 dB: ctl 165/130/180; mcmc 108/103/113; mrep 111/102/115; mreloc n/r/140/151; mnoise n/r/179/n/r; mreg 187/147/173; mlr 118/101/115; creset 228/143/205
- sensitivity, target −0.20 dB: ctl 128/123/140; mcmc 101/92/100; mrep 103/97/103; mreloc n/r/95/125; mnoise 169/114/160; mreg 128/102/121; mlr 103/85/100; creset 169/135/156
- sensitivity, reaches −0.05 dB and stays: ctl 219/202/186; mcmc 121/107/116; mrep 125/106/119; mreloc n/r/184/n/r; mnoise n/r/n/r/n/r; mreg n/r/188/n/r; mlr 127/105/140; creset n/r/228/257

### room

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 129.1 / 129.2 / 130.7 |  | 31.640 |  |  | 1.341 M | 144.9 | 1449.2 | 71.15 |
| mcmc | 74.4 / 81.3 / 85.2 | +42.4% / +37.0% / +34.8% | 32.083 | +0.443 | -0.0103 | 0.950 M | 132.9 | 1419.3 | 71.15 |
| mrep | 76.1 / 89.4 / 76.3 | +41.0% / +30.8% / +41.6% | 32.030 | +0.390 | -0.0103 | 0.950 M | 134.4 | 1425.1 | 71.15 |
| mreloc | 91.6 / n/r / n/r | +29.0% / n/r / n/r | 31.357 | -0.283 | +0.0160 | 0.950 M | 109.4 | 1391.4 | 71.15 |
| mnoise | 93.6 / 116.5 / 108.7 | +27.5% / +9.8% / +16.9% | 31.677 | +0.037 | -0.0153 | 0.950 M | 133.6 | 1432.2 | 71.15 |
| mreg | 79.8 / 109.7 / 107.3 | +38.2% / +15.1% / +17.9% | 31.693 | +0.053 | -0.0023 | 0.950 M | 120.0 | 1425.4 | 71.15 |
| mlr | 77.1 / 84.3 / 77.1 | +40.3% / +34.8% / +41.0% | 32.030 | +0.390 | -0.0147 | 0.950 M | 140.7 | 1429.0 | 71.15 |
| creset | 104.4 / n/r / 138.4 | +19.1% / n/r / -5.9% | 31.670 | +0.030 | -0.0030 | 1.878 M | 155.7 | 1473.2 | 71.15 |

- sensitivity, target −0.10 dB: ctl 127/125/128; mcmc 74/77/77; mrep 75/82/75; mreloc 85/n/r/n/r; mnoise 92/114/104; mreg 72/97/105; mlr 76/79/76; creset 102/138/127
- sensitivity, target −0.20 dB: ctl 99/120/114; mcmc 72/75/75; mrep 74/78/74; mreloc 82/n/r/n/r; mnoise 80/109/84; mreg 65/84/84; mlr 73/75/74; creset 90/127/108
- sensitivity, reaches −0.05 dB and stays: ctl 142/140/142; mcmc 74/81/85; mrep 76/89/76; mreloc n/r/n/r/n/r; mnoise 130/129/119; mreg 80/110/119; mlr 77/84/88; creset 126/n/r/138

## One-view failures at 30k (a test image > 3 dB below the median of the same image in the scene's other runs)

- room mlr s30: image 36 -4.7 dB
- room mnoise s30: image 36 -4.7 dB
- room creset s31: image 34 -4.3 dB
- room ctl s31: image 34 -3.1 dB
- room mreloc s31: image 34 -5.0 dB
- room mnoise s31: image 34 -4.5 dB
- room mreg s31: image 34 -3.6 dB
- room mrep s32: image 36 -7.9 dB
- room mreg s32: image 35 -3.5 dB

garden ctl: 0; garden mcmc: 0; garden mrep: 0; garden mreloc: 0; garden mnoise: 0; garden mreg: 0; garden mlr: 0; garden creset: 0; room ctl: 1; room mcmc: 0; room mrep: 1; room mreloc: 1; room mnoise: 2; room mreg: 2; room mlr: 1; room creset: 1

## Decision (phase3/EXP1A_PLAN.md)

### One-view failures (control-based reference: median of the scene's ctl and uninterrupted runs, excluding the run itself)

- room creset s31: image 34 -4.2 dB
- room ctl s31: image 34 -3.6 dB
- room mlr s30: image 36 -4.9 dB
- room mnoise s30: image 36 -5.0 dB
- room mnoise s31: image 34 -4.4 dB
- room mreg s31: image 34 -3.6 dB
- room mreloc s31: image 34 -4.9 dB
- room mrep s32: image 36 -8.1 dB

| arm | garden | room |
|---|---|---|
| ctl | 0 | 1 |
| mcmc | 0 | 0 |
| mrep | 0 | 1 |
| mreloc | 0 | 1 |
| mnoise | 0 | 2 |
| mreg | 0 | 1 |
| mlr | 0 | 1 |
| creset | 0 | 1 |

For comparison only (Experiment 1R's definition, median over all runs present, which depends on the arm set): garden ctl: 0; garden mcmc: 0; garden mrep: 0; garden mreloc: 0; garden mnoise: 0; garden mreg: 0; garden mlr: 0; garden creset: 0; room ctl: 1; room mcmc: 0; room mrep: 1; room mreloc: 1; room mnoise: 2; room mreg: 2; room mlr: 1; room creset: 1

**Step 1, replication: mcmc vs ctl by Experiment 1R's rule: PASS**
  - meets garden: median T reduction >= 10%
  - meets garden: mean ΔPSNR >= -0.10 dB
  - meets garden: mean ΔLPIPS <= +0.002
  - meets room: median T reduction >= 10%
  - meets room: mean ΔPSNR >= -0.10 dB
  - meets room: mean ΔLPIPS <= +0.002
  - meets T lower than ctl in >= 5 of 6 pairs (6)
  - meets one-view failures (control-based reference) not above ctl

### Quality of each arm against mcmc (ctl and creset against ctl), mean over seeds

| arm | garden ΔPSNR | garden ΔLPIPS | room ΔPSNR | room ΔLPIPS |
|---|---|---|---|---|
| mrep (vs mcmc) | -0.007 | +0.0000 | -0.053 | +0.0000 |
| mreloc (vs mcmc) | -0.337 | +0.0193 | -0.727 | +0.0263 |
| mnoise (vs mcmc) | -0.303 | +0.0107 | -0.407 | -0.0050 |
| mreg (vs mcmc) | -0.270 | +0.0090 | -0.390 | +0.0080 |
| mlr (vs mcmc) | -0.040 | +0.0020 | -0.053 | -0.0043 |
| creset (vs ctl) | -0.043 | -0.0010 | +0.030 | -0.0030 |

### Callback time (densify, Morton ordering, opacity resets; outside the training clock), seconds from the branch to the end

| arm | garden | room |
|---|---|---|
| ctl | 1.72 / 1.78 / 1.72 | 0.70 / 0.69 / 0.69 |
| mcmc | 0.32 / 0.34 / 0.33 | 0.19 / 0.19 / 0.20 |
| mrep | 0.32 / 0.34 / 0.32 | 0.19 / 0.19 / 0.20 |
| mreloc | 0.07 / 0.07 / 0.07 | 0.04 / 0.04 / 0.04 |
| mnoise | 0.31 / 0.35 / 0.32 | 0.19 / 0.18 / 0.19 |
| mreg | 0.32 / 0.34 / 0.33 | 0.18 / 0.19 / 0.18 |
| mlr | 0.32 / 0.34 / 0.32 | 0.19 / 0.18 / 0.20 |
| creset | 1.93 / 1.99 / 1.95 | 0.83 / 0.81 / 0.81 |

### Fractions at target −0.05 dB (primary), training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 30 | 49.9 | yes | +0.08 | not reached | not reached | not reached | +0.11 | not reached |
| garden | 31 | 56.5 | yes | -0.02 | +1.36 | not reached | +1.43 | -0.04 | -0.46 |
| garden | 32 | 69.7 | yes | +0.04 | not reached | not reached | not reached | +0.04 | -0.44 |
| room | 30 | 54.7 | yes | +0.03 | +0.31 | +0.35 | +0.10 | +0.05 | +0.45 |
| room | 31 | 47.9 | yes | +0.17 | not reached | +0.73 | +0.59 | +0.06 | not reached |
| room | 32 | 45.5 | yes | -0.19 | not reached | +0.52 | +0.49 | -0.18 | -0.17 |

### Fractions at target −0.10 dB, training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 30 | 57.2 | yes | +0.05 | not reached | not reached | +1.37 | +0.18 | -1.09 |
| garden | 31 | 27.5 | yes | -0.01 | +1.35 | +2.76 | +1.61 | -0.07 | -0.46 |
| garden | 32 | 66.4 | yes | +0.02 | +0.58 | not reached | +0.90 | +0.03 | -0.39 |
| room | 30 | 53.1 | yes | +0.03 | +0.21 | +0.35 | -0.04 | +0.04 | +0.46 |
| room | 31 | 48.3 | yes | +0.11 | not reached | +0.77 | +0.43 | +0.05 | -0.27 |
| room | 32 | 51.2 | yes | -0.02 | not reached | +0.54 | +0.56 | -0.01 | +0.01 |

### Fractions at target −0.20 dB, training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 30 | 26.3 | yes | +0.05 | not reached | +2.58 | +1.02 | +0.07 | -1.58 |
| garden | 31 | 31.2 | yes | +0.14 | +0.09 | +0.69 | +0.33 | -0.22 | -0.39 |
| garden | 32 | 40.2 | yes | +0.09 | +0.64 | +1.49 | +0.52 | +0.01 | -0.41 |
| room | 30 | 27.5 | yes | +0.07 | +0.38 | +0.30 | -0.24 | +0.05 | +0.35 |
| room | 31 | 45.8 | yes | +0.07 | not reached | +0.75 | +0.19 | +0.00 | -0.14 |
| room | 32 | 39.0 | yes | -0.04 | not reached | +0.23 | +0.22 | -0.04 | +0.15 |

### Fractions at target −0.05 dB and stays, training clock

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 30 | 97.4 | yes | +0.04 | not reached | not reached | not reached | +0.05 | not reached |
| garden | 31 | 94.6 | yes | -0.01 | +0.81 | not reached | +0.85 | -0.02 | -0.27 |
| garden | 32 | 69.7 | yes | +0.04 | not reached | not reached | not reached | +0.34 | -1.02 |
| room | 30 | 67.7 | yes | +0.03 | not reached | +0.82 | +0.08 | +0.04 | +0.24 |
| room | 31 | 58.6 | yes | +0.14 | not reached | +0.81 | +0.48 | +0.05 | not reached |
| room | 32 | 56.8 | yes | -0.16 | not reached | +0.59 | +0.60 | +0.06 | +0.06 |

### Fractions at target −0.05 dB (primary), training clock + callbacks

| scene | seed | A = T(ctl) − T(mcmc) (s) | used | f mrep | f mreloc (R) | f mnoise (Z) | f mreg (G) | f mlr (L) | g creset (O) |
|---|---|---|---|---|---|---|---|---|---|
| garden | 30 | 51.2 | yes | +0.08 | not reached | not reached | not reached | +0.10 | not reached |
| garden | 31 | 57.9 | yes | -0.02 | +1.32 | not reached | +1.40 | -0.04 | -0.45 |
| garden | 32 | 71.0 | yes | +0.04 | not reached | not reached | not reached | +0.04 | -0.44 |
| room | 30 | 55.2 | yes | +0.03 | +0.31 | +0.35 | +0.10 | +0.05 | +0.45 |
| room | 31 | 48.4 | yes | +0.17 | not reached | +0.73 | +0.59 | +0.06 | not reached |
| room | 32 | 46.0 | yes | -0.19 | not reached | +0.51 | +0.48 | -0.18 | -0.17 |

**Step 2, resolution (6 pairs used; median |f mrep| = 0.06, must be <= 0.20): PASS**

**Steps 3 and 4, classification at the primary target, training clock:**

- R (relocation, arm mreloc): **needed**
- Z (noise injection, arm mnoise): **needed**
- G (regularisation, arm mreg): **needed** — **sensitive to target selection** (−0.20 dB (training clock): not resolved)
- L (opacity learning rate 0.05, arm mlr): **not needed**
- O (opacity resets, removed from ctl, arm creset): **resets explain little**

Reading: "needed" means needed given the other components of this policy, on these two scenes and three seeds; leave-one-out fractions need not add up to 1. creset measures removing resets from the control policy, not the contribution of resets inside MCMC.
