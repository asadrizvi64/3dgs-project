## Validation

- garden: all checks pass
- room: all checks pass

## Check 1 (Experiment 1R): exact gates 1a-1d, paired report 1e

- garden: 1a-1d all pass

  1e (reported, not a gate): ctl - reference, per seed

  | seed | order | d PSNR (dB) | d PSNR / pilot SD | d N | d N (% of N) | largest per-view d (dB) |
  |---|---|---|---|---|---|---|
  | 10 | up first | -0.0989 | -1.96 | +287 | +0.007% | view 22 -3.33, view 7 +0.40 |
  | 11 | up last | +0.0588 | +1.17 | -2491 | -0.060% | view 19 +0.93, view 5 +0.47 |
  | 12 | up first | -0.0523 | -1.04 | -20671 | -0.493% | view 1 -0.41, view 4 -0.35 |

  mean d PSNR -0.0308 dB over 3 seed(s). Noise pilot, same-seed differences on this scene: SD 0.0504 dB. Passing 1a-1d establishes implementation fidelity, not statistical equivalence of final PSNR.

- room: 1a-1d all pass

  1e (reported, not a gate): ctl - reference, per seed

  | seed | order | d PSNR (dB) | d PSNR / pilot SD | d N | d N (% of N) | largest per-view d (dB) |
  |---|---|---|---|---|---|---|
  | 10 | up first | +0.1169 | +1.76 | +4679 | +0.363% | view 37 +1.57, view 14 +0.82 |
  | 11 | up last | +0.1809 | +2.72 | +5390 | +0.418% | view 35 +2.95, view 15 +1.16 |
  | 12 | up first | -0.0923 | -1.39 | +11793 | +0.913% | view 34 -3.33, view 16 +2.30 |

  mean d PSNR +0.0685 dB over 3 seed(s). Noise pilot, same-seed differences on this scene: SD 0.0664 dB. Passing 1a-1d establishes implementation fidelity, not statistical equivalence of final PSNR.


## Time to target (target = ctl PSNR at 30k minus 0.05 dB; training seconds from step 0)

### garden

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 134.7 / 144.8 / 172.4 |  | 27.457 |  |  | 4.170 M | 230.2 | 971.3 | 69.33 |
| stop | n/r / n/r / n/r | n/r / n/r / n/r | 27.313 | -0.143 | +0.0083 | 3.134 M | 190.7 | 888.8 | 69.33 |
| mcmc | 100.9 / 99.3 / 114.7 | +25.2% / +31.4% / +33.5% | 27.650 | +0.193 | -0.0047 | 3.078 M | 202.6 | 882.8 | 69.31 |
| oracle | 111.4 / 114.1 / 118.2 | +17.3% / +21.2% / +31.5% | 27.807 | +0.350 | -0.0120 | 4.170 M | 249.3 | 1005.6 | 69.31 |

- sensitivity, target −0.10 dB: ctl 129/142/166; stop 158/157/n/r; mcmc 97/97/113; oracle 109/109/115
- sensitivity, target −0.20 dB: ctl 123/113/132; stop 97/103/151; mcmc 92/87/108; oracle 84/92/96
- sensitivity, reaches −0.05 dB and stays: ctl 169/173/225; stop n/r/n/r/n/r; mcmc 101/99/137; oracle 111/114/118

### room

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 123.4 / 140.1 / 142.1 |  | 31.747 |  |  | 1.345 M | 144.5 | 1440.7 | 71.15 |
| stop | 116.2 / n/r / n/r | +5.8% / n/r / n/r | 31.653 | -0.093 | +0.0040 | 0.970 M | 122.4 | 1418.2 | 71.15 |
| mcmc | 84.7 / 90.4 / 86.9 | +31.3% / +35.5% / +38.9% | 32.043 | +0.297 | -0.0103 | 0.953 M | 134.2 | 1417.9 | 71.15 |
| oracle | 86.3 / 97.7 / 82.6 | +30.0% / +30.2% / +41.9% | 32.253 | +0.507 | -0.0153 | 1.345 M | 153.8 | 1454.3 | 71.15 |

- sensitivity, target −0.10 dB: ctl 110/130/139; stop 100/n/r/n/r; mcmc 83/80/85; oracle 84/96/82
- sensitivity, target −0.20 dB: ctl 98/110/123; stop 91/123/122; mcmc 80/77/77; oracle 82/89/82
- sensitivity, reaches −0.05 dB and stays: ctl 123/140/142; stop 116/n/r/n/r; mcmc 85/90/87; oracle 86/98/83

## One-view failures at 30k (a test image > 3 dB below the median of the same image in the scene's other runs)

- garden ctl s10: image 22 -3.5 dB
- room stop s12: image 34 -3.6 dB
- room stop s12: image 35 -3.6 dB

garden ctl: 1; garden stop: 0; garden mcmc: 0; garden oracle: 0; room ctl: 0; room stop: 2; room mcmc: 0; room oracle: 0

## Decision (phase3/PLAN.md)

**stop**: FAIL
  - misses garden: median T reduction >= 10%
  - misses garden: mean ΔPSNR >= -0.10 dB
  - misses garden: mean ΔLPIPS <= +0.002
  - misses room: median T reduction >= 10%
  - meets room: mean ΔPSNR >= -0.10 dB
  - misses room: mean ΔLPIPS <= +0.002
  - misses T lower than ctl in >= 5 of 6 pairs (1)
  - misses one-view failures not above ctl
**mcmc**: PASS
  - meets garden: median T reduction >= 10%
  - meets garden: mean ΔPSNR >= -0.10 dB
  - meets garden: mean ΔLPIPS <= +0.002
  - meets room: median T reduction >= 10%
  - meets room: mean ΔPSNR >= -0.10 dB
  - meets room: mean ΔLPIPS <= +0.002
  - meets T lower than ctl in >= 5 of 6 pairs (6)
  - meets one-view failures not above ctl
**oracle** (oracle: upper-bound reference, not a method): PASS
  - meets garden: median T reduction >= 10%
  - meets garden: mean ΔPSNR >= -0.10 dB
  - meets garden: mean ΔLPIPS <= +0.002
  - meets room: median T reduction >= 10%
  - meets room: mean ΔPSNR >= -0.10 dB
  - meets room: mean ΔLPIPS <= +0.002
  - meets T lower than ctl in >= 5 of 6 pairs (6)
  - meets one-view failures not above ctl

OUTCOME: a non-oracle arm passes -> Experiment 2 studies that mechanism as an adaptive, training-signal policy.
