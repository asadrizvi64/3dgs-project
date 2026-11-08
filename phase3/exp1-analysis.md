REPLACED from rerun logs (whole scene-seed blocks): room s1, room s2

## Validation

- garden: all checks pass
- room: all checks pass

## Check 1: ctl (branched) vs uninterrupted FasterGSUp runs

- garden: ctl PSNR 27.517 vs uninterrupted 27.500 (range 27.469–27.528) ok; final N 4.156 M vs 4.159 M (range 4.145–4.182 M) ok
- room: ctl PSNR 31.610 vs uninterrupted 31.711 (range 31.679–31.736) OUTSIDE; final N 1.299 M vs 1.297 M (range 1.287–1.308 M) ok
Criterion: |mean difference| <= the uninterrupted runs' range (max - min), for PSNR and for final N.

## Time to target (target = ctl PSNR at 30k minus 0.05 dB; training seconds from step 0)

### garden

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 172.8 / 180.7 / 156.8 |  | 27.517 |  |  | 4.172 M | 233.3 | 805.8 | 69.34 |
| stop | n/r / n/r / n/r | n/r / n/r / n/r | 27.340 | -0.177 | +0.0083 | 3.141 M | 192.3 | 758.0 | 69.34 |
| mcmc | 115.3 / 120.1 / 110.1 | +33.3% / +33.5% / +29.8% | 27.673 | +0.157 | -0.0043 | 3.087 M | 204.2 | 778.8 | 69.31 |
| oracle | 112.5 / 121.8 / 113.1 | +34.9% / +32.6% / +27.8% | 27.823 | +0.307 | -0.0113 | 4.172 M | 252.4 | 825.3 | 69.31 |

- sensitivity, target −0.10 dB: ctl 170/140/145; stop n/r/n/r/n/r; mcmc 102/100/95; oracle 106/98/110
- sensitivity, target −0.20 dB: ctl 146/122/108; stop n/r/150/130; mcmc 96/86/90; oracle 97/94/102
- sensitivity, reaches −0.05 dB and stays: ctl 225/181/226; stop n/r/n/r/n/r; mcmc 115/120/119; oracle 113/122/113

### room

| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| ctl | 121.9 / 116.5 / 126.4 |  | 31.610 |  |  | 1.346 M | 145.0 | 1422.8 | 71.16 |
| stop | n/r / 110.5 / 106.7 | n/r / +5.1% / +15.6% | 31.457 | -0.153 | +0.0037 | 0.969 M | 122.0 | 1388.8 | 71.16 |
| mcmc | 77.1 / 78.2 / 71.9 | +36.8% / +32.9% / +43.1% | 32.170 | +0.560 | -0.0113 | 0.953 M | 131.3 | 1399.3 | 71.15 |
| oracle | 85.3 / 83.8 / 82.7 | +30.1% / +28.1% / +34.6% | 32.227 | +0.617 | -0.0157 | 1.346 M | 152.2 | 1404.0 | 71.15 |

- sensitivity, target −0.10 dB: ctl 114/104/117; stop n/r/102/96; mcmc 76/77/71; oracle 83/83/82
- sensitivity, target −0.20 dB: ctl 104/101/101; stop 103/90/78; mcmc 73/75/70; oracle 81/81/82
- sensitivity, reaches −0.05 dB and stays: ctl 137/137/126; stop n/r/n/r/n/r; mcmc 77/78/72; oracle 85/84/83

## One-view failures at 30k (a test image > 3 dB below the median of the same image in the scene's other runs)

- room stop s0: image 35 -4.5 dB
- room stop s1: image 35 -4.0 dB
- room ctl s1: image 35 -4.5 dB
- room ctl s2: image 35 -4.4 dB
- room stop s2: image 34 -3.8 dB
- room stop s2: image 35 -3.9 dB

garden ctl: 0; garden stop: 0; garden mcmc: 0; garden oracle: 0; room ctl: 2; room stop: 4; room mcmc: 0; room oracle: 0

## Decision (phase3/PLAN.md)

NOT EVALUATED: validation or check 1 failed (see above). Fix the cause before reading the arms.

**Descriptive only (--descriptive): the rule applied anyway. This is NOT the pre-registered decision.**

**stop**: FAIL
  - misses garden: median T reduction >= 10%
  - misses garden: mean ΔPSNR >= -0.10 dB
  - misses garden: mean ΔLPIPS <= +0.002
  - misses room: median T reduction >= 10%
  - misses room: mean ΔPSNR >= -0.10 dB
  - misses room: mean ΔLPIPS <= +0.002
  - misses T lower than ctl in >= 5 of 6 pairs (2)
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
