## Adjudication

Part 1: FAIL part 1: identical training-view sequences 0-7999 with and without evaluation

Part 2, check 1 on room, seeds 0 1 2 3 4 5 6 7 8:

| seed | ctl PSNR | up PSNR | ctl N | up N |
|---|---|---|---|---|
| 0 | 31.640 | 31.736 | 1291243 | 1294841 |
| 1 | 31.670 | 31.717 | 1304632 | 1307852 |
| 2 | 31.520 | 31.679 | 1302293 | 1287454 |
| 3 | 31.420 | 31.573 | 1295386 | 1271790 |
| 4 | 31.410 | 31.691 | 1296930 | 1295427 |
| 5 | 31.520 | 31.733 | 1291975 | 1285832 |
| 6 | 31.680 | 31.536 | 1287955 | 1287994 |
| 7 | 31.710 | 31.716 | 1295592 | 1301634 |
| 8 | 31.700 | 31.611 | 1296812 | 1284414 |

PSNR: ctl 31.586 vs up 31.666, difference -0.080, allowed 0.200 -> pass
N: ctl 1.296 M vs up 1.291 M, difference +0.005 M, allowed 0.036 M -> pass

Part 1 FAIL, part 2 PASS: Experiment 1 stays invalid; the arms are not compared.

## Supplementary diagnostics (declared before results; they do not change the rule above)

Part 1 at step 8000 (with evaluation / without):
- N 1194125 / 1207452 (difference -13327); PSNR 29.844 / 29.886 (-0.041); SSIM 0.9026 / 0.9035; LPIPS 0.3152 / 0.3146
- per-view PSNR difference: largest +1.63 dB (view 29); views 34 / 35: -0.96 / +1.19
- Two runs without evaluation would also differ here (atomic-addition order), and no such pair exists at 8000, so these numbers have no noise reference. Descriptive only.

Part 2, ctl minus up paired by seed, full-precision final PSNR (RESULT line), and per view for 34 and 35:

| seed | ΔPSNR | Δ without views 34, 35 | Δ view 34 | Δ view 35 |
|---|---|---|---|---|
| 0 | -0.100 | -0.057 | -0.29 | -1.48 |
| 1 | -0.050 | +0.020 | -0.36 | -2.32 |
| 2 | -0.164 | -0.052 | -1.87 | -2.59 |
| 3 | -0.158 | -0.083 | -1.94 | -1.10 |
| 4 | -0.284 | -0.189 | +0.07 | -4.14 |
| 5 | -0.215 | -0.188 | -0.14 | -1.30 |
| 6 | +0.146 | +0.182 | -0.17 | -0.85 |
| 7 | -0.002 | +0.007 | -0.17 | -0.20 |
| 8 | +0.090 | +0.073 | -0.87 | +1.70 |
- ΔPSNR: mean -0.082 dB, 95% t interval [-0.191, +0.027] (n=9), 7 of 9 seeds negative
- Δ without views 34, 35 (per-view values at 0.01 dB): mean -0.032 dB, 95% t interval [-0.123, +0.060] (n=9), 5 of 9 seeds negative
- The rule's allowed difference is the range of the uninterrupted runs, which tends to grow with the number of seeds; the interval above shows whether a systematic gap remains.
