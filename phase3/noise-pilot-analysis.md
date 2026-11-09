## garden: 4 same-seed pairs (seeds 20 21 22 23)

| seed | views identical | d PSNR (dB) | d N | largest per-view d (dB, view) |
|---|---|---|---|---|
| 20 | yes | -0.0123 | +5487 | -0.10 (view 4) |
| 21 | yes | -0.0779 | -1001 | -1.10 (view 1) |
| 22 | yes | -0.1245 | -28294 | -0.97 (view 5) |
| 23 | yes | -0.0302 | -20926 | -0.59 (view 5) |

SD of same-seed differences: PSNR 0.0504 dB, N 0.385% of mean N (from 4 pairs; an SD from this few pairs is imprecise)

TOST power of check 1 at true difference 0 (margins: PSNR 0.05 dB, N 1.0%):

| seeds per scene | power PSNR | power N |
|---|---|---|
| 3 | 0.12 | 0.78 |
| 4 | 0.19 | 0.97 |
| 5 | 0.28 | 1.00 |
| 6 | 0.40 | 1.00 |
| 7 | 0.52 | 1.00 |
| 8 | 0.63 | 1.00 |
| 9 | 0.71 | 1.00 |
| 10 | 0.78 | 1.00 |
| 11 | 0.84 | 1.00 |
| 12 | 0.88 | 1.00 |

smallest n with >= 80% power on both: 11

## room: 4 same-seed pairs (seeds 20 21 22 23)

| seed | views identical | d PSNR (dB) | d N | largest per-view d (dB, view) |
|---|---|---|---|---|
| 20 | yes | -0.0734 | +4709 | -1.97 (view 35) |
| 21 | yes | -0.1500 | -5188 | -2.85 (view 35) |
| 22 | yes | +0.0116 | -1363 | +1.66 (view 37) |
| 23 | yes | -0.0839 | +4233 | -2.40 (view 35) |

SD of same-seed differences: PSNR 0.0664 dB, N 0.364% of mean N (from 4 pairs; an SD from this few pairs is imprecise)

TOST power of check 1 at true difference 0 (margins: PSNR 0.05 dB, N 1.0%):

| seeds per scene | power PSNR | power N |
|---|---|---|
| 3 | 0.06 | 0.83 |
| 4 | 0.08 | 0.98 |
| 5 | 0.11 | 1.00 |
| 6 | 0.16 | 1.00 |
| 7 | 0.21 | 1.00 |
| 8 | 0.28 | 1.00 |
| 9 | 0.35 | 1.00 |
| 10 | 0.43 | 1.00 |
| 11 | 0.52 | 1.00 |
| 12 | 0.58 | 1.00 |

smallest n with >= 80% power on both: more than 12

Seeds per scene for check 1, by the rule in EXP1R_PLAN.md: more than 12 on at least one scene: check 1 cannot be made conclusive at these margins; the plan returns to review before launch
