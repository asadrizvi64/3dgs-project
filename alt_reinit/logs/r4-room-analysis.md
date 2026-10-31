
## R4 confirmation (single evaluation per run, no other intermediate evaluation)

### room

ref confirmation (3 seeds) at 30000: PSNR 31.6723, SSIM 0.92070, LPIPS 0.28249, track 152.7 s

| arm | step | seeds | PSNR | SSIM | LPIPS | track s | verdict |
|---|---|---|---|---|---|---|---|
| reinit-same | 26500 | 3 | 31.4909 | 0.91942 | 0.27840 | 115.0 | not reproduced: psnr, ssim below target |
| reinit-target | 24500 | 3 | 31.5626 | 0.92261 | 0.26727 | 120.8 | not reproduced: psnr below target |

