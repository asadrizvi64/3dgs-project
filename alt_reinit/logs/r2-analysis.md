# Alternative track R, R2 analysis (k = 5000)

Jobs: 4504373, 4504374; repo 04de1b6.
Track clock = SEGCLOCK (training steps) + CBCLOCK (callbacks between steps, reinit included); evaluations excluded.
Target = ref 3-seed means at 30k; candidate = first evaluation where the arm's 3-seed means meet all three. Any candidate is pending R4 confirmation.

## garden

Target (ref, 3 seeds, at 30k): PSNR 27.4175 (sd 0.0653), SSIM 0.86685 (sd 0.00040), LPIPS 0.12043 (sd 0.00043). Images failing (> 3 dB below the ref median) in a ref run: [22].

| arm | candidate step | track s at candidate | track s to 30k | 30k PSNR | 30k SSIM | 30k LPIPS | N at 30k | margins at candidate (or 30k) | outcome |
|---|---|---|---|---|---|---|---|---|---|
| ref | 30000 | 256.0 | 256.0 | 27.418 ± 0.065 | 0.8668 ± 0.0004 | 0.1204 ± 0.0004 | 4,167,075 | psnr +0.0000, ssim +0.0000, lpips -0.0000 | reference |
| stop | none by 30k | - | 209.7 | 27.383 ± 0.021 | 0.8618 ± 0.0002 | 0.1282 ± 0.0003 | 3,133,302 | psnr -0.0350, ssim -0.0051, lpips -0.0077 | reduces quality (psnr below target, ssim below target by more than the ref sd, lpips below target by more than the ref sd) |
| reinit-same | none by 30k | - | 225.7 | 27.407 ± 0.066 | 0.8642 ± 0.0006 | 0.1175 ± 0.0005 | 3,139,765 | psnr -0.0104, ssim -0.0026, lpips +0.0029 | reduces quality (psnr below target, ssim below target by more than the ref sd) |
| reinit-target | none by 30k | - | 282.6 | 27.406 ± 0.063 | 0.8654 ± 0.0004 | 0.1133 ± 0.0003 | 4,166,000 | psnr -0.0117, ssim -0.0014, lpips +0.0071 | reduces quality (psnr below target, ssim below target by more than the ref sd) |

Per seed (exploratory only): first evaluation where that seed alone meets the target, and its track time:

- ref: s0: none; s1: none; s2: 28000 at 237.4 s
- stop: s0: none; s1: none; s2: none
- reinit-same: s0: none; s1: none; s2: none
- reinit-target: s0: none; s1: none; s2: 22000 at 200.3 s

Images (index in the test set) more than 1 dB below the ref median, at the candidate step or 30k; new failures (> 3 dB, never failing in ref) in bold:

- ref: 19, 22
- stop: none
- reinit-same: none
- reinit-target: 22

Reinitialization (per seed): cost, rows before / removed / added / after:

- reinit-same s0: 1.63 s; 3127901 / 1306970 / 1306970 / 3127901
- reinit-same s1: 1.58 s; 3147719 / 1446296 / 1446296 / 3147719
- reinit-same s2: 1.57 s; 3143674 / 1417731 / 1417731 / 3143674
- reinit-target s0: 1.55 s; 3122004 / 1427240 / 2471236 / 4166000
- reinit-target s1: 1.63 s; 3135976 / 1321919 / 2351943 / 4166000
- reinit-target s2: 1.67 s; 3138606 / 1434729 / 2462123 / 4166000

Evaluator: 612 calls, median 0.53 s, max 3.65 s (outside the clock).

Pairwise at 30k (3-seed mean differences, PLAN.md R2 reading):

- stop - ref: PSNR -0.035, SSIM -0.0051, LPIPS +0.0077
- reinit-same - stop: PSNR +0.025, SSIM +0.0024, LPIPS -0.0106
- reinit-target - reinit-same: PSNR -0.001, SSIM +0.0012, LPIPS -0.0042
- reinit-target - ref: PSNR -0.012, SSIM -0.0014, LPIPS -0.0071

## room

Target (ref, 3 seeds, at 30k): PSNR 31.5315 (sd 0.2095), SSIM 0.91803 (sd 0.00391), LPIPS 0.28350 (sd 0.00198). Images failing (> 3 dB below the ref median) in a ref run: [17].

| arm | candidate step | track s at candidate | track s to 30k | 30k PSNR | 30k SSIM | 30k LPIPS | N at 30k | margins at candidate (or 30k) | outcome |
|---|---|---|---|---|---|---|---|---|---|
| ref | 30000 | 150.8 | 150.8 | 31.531 ± 0.209 | 0.9180 ± 0.0039 | 0.2835 ± 0.0020 | 1,347,407 | psnr +0.0000, ssim +0.0000, lpips -0.0000 | reference |
| stop | none by 30k | - | 126.5 | 31.562 ± 0.073 | 0.9184 ± 0.0006 | 0.2861 ± 0.0007 | 971,506 | psnr +0.0308, ssim +0.0004, lpips -0.0026 | reduces quality (lpips below target by more than the ref sd) |
| reinit-same | 26500 | 113.4 | 128.2 | 31.601 ± 0.099 | 0.9197 ± 0.0004 | 0.2773 ± 0.0010 | 964,090 | psnr +0.0072, ssim +0.0013, lpips +0.0055 | improves time-to-quality (pending R4); near-tie: inconclusive |
| reinit-target | 24500 | 119.1 | 146.3 | 31.654 ± 0.201 | 0.9228 ± 0.0001 | 0.2648 ± 0.0012 | 1,345,000 | psnr +0.0064, ssim +0.0043, lpips +0.0166 | improves time-to-quality (pending R4); near-tie: inconclusive |

Per seed (exploratory only): first evaluation where that seed alone meets the target, and its track time:

- ref: s0: 29000 at 144.7 s; s1: 25500 at 128.6 s; s2: none
- stop: s0: none; s1: none; s2: none
- reinit-same: s0: 28000 at 120.2 s; s1: 26000 at 111.8 s; s2: 22000 at 93.6 s
- reinit-target: s0: 23000 at 112.2 s; s1: none; s2: 21000 at 101.5 s

Images (index in the test set) more than 1 dB below the ref median, at the candidate step or 30k; new failures (> 3 dB, never failing in ref) in bold:

- ref: 14, 17, 34, 35, 38
- stop: 5, 6, 7, 28, 34
- reinit-same: 5, 32, 33, 34
- reinit-target: 5, 7, 13, 16, 28, 34

Reinitialization (per seed): cost, rows before / removed / added / after:

- reinit-same s0: 2.20 s; 966807 / 204216 / 204216 / 966807
- reinit-same s1: 2.20 s; 968200 / 199002 / 199002 / 968200
- reinit-same s2: 2.13 s; 957263 / 205921 / 205921 / 957263
- reinit-target s0: 2.27 s; 960967 / 185809 / 569842 / 1345000
- reinit-target s1: 2.21 s; 971397 / 193761 / 567364 / 1345000
- reinit-target s2: 2.16 s; 969616 / 178636 / 554020 / 1345000

Evaluator: 612 calls, median 1.20 s, max 3.03 s (outside the clock).

Pairwise at 30k (3-seed mean differences, PLAN.md R2 reading):

- stop - ref: PSNR +0.031, SSIM +0.0004, LPIPS +0.0026
- reinit-same - stop: PSNR +0.039, SSIM +0.0012, LPIPS -0.0088
- reinit-target - reinit-same: PSNR +0.053, SSIM +0.0032, LPIPS -0.0125
- reinit-target - ref: PSNR +0.123, SSIM +0.0048, LPIPS -0.0187

## R4 confirmation needed

Rerun 3 seeds each with `FGS_EVAL=at=<step>` (no other intermediate evaluation):

- garden ref at step 30000
- room ref at step 30000
- room reinit-same at step 26500
- room reinit-target at step 24500

