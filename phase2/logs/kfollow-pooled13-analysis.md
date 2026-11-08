## Validation: 1/9 expected scenes pass (kitchen)
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s0: test image 6 at 27.45 dB; same image in the other runs: median 28.68 dB, same seed A+B+C+D+S 28.96
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s1: test image 15 at 30.48 dB; same image in the other runs: median 31.91 dB, same seed A+B+C+D+S 31.49
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s3: test image 6 at 26.52 dB; same image in the other runs: median 28.68 dB, same seed A+B+C+D+S 29.07
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s3: test image 23 at 26.57 dB; same image in the other runs: median 33.00 dB, same seed A+B+C+D+S 33.00
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s4: test image 25 at 31.80 dB; same image in the other runs: median 32.92 dB, same seed A+B+C+D+S 31.90
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s4: test image 28 at 30.67 dB; same image in the other runs: median 32.37 dB, same seed A+B+C+D+S 31.89
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s5: test image 2 at 27.09 dB; same image in the other runs: median 31.87 dB, same seed A+B+C+D+S 30.12
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s6: test image 7 at 27.58 dB; same image in the other runs: median 28.61 dB, same seed A+B+C+D+S 28.29
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s8: test image 0 at 25.89 dB; same image in the other runs: median 26.90 dB, same seed A+B+C+D+S 26.87
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s8: test image 1 at 27.08 dB; same image in the other runs: median 28.88 dB, same seed A+B+C+D+S 28.85
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s8: test image 2 at 29.54 dB; same image in the other runs: median 31.87 dB, same seed A+B+C+D+S 31.68
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s8: test image 4 at 29.12 dB; same image in the other runs: median 30.41 dB, same seed A+B+C+D+S 29.88
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s10: test image 23 at 31.93 dB; same image in the other runs: median 33.00 dB, same seed A+B+C+D+S 33.10
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s11: test image 9 at 19.22 dB; same image in the other runs: median 31.49 dB, same seed A+B+C+D+S 31.36
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s11: test image 18 at 30.01 dB; same image in the other runs: median 31.15 dB, same seed A+B+C+D+S 30.62
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s11: test image 19 at 30.43 dB; same image in the other runs: median 31.64 dB, same seed A+B+C+D+S 31.24
  - kitchen: FLAG possible one-view artifact: b_kitchen_stock_s12: test image 21 at 30.58 dB; same image in the other runs: median 33.10 dB, same seed A+B+C+D+S 33.13
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s0: test image 7 at 27.39 dB; same image in the other runs: median 28.61 dB, same seed stock Faster-GS 28.13
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s0: test image 21 at 26.06 dB; same image in the other runs: median 33.10 dB, same seed stock Faster-GS 33.10
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s1: test image 4 at 26.27 dB; same image in the other runs: median 30.41 dB, same seed stock Faster-GS 31.11
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s4: test image 4 at 29.08 dB; same image in the other runs: median 30.41 dB, same seed stock Faster-GS 30.08
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s4: test image 25 at 31.90 dB; same image in the other runs: median 32.92 dB, same seed stock Faster-GS 31.80
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s5: test image 2 at 30.12 dB; same image in the other runs: median 31.87 dB, same seed stock Faster-GS 27.09
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s5: test image 6 at 27.67 dB; same image in the other runs: median 28.68 dB, same seed stock Faster-GS 27.68
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s5: test image 33 at 28.83 dB; same image in the other runs: median 30.95 dB, same seed stock Faster-GS 30.86
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s9: test image 34 at 26.84 dB; same image in the other runs: median 28.15 dB, same seed stock Faster-GS 27.65
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s10: test image 10 at 32.39 dB; same image in the other runs: median 33.66 dB, same seed stock Faster-GS 33.57
  - kitchen: FLAG possible one-view artifact: b_kitchen_abcds_s10: test image 34 at 25.65 dB; same image in the other runs: median 28.15 dB, same seed stock Faster-GS 28.98
  - not in these logs: bicycle, flowers, garden, stump, treehill, room, counter, bonsai

## kitchen (job 4551359 (c34); per seed in brackets; deltas paired against stock)

| arm | train s | time reduction | ms/it | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | N at 15k | N at 30k, before cleanup | N final | train peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | 201.0 (202.2 / 200.0 / 199.2 / 201.6 / 198.6 / 204.2 / 203.6 / 201.7 / 199.5 / 201.7 / 199.8 / 200.7 / 200.7) |  | 6.75 | 31.420 |  |  |  | 1.616 M | 1.616 M | 1.591 M | 6.53 |
| A+B+C+D+S | 142.8 (142.4 / 144.1 / 144.8 / 141.8 / 144.4 / 142.7 / 142.9 / 144.0 / 144.2 / 142.1 / 141.4 / 141.5 / 140.0) | 29.0 (29.6 / 27.9 / 27.3 / 29.7 / 27.3 / 30.1 / 29.8 / 28.6 / 27.7 / 29.6 / 29.2 / 29.5 / 30.2)% | 4.80 | 31.455 | +0.036 (-0.316 / -0.174 / -0.023 / +0.268 / +0.030 / -0.117 / +0.069 / +0.039 / +0.216 / -0.058 / +0.055 / +0.475 / +0.001) | +0.0002 | -0.0002 | 1.612 M | 1.612 M | 1.607 M | 6.49 |

## Across scenes (each scene = mean of its paired seeds) - PARTIAL, 1/9 scenes

| scene | A+B+C+D+S time | A+B+C+D+S ΔPSNR | A+B+C+D+S ΔLPIPS |
|---|---|---|---|
| kitchen | −29.0% | +0.036 | -0.0002 |

**A+B+C+D+S** over 1 scenes: time reduction median 29.0%, mean 29.0%, range 29.0–29.0%; ΔPSNR mean +0.036, worst kitchen +0.036; ΔLPIPS mean -0.0002; paired runs faster: 13/13
  - meets median time reduction >= 20% (partial data, not a verdict)
  - meets mean dPSNR >= -0.05 dB (partial data, not a verdict)
  - meets no scene dPSNR < -0.10 dB (partial data, not a verdict)
  - meets mean dLPIPS <= +0.002 (partial data, not a verdict)

## Outdoor / indoor and speedup factors (descriptive; scene speedup = mean over seeds of stock time / arm time)

| arm | group | scenes | time reduction median / mean | speedup median / mean | speedup range | aggregate speedup | ΔPSNR mean |
|---|---|---|---|---|---|---|---|
| A+B+C+D+S | all | 1 | 29.0% / 29.0% | 1.408× / 1.408× | 1.408–1.408× | 1.408× | +0.036 |
| A+B+C+D+S | indoor | 1 | 29.0% / 29.0% | 1.408× / 1.408× | 1.408–1.408× | 1.408× | +0.036 |

## Sensitivity: one-view failures (a test image > 3 dB below the same image in the scene's other runs)

- stock Faster-GS: 3 failing image(s) in 3 of 13 runs: kitchen s3 image 23 (-6.4 dB); kitchen s5 image 2 (-4.8 dB); kitchen s11 image 9 (-12.3 dB)
- A+B+C+D+S: 2 failing image(s) in 2 of 13 runs: kitchen s0 image 21 (-7.0 dB); kitchen s1 image 4 (-4.1 dB)

These happen without the changes (stock has them), but these counts cannot show whether the changes alter how often they happen. The exclusion below is a secondary analysis; the verdict uses all images.

Deltas with those images removed from every run of the scene (paired per seed, mean of 13 seeds):

| scene | images removed | A+B+C+D+S ΔPSNR | A+B+C+D+S ΔLPIPS |
|---|---|---|---|
| kitchen | 2, 4, 9, 21, 23 | +0.015 | -0.0000 |

A+B+C+D+S with one-view failures removed: ΔPSNR mean +0.015, worst kitchen +0.015, ΔLPIPS mean -0.0000; tolerances still met

VERDICT: incomplete - not evaluated (1/9 scenes and 26/234 runs validated)
