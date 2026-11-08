## Validation: 2/9 expected scenes pass (garden, room)
  - replaced by --rerun jobs (all arms of those seeds come from the rerun): garden s2 ups (job 4552203); room s2 ups (job 4552204); room s2 stock (job 4552204); room s2 up (job 4552204)
  - room: FLAG possible one-view artifact: b_room_stock_s1: test image 13 at 27.88 dB; same image in the other runs: median 32.56 dB, same seed upstream main 32.24, upstream main + S loads 32.18
  - room: FLAG possible one-view artifact: b_room_stock_s1: test image 34 at 24.06 dB; same image in the other runs: median 25.13 dB, same seed upstream main 25.55, upstream main + S loads 25.49
  - room: FLAG possible one-view artifact: b_room_stock_s1: test image 35 at 27.06 dB; same image in the other runs: median 28.54 dB, same seed upstream main 28.73, upstream main + S loads 27.36
  - room: FLAG possible one-view artifact: b_room_stock_s2: test image 35 at 26.39 dB; same image in the other runs: median 28.54 dB, same seed upstream main 29.04, upstream main + S loads 31.12
  - room: FLAG possible one-view artifact: b_room_up_s0: test image 34 at 23.82 dB; same image in the other runs: median 25.13 dB, same seed stock Faster-GS 24.78, upstream main + S loads 23.79
  - room: FLAG possible one-view artifact: b_room_ups_s0: test image 32 at 27.62 dB; same image in the other runs: median 28.82 dB, same seed stock Faster-GS 28.39, upstream main 28.96
  - room: FLAG possible one-view artifact: b_room_ups_s0: test image 34 at 23.79 dB; same image in the other runs: median 25.13 dB, same seed stock Faster-GS 24.78, upstream main 23.82
  - room: FLAG possible one-view artifact: b_room_ups_s1: test image 35 at 27.36 dB; same image in the other runs: median 28.54 dB, same seed stock Faster-GS 27.06, upstream main 28.73
  - not in these logs: bicycle, flowers, stump, treehill, counter, kitchen, bonsai

## garden (job 4552280 (c34); per seed in brackets; deltas paired against stock)

| arm | train s | time reduction | ms/it | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | N at 15k | N at 30k, before cleanup | N final | train peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | 252.3 (251.0 / 252.7 / 253.3) |  | 8.46 | 27.504 |  |  |  | 4.173 M | 4.173 M | 4.157 M | 6.47 |
| upstream main | 230.0 (229.0 / 230.0 / 231.1) | 8.8 (8.8 / 9.0 / 8.8)% | 7.71 | 27.500 | -0.004 (+0.026 / -0.000 / -0.040) | -0.0001 | +0.0003 | 4.175 M | 4.175 M | 4.159 M | 6.47 |
| upstream main + S loads | 229.7 (227.8 / 230.5 / 230.9) | 9.0 (9.2 / 8.8 / 8.8)% | 7.70 | 27.476 | -0.029 (-0.054 / -0.014 / -0.018) | -0.0005 | +0.0004 | 4.174 M | 4.174 M | 4.158 M | 6.47 |

## room (job 4552281 (c133); per seed in brackets; deltas paired against stock)

| arm | train s | time reduction | ms/it | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | N at 15k | N at 30k, before cleanup | N final | train peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | 150.7 (151.2 / 150.4 / 150.7) |  | 5.06 | 31.581 |  |  |  | 1.344 M | 1.344 M | 1.297 M | 6.64 |
| upstream main | 145.0 (144.0 / 146.1 / 145.0) | 3.8 (4.8 / 2.8 / 3.8)% | 4.88 | 31.711 | +0.130 (+0.069 / +0.269 / +0.051) | +0.0004 | -0.0004 | 1.343 M | 1.343 M | 1.297 M | 6.64 |
| upstream main + S loads | 145.4 (145.3 / 145.6 / 145.3) | 3.6 (3.9 / 3.2 / 3.6)% | 4.89 | 31.661 | +0.080 (-0.122 / +0.150 / +0.212) | +0.0001 | -0.0002 | 1.349 M | 1.349 M | 1.303 M | 6.65 |

## Across scenes (each scene = mean of its paired seeds) - PARTIAL, 2/9 scenes

| scene | upstream main time | upstream main ΔPSNR | upstream main ΔLPIPS | upstream main + S loads time | upstream main + S loads ΔPSNR | upstream main + S loads ΔLPIPS |
|---|---|---|---|---|---|---|
| garden | −8.8% | -0.004 | +0.0003 | −9.0% | -0.029 | +0.0004 |
| room | −3.8% | +0.130 | -0.0004 | −3.6% | +0.080 | -0.0002 |

**upstream main** over 2 scenes: time reduction median 6.3%, mean 6.3%, range 3.8–8.8%; ΔPSNR mean +0.063, worst garden -0.004; ΔLPIPS mean -0.0001; paired runs faster: 6/6
  - misses median time reduction >= 20% (partial data, not a verdict)
  - meets mean dPSNR >= -0.05 dB (partial data, not a verdict)
  - meets no scene dPSNR < -0.10 dB (partial data, not a verdict)
  - meets mean dLPIPS <= +0.002 (partial data, not a verdict)
**upstream main + S loads** over 2 scenes: time reduction median 6.3%, mean 6.3%, range 3.6–9.0%; ΔPSNR mean +0.026, worst garden -0.029; ΔLPIPS mean +0.0001; paired runs faster: 6/6
  - misses median time reduction >= 20% (partial data, not a verdict)
  - meets mean dPSNR >= -0.05 dB (partial data, not a verdict)
  - meets no scene dPSNR < -0.10 dB (partial data, not a verdict)
  - meets mean dLPIPS <= +0.002 (partial data, not a verdict)

## Outdoor / indoor and speedup factors (descriptive; scene speedup = mean over seeds of stock time / arm time)

| arm | group | scenes | time reduction median / mean | speedup median / mean | speedup range | aggregate speedup | ΔPSNR mean |
|---|---|---|---|---|---|---|---|
| upstream main | all | 2 | 6.3% / 6.3% | 1.068× / 1.068× | 1.039–1.097× | 1.075× | +0.063 |
| upstream main | outdoor | 1 | 8.8% / 8.8% | 1.097× / 1.097× | 1.097–1.097× | 1.097× | -0.004 |
| upstream main | indoor | 1 | 3.8% / 3.8% | 1.039× / 1.039× | 1.039–1.039× | 1.039× | +0.130 |
| upstream main + S loads | all | 2 | 6.3% / 6.3% | 1.068× / 1.068× | 1.037–1.098× | 1.075× | +0.026 |
| upstream main + S loads | outdoor | 1 | 9.0% / 9.0% | 1.098× / 1.098× | 1.098–1.098× | 1.098× | -0.029 |
| upstream main + S loads | indoor | 1 | 3.6% / 3.6% | 1.037× / 1.037× | 1.037–1.037× | 1.037× | +0.080 |

## Sensitivity: one-view failures (a test image > 3 dB below the same image in the scene's other runs)

- stock Faster-GS: 1 failing image(s) in 1 of 6 runs: room s1 image 13 (-4.7 dB)
- upstream main: 0 failing image(s) in 0 of 6 runs
- upstream main + S loads: 0 failing image(s) in 0 of 6 runs

These happen without the changes (stock has them), but these counts cannot show whether the changes alter how often they happen. The exclusion below is a secondary analysis; the verdict uses all images.

Deltas with those images removed from every run of the scene (paired per seed, mean of 3 seeds):

| scene | images removed | upstream main ΔPSNR | upstream main ΔLPIPS | upstream main + S loads ΔPSNR | upstream main + S loads ΔLPIPS |
|---|---|---|---|---|---|
| garden | - | -0.005 | +0.0004 | -0.029 | +0.0004 |
| room | 13 | +0.093 | -0.0003 | +0.044 | -0.0002 |

upstream main with one-view failures removed: ΔPSNR mean +0.044, worst garden -0.005, ΔLPIPS mean +0.0000; tolerances still met

upstream main + S loads with one-view failures removed: ΔPSNR mean +0.007, worst garden -0.029, ΔLPIPS mean +0.0001; tolerances still met

VERDICT: incomplete - not evaluated (2/9 scenes and 18/81 runs validated)
