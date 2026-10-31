# Alternative track R, R1 audit analysis: job 4503153 (node c95, repo 4607ea4)

Runs in the log: 2 (2 audit). Other runs: none

## Validation

- ar1_garden_audit_s0: valid
- ar1_room_audit_s0: valid

## garden: ar1_garden_audit_s0 (30000 iterations, seed 0)

Settings: `mode=audit,iters=2000:5000,target=4166000/match`

### Cost and budget

Reference estimate (stock seed-0 stage run, time to its own 30k PSNR): 208.6 s.

| k | warm-up s (steps + callbacks) | reinit s | timed target | stride | depth | candidates | neighbours | filter | dedupe | select | knn | rebuild | morton | budget left s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 5.2 (0.16 callbacks) | 1.54 | 1138286 | 1 | 0.17 | 0.12 | 0.21 | 0.89 | 0.03 | 0.07 | 0.04 | 0.01 | 0.00 | 201.9 |
| 5000 | 22.8 (0.48 callbacks) | 1.48 | 4166000 | 2 | 0.22 | 0.03 | 0.12 | 0.78 | 0.02 | 0.15 | 0.12 | 0.02 | 0.02 | 184.4 |

### Counts and targets

| k | N before | grid candidates | after filter | after dedupe | removed (covered) | survivors | added | N after | targets |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 1138286 | 175406280 | 1075435 | 954026 | 10886 | 1127400 | 10886 | 1138286 | 4166000: UNREACHABLE (max 2081426); 1138286: reachable (max 2081426) |
| 5000 | 3119719 | 43817760 | 12742478 | 5314238 | 1383720 | 1735999 | 2430001 | 4166000 | 4166000: reachable (max 7050237); 3119719: reachable (max 7050237) |

### Status of the grid candidates

| k | kept | empty | low alpha | mixture | geometry | neighbour outcomes (agree / out of view / target invalid / depth / reprojection) |
|---|---|---|---|---|---|---|
| 2000 | 0.6% | 0.0% | 25.3% | 69.9% | 4.2% | 12.0% / 0.5% / 82.5% / 1.9% / 3.0% |
| 5000 | 29.1% | 0.0% | 34.0% | 19.5% | 17.4% | 51.5% / 6.5% / 29.8% / 6.3% / 5.9% |

### k = 2000: where candidates are rejected

By depth (scene units; share of the candidates in each bin):

| depth bin | candidates | kept | mixture | geometry | low alpha |
|---|---|---|---|---|---|
| 1.6-3.2 | 7445922 | 4.6% | 70.2% | 25.1% | 0.2% |
| 3.2-6.4 | 90007099 | 0.8% | 70.0% | 5.8% | 23.3% |
| 6.4-12.8 | 62286125 | 0.0% | 72.8% | 0.4% | 26.7% |
| 12.8-25.6 | 14647120 | 0.0% | 58.1% | 0.0% | 41.8% |
| 25.6-51.2 | 1007574 | 0.0% | 44.4% | 0.0% | 55.6% |
| 51.2-102.4 | 12440 | 0.0% | 3.3% | 0.0% | 96.7% |

By image region (kept share; rows top to bottom):

| 0.0% | 0.4% | 0.0% |
|---|---|---|
| 0.5% | 2.5% | 0.9% |
| 0.1% | 0.9% | 0.2% |

Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view's neighbours; the same statistics for every class):

| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |
|---|---|---|---|---|---|---|---|
| kept | 4143 | 2878 | 18.7% | 35.3% | 45.9% | 0.340 | 0.671 |
| geometry | 11250 | 8383 | 4.9% | 20.4% | 74.8% | 0.076 | 0.340 |
| mixture | 11250 | 6726 | 4.7% | 19.0% | 76.3% | 0.088 | 0.289 |
| low_alpha | 11250 | 6764 | 6.3% | 20.1% | 73.6% | 0.099 | 0.300 |

COLMAP points (independent of the model; sparse, biased to texture): 85228 observations in the diagnostic views; transform deviation 1.8e-15; reprojection median 0.293 px, q90 0.640 px (checks the camera conventions).

| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 57 | 0.0071 | 73.7% | 100.0% |
| low_alpha | 19369 | 0.0214 | 26.4% | 84.5% |
| mixture | 63256 | 0.0273 | 11.4% | 87.8% |
| geometry | 2546 | 0.0065 | 73.0% | 99.8% |

Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes 0.0% of the previously covered pixels, newly covered 0.0%.

Agreement with the final model's depth (agreement with another reconstruction, not accuracy):

| audited status | pixels | rel. difference median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 18223 | 0.0085 | 59.6% | 99.9% |
| low_alpha | 1732859 | 0.0163 | 33.7% | 89.9% |
| mixture | 10000000 | 0.0242 | 16.7% | 88.4% |
| geometry | 798735 | 0.0070 | 66.0% | 99.4% |

Neighbours: 71 views without any; overlap median 1.000, triangulation angle median 88.1 deg (q10 11.5, q90 138.6).

Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | relative depth std, white = 0.2+): `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_02000/masks/view000.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_02000/masks/view043.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_02000/masks/view085.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_02000/masks/view128.jpg`

### k = 5000: where candidates are rejected

By depth (scene units; share of the candidates in each bin):

| depth bin | candidates | kept | mixture | geometry | low alpha |
|---|---|---|---|---|---|
| 0-0.1 | 714 | 0.0% | 0.0% | 0.0% | 0.0% |
| 1.6-3.2 | 1737934 | 48.2% | 32.2% | 19.6% | 0.1% |
| 3.2-6.4 | 22138095 | 40.8% | 14.7% | 20.1% | 24.3% |
| 6.4-12.8 | 15669752 | 17.5% | 24.7% | 15.8% | 42.0% |
| 12.8-25.6 | 4182908 | 3.1% | 19.5% | 8.5% | 69.0% |
| 25.6-51.2 | 87006 | 0.0% | 28.9% | 5.0% | 66.1% |
| 51.2-102.4 | 1351 | 0.0% | 4.4% | 14.4% | 81.1% |

By image region (kept share; rows top to bottom):

| 13.6% | 16.7% | 14.1% |
|---|---|---|
| 28.1% | 48.1% | 26.8% |
| 38.0% | 39.2% | 37.3% |

Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view's neighbours; the same statistics for every class):

| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |
|---|---|---|---|---|---|---|---|
| kept | 20125 | 16973 | 34.2% | 47.9% | 18.0% | 0.598 | 0.807 |
| geometry | 20125 | 17703 | 25.4% | 47.0% | 27.6% | 0.507 | 0.747 |
| mixture | 20125 | 16420 | 12.9% | 35.4% | 51.7% | 0.283 | 0.565 |
| low_alpha | 20125 | 17757 | 27.3% | 42.4% | 30.3% | 0.506 | 0.726 |

COLMAP points (independent of the model; sparse, biased to texture): 85228 observations in the diagnostic views; transform deviation 1.8e-15; reprojection median 0.293 px, q90 0.640 px (checks the camera conventions).

| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 30799 | 0.0035 | 91.2% | 99.9% |
| low_alpha | 28568 | 0.0059 | 67.7% | 96.4% |
| mixture | 12182 | 0.0125 | 37.6% | 96.4% |
| geometry | 13679 | 0.0041 | 85.0% | 99.7% |

Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes 0.0% of the previously covered pixels, newly covered 0.0%.

Agreement with the final model's depth (agreement with another reconstruction, not accuracy):

| audited status | pixels | rel. difference median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 4978836 | 0.0032 | 91.8% | 100.0% |
| low_alpha | 1980430 | 0.0045 | 76.1% | 97.4% |
| mixture | 3002707 | 0.0107 | 46.6% | 93.4% |
| geometry | 2730209 | 0.0038 | 87.5% | 99.7% |

Neighbours: 0 views without any; overlap median 0.941, triangulation angle median 8.6 deg (q10 5.0, q90 15.5).

Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | relative depth std, white = 0.2+): `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_05000/masks/view000.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_05000/masks/view043.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_05000/masks/view085.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_garden_audit_s0/reinit_audit/k_05000/masks/view128.jpg`

## room: ar1_room_audit_s0 (30000 iterations, seed 0)

Settings: `mode=audit,iters=2000:5000,target=1345000/match`

### Cost and budget

Reference estimate (stock seed-0 stage run, time to its own 30k PSNR): 149.4 s.

| k | warm-up s (steps + callbacks) | reinit s | timed target | stride | depth | candidates | neighbours | filter | dedupe | select | knn | rebuild | morton | budget left s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 6.6 (0.12 callbacks) | 3.58 | 1345000 | 1 | 0.42 | 0.22 | 0.41 | 2.32 | 0.03 | 0.12 | 0.06 | 0.00 | 0.01 | 139.2 |
| 5000 | 18.5 (0.27 callbacks) | 2.15 | 1345000 | 2 | 0.32 | 0.06 | 0.22 | 1.35 | 0.02 | 0.11 | 0.05 | 0.00 | 0.00 | 128.8 |

### Counts and targets

| k | N before | grid candidates | after filter | after dedupe | removed (covered) | survivors | added | N after | targets |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 453080 | 439597152 | 2405210 | 2062029 | 4270 | 448810 | 896190 | 1345000 | 1345000: reachable (max 2510839); 453080: reachable (max 2510839) |
| 5000 | 968429 | 109828704 | 13559782 | 7182142 | 204756 | 763673 | 581327 | 1345000 | 1345000: reachable (max 7945815); 968429: reachable (max 7945815) |

### Status of the grid candidates

| k | kept | empty | low alpha | mixture | geometry | neighbour outcomes (agree / out of view / target invalid / depth / reprojection) |
|---|---|---|---|---|---|---|
| 2000 | 0.5% | 0.0% | 5.3% | 89.5% | 4.6% | 9.8% / 6.6% / 73.6% / 4.3% / 5.7% |
| 5000 | 12.3% | 0.0% | 11.9% | 49.8% | 25.9% | 27.6% / 8.1% / 39.2% / 8.8% / 16.3% |

### k = 2000: where candidates are rejected

By depth (scene units; share of the candidates in each bin):

| depth bin | candidates | kept | mixture | geometry | low alpha |
|---|---|---|---|---|---|
| 0.8-1.6 | 312334 | 0.2% | 86.6% | 13.2% | 0.0% |
| 1.6-3.2 | 12832070 | 1.6% | 79.7% | 18.0% | 0.7% |
| 3.2-6.4 | 231761256 | 0.8% | 85.7% | 6.6% | 6.9% |
| 6.4-12.8 | 180083124 | 0.2% | 94.9% | 1.5% | 3.4% |
| 12.8-25.6 | 13516173 | 0.0% | 94.2% | 0.0% | 5.7% |
| 25.6-51.2 | 947959 | 0.0% | 82.8% | 0.0% | 17.2% |
| 51.2-102.4 | 144236 | 0.0% | 85.4% | 0.0% | 14.6% |

By image region (kept share; rows top to bottom):

| 1.4% | 0.4% | 0.1% |
|---|---|---|
| 1.1% | 0.4% | 0.2% |
| 0.6% | 0.4% | 0.3% |

Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view's neighbours; the same statistics for every class):

| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |
|---|---|---|---|---|---|---|---|
| kept | 10247 | 3537 | 32.7% | 30.1% | 37.2% | 0.474 | 0.720 |
| geometry | 16589 | 4846 | 23.7% | 27.9% | 48.4% | 0.319 | 0.613 |
| mixture | 16724 | 3467 | 20.9% | 22.1% | 57.0% | 0.206 | 0.443 |
| low_alpha | 16650 | 2817 | 3.8% | 12.6% | 83.7% | 0.023 | 0.175 |

COLMAP points (independent of the model; sparse, biased to texture): 50720 observations in the diagnostic views; transform deviation 1.8e-15; reprojection median 0.320 px, q90 0.969 px (checks the camera conventions).

| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 110 | 0.0071 | 74.5% | 99.1% |
| low_alpha | 1597 | 0.2484 | 9.4% | 32.3% |
| mixture | 46280 | 0.0291 | 15.5% | 70.5% |
| geometry | 2733 | 0.0091 | 55.5% | 99.2% |

Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes 0.0% of the previously covered pixels, newly covered 0.0%.

Agreement with the final model's depth (agreement with another reconstruction, not accuracy):

| audited status | pixels | rel. difference median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 41054 | 0.0075 | 60.2% | 85.5% |
| low_alpha | 970675 | 0.0499 | 19.8% | 50.0% |
| mixture | 10000000 | 0.0288 | 18.0% | 67.8% |
| geometry | 1002924 | 0.0071 | 63.1% | 96.6% |

Neighbours: 46 views without any; overlap median 0.996, triangulation angle median 35.1 deg (q10 4.7, q90 112.1).

Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | relative depth std, white = 0.2+): `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_02000/masks/view000.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_02000/masks/view072.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_02000/masks/view145.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_02000/masks/view217.jpg`

### k = 5000: where candidates are rejected

By depth (scene units; share of the candidates in each bin):

| depth bin | candidates | kept | mixture | geometry | low alpha |
|---|---|---|---|---|---|
| 0-0.1 | 390 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.4-0.8 | 383 | 0.0% | 0.0% | 0.0% | 100.0% |
| 0.8-1.6 | 74402 | 0.2% | 44.1% | 45.4% | 10.3% |
| 1.6-3.2 | 3671482 | 5.2% | 59.4% | 30.3% | 5.1% |
| 3.2-6.4 | 64473505 | 9.5% | 47.6% | 28.9% | 14.0% |
| 6.4-12.8 | 39941993 | 18.1% | 51.1% | 21.6% | 9.1% |
| 12.8-25.6 | 1662278 | 0.9% | 83.2% | 2.7% | 13.2% |
| 25.6-51.2 | 4271 | 0.0% | 69.5% | 0.0% | 30.5% |

By image region (kept share; rows top to bottom):

| 13.3% | 14.1% | 11.5% |
|---|---|---|
| 12.3% | 14.7% | 13.1% |
| 10.4% | 12.6% | 9.1% |

Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view's neighbours; the same statistics for every class):

| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |
|---|---|---|---|---|---|---|---|
| kept | 20116 | 6462 | 46.5% | 26.9% | 26.6% | 0.660 | 0.829 |
| geometry | 20128 | 5698 | 33.9% | 28.8% | 37.3% | 0.485 | 0.718 |
| mixture | 20128 | 4422 | 21.9% | 27.7% | 50.3% | 0.295 | 0.542 |
| low_alpha | 20128 | 5190 | 27.9% | 20.8% | 51.3% | 0.280 | 0.533 |

COLMAP points (independent of the model; sparse, biased to texture): 50720 observations in the diagnostic views; transform deviation 1.8e-15; reprojection median 0.320 px, q90 0.969 px (checks the camera conventions).

| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 9138 | 0.0045 | 82.5% | 99.2% |
| low_alpha | 3154 | 0.0181 | 34.1% | 78.0% |
| mixture | 24158 | 0.0155 | 30.3% | 87.7% |
| geometry | 14270 | 0.0061 | 69.7% | 98.1% |

Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes 0.0% of the previously covered pixels, newly covered 0.0%.

Agreement with the final model's depth (agreement with another reconstruction, not accuracy):

| audited status | pixels | rel. difference median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 2656000 | 0.0036 | 86.4% | 99.9% |
| low_alpha | 1501071 | 0.0120 | 44.3% | 88.1% |
| mixture | 10000000 | 0.0127 | 41.3% | 86.9% |
| geometry | 6477745 | 0.0053 | 72.5% | 98.1% |

Neighbours: 0 views without any; overlap median 0.938, triangulation angle median 14.8 deg (q10 4.0, q90 98.6).

Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | relative depth std, white = 0.2+): `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_05000/masks/view000.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_05000/masks/view072.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_05000/masks/view145.jpg`, `alt_reinit/logs/run_4503153/runs/ar1_room_audit_s0/reinit_audit/k_05000/masks/view217.jpg`

## R1 gates (alt_reinit/PLAN.md)

G1: warm-up + reinit cost leaves a positive budget against the reference estimate. G2: kept candidates are more photo-consistent than geometry-rejected ones (consistent share and median, mean over neighbours). G3: rejection masks inspected (manual).

| scene | k | G1 budget left s | G1 | G2 kept / rejected consistent | G2 | G3 |
|---|---|---|---|---|---|---|
| garden | 2000 | 201.9 | PASS | 19% / 5% (medians 0.34 / 0.08) | PASS | inspect 4 images |
| garden | 5000 | 184.4 | PASS | 34% / 25% (medians 0.60 / 0.51) | PASS | inspect 4 images |
| room | 2000 | 139.2 | PASS | 33% / 24% (medians 0.47 / 0.32) | PASS | inspect 4 images |
| room | 5000 | 128.8 | PASS | 46% / 34% (medians 0.66 / 0.48) | PASS | inspect 4 images |

- garden: earliest k passing G1 and G2: 2000 (G3 pending manual inspection)
- room: earliest k passing G1 and G2: 2000 (G3 pending manual inspection)

