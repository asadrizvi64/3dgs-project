# Alternative track R, R1 audit analysis: job 4503144 (node c68, repo 4607ea4)

Runs in the log: 5 (1 audit). Other runs: ar0_garden_stock_s0, ar0_garden_off_s0, ar0_garden_applymatch_s0, ar0_garden_applyplus_s0

## Validation

- ar0_garden_audit_s0: valid

## garden: ar0_garden_audit_s0 (2100 iterations, seed 0)

Settings: `mode=audit,iters=2000,target=match/4166000`

### Cost and budget

Reference estimate (stock seed-0 stage run, time to its own 30k PSNR): 208.6 s.

| k | warm-up s (steps + callbacks) | reinit s | timed target | stride | depth | candidates | neighbours | filter | dedupe | select | knn | rebuild | morton | budget left s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 5.3 (0.15 callbacks) | 0.97 | 1137951 | 2 | 0.17 | 0.07 | 0.12 | 0.48 | 0.02 | 0.05 | 0.05 | 0.01 | 0.00 | 202.4 |

### Counts and targets

| k | N before | grid candidates | after filter | after dedupe | removed (covered) | survivors | added | N after | targets |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 1137951 | 43817760 | 791201 | 561797 | 54829 | 1083122 | 54829 | 1137951 | 1137951: reachable (max 1644919); 4166000: UNREACHABLE (max 1644919) |

### Status of the grid candidates

| k | kept | empty | low alpha | mixture | geometry | neighbour outcomes (agree / out of view / target invalid / depth / reprojection) |
|---|---|---|---|---|---|---|
| 2000 | 1.8% | 0.0% | 24.9% | 68.1% | 5.2% | 21.1% / 1.1% / 63.1% / 8.3% / 6.4% |

### k = 2000: where candidates are rejected

By depth (scene units; share of the candidates in each bin):

| depth bin | candidates | kept | mixture | geometry | low alpha |
|---|---|---|---|---|---|
| 1.6-3.2 | 1842300 | 13.1% | 58.8% | 28.1% | 0.1% |
| 3.2-6.4 | 22331433 | 2.3% | 67.9% | 7.2% | 22.5% |
| 6.4-12.8 | 15746769 | 0.2% | 72.1% | 1.0% | 26.8% |
| 12.8-25.6 | 3695813 | 0.0% | 58.3% | 0.1% | 41.6% |
| 25.6-51.2 | 196660 | 0.0% | 46.1% | 0.0% | 53.9% |
| 51.2-102.4 | 4785 | 0.0% | 40.6% | 0.0% | 59.4% |

By image region (kept share; rows top to bottom):

| 0.0% | 1.2% | 0.5% |
|---|---|---|
| 1.4% | 6.2% | 2.3% |
| 0.7% | 2.6% | 1.4% |

Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view's neighbours; the same statistics for every class):

| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |
|---|---|---|---|---|---|---|---|
| kept | 6798 | 4373 | 20.9% | 34.4% | 44.7% | 0.368 | 0.696 |
| geometry | 12117 | 9019 | 9.1% | 25.2% | 65.7% | 0.143 | 0.400 |
| mixture | 12375 | 7853 | 6.8% | 24.1% | 69.2% | 0.126 | 0.348 |
| low_alpha | 12375 | 7737 | 7.4% | 23.0% | 69.6% | 0.121 | 0.321 |

COLMAP points (independent of the model; sparse, biased to texture): 85228 observations in the diagnostic views; transform deviation 1.8e-15; reprojection median 0.293 px, q90 0.640 px (checks the camera conventions).

| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 934 | 0.0049 | 86.9% | 100.0% |
| low_alpha | 19187 | 0.0248 | 23.3% | 77.4% |
| mixture | 62061 | 0.0255 | 15.6% | 88.3% |
| geometry | 3046 | 0.0064 | 75.6% | 99.9% |

Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes 0.0% of the previously covered pixels, newly covered 0.0%.

Agreement with the final model's depth (agreement with another reconstruction, not accuracy):

| audited status | pixels | rel. difference median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 276246 | 0.0014 | 99.0% | 100.0% |
| low_alpha | 339321 | 0.0064 | 64.6% | 95.7% |
| mixture | 10000000 | 0.0042 | 78.2% | 97.2% |
| geometry | 776044 | 0.0017 | 97.8% | 100.0% |

Neighbours: 62 views without any; overlap median 1.000, triangulation angle median 52.1 deg (q10 9.0, q90 124.8).

Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | relative depth std, white = 0.2+): `alt_reinit/logs/run_4503144/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view000.jpg`, `alt_reinit/logs/run_4503144/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view043.jpg`, `alt_reinit/logs/run_4503144/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view085.jpg`, `alt_reinit/logs/run_4503144/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view128.jpg`

## R1 gates (alt_reinit/PLAN.md)

G1: warm-up + reinit cost leaves a positive budget against the reference estimate. G2: kept candidates are more photo-consistent than geometry-rejected ones (consistent share and median, mean over neighbours). G3: rejection masks inspected (manual).

| scene | k | G1 budget left s | G1 | G2 kept / rejected consistent | G2 | G3 |
|---|---|---|---|---|---|---|
| garden | 2000 | 202.4 | PASS | 21% / 9% (medians 0.37 / 0.14) | PASS | inspect 4 images |

- garden: earliest k passing G1 and G2: 2000 (G3 pending manual inspection)

