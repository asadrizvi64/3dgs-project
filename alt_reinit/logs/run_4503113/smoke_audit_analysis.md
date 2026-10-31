# Alternative track R, R1 audit analysis: job 4503113 (node c10, repo fcda5ea)

Runs in the log: 5 (1 audit). Other runs: ar0_garden_stock_s0, ar0_garden_off_s0, ar0_garden_applymatch_s0, ar0_garden_applyplus_s0

## Validation

- ar0_garden_audit_s0: valid

## garden: ar0_garden_audit_s0 (2100 iterations, seed 0)

Settings: `mode=audit,iters=2000,target=match/4166000`

### Cost and budget

Reference estimate (stock seed-0 stage run, time to its own 30k PSNR): 208.6 s.

| k | warm-up s (steps + callbacks) | reinit s | timed target | stride | depth | candidates | neighbours | filter | dedupe | select | knn | rebuild | morton | budget left s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 5.4 (0.14 callbacks) | 0.88 | 1138781 | 2 | 0.17 | 0.06 | 0.11 | 0.41 | 0.02 | 0.04 | 0.05 | 0.01 | 0.00 | 202.4 |

### Counts and targets

| k | N before | grid candidates | after filter | after dedupe | removed (covered) | survivors | added | N after | targets |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 1138781 | 43817760 | 411551 | 327676 | 29493 | 1109288 | 29493 | 1138781 | 1138781: reachable (max 1436964); 4166000: UNREACHABLE (max 1436964) |

### Status of the grid candidates

| k | kept | empty | low alpha | mixture | geometry | neighbour outcomes (agree / out of view / target invalid / depth / reprojection) |
|---|---|---|---|---|---|---|
| 2000 | 0.9% | 0.0% | 24.0% | 70.8% | 4.3% | 16.5% / 1.2% / 71.1% / 6.3% / 5.0% |

### k = 2000: where candidates are rejected

By depth (scene units; share of the candidates in each bin):

| depth bin | candidates | kept | mixture | geometry | low alpha |
|---|---|---|---|---|---|
| 1.6-3.2 | 1945537 | 8.3% | 68.5% | 22.9% | 0.2% |
| 3.2-6.4 | 22136189 | 1.1% | 72.2% | 6.2% | 20.4% |
| 6.4-12.8 | 15594165 | 0.0% | 73.5% | 0.2% | 26.2% |
| 12.8-25.6 | 3849935 | 0.0% | 55.3% | 0.0% | 44.7% |
| 25.6-51.2 | 288094 | 0.0% | 41.7% | 0.0% | 58.3% |
| 51.2-102.4 | 3840 | 0.0% | 69.1% | 0.0% | 30.9% |

By image region (kept share; rows top to bottom):

| 0.0% | 0.3% | 0.0% |
|---|---|---|
| 0.9% | 2.6% | 1.3% |
| 0.5% | 1.9% | 0.9% |

Photo evidence (training photos, 7x7 ZNCC between a candidate and its projections into the view's neighbours; the same statistics for every class):

| class | samples | verifiable | consistent (>= 0.7) | ambiguous | inconsistent (< 0.3) | median, mean over neighbours | median, best neighbour |
|---|---|---|---|---|---|---|---|
| kept | 4456 | 2991 | 16.9% | 37.7% | 45.4% | 0.347 | 0.708 |
| geometry | 10349 | 7521 | 6.1% | 22.4% | 71.6% | 0.105 | 0.367 |
| mixture | 10500 | 6546 | 6.4% | 20.9% | 72.7% | 0.105 | 0.317 |
| low_alpha | 10500 | 6117 | 7.3% | 22.9% | 69.8% | 0.118 | 0.309 |

COLMAP points (independent of the model; sparse, biased to texture): 85228 observations in the diagnostic views; transform deviation 1.8e-15; reprojection median 0.293 px, q90 0.640 px (checks the camera conventions).

| status at the observed pixel | points | depth rel. error median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 779 | 0.0047 | 85.0% | 99.9% |
| low_alpha | 18358 | 0.0308 | 21.2% | 65.6% |
| mixture | 62879 | 0.0324 | 12.4% | 76.1% |
| geometry | 3212 | 0.0057 | 81.1% | 99.8% |

Coverage after the replacement (all opacities forced to 0.99, diagnostic views): holes 0.0% of the previously covered pixels, newly covered 0.0%.

Agreement with the final model's depth (agreement with another reconstruction, not accuracy):

| audited status | pixels | rel. difference median | within 1% | within 5% |
|---|---|---|---|---|
| kept | 223795 | 0.0015 | 98.9% | 100.0% |
| low_alpha | 346773 | 0.0068 | 59.4% | 88.6% |
| mixture | 10000000 | 0.0044 | 73.3% | 97.4% |
| geometry | 894517 | 0.0016 | 98.0% | 100.0% |

Neighbours: 77 views without any; overlap median 1.000, triangulation angle median 66.5 deg (q10 9.9, q90 140.1).

Masks to inspect (photo | status: green kept, red geometry, orange mixture, blue low alpha, black empty | relative depth std, white = 0.2+): `alt_reinit/logs/run_4503113/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view000.jpg`, `alt_reinit/logs/run_4503113/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view043.jpg`, `alt_reinit/logs/run_4503113/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view085.jpg`, `alt_reinit/logs/run_4503113/runs/ar0_garden_audit_s0/reinit_audit/k_02000/masks/view128.jpg`

## R1 gates (alt_reinit/PLAN.md)

G1: warm-up + reinit cost leaves a positive budget against the reference estimate. G2: kept candidates are more photo-consistent than geometry-rejected ones (consistent share and median, mean over neighbours). G3: rejection masks inspected (manual).

| scene | k | G1 budget left s | G1 | G2 kept / rejected consistent | G2 | G3 |
|---|---|---|---|---|---|---|
| garden | 2000 | 202.4 | PASS | 17% / 6% (medians 0.35 / 0.10) | PASS | inspect 4 images |

- garden: earliest k passing G1 and G2: 2000 (G3 pending manual inspection)

