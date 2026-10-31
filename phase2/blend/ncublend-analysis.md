### garden: forward blend_cu  (forward::blend_cu vs void forward::blend_cu<1>)

| metric | A+B+C | culling | change |
|---|---|---|---|
| duration (us, ncu clock) | 685 | 972 | +41.8% |
| warp instructions executed (M) | 380.9 | 409.2 | +7.4% |
| issue slots busy (%) | 75.3 | 53.9 | -28.4% |
| IPC (per SM) | 3.01 | 2.15 | -28.5% |
| warp cycles per issued instruction | 14.35 | 10.90 | -24.1% |
| active threads per instruction | 25.5 | 23.3 | -8.8% |
| registers per thread | 40 | 40 | +0.0% |
| theoretical occupancy (%) | 75.0 | 37.5 | -50.0% |
| achieved occupancy (%) | 67.6 | 36.7 | -45.7% |
| L1/TEX throughput (%) | 44.6 | 29.0 | -35.1% |
| DRAM throughput (%) | 23.8 | 15.9 | -33.2% |
| **stalls** (warps per issued instruction) | | | |
| stall barrier | 4.81 | 3.89 | -0.92 |
| stall wait | 2.57 | 2.17 | -0.39 |
| stall short_scoreboard | 1.99 | 1.89 | -0.11 |
| stall not_selected | 1.94 | 0.60 | -1.34 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall long_scoreboard | 0.80 | 0.52 | -0.28 |

### garden: backward macro  (void backward::blend_backward_macro_cu<0, 4, 1, 0> vs void backward::blend_backward_macro_cu<0, 4, 1, 0, 1>)

| metric | A+B+C | culling | change |
|---|---|---|---|
| duration (us, ncu clock) | 1597 | 1643 | +2.9% |
| warp instructions executed (M) | 1070.3 | 1131.1 | +5.7% |
| issue slots busy (%) | 80.1 | 82.5 | +2.9% |
| IPC (per SM) | 3.19 | 3.28 | +2.8% |
| warp cycles per issued instruction | 7.78 | 8.54 | +9.8% |
| active threads per instruction | 21.5 | 21.2 | -1.7% |
| registers per thread | 71 | 64 | -9.9% |
| theoretical occupancy (%) | 43.8 | 50.0 | +14.3% |
| achieved occupancy (%) | 39.0 | 44.0 | +12.9% |
| L1/TEX throughput (%) | 69.3 | 68.1 | -1.7% |
| DRAM throughput (%) | 12.3 | 12.0 | -2.2% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.00 | 2.57 | +0.57 |
| stall wait | 1.76 | 1.76 | +0.00 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall long_scoreboard | 0.77 | 0.61 | -0.17 |
| stall short_scoreboard | 0.68 | 0.74 | +0.05 |
| stall branch_resolving | 0.66 | 0.71 | +0.05 |

### room: forward blend_cu  (forward::blend_cu vs void forward::blend_cu<1>)

| metric | A+B+C | culling | change |
|---|---|---|---|
| duration (us, ncu clock) | 750 | 907 | +20.9% |
| warp instructions executed (M) | 435.0 | 445.1 | +2.3% |
| issue slots busy (%) | 80.4 | 63.2 | -21.4% |
| IPC (per SM) | 3.22 | 2.53 | -21.5% |
| warp cycles per issued instruction | 13.18 | 9.16 | -30.5% |
| active threads per instruction | 27.4 | 26.4 | -3.7% |
| registers per thread | 40 | 40 | +0.0% |
| theoretical occupancy (%) | 75.0 | 37.5 | -50.0% |
| achieved occupancy (%) | 66.2 | 36.2 | -45.3% |
| L1/TEX throughput (%) | 45.8 | 33.4 | -26.9% |
| DRAM throughput (%) | 14.1 | 11.6 | -17.6% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.93 | 0.83 | -2.10 |
| stall barrier | 2.85 | 2.04 | -0.81 |
| stall wait | 2.55 | 2.17 | -0.38 |
| stall short_scoreboard | 2.01 | 1.97 | -0.04 |
| stall selected | 1.00 | 1.00 | -0.00 |
| stall branch_resolving | 0.44 | 0.41 | -0.03 |

### room: backward macro  (void backward::blend_backward_macro_cu<0, 4, 1, 0> vs void backward::blend_backward_macro_cu<0, 4, 1, 0, 1>)

| metric | A+B+C | culling | change |
|---|---|---|---|
| duration (us, ncu clock) | 1423 | 1463 | +2.8% |
| warp instructions executed (M) | 953.9 | 995.7 | +4.4% |
| issue slots busy (%) | 80.1 | 81.5 | +1.7% |
| IPC (per SM) | 3.19 | 3.24 | +1.6% |
| warp cycles per issued instruction | 7.47 | 8.21 | +9.9% |
| active threads per instruction | 23.2 | 22.9 | -1.6% |
| registers per thread | 71 | 64 | -9.9% |
| theoretical occupancy (%) | 43.8 | 50.0 | +14.3% |
| achieved occupancy (%) | 37.4 | 41.8 | +11.8% |
| L1/TEX throughput (%) | 62.6 | 61.4 | -1.9% |
| DRAM throughput (%) | 9.3 | 9.1 | -2.7% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.07 | 2.52 | +0.45 |
| stall wait | 1.77 | 1.77 | +0.00 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall short_scoreboard | 0.67 | 0.76 | +0.09 |
| stall long_scoreboard | 0.63 | 0.53 | -0.11 |
| stall branch_resolving | 0.50 | 0.57 | +0.07 |

CHECK PASSED: 2 launches of each kernel per file
