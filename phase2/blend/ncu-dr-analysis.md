### garden: forward blend_cu  (forward::blend_cu vs forward::blend_cu)

| metric | A+B+C | abcd | change |
|---|---|---|---|
| duration (us, ncu clock) | 684 | 687 | +0.4% |
| warp instructions executed (M) | 382.2 | 382.7 | +0.1% |
| issue slots busy (%) | 75.3 | 75.4 | +0.1% |
| IPC (per SM) | 3.01 | 3.01 | +0.1% |
| warp cycles per issued instruction | 14.35 | 14.36 | +0.0% |
| active threads per instruction | 25.5 | 25.5 | +0.0% |
| registers per thread | 40 | 40 | +0.0% |
| theoretical occupancy (%) | 75.0 | 75.0 | +0.0% |
| achieved occupancy (%) | 67.5 | 67.7 | +0.2% |
| L1/TEX throughput (%) | 44.6 | 44.7 | +0.1% |
| DRAM throughput (%) | 24.0 | 23.9 | -0.4% |
| **stalls** (warps per issued instruction) | | | |
| stall barrier | 4.81 | 4.81 | +0.00 |
| stall wait | 2.57 | 2.57 | -0.00 |
| stall short_scoreboard | 1.99 | 1.99 | -0.00 |
| stall not_selected | 1.93 | 1.93 | -0.00 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall long_scoreboard | 0.80 | 0.79 | -0.00 |

### garden: backward macro  (void backward::blend_backward_macro_cu<0, 4, 1, 0> vs void backward::blend_backward_macro_cu<0, 4, 1, 0, 1>)

| metric | A+B+C | abcd | change |
|---|---|---|---|
| duration (us, ncu clock) | 1603 | 2161 | +34.8% |
| warp instructions executed (M) | 1073.4 | 1088.6 | +1.4% |
| issue slots busy (%) | 80.2 | 60.1 | -25.0% |
| IPC (per SM) | 3.19 | 2.39 | -25.0% |
| warp cycles per issued instruction | 7.78 | 6.03 | -22.5% |
| active threads per instruction | 21.5 | 22.7 | +5.5% |
| registers per thread | 71 | 84 | +18.3% |
| theoretical occupancy (%) | 43.8 | 25.0 | -42.9% |
| achieved occupancy (%) | 39.0 | 22.7 | -41.8% |
| L1/TEX throughput (%) | 69.3 | 30.2 | -56.5% |
| DRAM throughput (%) | 12.3 | 9.1 | -25.6% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.00 | 0.62 | -1.38 |
| stall wait | 1.76 | 1.90 | +0.14 |
| stall selected | 1.00 | 1.00 | -0.00 |
| stall long_scoreboard | 0.77 | 0.93 | +0.16 |
| stall short_scoreboard | 0.68 | 0.38 | -0.31 |
| stall branch_resolving | 0.66 | 0.67 | +0.01 |

### room: forward blend_cu  (forward::blend_cu vs forward::blend_cu)

| metric | A+B+C | abcd | change |
|---|---|---|---|
| duration (us, ncu clock) | 733 | 744 | +1.6% |
| warp instructions executed (M) | 431.0 | 439.5 | +2.0% |
| issue slots busy (%) | 80.3 | 80.6 | +0.5% |
| IPC (per SM) | 3.21 | 3.22 | +0.5% |
| warp cycles per issued instruction | 13.19 | 13.17 | -0.2% |
| active threads per instruction | 27.4 | 27.4 | +0.2% |
| registers per thread | 40 | 40 | +0.0% |
| theoretical occupancy (%) | 75.0 | 75.0 | +0.0% |
| achieved occupancy (%) | 66.2 | 66.4 | +0.3% |
| L1/TEX throughput (%) | 45.6 | 45.9 | +0.6% |
| DRAM throughput (%) | 14.4 | 14.4 | -0.4% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.91 | 2.94 | +0.03 |
| stall barrier | 2.87 | 2.82 | -0.05 |
| stall wait | 2.55 | 2.55 | -0.00 |
| stall short_scoreboard | 2.01 | 2.01 | +0.00 |
| stall selected | 1.00 | 1.00 | -0.00 |
| stall branch_resolving | 0.45 | 0.44 | -0.00 |

### room: backward macro  (void backward::blend_backward_macro_cu<0, 4, 1, 0> vs void backward::blend_backward_macro_cu<0, 4, 1, 0, 1>)

| metric | A+B+C | abcd | change |
|---|---|---|---|
| duration (us, ncu clock) | 1415 | 1895 | +34.0% |
| warp instructions executed (M) | 947.3 | 972.0 | +2.6% |
| issue slots busy (%) | 80.2 | 61.1 | -23.8% |
| IPC (per SM) | 3.20 | 2.44 | -23.8% |
| warp cycles per issued instruction | 7.49 | 5.74 | -23.4% |
| active threads per instruction | 23.2 | 24.3 | +5.0% |
| registers per thread | 71 | 84 | +18.3% |
| theoretical occupancy (%) | 43.8 | 25.0 | -42.9% |
| achieved occupancy (%) | 37.6 | 21.9 | -41.6% |
| L1/TEX throughput (%) | 62.7 | 28.8 | -54.1% |
| DRAM throughput (%) | 9.4 | 7.0 | -24.8% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.09 | 0.63 | -1.46 |
| stall wait | 1.77 | 1.89 | +0.12 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall long_scoreboard | 0.64 | 0.80 | +0.17 |
| stall short_scoreboard | 0.67 | 0.41 | -0.26 |
| stall branch_resolving | 0.51 | 0.54 | +0.04 |

CHECK PASSED: 2 launches of each kernel per file
