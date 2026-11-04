### garden: forward blend_cu  (forward::blend_cu vs forward::blend_cu)

| metric | A+B+C | abcd2 | change |
|---|---|---|---|
| duration (us, ncu clock) | 704 | 678 | -3.7% |
| warp instructions executed (M) | 382.1 | 381.6 | -0.1% |
| issue slots busy (%) | 75.1 | 75.3 | +0.3% |
| IPC (per SM) | 3.00 | 3.01 | +0.3% |
| warp cycles per issued instruction | 14.39 | 14.34 | -0.3% |
| active threads per instruction | 25.5 | 25.5 | +0.1% |
| registers per thread | 40 | 40 | +0.0% |
| theoretical occupancy (%) | 75.0 | 75.0 | +0.0% |
| achieved occupancy (%) | 67.4 | 67.6 | +0.3% |
| L1/TEX throughput (%) | 44.5 | 44.7 | +0.3% |
| DRAM throughput (%) | 23.4 | 24.0 | +2.5% |
| **stalls** (warps per issued instruction) | | | |
| stall barrier | 4.84 | 4.81 | -0.03 |
| stall wait | 2.57 | 2.57 | +0.00 |
| stall short_scoreboard | 1.99 | 1.99 | +0.00 |
| stall not_selected | 1.93 | 1.94 | +0.02 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall long_scoreboard | 0.80 | 0.79 | -0.00 |

### garden: backward macro  (void backward::blend_backward_macro_cu<0, 4, 1, 0> vs void backward::blend_backward_macro_cu<0, 4, 1, 0, 1>)

| metric | A+B+C | abcd2 | change |
|---|---|---|---|
| duration (us, ncu clock) | 1601 | 1471 | -8.1% |
| warp instructions executed (M) | 1074.0 | 962.9 | -10.3% |
| issue slots busy (%) | 80.2 | 78.5 | -2.1% |
| IPC (per SM) | 3.19 | 3.12 | -2.2% |
| warp cycles per issued instruction | 7.78 | 7.98 | +2.5% |
| active threads per instruction | 21.5 | 21.2 | -1.3% |
| registers per thread | 71 | 72 | +1.4% |
| theoretical occupancy (%) | 43.8 | 43.8 | +0.0% |
| achieved occupancy (%) | 39.0 | 39.1 | +0.3% |
| L1/TEX throughput (%) | 69.3 | 45.5 | -34.3% |
| DRAM throughput (%) | 12.3 | 13.3 | +8.5% |
| **stalls** (warps per issued instruction) | | | |
| stall wait | 1.76 | 2.14 | +0.38 |
| stall not_selected | 2.00 | 1.62 | -0.39 |
| stall selected | 1.00 | 1.00 | +0.00 |
| stall long_scoreboard | 0.77 | 0.97 | +0.19 |
| stall branch_resolving | 0.66 | 0.80 | +0.14 |
| stall short_scoreboard | 0.68 | 0.55 | -0.13 |

### room: forward blend_cu  (forward::blend_cu vs forward::blend_cu)

| metric | A+B+C | abcd2 | change |
|---|---|---|---|
| duration (us, ncu clock) | 747 | 742 | -0.7% |
| warp instructions executed (M) | 434.3 | 433.7 | -0.2% |
| issue slots busy (%) | 80.4 | 80.4 | +0.0% |
| IPC (per SM) | 3.21 | 3.21 | +0.0% |
| warp cycles per issued instruction | 13.17 | 13.18 | +0.1% |
| active threads per instruction | 27.4 | 27.4 | -0.0% |
| registers per thread | 40 | 40 | +0.0% |
| theoretical occupancy (%) | 75.0 | 75.0 | +0.0% |
| achieved occupancy (%) | 66.2 | 66.2 | -0.0% |
| L1/TEX throughput (%) | 45.7 | 45.7 | +0.0% |
| DRAM throughput (%) | 14.2 | 14.3 | +0.6% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.92 | 2.91 | -0.01 |
| stall barrier | 2.84 | 2.87 | +0.03 |
| stall wait | 2.55 | 2.55 | +0.00 |
| stall short_scoreboard | 2.01 | 2.01 | -0.00 |
| stall selected | 1.00 | 1.00 | -0.00 |
| stall branch_resolving | 0.44 | 0.44 | +0.00 |

### room: backward macro  (void backward::blend_backward_macro_cu<0, 4, 1, 0> vs void backward::blend_backward_macro_cu<0, 4, 1, 0, 1>)

| metric | A+B+C | abcd2 | change |
|---|---|---|---|
| duration (us, ncu clock) | 1420 | 1308 | -7.9% |
| warp instructions executed (M) | 951.8 | 866.9 | -8.9% |
| issue slots busy (%) | 80.2 | 79.3 | -1.2% |
| IPC (per SM) | 3.20 | 3.16 | -1.3% |
| warp cycles per issued instruction | 7.48 | 7.60 | +1.6% |
| active threads per instruction | 23.2 | 23.2 | -0.2% |
| registers per thread | 71 | 72 | +1.4% |
| theoretical occupancy (%) | 43.8 | 43.8 | +0.0% |
| achieved occupancy (%) | 37.5 | 37.7 | +0.3% |
| L1/TEX throughput (%) | 62.7 | 42.2 | -32.7% |
| DRAM throughput (%) | 9.3 | 10.1 | +8.4% |
| **stalls** (warps per issued instruction) | | | |
| stall not_selected | 2.08 | 1.71 | -0.37 |
| stall wait | 1.77 | 2.08 | +0.31 |
| stall selected | 1.00 | 1.00 | -0.00 |
| stall long_scoreboard | 0.63 | 0.78 | +0.15 |
| stall short_scoreboard | 0.67 | 0.59 | -0.08 |
| stall branch_resolving | 0.50 | 0.64 | +0.14 |

CHECK PASSED: 2 launches of each kernel per file
