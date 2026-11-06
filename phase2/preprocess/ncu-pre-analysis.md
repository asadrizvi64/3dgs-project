### garden (abcd, step 20000)

| metric | forward preprocess | preprocess backward |
|---|---|---|
| duration (us) | 270.8 | 1213.4 |
| threads launched | 4.15 M | 4.15 M |
| registers / thread | 56 | 80 |
| instructions (M) | 85.2 | 41.8 |
| DRAM read (MB) | 428 | 476 |
| DRAM write (MB) | 89 | 406 |
| DRAM bandwidth (GB/s) | 1908 | 727 |
| DRAM throughput (% peak) | 78.0 | 29.7 |
| memory SOL (%) | 78.0 | 78.7 |
| L2 SOL (%) | 58.1 | 78.7 |
| SM SOL (%) | 37.5 | 7.1 |
| issue slots busy (%) | 38.8 | 4.2 |
| occupancy theo / achieved (%) | 56.2 / 47.3 | 37.5 / 33.4 |
| global load sectors / request | 21.5 | 22.8 |
| global store sectors / request | 7.7 | 27.4 |
| active threads / warp instr. | 26.2 | 30.9 |

- forward preprocess stalls (warps per issued instr., total 19.5): long_scoreboard 9.52, mio_throttle 2.56, wait 1.86, short_scoreboard 1.26, lg_throttle 1.16
- preprocess backward stalls (warps per issued instr., total 129.4): lg_throttle 59.26, long_scoreboard 53.69, mio_throttle 6.92, short_scoreboard 3.62, drain 3.29

Preprocess backward by instruction class (one launch, `garden_abcd.pre_sass.csv`):

| class | instructions (M) | share | stall samples share |
|---|---|---|---|
| fp32 arithmetic | 30.71 | 72.1% | 31.4% |
| other (int, move, predicate) | 3.30 | 7.7% | 9.3% |
| global load | 3.22 | 7.6% | 34.2% |
| global store | 2.62 | 6.1% | 16.3% |
| shared/const load | 1.70 | 4.0% | 5.0% |
| control | 0.55 | 1.3% | 2.5% |
| MUFU (exp, rcp, sqrt) | 0.52 | 1.2% | 1.2% |

- most stall samples by opcode: LDG 34.2%, FMUL 23.6%, STG 16.3%, ISETP 6.4%, LDC 5.0%, FFMA 4.6%

### room (abcd, step 20000)

| metric | forward preprocess | preprocess backward |
|---|---|---|
| duration (us) | 196.8 | 345.6 |
| threads launched | 1.35 M | 1.35 M |
| registers / thread | 56 | 80 |
| instructions (M) | 57.3 | 11.8 |
| DRAM read (MB) | 130 | 131 |
| DRAM write (MB) | 22 | 97 |
| DRAM bandwidth (GB/s) | 769 | 661 |
| DRAM throughput (% peak) | 34.4 | 27.0 |
| memory SOL (%) | 34.4 | 75.3 |
| L2 SOL (%) | 25.8 | 75.3 |
| SM SOL (%) | 38.9 | 6.8 |
| issue slots busy (%) | 60.2 | 4.3 |
| occupancy theo / achieved (%) | 56.2 / 39.7 | 37.5 / 32.4 |
| global load sectors / request | 20.0 | 22.1 |
| global store sectors / request | 7.4 | 26.6 |
| active threads / warp instr. | 26.6 | 30.2 |

- forward preprocess stalls (warps per issued instr., total 10.5): long_scoreboard 3.16, wait 2.28, not_selected 1.96, selected 1.00, math_pipe_throttle 0.68
- preprocess backward stalls (warps per issued instr., total 120.4): lg_throttle 55.34, long_scoreboard 51.02, mio_throttle 5.23, short_scoreboard 3.21, drain 2.85

Preprocess backward by instruction class (one launch, `room_abcd.pre_sass.csv`):

| class | instructions (M) | share | stall samples share |
|---|---|---|---|
| fp32 arithmetic | 9.86 | 72.0% | 33.2% |
| other (int, move, predicate) | 1.06 | 7.8% | 9.7% |
| global load | 1.04 | 7.6% | 32.3% |
| global store | 0.84 | 6.1% | 15.5% |
| shared/const load | 0.55 | 4.0% | 5.4% |
| control | 0.18 | 1.3% | 2.6% |
| MUFU (exp, rcp, sqrt) | 0.17 | 1.2% | 1.3% |

- most stall samples by opcode: LDG 32.3%, FMUL 25.0%, STG 15.5%, ISETP 6.7%, LDC 5.3%, FFMA 4.9%

CHECK OK
