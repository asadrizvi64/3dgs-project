### garden (abcds, step 20000)

| metric | forward preprocess | preprocess backward |
|---|---|---|
| duration (us) | 272.1 | 341.1 |
| threads launched | 4.16 M | 4.16 M |
| registers / thread | 56 | 72 |
| instructions (M) | 85.2 | 54.9 |
| DRAM read (MB) | 428 | 392 |
| DRAM write (MB) | 88 | 321 |
| DRAM bandwidth (GB/s) | 1900 | 2090 |
| DRAM throughput (% peak) | 77.7 | 85.4 |
| memory SOL (%) | 77.7 | 85.4 |
| L2 SOL (%) | 57.9 | 54.5 |
| SM SOL (%) | 36.5 | 18.8 |
| issue slots busy (%) | 38.1 | 19.1 |
| occupancy theo / achieved (%) | 56.2 / 47.2 | 43.8 / 38.2 |
| global load sectors / request | 21.5 | 9.0 |
| global store sectors / request | 7.7 | 12.0 |
| active threads / warp instr. | 26.2 | 30.3 |

- forward preprocess stalls (warps per issued instr., total 19.7): long_scoreboard 9.55, mio_throttle 2.65, wait 1.86, short_scoreboard 1.30, lg_throttle 1.19
- preprocess backward stalls (warps per issued instr., total 31.9): long_scoreboard 28.07, selected 1.00, wait 0.98, short_scoreboard 0.63, not_selected 0.43

Preprocess backward by instruction class (one launch, `garden_abcds.pre_sass.csv`):

| class | instructions (M) | share | stall samples share |
|---|---|---|---|
| fp32 arithmetic | 30.92 | 55.3% | 36.5% |
| other (int, move, predicate) | 10.42 | 18.6% | 16.0% |
| shared/const load | 8.37 | 15.0% | 44.4% |
| control | 2.31 | 4.1% | 1.3% |
| global load | 2.09 | 3.7% | 0.7% |
| global store | 1.24 | 2.2% | 0.8% |
| MUFU (exp, rcp, sqrt) | 0.57 | 1.0% | 0.2% |

- most stall samples by opcode: STS 42.2%, FMUL 23.6%, ISETP 12.8%, FADD 11.5%, FFMA 1.4%, IADD3 1.2%

### room (abcds, step 20000)

| metric | forward preprocess | preprocess backward |
|---|---|---|
| duration (us) | 175.5 | 94.2 |
| threads launched | 1.34 M | 1.34 M |
| registers / thread | 56 | 72 |
| instructions (M) | 56.9 | 15.5 |
| DRAM read (MB) | 129 | 110 |
| DRAM write (MB) | 21 | 72 |
| DRAM bandwidth (GB/s) | 856 | 1929 |
| DRAM throughput (% peak) | 35.4 | 78.5 |
| memory SOL (%) | 35.4 | 78.5 |
| L2 SOL (%) | 26.6 | 54.6 |
| SM SOL (%) | 38.1 | 19.4 |
| issue slots busy (%) | 60.9 | 20.5 |
| occupancy theo / achieved (%) | 56.2 / 40.0 | 43.8 / 36.3 |
| global load sectors / request | 20.0 | 8.8 |
| global store sectors / request | 7.4 | 11.9 |
| active threads / warp instr. | 26.6 | 29.8 |

- forward preprocess stalls (warps per issued instr., total 10.5): long_scoreboard 3.09, wait 2.28, not_selected 2.02, selected 1.00, math_pipe_throttle 0.69
- preprocess backward stalls (warps per issued instr., total 28.4): long_scoreboard 23.77, wait 1.01, selected 1.00, short_scoreboard 0.72, not_selected 0.65

Preprocess backward by instruction class (one launch, `room_abcds.pre_sass.csv`):

| class | instructions (M) | share | stall samples share |
|---|---|---|---|
| fp32 arithmetic | 9.85 | 55.0% | 33.9% |
| other (int, move, predicate) | 3.33 | 18.6% | 16.9% |
| shared/const load | 2.72 | 15.2% | 45.4% |
| control | 0.74 | 4.1% | 1.4% |
| global load | 0.67 | 3.7% | 0.8% |
| global store | 0.40 | 2.2% | 1.2% |
| MUFU (exp, rcp, sqrt) | 0.18 | 1.0% | 0.3% |

- most stall samples by opcode: STS 42.2%, FMUL 21.8%, ISETP 13.2%, FADD 10.2%, FFMA 1.8%, LDS 1.5%

CHECK OK
