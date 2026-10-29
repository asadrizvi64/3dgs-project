## Validation: 1/1 scenes pass (garden)
  - all runs: both wall windows, clock, results present; timer state confirmed; view sequences identical per seed

## garden (job 4485369 (c136); seeds 0)

| arm | timer | wall 0–end, s | wall first half | wall second half | summed step events (SEGCLOCK), s | wall − SEGCLOCK, s | NeRFICG ms/it | PSNR |
|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | ON | 2.8 (2.8) | 1.4 | 1.4 | 2.6 (2.6) | 0.2 | 2.22 | 22.974 |
| stock Faster-GS | OFF | 2.7 (2.7) | 1.3 | 1.3 | 2.5 (2.5) | 0.1 |  | 23.025 |
| A+B+C | ON | 2.4 (2.4) | 1.2 | 1.2 | 2.3 (2.3) | 0.1 | 1.94 | 23.007 |
| A+B+C | OFF | 2.3 (2.3) | 1.1 | 1.2 | 2.2 (2.2) | 0.1 |  | 22.969 |

Paired effects (per seed in brackets):

- stock Faster-GS, timer OFF vs ON: wall -4.4 (-4.4)%, SEGCLOCK -2.5 (-2.5)%, ΔPSNR +0.051 (+0.051)
- A+B+C, timer OFF vs ON: wall -5.0 (-5.0)%, SEGCLOCK -3.0 (-3.0)%, ΔPSNR -0.038 (-0.038)
- A+B+C vs stock, timer ON: wall -12.2 (-12.2)%, SEGCLOCK -12.5 (-12.5)%
- A+B+C vs stock, timer OFF: wall -12.8 (-12.8)%, SEGCLOCK -13.0 (-13.0)%
