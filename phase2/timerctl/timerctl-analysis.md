## Validation: 2/2 scenes pass (garden, room)
  - all runs: both wall windows, clock, results present; timer state confirmed; view sequences identical per seed

## garden (job 4485370 (c136); seeds 0, 1, 2)

| arm | timer | wall 0–end, s | wall first half | wall second half | summed step events (SEGCLOCK), s | wall − SEGCLOCK, s | NeRFICG ms/it | PSNR |
|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | ON | 255.8 (255.4 / 257.3 / 254.6) | 115.5 | 140.3 | 251.6 (251.2 / 253.1 / 250.4) | 4.2 | 8.44 | 27.488 |
| stock Faster-GS | OFF | 252.7 (252.8 / 253.4 / 251.8) | 113.9 | 138.7 | 250.4 (250.6 / 251.1 / 249.6) | 2.2 |  | 27.498 |
| A+B+C | ON | 194.6 (196.1 / 193.2 / 194.4) | 96.0 | 98.6 | 190.7 (192.2 / 189.3 / 190.5) | 3.9 | 6.40 | 27.526 |
| A+B+C | OFF | 190.3 (190.3 / 190.7 / 190.0) | 93.8 | 96.5 | 188.1 (188.1 / 188.5 / 187.7) | 2.2 |  | 27.530 |

Paired effects (per seed in brackets):

- stock Faster-GS, timer OFF vs ON: wall -1.2 (-1.0 / -1.5 / -1.1)%, SEGCLOCK -0.5 (-0.3 / -0.8 / -0.3)%, ΔPSNR +0.010 (+0.077 / -0.013 / -0.035)
- A+B+C, timer OFF vs ON: wall -2.2 (-2.9 / -1.3 / -2.3)%, SEGCLOCK -1.3 (-2.1 / -0.4 / -1.5)%, ΔPSNR +0.004 (+0.007 / -0.028 / +0.033)
- A+B+C vs stock, timer ON: wall -23.9 (-23.2 / -24.9 / -23.6)%, SEGCLOCK -24.2 (-23.5 / -25.2 / -23.9)%
- A+B+C vs stock, timer OFF: wall -24.7 (-24.7 / -24.7 / -24.6)%, SEGCLOCK -24.9 (-24.9 / -24.9 / -24.8)%

## room (job 4485371 (c79); seeds 0, 1, 2)

| arm | timer | wall 0–end, s | wall first half | wall second half | summed step events (SEGCLOCK), s | wall − SEGCLOCK, s | NeRFICG ms/it | PSNR |
|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | ON | 151.1 (152.0 / 151.5 / 149.8) | 71.3 | 79.8 | 148.4 (149.2 / 148.9 / 147.1) | 2.7 | 4.99 | 31.665 |
| stock Faster-GS | OFF | 148.2 (148.5 / 148.1 / 147.9) | 69.8 | 78.4 | 147.2 (147.5 / 147.1 / 146.9) | 1.0 |  | 31.546 |
| A+B+C | ON | 121.8 (121.5 / 120.9 / 123.1) | 60.1 | 61.7 | 119.4 (119.0 / 118.5 / 120.6) | 2.5 | 4.02 | 31.682 |
| A+B+C | OFF | 117.8 (117.7 / 117.9 / 118.0) | 58.2 | 59.6 | 116.8 (116.7 / 116.9 / 117.0) | 1.0 |  | 31.790 |

Paired effects (per seed in brackets):

- stock Faster-GS, timer OFF vs ON: wall -1.9 (-2.3 / -2.3 / -1.2)%, SEGCLOCK -0.8 (-1.1 / -1.2 / -0.1)%, ΔPSNR -0.118 (-0.085 / -0.163 / -0.107)
- A+B+C, timer OFF vs ON: wall -3.3 (-3.1 / -2.5 / -4.2)%, SEGCLOCK -2.1 (-2.0 / -1.3 / -3.0)%, ΔPSNR +0.107 (+0.119 / +0.065 / +0.138)
- A+B+C vs stock, timer ON: wall -19.4 (-20.1 / -20.2 / -17.8)%, SEGCLOCK -19.6 (-20.2 / -20.4 / -18.0)%
- A+B+C vs stock, timer OFF: wall -20.5 (-20.8 / -20.4 / -20.3)%, SEGCLOCK -20.6 (-20.9 / -20.5 / -20.4)%
