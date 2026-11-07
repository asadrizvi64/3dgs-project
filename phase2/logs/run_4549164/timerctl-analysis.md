## Validation: 1/1 scenes pass (garden)
  - all runs: both wall windows, clock, results present; timer state confirmed; view sequences identical per seed

## garden (job 4549164 (c94); seeds 0, 1, 2)

| arm | timer | wall 0–end, s | wall first half | wall second half | summed step events (SEGCLOCK), s | wall − SEGCLOCK, s | NeRFICG ms/it | PSNR |
|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | ON | 259.6 (259.2 / 258.8 / 260.8) | 117.3 | 142.3 | 255.1 (254.7 / 254.4 / 256.4) | 4.4 | 8.56 | 27.480 |
| stock Faster-GS | OFF | 255.9 (258.3 / 254.6 / 254.8) | 115.2 | 140.7 | 253.7 (256.1 / 252.4 / 252.6) | 2.2 |  | 27.519 |
| A+B+C+D+S | ON | 171.5 (172.6 / 172.2 / 169.8) | 87.0 | 84.5 | 167.6 (168.6 / 168.3 / 165.9) | 3.9 | 5.63 | 27.519 |
| A+B+C+D+S | OFF | 169.8 (168.2 / 169.4 / 171.9) | 86.3 | 83.5 | 167.6 (166.0 / 167.2 / 169.7) | 2.2 |  | 27.539 |

Paired effects (per seed in brackets):

- stock Faster-GS, timer OFF vs ON: wall -1.4 (-0.4 / -1.6 / -2.3)%, SEGCLOCK -0.6 (+0.5 / -0.8 / -1.5)%, ΔPSNR +0.039 (+0.082 / -0.002 / +0.038)
- A+B+C+D+S, timer OFF vs ON: wall -1.0 (-2.5 / -1.6 / +1.2)%, SEGCLOCK +0.0 (-1.6 / -0.7 / +2.3)%, ΔPSNR +0.019 (+0.044 / +0.048 / -0.035)
- A+B+C+D+S vs stock, timer ON: wall -33.9 (-33.4 / -33.5 / -34.9)%, SEGCLOCK -34.3 (-33.8 / -33.8 / -35.3)%
- A+B+C+D+S vs stock, timer OFF: wall -33.6 (-34.9 / -33.5 / -32.6)%, SEGCLOCK -33.9 (-35.2 / -33.7 / -32.8)%
## Validation: 1/1 scenes pass (room)
  - all runs: both wall windows, clock, results present; timer state confirmed; view sequences identical per seed

## room (job 4549165 (c136); seeds 0, 1, 2)

| arm | timer | wall 0–end, s | wall first half | wall second half | summed step events (SEGCLOCK), s | wall − SEGCLOCK, s | NeRFICG ms/it | PSNR |
|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | ON | 153.3 (153.0 / 153.7 / 153.1) | 72.0 | 81.3 | 150.7 (150.4 / 151.1 / 150.5) | 2.6 | 5.06 | 31.607 |
| stock Faster-GS | OFF | 148.8 (149.3 / 148.9 / 148.2) | 70.0 | 78.8 | 147.8 (148.3 / 147.9 / 147.3) | 1.0 |  | 31.641 |
| A+B+C+D+S | ON | 115.3 (114.7 / 115.4 / 115.9) | 57.8 | 57.5 | 112.9 (112.3 / 112.9 / 113.5) | 2.4 | 3.80 | 31.684 |
| A+B+C+D+S | OFF | 113.2 (111.9 / 115.3 / 112.5) | 56.7 | 56.5 | 112.2 (110.9 / 114.2 / 111.5) | 1.0 |  | 31.765 |

Paired effects (per seed in brackets):

- stock Faster-GS, timer OFF vs ON: wall -2.9 (-2.4 / -3.1 / -3.2)%, SEGCLOCK -1.9 (-1.4 / -2.1 / -2.2)%, ΔPSNR +0.034 (+0.042 / -0.048 / +0.108)
- A+B+C+D+S, timer OFF vs ON: wall -1.9 (-2.5 / -0.1 / -3.0)%, SEGCLOCK -0.6 (-1.2 / +1.2 / -1.8)%, ΔPSNR +0.082 (+0.171 / +0.205 / -0.130)
- A+B+C+D+S vs stock, timer ON: wall -24.8 (-25.0 / -24.9 / -24.3)%, SEGCLOCK -25.1 (-25.4 / -25.3 / -24.6)%
- A+B+C+D+S vs stock, timer OFF: wall -23.9 (-25.1 / -22.6 / -24.1)%, SEGCLOCK -24.1 (-25.2 / -22.8 / -24.3)%
