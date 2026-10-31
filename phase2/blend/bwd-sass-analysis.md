### garden.bwd_sass.csv

One launch: 1078.7M warp instructions; 113,616 launched buckets, 3,192,882 splat steps (reduction run on 2,953,919 = 92.5%), 11,385,096 warp row evaluations, 7,978,279 warp row gradients.

| region | instructions executed | share | static instructions | per splat step | stall samples share |
|---|---|---|---|---|---|
| staging | 51.1M | 4.7% | 517 | 16.0 | 16.7% |
| loop head | 92.6M | 8.6% | 29 | 29.0 | 5.8% |
| row check | 69.7M | 6.5% | 18 | 21.8 | 14.3% |
| evaluate | 193.5M | 17.9% | 102 | 60.6 | 21.5% |
| gradient | 240.7M | 22.3% | 181 | 75.4 | 14.3% |
| vote | 9.6M | 0.9% | 3 | 3.0 | 2.3% |
| reduce | 395.8M | 36.7% | 133 | 124.0 | 22.4% |
| loop tail | 25.6M | 2.4% | 9 | 8.0 | 2.7% |
| **total** | 1078.7M | | 992 | 337.8 | |

### room.bwd_sass.csv

One launch: 1107.8M warp instructions; 95,008 launched buckets, 2,781,695 splat steps (reduction run on 2,628,106 = 94.5%), 13,140,294 warp row evaluations, 10,365,183 warp row gradients.

| region | instructions executed | share | static instructions | per splat step | stall samples share |
|---|---|---|---|---|---|
| staging | 45.3M | 4.1% | 517 | 16.3 | 14.9% |
| loop head | 80.7M | 7.3% | 29 | 29.0 | 5.1% |
| row check | 62.9M | 5.7% | 18 | 22.6 | 12.4% |
| evaluate | 223.4M | 20.2% | 102 | 80.3 | 25.0% |
| gradient | 312.7M | 28.2% | 181 | 112.4 | 18.5% |
| vote | 8.3M | 0.8% | 3 | 3.0 | 1.9% |
| reduce | 352.2M | 31.8% | 133 | 126.6 | 19.7% |
| loop tail | 22.3M | 2.0% | 9 | 8.0 | 2.3% |
| **total** | 1107.8M | | 992 | 398.2 | |

CHECK PASSED: regions add up to the total
