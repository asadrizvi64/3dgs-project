GPU: NVIDIA H100  torch 2.11.0+cu128  CUDA 12.8

| scene | backend | forward | means | conic | color | opacity |
|---|---|---|---|---|---|---|
| tiny | pixel | 7.4e-08 | 2.9e-07 | 5.8e-07 | 3.6e-07 | 2.5e-07 |
| tiny | splat | 7.4e-08 | 6.2e-07 | 8.6e-07 | 3.0e-07 | 5.7e-07 |
| sparse_odd | pixel | 6.9e-08 | 6.5e-07 | 5.2e-07 | 4.4e-07 | 4.4e-07 |
| sparse_odd | splat | 6.9e-08 | 7.0e-07 | 6.3e-07 | 4.2e-07 | 5.3e-07 |
| dense | pixel | 1.7e-07 | 6.4e-07 | 5.2e-07 | 3.3e-07 | 6.5e-07 |
| dense | splat | 1.7e-07 | 3.2e-06 | 2.4e-06 | 2.4e-07 | 4.2e-06 |
| opaque | pixel | 1.1e-07 | 4.9e-07 | 5.1e-07 | 4.7e-07 | 5.0e-07 |
| opaque | splat | 1.1e-07 | 2.3e-06 | 1.4e-06 | 3.1e-07 | 7.4e-06 |

Worst relative error: 7.44e-06  ->  G0 accuracy PASS (< 1e-05)
