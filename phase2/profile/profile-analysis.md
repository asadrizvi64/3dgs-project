# Profiling: where the training step goes (seed 0)

## garden (job 4484137)

Step split from CUDA events (mean ms per step over the sampled steps):

| arm | phase | forward | backward | optimizer | sum | forward % | backward % | optimizer % | Gaussians (mean) |
|---|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | densifying (<15k) | 1.60 | 3.59 | 2.38 | 7.57 | 21% | 47% | 31% | 3.08 M |
| stock Faster-GS | after (>=15k) | 1.79 | 4.41 | 3.21 | 9.41 | 19% | 47% | 34% | 4.16 M |
| stock Faster-GS | whole run | 1.69 | 3.99 | 2.78 | 8.46 | 20% | 47% | 33% | 3.60 M |
| A+B+C | densifying (<15k) | 1.56 | 2.74 | 2.58 | 6.89 | 23% | 40% | 38% | 3.01 M |
| A+B+C | after (>=15k) | 1.72 | 3.40 | 1.38 | 6.49 | 26% | 52% | 21% | 4.15 M |
| A+B+C | whole run | 1.65 | 3.09 | 1.95 | 6.68 | 25% | 46% | 29% | 3.61 M |

Kernels, stock Faster-GS, steps 5000–5050: 6.73 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 2.399 | 35.7% |
| backward: blend | 1.936 | 28.8% |
| backward: preprocess | 0.866 | 12.9% |
| backward: other | 0.093 | 1.4% |
| forward: blend | 0.455 | 6.8% |
| forward: preprocess | 0.212 | 3.1% |
| forward: sort | 0.148 | 2.2% |
| PyTorch ops (loss, fills, copies) | 0.478 | 7.1% |
| other | 0.141 | 2.1% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 35.7% | 2.399 | 6 | `faster_gs::adam::adam_step_cu` |
| 28.8% | 1.936 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_cu<` |
| 12.9% | 0.866 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 6.8% | 0.455 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 5.4% | 0.365 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 3.1% | 0.212 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 1.4% | 0.093 | 1 | `fusedssim_backwardCUDA` |
| 0.9% | 0.062 | 1 | `fusedssimCUDA` |

Kernels, stock Faster-GS, steps 20000–20050: 8.49 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 3.197 | 37.7% |
| backward: blend | 2.277 | 26.8% |
| backward: preprocess | 1.174 | 13.8% |
| backward: other | 0.093 | 1.1% |
| forward: blend | 0.551 | 6.5% |
| forward: preprocess | 0.274 | 3.2% |
| forward: sort | 0.174 | 2.0% |
| PyTorch ops (loss, fills, copies) | 0.594 | 7.0% |
| other | 0.158 | 1.9% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 37.7% | 3.197 | 6 | `faster_gs::adam::adam_step_cu` |
| 26.8% | 2.277 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_cu<` |
| 13.8% | 1.174 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 6.5% | 0.551 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 5.7% | 0.480 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 3.2% | 0.274 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 1.1% | 0.093 | 1 | `fusedssim_backwardCUDA` |
| 0.9% | 0.074 | 1 | `faster_gs::rasterization::kernels::forward::create_instances_cu<unsigned short>` |

Kernels, A+B+C, steps 5000–5050: 5.41 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 1.900 | 35.1% |
| backward: blend | 1.100 | 20.3% |
| backward: preprocess | 0.865 | 16.0% |
| backward: other | 0.093 | 1.7% |
| forward: blend | 0.455 | 8.4% |
| forward: preprocess | 0.210 | 3.9% |
| forward: sort | 0.148 | 2.7% |
| PyTorch ops (loss, fills, copies) | 0.498 | 9.2% |
| other | 0.141 | 2.6% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 24.5% | 1.327 | 1 | `faster_gs::adam::adam_step_sparse_cu<` |
| 20.3% | 1.100 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_macro_cu<` |
| 16.0% | 0.865 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 10.6% | 0.573 | 5 | `faster_gs::adam::adam_step_sparse_cu<` |
| 8.4% | 0.455 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 6.8% | 0.368 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 3.9% | 0.210 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 1.7% | 0.093 | 1 | `fusedssim_backwardCUDA` |

Kernels, A+B+C, steps 20000–20050: 5.72 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 1.416 | 24.7% |
| backward: blend | 1.297 | 22.7% |
| backward: preprocess | 1.169 | 20.4% |
| backward: other | 0.093 | 1.6% |
| forward: blend | 0.552 | 9.6% |
| forward: preprocess | 0.274 | 4.8% |
| forward: sort | 0.173 | 3.0% |
| PyTorch ops (loss, fills, copies) | 0.588 | 10.3% |
| other | 0.158 | 2.8% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 22.7% | 1.297 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_macro_cu<` |
| 20.4% | 1.169 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 18.0% | 1.027 | 1 | `faster_gs::adam::adam_step_sparse_cu<` |
| 9.6% | 0.552 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 8.3% | 0.476 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 6.8% | 0.388 | 5 | `faster_gs::adam::adam_step_sparse_cu<` |
| 4.8% | 0.274 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 1.6% | 0.093 | 1 | `fusedssim_backwardCUDA` |

## room (job 4484138)

Step split from CUDA events (mean ms per step over the sampled steps):

| arm | phase | forward | backward | optimizer | sum | forward % | backward % | optimizer % | Gaussians (mean) |
|---|---|---|---|---|---|---|---|---|---|
| stock Faster-GS | densifying (<15k) | 1.48 | 2.21 | 0.78 | 4.47 | 33% | 49% | 17% | 1.00 M |
| stock Faster-GS | after (>=15k) | 1.60 | 2.63 | 1.06 | 5.29 | 30% | 50% | 20% | 1.36 M |
| stock Faster-GS | whole run | 1.54 | 2.41 | 0.92 | 4.88 | 32% | 50% | 19% | 1.17 M |
| A+B+C | densifying (<15k) | 1.52 | 1.68 | 0.87 | 4.06 | 37% | 41% | 21% | 0.99 M |
| A+B+C | after (>=15k) | 1.70 | 2.11 | 0.39 | 4.20 | 41% | 50% | 9% | 1.35 M |
| A+B+C | whole run | 1.61 | 1.90 | 0.62 | 4.13 | 39% | 46% | 15% | 1.18 M |

Kernels, stock Faster-GS, steps 5000–5050: 3.61 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 0.739 | 20.5% |
| backward: blend | 1.366 | 37.9% |
| backward: preprocess | 0.202 | 5.6% |
| backward: other | 0.134 | 3.7% |
| forward: blend | 0.443 | 12.3% |
| forward: preprocess | 0.124 | 3.4% |
| forward: sort | 0.105 | 2.9% |
| PyTorch ops (loss, fills, copies) | 0.311 | 8.6% |
| other | 0.183 | 5.1% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 37.9% | 1.366 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_cu<` |
| 20.5% | 0.739 | 6 | `faster_gs::adam::adam_step_cu` |
| 12.3% | 0.443 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 5.6% | 0.202 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 4.1% | 0.147 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 3.7% | 0.134 | 1 | `fusedssim_backwardCUDA` |
| 3.4% | 0.124 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 2.5% | 0.089 | 1 | `fusedssimCUDA` |

Kernels, stock Faster-GS, steps 20000–20050: 4.68 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 1.045 | 22.3% |
| backward: blend | 1.795 | 38.3% |
| backward: preprocess | 0.306 | 6.5% |
| backward: other | 0.134 | 2.9% |
| forward: blend | 0.576 | 12.3% |
| forward: preprocess | 0.166 | 3.5% |
| forward: sort | 0.122 | 2.6% |
| PyTorch ops (loss, fills, copies) | 0.354 | 7.6% |
| other | 0.184 | 3.9% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 38.3% | 1.795 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_cu<` |
| 22.3% | 1.045 | 6 | `faster_gs::adam::adam_step_cu` |
| 12.3% | 0.576 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 6.5% | 0.306 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 4.0% | 0.189 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 3.5% | 0.166 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 2.9% | 0.134 | 1 | `fusedssim_backwardCUDA` |
| 1.9% | 0.089 | 1 | `fusedssimCUDA` |

Kernels, A+B+C, steps 5000–5050: 2.96 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 0.589 | 19.9% |
| backward: blend | 0.865 | 29.3% |
| backward: preprocess | 0.201 | 6.8% |
| backward: other | 0.134 | 4.5% |
| forward: blend | 0.452 | 15.3% |
| forward: preprocess | 0.119 | 4.0% |
| forward: sort | 0.106 | 3.6% |
| PyTorch ops (loss, fills, copies) | 0.314 | 10.6% |
| other | 0.177 | 6.0% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 29.3% | 0.865 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_macro_cu<` |
| 15.3% | 0.452 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 13.7% | 0.404 | 1 | `faster_gs::adam::adam_step_sparse_cu<` |
| 6.8% | 0.201 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 6.2% | 0.185 | 5 | `faster_gs::adam::adam_step_sparse_cu<` |
| 4.9% | 0.146 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 4.5% | 0.134 | 1 | `fusedssim_backwardCUDA` |
| 4.0% | 0.119 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |

Kernels, A+B+C, steps 20000–20050: 3.45 ms of GPU kernel time per step

| kernel group | ms/step | share |
|---|---|---|
| optimizer (Adam) | 0.396 | 11.5% |
| backward: blend | 1.162 | 33.7% |
| backward: preprocess | 0.308 | 8.9% |
| backward: other | 0.134 | 3.9% |
| forward: blend | 0.600 | 17.4% |
| forward: preprocess | 0.186 | 5.4% |
| forward: sort | 0.124 | 3.6% |
| PyTorch ops (loss, fills, copies) | 0.351 | 10.2% |
| other | 0.188 | 5.4% |

Top kernels:

| share | ms/step | calls/step | kernel |
|---|---|---|---|
| 33.7% | 1.162 | 1 | `faster_gs::rasterization::kernels::backward::blend_backward_macro_cu<` |
| 17.4% | 0.600 | 1 | `faster_gs::rasterization::kernels::forward::blend_cu` |
| 8.9% | 0.308 | 1 | `faster_gs::rasterization::kernels::backward::preprocess_backward_cu` |
| 8.5% | 0.292 | 1 | `faster_gs::adam::adam_step_sparse_cu<` |
| 5.4% | 0.187 | 14 | `at::native::vectorized_elementwise_kernel<` |
| 5.4% | 0.186 | 1 | `faster_gs::rasterization::kernels::forward::preprocess_cu` |
| 3.9% | 0.134 | 1 | `fusedssim_backwardCUDA` |
| 3.0% | 0.104 | 5 | `faster_gs::adam::adam_step_sparse_cu<` |

(figure: phase2/profile/stage_split.png)
