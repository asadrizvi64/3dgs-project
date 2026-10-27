ncu step 5000 (3.12M Gaussians), 69 launches: kernel time 6.54 ms, DRAM read 4.39 GB + write 3.31 GB = 7.70 GB/step, avg 1176 GB/s; ncu peak DRAM 2447 GB/s
| % time | ms | launches | DRAM read MB | DRAM write MB | GB/s | % of peak BW | kernel |
|---|---|---|---|---|---|---|---|
| 35.4 | 2.315 | 6 | 2950 | 2065 | 2166 | 89 | `adam::adam_step_cu` |
| 29.1 | 1.906 | 1 | 367 | 34 | 210 | 9 | `backward::blend_backward_cu<0>` |
| 12.0 | 0.783 | 1 | 323 | 271 | 759 | 31 | `backward::preprocess_backward_cu` |
| 7.1 | 0.462 | 1 | 76 | 227 | 657 | 27 | `forward::blend_cu` |
| 5.1 | 0.332 | 14 | 0 | 601 | 1812 | 74 | `at::vectorized_elementwise_kernel<4, at::FillFunctor<float>, std::arra` |
| 2.9 | 0.190 | 1 | 306 | 60 | 1922 | 79 | `forward::preprocess_cu` |
| 1.4 | 0.091 | 1 | 78 | 11 | 976 | 40 | `fusedssim_backwardCUDA` |
| 1.0 | 0.067 | 4 | 31 | 0 | 468 | 19 | `cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned` |
| 0.9 | 0.061 | 1 | 26 | 29 | 905 | 37 | `fusedssimCUDA` |
| 0.9 | 0.058 | 1 | 39 | 8 | 826 | 34 | `forward::create_instances_cu<unsigned short>` |
