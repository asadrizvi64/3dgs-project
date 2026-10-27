// launch.h - SPLAT2D_LAUNCH(kernel, grid, block, smem)(args...)
//
// Under nvcc this is kernel<<<grid, block, smem>>>(args...). g++ cannot parse <<<>>>, so
// under SPLAT2D_MOCK_CUDA it becomes a plain host call of the kernel function (the launch
// configuration is only type-checked). Only the mock build ever takes the second branch.
#pragma once

#if defined(SPLAT2D_MOCK_CUDA)
#define SPLAT2D_LAUNCH(kernel, grid, block, smem) \
  (::splat2d_mock::launch_config(&kernel, (grid), (block), (smem)))
#else
#define SPLAT2D_LAUNCH(kernel, grid, block, smem) kernel<<<(grid), (block), (smem)>>>
#endif
