// cuda_shim.h - lets g++ parse the .cu files for the mock build (tools/mock_build.py).
//
// Included ONLY under -DSPLAT2D_MOCK_CUDA (via `g++ -include`). It turns CUDA qualifiers
// into nothing and provides single-thread host stand-ins for the built-ins the kernels use.
// This proves syntax and types only: there are no real warps, no shared-memory semantics
// and no memory model here. Never include it in a real nvcc or host build.
#pragma once

#ifndef SPLAT2D_MOCK_CUDA
#error "cuda_shim.h is only for the mock-CUDA syntax check (-DSPLAT2D_MOCK_CUDA)"
#endif

#include <stddef.h>

#define __global__
#define __device__
#define __host__
#define __shared__

struct uint3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

inline uint3 threadIdx = {0, 0, 0};
inline uint3 blockIdx = {0, 0, 0};
inline dim3 blockDim = dim3(1, 1, 1);
inline dim3 gridDim = dim3(1, 1, 1);

inline void __syncthreads() {}
inline void __syncwarp(unsigned int = 0xffffffffu) {}

template <typename T>
inline T __shfl_down_sync(unsigned int, T var, unsigned int, int = 32) {
  return var;
}

inline unsigned int __ballot_sync(unsigned int, int predicate) { return predicate ? 1u : 0u; }

inline float atomicAdd(float* address, float val) {
  const float old = *address;
  *address = old + val;
  return old;
}

inline double atomicAdd(double* address, double val) {
  const double old = *address;
  *address = old + val;
  return old;
}

namespace splat2d_mock {
// Host stand-in for the <<<grid, block, smem>>> configuration; see launch.h. Takes the
// kernel as a function pointer and returns it, so the launch becomes a plain host call.
template <typename Kernel>
inline Kernel* launch_config(Kernel* kernel, dim3, dim3, size_t) {
  return kernel;
}
}  // namespace splat2d_mock
