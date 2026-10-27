// splat2d_bwd_splat.cu - backward (B): per-splat, warp-bucketed (Taming-3DGS style).
//
// One block of exactly one warp per (tile, 32-splat bucket). The warp sweeps the tile's
// 256 pixels in 8 rounds of 32 (lane = pixel). Each lane restores its pixel's exact (C, T)
// from the bucket's forward checkpoint and replays the bucket front to back with the same
// composite_step as the forward, so T_j is reconstructed exactly. Per splat the 9 gradient
// terms are warp-reduced over the 32 pixels (__shfl_down_sync) and summed in shared memory;
// at the end lane j issues one atomicAdd per parameter for splat j if it touched the tile:
// sum_tiles sum_buckets |splats in bucket that touch the tile| * 9 atomics in total.
#include "launch.h"
#include "splat2d_common.h"

namespace splat2d {

constexpr unsigned int kFullMask = 0xffffffffu;

template <typename S>
__global__ void backward_splat_kernel(SceneView<S> v, const S* grad_image, const S* image,
                                      const S* ckpt, Grads<S> grads) {
  __shared__ S s_acc[kBucket][kNumParams];
  __shared__ int s_touched[kBucket];

  const int lane = static_cast<int>(threadIdx.x);
  const int bucket = static_cast<int>(blockIdx.x);
  const int tile = v.bucket_tile[bucket];
  const int start = v.tile_off[tile] + (bucket - v.bucket_off[tile]) * kBucket;
  const int remaining = v.tile_off[tile + 1] - start;
  const int bsize = remaining < kBucket ? remaining : kBucket;

  for (int k = 0; k < kNumParams; ++k) s_acc[lane][k] = S(0);
  s_touched[lane] = 0;
  __syncthreads();

  for (int round = 0; round < kTilePixels / kBucket; ++round) {
    const int pix_lane = round * kBucket + lane;
    const PixelCoord p = pixel_of(tile, pix_lane, v.tiles_x, v.height, v.width);
    const S px = S(p.x) + S(0.5);
    const S py = S(p.y) + S(0.5);
    S C[3];
    S T;
    load_checkpoint(ckpt, bucket, pix_lane, C, T);
    S G[3] = {S(0), S(0), S(0)};
    S C_final[3] = {S(0), S(0), S(0)};
    if (p.inside) {
      const int pix = p.y * v.width + p.x;
      for (int ch = 0; ch < 3; ++ch) {
        G[ch] = grad_image[3 * pix + ch];
        C_final[ch] = image[3 * pix + ch];
      }
    }

    for (int jj = 0; jj < bsize; ++jj) {
      const int id = v.tile_ids[start + jj];
      S g[kNumParams];
      for (int k = 0; k < kNumParams; ++k) g[k] = S(0);
      bool contributes = false;
      if (p.inside && T >= S(kTStop)) {
        const SplatEval<S> e = eval_splat(v, id, px, py);
        if (e.alpha >= S(kAlphaMin)) {
          contributes = true;
          S dL_dc[3];
          const S dL_dalpha =
              backprop_replay_step(e, v.colors + 3 * id, G, C_final, C, T, dL_dc);
          chain_rule(v, id, e, dL_dalpha, dL_dc, g);
        }
      }
      // Warp-uniform: skip the reduction when no pixel of this round uses splat jj.
      if (__ballot_sync(kFullMask, contributes ? 1 : 0) == 0u) continue;
      for (int k = 0; k < kNumParams; ++k) {
        S val = g[k];
        for (unsigned int offset = kBucket / 2; offset > 0u; offset /= 2u) {
          val += __shfl_down_sync(kFullMask, val, offset);
        }
        if (lane == 0) s_acc[jj][k] += val;
      }
      if (lane == 0) s_touched[jj] = 1;
    }
  }
  __syncthreads();

  if (lane < bsize && s_touched[lane] != 0) {
    const int id = v.tile_ids[start + lane];
    for (int k = 0; k < kNumParams; ++k) {
      atomicAdd(grad_slot(grads, id, k), s_acc[lane][k]);
    }
  }
}

template <typename S>
void backward_splat_cuda(const SceneView<S>& v, const S* grad_image, const S* image,
                         const S* ckpt, const Grads<S>& grads) {
  if (v.num_buckets == 0) return;
  const dim3 grid(static_cast<unsigned int>(v.num_buckets));
  const dim3 block(static_cast<unsigned int>(kBucket));
  SPLAT2D_LAUNCH(backward_splat_kernel<S>, grid, block, 0)(v, grad_image, image, ckpt, grads);
}

template void backward_splat_cuda<float>(const SceneView<float>&, const float*, const float*,
                                         const float*, const Grads<float>&);
template void backward_splat_cuda<double>(const SceneView<double>&, const double*,
                                          const double*, const double*, const Grads<double>&);

}  // namespace splat2d
