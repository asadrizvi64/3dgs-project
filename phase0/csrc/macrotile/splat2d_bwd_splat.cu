// splat2d_bwd_splat.cu - backward (B): per-splat, warp-bucketed (Taming-3DGS style).
//
// One warp per (tile, 32-splat bucket); a block holds kBucketWarpsPerBlock independent
// warps. The bucket's splat parameters and the tile's upstream gradients G are staged in
// shared memory. Each lane owns 8 pixels of the tile (pixel = r * 32 + lane: one column,
// rows 2r + lane / 16). From the bucket's forward checkpoint (C, T) it keeps only two
// registers per pixel: T, replayed front to back with the forward's exact update, and
// rho = G . (C_final - C) (see backprop_replay_step). For each splat a lane sums its
// pixels' gradient terms in registers, the warp reduces the 9 sums over the lanes
// (__shfl_down_sync), and lane 0 issues one atomicAdd per parameter if any pixel used the
// splat: sum_tiles sum_buckets |splats in bucket that touch the tile| * 9 atomics.
#include "launch.h"
#include "splat2d_common.h"

namespace splat2d {

constexpr unsigned int kFullMask = 0xffffffffu;

template <typename S>
__global__ void backward_splat_kernel(SceneView<S> v, const S* grad_image, const S* image,
                                      const S* ckpt, Grads<S> grads) {
  __shared__ SplatParams<S> s_splat[kBucketWarpsPerBlock][kBucket];
  __shared__ S s_grad[kBucketWarpsPerBlock][kTilePixels][3];

  const int warp = static_cast<int>(threadIdx.x) / kBucket;
  const int lane = static_cast<int>(threadIdx.x) % kBucket;
  const int bucket = static_cast<int>(blockIdx.x) * kBucketWarpsPerBlock + warp;
  if (bucket >= v.num_buckets) return;  // whole warp; warps never sync with each other
  const int tile = v.bucket_tile[bucket];
  const int start = v.tile_off[tile] + (bucket - v.bucket_off[tile]) * kBucket;
  const int remaining = v.tile_off[tile + 1] - start;
  const int bsize = remaining < kBucket ? remaining : kBucket;

  if (lane < bsize) s_splat[warp][lane] = load_splat(v, v.tile_ids[start + lane]);

  const int x = (tile % v.tiles_x) * kTile + lane % kTile;  // same column for all 8 pixels
  const int y0 = (tile / v.tiles_x) * kTile + lane / kTile;  // row of pixel r is y0 + 2r
  const S px = S(x) + S(0.5);
  unsigned int live = 0u;  // bit r: pixel r is inside the image
  S T[kPixelsPerLane];
  S rho[kPixelsPerLane];
  for (int r = 0; r < kPixelsPerLane; ++r) {
    const int pix_lane = r * kBucket + lane;
    const int y = y0 + 2 * r;
    S C[3];
    load_checkpoint(ckpt, bucket, pix_lane, C, T[r]);
    S G[3] = {S(0), S(0), S(0)};
    rho[r] = S(0);
    if (x < v.width && y < v.height) {
      live |= 1u << r;
      const int pix = y * v.width + x;
      S C_final[3];
      for (int ch = 0; ch < 3; ++ch) {
        G[ch] = grad_image[3 * pix + ch];
        C_final[ch] = image[3 * pix + ch];
      }
      rho[r] = replay_rho(G, C_final, C);
    }
    for (int ch = 0; ch < 3; ++ch) s_grad[warp][pix_lane][ch] = G[ch];
  }
  __syncwarp();

  for (int jj = 0; jj < bsize; ++jj) {
    const SplatParams<S>& sp = s_splat[warp][jj];
    S acc[kNumParams];
    for (int k = 0; k < kNumParams; ++k) acc[k] = S(0);
    bool used = false;
    for (int r = 0; r < kPixelsPerLane; ++r) {
      if (((live >> r) & 1u) == 0u || T[r] < S(kTStop)) continue;
      const SplatEval<S> e = eval_splat(sp, px, S(y0 + 2 * r) + S(0.5));
      if (e.alpha < S(kAlphaMin)) continue;
      used = true;
      S dL_dc[3];
      const S dL_dalpha = backprop_replay_step(e, sp.color, s_grad[warp][r * kBucket + lane],
                                               rho[r], T[r], dL_dc);
      S g[kNumParams];
      chain_rule(sp, e, dL_dalpha, dL_dc, g);
      for (int k = 0; k < kNumParams; ++k) acc[k] += g[k];
    }
    // Warp-uniform: skip splats no pixel of the tile uses.
    if (__ballot_sync(kFullMask, used ? 1 : 0) == 0u) continue;
    const int id = v.tile_ids[start + jj];
    for (int k = 0; k < kNumParams; ++k) {
      S val = acc[k];
      for (unsigned int offset = kBucket / 2; offset > 0u; offset /= 2u) {
        val += __shfl_down_sync(kFullMask, val, offset);
      }
      if (lane == 0) atomicAdd(grad_slot(grads, id, k), val);
    }
  }
}

template <typename S>
void backward_splat_cuda(const SceneView<S>& v, const S* grad_image, const S* image,
                         const S* ckpt, const Grads<S>& grads) {
  if (v.num_buckets == 0) return;
  const int blocks = (v.num_buckets + kBucketWarpsPerBlock - 1) / kBucketWarpsPerBlock;
  const dim3 grid(static_cast<unsigned int>(blocks));
  const dim3 block(static_cast<unsigned int>(kBucket * kBucketWarpsPerBlock));
  SPLAT2D_LAUNCH(backward_splat_kernel<S>, grid, block, 0)(v, grad_image, image, ckpt, grads);
}

template void backward_splat_cuda<float>(const SceneView<float>&, const float*, const float*,
                                         const float*, const Grads<float>&);
template void backward_splat_cuda<double>(const SceneView<double>&, const double*,
                                          const double*, const double*, const Grads<double>&);

}  // namespace splat2d
