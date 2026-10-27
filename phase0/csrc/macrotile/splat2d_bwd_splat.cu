// splat2d_bwd_splat.cu - backward (B): per-splat, warp-bucketed (Taming-3DGS style).
//
// One warp per (tile, 32-splat bucket); a block holds kBucketWarpsPerBlock independent
// warps. The bucket's splat parameters are staged in shared memory. Each lane owns
// 8 pixels of the tile (pixel = r * 32 + lane), restores their exact (C, T) from the
// bucket's forward checkpoint and replays the bucket front to back with the same
// composite_step as the forward, so T_j is reconstructed exactly. For each splat a lane
// sums its 8 pixels' gradient terms in registers, the warp reduces the 9 sums over the
// lanes (__shfl_down_sync), and lane 0 issues one atomicAdd per parameter if any pixel
// used the splat: sum_tiles sum_buckets |splats in bucket that touch the tile| * 9.
#include "launch.h"
#include "splat2d_common.h"

namespace splat2d {

constexpr unsigned int kFullMask = 0xffffffffu;

template <typename S>
__global__ void backward_splat_kernel(SceneView<S> v, const S* grad_image, const S* image,
                                      const S* ckpt, Grads<S> grads) {
  __shared__ SplatParams<S> s_splat[kBucketWarpsPerBlock][kBucket];

  const int warp = static_cast<int>(threadIdx.x) / kBucket;
  const int lane = static_cast<int>(threadIdx.x) % kBucket;
  const int bucket = static_cast<int>(blockIdx.x) * kBucketWarpsPerBlock + warp;
  if (bucket >= v.num_buckets) return;  // whole warp; warps never sync with each other
  const int tile = v.bucket_tile[bucket];
  const int start = v.tile_off[tile] + (bucket - v.bucket_off[tile]) * kBucket;
  const int remaining = v.tile_off[tile + 1] - start;
  const int bsize = remaining < kBucket ? remaining : kBucket;

  if (lane < bsize) s_splat[warp][lane] = load_splat(v, v.tile_ids[start + lane]);
  __syncwarp();

  // Per-lane state for its 8 pixels.
  S px[kPixelsPerLane];
  S py[kPixelsPerLane];
  bool inside[kPixelsPerLane];
  S C[kPixelsPerLane][3];
  S T[kPixelsPerLane];
  S G[kPixelsPerLane][3];
  S C_final[kPixelsPerLane][3];
  for (int r = 0; r < kPixelsPerLane; ++r) {
    const int pix_lane = r * kBucket + lane;
    const PixelCoord p = pixel_of(tile, pix_lane, v.tiles_x, v.height, v.width);
    px[r] = S(p.x) + S(0.5);
    py[r] = S(p.y) + S(0.5);
    inside[r] = p.inside;
    load_checkpoint(ckpt, bucket, pix_lane, C[r], T[r]);
    const int pix = p.y * v.width + p.x;
    for (int ch = 0; ch < 3; ++ch) {
      G[r][ch] = p.inside ? grad_image[3 * pix + ch] : S(0);
      C_final[r][ch] = p.inside ? image[3 * pix + ch] : S(0);
    }
  }

  for (int jj = 0; jj < bsize; ++jj) {
    const SplatParams<S>& sp = s_splat[warp][jj];
    S acc[kNumParams];
    for (int k = 0; k < kNumParams; ++k) acc[k] = S(0);
    bool used = false;
    for (int r = 0; r < kPixelsPerLane; ++r) {
      if (!inside[r] || T[r] < S(kTStop)) continue;
      const SplatEval<S> e = eval_splat(sp, px[r], py[r]);
      if (e.alpha < S(kAlphaMin)) continue;
      used = true;
      S dL_dc[3];
      const S dL_dalpha = backprop_replay_step(e, sp.color, G[r], C_final[r], C[r], T[r], dL_dc);
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
