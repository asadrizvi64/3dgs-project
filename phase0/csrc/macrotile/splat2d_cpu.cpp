// splat2d_cpu.cpp - CPU reference for the three kernels.
//
// Uses exactly the math in splat2d_common.h. The loops emulate the GPU thread mappings:
// blocks run one after another, threads (lanes) of a block run in lane order, and backward
// (B) keeps 32 lanes x 8 pixels of register state, does the checkpoint replay, sums each
// lane's 8 pixels in lane order and emulates the __shfl_down_sync tree. (Shared-memory
// staging of splat parameters does not change the math and is not emulated.) Every
// emulated atomicAdd and every (splat, pixel) evaluation increments the op counters.
// No torch dependency.
#include <stdint.h>

#include "splat2d_common.h"

namespace splat2d {

Counters& counters() {
  static Counters c;
  return c;
}

namespace {

template <typename S>
void emulated_atomic_add(S* address, S val) {
  *address = *address + val;
  ++counters().atomic_adds;
}

}  // namespace

template <typename S>
void forward_cpu(const SceneView<S>& v, const FwdOut<S>& out) {
  counters().checkpoint_bytes += static_cast<int64_t>(v.num_buckets) * kTilePixels *
                                 kCkptStride * static_cast<int64_t>(sizeof(S));
  for (int tile = 0; tile < v.num_tiles; ++tile) {   // blockIdx.x
    for (int lane = 0; lane < kTilePixels; ++lane) {  // threadIdx.x
      const PixelCoord p = pixel_of(tile, lane, v.tiles_x, v.height, v.width);
      const S px = S(p.x) + S(0.5);
      const S py = S(p.y) + S(0.5);
      S C[3] = {S(0), S(0), S(0)};
      S T = S(1);
      int count = 0;
      int last = 0;
      const int begin = v.tile_off[tile];
      const int end = v.tile_off[tile + 1];
      const int bucket0 = v.bucket_off[tile];
      for (int k = begin; k < end; ++k) {
        const int local = k - begin;
        if (local % kBucket == 0) {
          store_checkpoint(out.ckpt, bucket0 + local / kBucket, lane, C, T);
        }
        if (!p.inside || T < S(kTStop)) continue;
        const SplatParams<S> sp = load_splat(v, v.tile_ids[k]);
        const SplatEval<S> e = eval_splat(sp, px, py);
        ++counters().splat_pixel_evals;
        if (e.alpha < S(kAlphaMin)) continue;
        composite_step(C, T, sp.color, e.alpha);
        ++count;
        last = local + 1;
      }
      if (p.inside) {
        const int pix = p.y * v.width + p.x;
        out.image[3 * pix + 0] = C[0];
        out.image[3 * pix + 1] = C[1];
        out.image[3 * pix + 2] = C[2];
        out.t_final[pix] = T;
        out.n_contrib[pix] = count;
        out.last_contrib[pix] = last;
      }
    }
  }
}

template <typename S>
void backward_pixel_cpu(const SceneView<S>& v, const S* grad_image, const S* t_final,
                        const int32_t* last_contrib, const Grads<S>& grads) {
  for (int tile = 0; tile < v.num_tiles; ++tile) {   // blockIdx.x
    for (int lane = 0; lane < kTilePixels; ++lane) {  // threadIdx.x
      const PixelCoord p = pixel_of(tile, lane, v.tiles_x, v.height, v.width);
      if (!p.inside) continue;
      const int pix = p.y * v.width + p.x;
      const S px = S(p.x) + S(0.5);
      const S py = S(p.y) + S(0.5);
      const S G[3] = {grad_image[3 * pix], grad_image[3 * pix + 1], grad_image[3 * pix + 2]};
      S T_after = t_final[pix];
      S S_acc[3] = {S(0), S(0), S(0)};
      const int begin = v.tile_off[tile];
      for (int local = last_contrib[pix] - 1; local >= 0; --local) {
        const int id = v.tile_ids[begin + local];
        const SplatParams<S> sp = load_splat(v, id);
        const SplatEval<S> e = eval_splat(sp, px, py);
        ++counters().splat_pixel_evals;
        if (e.alpha < S(kAlphaMin)) continue;
        S dL_dc[3];
        const S dL_dalpha = backprop_pixel_step(e, sp.color, G, T_after, S_acc, dL_dc);
        S g[kNumParams];
        chain_rule(sp, e, dL_dalpha, dL_dc, g);
        for (int k = 0; k < kNumParams; ++k) {
          emulated_atomic_add(grad_slot(grads, id, k), g[k]);
        }
      }
    }
  }
}

template <typename S>
void backward_splat_cpu(const SceneView<S>& v, const S* grad_image, const S* image,
                        const S* ckpt, const Grads<S>& grads) {
  for (int bucket = 0; bucket < v.num_buckets; ++bucket) {  // one warp per bucket
    const int tile = v.bucket_tile[bucket];
    const int start = v.tile_off[tile] + (bucket - v.bucket_off[tile]) * kBucket;
    const int remaining = v.tile_off[tile + 1] - start;
    const int bsize = remaining < kBucket ? remaining : kBucket;

    // Per-lane registers: lane owns pixels r * 32 + lane, r = 0..7.
    S px[kBucket][kPixelsPerLane];
    S py[kBucket][kPixelsPerLane];
    bool inside[kBucket][kPixelsPerLane];
    S C[kBucket][kPixelsPerLane][3];
    S T[kBucket][kPixelsPerLane];
    S G[kBucket][kPixelsPerLane][3];
    S C_final[kBucket][kPixelsPerLane][3];
    for (int lane = 0; lane < kBucket; ++lane) {
      for (int r = 0; r < kPixelsPerLane; ++r) {
        const int pix_lane = r * kBucket + lane;
        const PixelCoord p = pixel_of(tile, pix_lane, v.tiles_x, v.height, v.width);
        px[lane][r] = S(p.x) + S(0.5);
        py[lane][r] = S(p.y) + S(0.5);
        inside[lane][r] = p.inside;
        load_checkpoint(ckpt, bucket, pix_lane, C[lane][r], T[lane][r]);
        const int pix = p.y * v.width + p.x;
        for (int ch = 0; ch < 3; ++ch) {
          G[lane][r][ch] = p.inside ? grad_image[3 * pix + ch] : S(0);
          C_final[lane][r][ch] = p.inside ? image[3 * pix + ch] : S(0);
        }
      }
    }

    for (int jj = 0; jj < bsize; ++jj) {
      const int id = v.tile_ids[start + jj];
      const SplatParams<S> sp = load_splat(v, id);  // shared memory on the GPU
      S acc[kBucket][kNumParams];
      bool any_used = false;
      for (int lane = 0; lane < kBucket; ++lane) {
        for (int k = 0; k < kNumParams; ++k) acc[lane][k] = S(0);
        for (int r = 0; r < kPixelsPerLane; ++r) {
          if (!inside[lane][r] || T[lane][r] < S(kTStop)) continue;
          const SplatEval<S> e = eval_splat(sp, px[lane][r], py[lane][r]);
          ++counters().splat_pixel_evals;
          if (e.alpha < S(kAlphaMin)) continue;
          any_used = true;
          S dL_dc[3];
          const S dL_dalpha = backprop_replay_step(e, sp.color, G[lane][r], C_final[lane][r],
                                                   C[lane][r], T[lane][r], dL_dc);
          S g[kNumParams];
          chain_rule(sp, e, dL_dalpha, dL_dc, g);
          for (int k = 0; k < kNumParams; ++k) acc[lane][k] += g[k];
        }
      }
      if (!any_used) continue;  // __ballot_sync == 0
      for (int k = 0; k < kNumParams; ++k) {
        // __shfl_down_sync tree: lanes whose source is out of range keep their own value.
        S val[kBucket];
        for (int lane = 0; lane < kBucket; ++lane) val[lane] = acc[lane][k];
        for (int offset = kBucket / 2; offset > 0; offset /= 2) {
          S next[kBucket];
          for (int lane = 0; lane < kBucket; ++lane) {
            const int src = lane + offset;
            next[lane] = val[lane] + (src < kBucket ? val[src] : val[lane]);
          }
          for (int lane = 0; lane < kBucket; ++lane) val[lane] = next[lane];
        }
        emulated_atomic_add(grad_slot(grads, id, k), val[0]);  // lane 0
      }
    }
  }
}

template void forward_cpu<float>(const SceneView<float>&, const FwdOut<float>&);
template void forward_cpu<double>(const SceneView<double>&, const FwdOut<double>&);
template void backward_pixel_cpu<float>(const SceneView<float>&, const float*, const float*,
                                        const int32_t*, const Grads<float>&);
template void backward_pixel_cpu<double>(const SceneView<double>&, const double*,
                                         const double*, const int32_t*, const Grads<double>&);
template void backward_splat_cpu<float>(const SceneView<float>&, const float*, const float*,
                                        const float*, const Grads<float>&);
template void backward_splat_cpu<double>(const SceneView<double>&, const double*,
                                         const double*, const double*, const Grads<double>&);

}  // namespace splat2d
