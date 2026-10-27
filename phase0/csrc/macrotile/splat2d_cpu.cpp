// splat2d_cpu.cpp - CPU reference for the three kernels.
//
// Uses exactly the math in splat2d_common.h. The loops emulate the GPU thread mappings:
// blocks run one after another, threads (lanes) of a block run in lane order, and backward
// (B) keeps 32 lanes of registers, does the checkpoint replay, emulates the
// __shfl_down_sync tree and the shared-memory accumulators. Every emulated atomicAdd and
// every (splat, pixel) evaluation increments the op counters. No torch dependency.
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
  for (int tile = 0; tile < v.num_tiles; ++tile) {      // blockIdx.x
    for (int lane = 0; lane < kTilePixels; ++lane) {     // threadIdx.x
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
        const int id = v.tile_ids[k];
        const SplatEval<S> e = eval_splat(v, id, px, py);
        ++counters().splat_pixel_evals;
        if (e.alpha < S(kAlphaMin)) continue;
        composite_step(C, T, v.colors + 3 * id, e.alpha);
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
  for (int tile = 0; tile < v.num_tiles; ++tile) {      // blockIdx.x
    for (int lane = 0; lane < kTilePixels; ++lane) {     // threadIdx.x
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
        const SplatEval<S> e = eval_splat(v, id, px, py);
        ++counters().splat_pixel_evals;
        if (e.alpha < S(kAlphaMin)) continue;
        S dL_dc[3];
        const S dL_dalpha = backprop_pixel_step(e, v.colors + 3 * id, G, T_after, S_acc, dL_dc);
        S g[kNumParams];
        chain_rule(v, id, e, dL_dalpha, dL_dc, g);
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
  constexpr int kRounds = kTilePixels / kBucket;
  for (int bucket = 0; bucket < v.num_buckets; ++bucket) {  // blockIdx.x (one warp)
    const int tile = v.bucket_tile[bucket];
    const int start = v.tile_off[tile] + (bucket - v.bucket_off[tile]) * kBucket;
    const int remaining = v.tile_off[tile + 1] - start;
    const int bsize = remaining < kBucket ? remaining : kBucket;

    S s_acc[kBucket][kNumParams];  // __shared__
    int s_touched[kBucket];        // __shared__
    for (int lane = 0; lane < kBucket; ++lane) {
      for (int k = 0; k < kNumParams; ++k) s_acc[lane][k] = S(0);
      s_touched[lane] = 0;
    }

    for (int round = 0; round < kRounds; ++round) {
      // Per-lane registers.
      PixelCoord p[kBucket];
      S px[kBucket];
      S py[kBucket];
      S C[kBucket][3];
      S T[kBucket];
      S G[kBucket][3];
      S C_final[kBucket][3];
      for (int lane = 0; lane < kBucket; ++lane) {
        const int pix_lane = round * kBucket + lane;
        p[lane] = pixel_of(tile, pix_lane, v.tiles_x, v.height, v.width);
        px[lane] = S(p[lane].x) + S(0.5);
        py[lane] = S(p[lane].y) + S(0.5);
        load_checkpoint(ckpt, bucket, pix_lane, C[lane], T[lane]);
        for (int ch = 0; ch < 3; ++ch) {
          G[lane][ch] = S(0);
          C_final[lane][ch] = S(0);
        }
        if (p[lane].inside) {
          const int pix = p[lane].y * v.width + p[lane].x;
          for (int ch = 0; ch < 3; ++ch) {
            G[lane][ch] = grad_image[3 * pix + ch];
            C_final[lane][ch] = image[3 * pix + ch];
          }
        }
      }

      for (int jj = 0; jj < bsize; ++jj) {
        const int id = v.tile_ids[start + jj];
        S g[kBucket][kNumParams];
        unsigned int ballot = 0u;
        for (int lane = 0; lane < kBucket; ++lane) {
          for (int k = 0; k < kNumParams; ++k) g[lane][k] = S(0);
          if (!p[lane].inside || T[lane] < S(kTStop)) continue;
          const SplatEval<S> e = eval_splat(v, id, px[lane], py[lane]);
          ++counters().splat_pixel_evals;
          if (e.alpha < S(kAlphaMin)) continue;
          ballot |= 1u << lane;
          S dL_dc[3];
          const S dL_dalpha = backprop_replay_step(e, v.colors + 3 * id, G[lane], C_final[lane],
                                                   C[lane], T[lane], dL_dc);
          chain_rule(v, id, e, dL_dalpha, dL_dc, g[lane]);
        }
        if (ballot == 0u) continue;
        for (int k = 0; k < kNumParams; ++k) {
          // __shfl_down_sync tree: lanes whose source is out of range keep their own value.
          S val[kBucket];
          for (int lane = 0; lane < kBucket; ++lane) val[lane] = g[lane][k];
          for (int offset = kBucket / 2; offset > 0; offset /= 2) {
            S next[kBucket];
            for (int lane = 0; lane < kBucket; ++lane) {
              const int src = lane + offset;
              next[lane] = val[lane] + (src < kBucket ? val[src] : val[lane]);
            }
            for (int lane = 0; lane < kBucket; ++lane) val[lane] = next[lane];
          }
          s_acc[jj][k] += val[0];  // lane 0
        }
        s_touched[jj] = 1;
      }
    }

    for (int lane = 0; lane < bsize; ++lane) {
      if (s_touched[lane] == 0) continue;
      const int id = v.tile_ids[start + lane];
      for (int k = 0; k < kNumParams; ++k) {
        emulated_atomic_add(grad_slot(grads, id, k), s_acc[lane][k]);
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
