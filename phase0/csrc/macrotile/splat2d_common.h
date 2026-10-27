// splat2d_common.h - math shared by the CPU reference and the CUDA kernels.
//
// Every function here is HD-qualified so the exact same code runs on host and device.
// HD expands to __host__ __device__ under nvcc and under the mock-CUDA syntax check
// (where cuda_shim.h defines those qualifiers as empty), and to nothing in plain C++.
//
// MODEL (fixed): splat i has mean mu_i, inverse-covariance Cholesky params (l0, l1, l2)
// with q = (l0*dx + l1*dy)^2 + (l2*dy)^2, opacity o_i = sigmoid(z_i), colour c_i (RGB).
// alpha_i = min(0.99, o_i * exp(-q/2)); skipped if alpha_i < 1/255.
// Front to back: C += c_i * alpha_i * T_i, T_{i+1} = T_i * (1 - alpha_i); a pixel stops
// once T < 1e-4. Pixel centres are at (x + 0.5, y + 0.5). 16x16 tiles, 32-splat buckets.
#pragma once

#include <math.h>
#include <stdint.h>

#ifndef HD
#if defined(__CUDACC__) || defined(SPLAT2D_MOCK_CUDA)
#define HD __host__ __device__
#else
#define HD
#endif
#endif

namespace splat2d {

constexpr int kTile = 16;
constexpr int kTilePixels = kTile * kTile;  // 256 threads per forward / pixel-backward block
constexpr int kBucket = 32;                 // splats per bucket == warp size
constexpr int kNumParams = 9;               // P: mu(2) + chol(3) + opacity logit(1) + rgb(3)
constexpr int kCkptStride = 4;              // checkpoint = (C.r, C.g, C.b, T)
constexpr double kAlphaMin = 1.0 / 255.0;
constexpr double kAlphaMax = 0.99;
constexpr double kTStop = 1e-4;

// Read-only scene description. Splats are pre-sorted by depth (in Python); tiles hold
// CSR lists of splat ids in that order. Bucket b of tile t covers list entries
// [32*(b - bucket_off[t]), +32) and bucket_tile[b] == t.
template <typename S>
struct SceneView {
  const S* means;          // [N, 2]
  const S* chol;           // [N, 3] (l0, l1, l2)
  const S* opacity_logit;  // [N]
  const S* colors;         // [N, 3]
  const int32_t* tile_off;     // [num_tiles + 1]
  const int32_t* tile_ids;     // [tile_off[num_tiles]]
  const int32_t* bucket_off;   // [num_tiles + 1]
  const int32_t* bucket_tile;  // [num_buckets]
  int height;
  int width;
  int tiles_x;
  int num_tiles;
  int num_buckets;
};

template <typename S>
struct FwdOut {
  S* image;              // [H, W, 3]
  S* t_final;            // [H, W]
  int32_t* n_contrib;    // [H, W] number of contributing splats
  int32_t* last_contrib; // [H, W] 1 + tile-list position of the last contributing splat
  S* ckpt;               // [num_buckets, 256, 4] (C, T) at the start of every bucket
};

template <typename S>
struct Grads {
  S* means;          // [N, 2]
  S* chol;           // [N, 3]
  S* opacity_logit;  // [N]
  S* colors;         // [N, 3]
};

struct PixelCoord {
  int x;
  int y;
  bool inside;
};

HD inline PixelCoord pixel_of(int tile, int lane, int tiles_x, int height, int width) {
  PixelCoord p;
  p.x = (tile % tiles_x) * kTile + lane % kTile;
  p.y = (tile / tiles_x) * kTile + lane / kTile;
  p.inside = p.x < width && p.y < height;
  return p;
}

HD inline float sp_exp(float x) { return expf(x); }
HD inline double sp_exp(double x) { return exp(x); }

template <typename S>
HD inline S sp_sigmoid(S z) {
  return S(1) / (S(1) + sp_exp(-z));
}

template <typename S>
struct SplatEval {
  S dx;
  S dy;
  S a;  // l0*dx + l1*dy
  S b;  // l2*dy
  S g;  // exp(-q/2)
  S o;  // sigmoid(z)
  S alpha;
  bool clamped;  // o*g > 0.99, so alpha does not depend on the splat parameters
};

template <typename S>
HD inline SplatEval<S> eval_splat(const SceneView<S>& v, int id, S px, S py) {
  SplatEval<S> e;
  e.dx = px - v.means[2 * id];
  e.dy = py - v.means[2 * id + 1];
  e.a = v.chol[3 * id] * e.dx + v.chol[3 * id + 1] * e.dy;
  e.b = v.chol[3 * id + 2] * e.dy;
  const S q = e.a * e.a + e.b * e.b;
  e.g = sp_exp(S(-0.5) * q);
  e.o = sp_sigmoid(v.opacity_logit[id]);
  const S raw = e.o * e.g;
  e.clamped = raw > S(kAlphaMax);
  e.alpha = e.clamped ? S(kAlphaMax) : raw;
  return e;
}

// One front-to-back compositing step. The forward pass and the checkpoint replay of
// backward (B) both call this, so the replayed (C, T) are the forward's values exactly.
template <typename S>
HD inline void composite_step(S* C, S& T, const S* c, S alpha) {
  for (int ch = 0; ch < 3; ++ch) {
    C[ch] = C[ch] + c[ch] * alpha * T;
  }
  T = T * (S(1) - alpha);
}

template <typename S>
HD inline void store_checkpoint(S* ckpt, int bucket, int lane, const S* C, S T) {
  S* dst = ckpt + (static_cast<int64_t>(bucket) * kTilePixels + lane) * kCkptStride;
  dst[0] = C[0];
  dst[1] = C[1];
  dst[2] = C[2];
  dst[3] = T;
}

template <typename S>
HD inline void load_checkpoint(const S* ckpt, int bucket, int lane, S* C, S& T) {
  const S* src = ckpt + (static_cast<int64_t>(bucket) * kTilePixels + lane) * kCkptStride;
  C[0] = src[0];
  C[1] = src[1];
  C[2] = src[2];
  T = src[3];
}

// Backward (A) step, walking back to front. T_after is T_{j+1} on entry and T_j on exit;
// S_acc is the normalised colour behind splat j. Returns dL/dalpha_j.
template <typename S>
HD inline S backprop_pixel_step(const SplatEval<S>& e, const S* c, const S* G, S& T_after,
                                S* S_acc, S* dL_dc) {
  const S one_minus = S(1) - e.alpha;
  const S T = T_after / one_minus;
  S dot = S(0);
  for (int ch = 0; ch < 3; ++ch) {
    dL_dc[ch] = G[ch] * e.alpha * T;
    dot += G[ch] * (c[ch] - S_acc[ch]);
  }
  for (int ch = 0; ch < 3; ++ch) {
    S_acc[ch] = e.alpha * c[ch] + one_minus * S_acc[ch];
  }
  T_after = T;
  return T * dot;
}

// Backward (B) step, replaying front to back from a checkpoint. (C, T) is the state in
// front of splat j on entry and behind it on exit; C_final is the rendered pixel.
// Uses C_final - C_{<=j} = T_{j+1} * (colour behind j). Returns dL/dalpha_j.
template <typename S>
HD inline S backprop_replay_step(const SplatEval<S>& e, const S* c, const S* G, const S* C_final,
                                 S* C, S& T, S* dL_dc) {
  const S T_j = T;
  composite_step(C, T, c, e.alpha);
  S dot_c = S(0);
  S dot_rest = S(0);
  for (int ch = 0; ch < 3; ++ch) {
    dL_dc[ch] = G[ch] * e.alpha * T_j;
    dot_c += G[ch] * c[ch];
    dot_rest += G[ch] * (C_final[ch] - C[ch]);
  }
  return T_j * dot_c - dot_rest / (S(1) - e.alpha);
}

// Chain rule from (dL/dalpha, dL/dc) to the 9 parameters, in the order
// mu_x, mu_y, l0, l1, l2, z, c_r, c_g, c_b.
template <typename S>
HD inline void chain_rule(const SceneView<S>& v, int id, const SplatEval<S>& e, S dL_dalpha,
                          const S* dL_dc, S* out) {
  const S d_alpha = e.clamped ? S(0) : dL_dalpha;
  const S d_z = d_alpha * e.g * e.o * (S(1) - e.o);
  const S d_q = S(-0.5) * e.g * e.o * d_alpha;
  const S l0 = v.chol[3 * id];
  const S l1 = v.chol[3 * id + 1];
  const S l2 = v.chol[3 * id + 2];
  const S two_a = S(2) * e.a;
  const S two_b = S(2) * e.b;
  out[0] = -d_q * two_a * l0;
  out[1] = -d_q * (two_a * l1 + two_b * l2);
  out[2] = d_q * two_a * e.dx;
  out[3] = d_q * two_a * e.dy;
  out[4] = d_q * two_b * e.dy;
  out[5] = d_z;
  out[6] = dL_dc[0];
  out[7] = dL_dc[1];
  out[8] = dL_dc[2];
}

template <typename S>
HD inline S* grad_slot(const Grads<S>& g, int id, int k) {
  if (k < 2) return g.means + 2 * id + k;
  if (k < 5) return g.chol + 3 * id + (k - 2);
  if (k == 5) return g.opacity_logit + id;
  return g.colors + 3 * id + (k - 6);
}

// ---------------------------------------------------------------------------------------
// Host entry points. CPU versions live in splat2d_cpu.cpp, CUDA launchers in the .cu files.
// ---------------------------------------------------------------------------------------

// Op counters (CPU emulation only; real instruction counts come from Nsight on the GPU).
struct Counters {
  int64_t atomic_adds = 0;
  int64_t checkpoint_bytes = 0;
  int64_t splat_pixel_evals = 0;
};
Counters& counters();

template <typename S>
void forward_cpu(const SceneView<S>& v, const FwdOut<S>& out);
template <typename S>
void backward_pixel_cpu(const SceneView<S>& v, const S* grad_image, const S* t_final,
                        const int32_t* last_contrib, const Grads<S>& grads);
template <typename S>
void backward_splat_cpu(const SceneView<S>& v, const S* grad_image, const S* image,
                        const S* ckpt, const Grads<S>& grads);

template <typename S>
void forward_cuda(const SceneView<S>& v, const FwdOut<S>& out);
template <typename S>
void backward_pixel_cuda(const SceneView<S>& v, const S* grad_image, const S* t_final,
                         const int32_t* last_contrib, const Grads<S>& grads);
template <typename S>
void backward_splat_cuda(const SceneView<S>& v, const S* grad_image, const S* image,
                         const S* ckpt, const Grads<S>& grads);

}  // namespace splat2d
