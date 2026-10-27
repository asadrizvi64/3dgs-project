// splat2d_bwd_pixel.cu - backward (A): the 3DGS-style per-pixel baseline.
//
// One thread per pixel walks its tile list back to front from last_contrib, recovering
// T_j = T_{j+1} / (1 - alpha_j) from T_final, and issues one atomicAdd per
// (pixel, contributing splat, parameter): sum_p n_contrib(p) * 9 atomics in total.
// Like 3DGS, the block stages batches of 256 splats (taken from the back of the tile
// list) in shared memory.
#include "launch.h"
#include "splat2d_common.h"

namespace splat2d {

template <typename S>
__global__ void backward_pixel_kernel(SceneView<S> v, const S* grad_image, const S* t_final,
                                      const int32_t* last_contrib, Grads<S> grads) {
  __shared__ SplatParams<S> s_splat[kTilePixels];
  __shared__ int s_id[kTilePixels];

  const int tile = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const PixelCoord p = pixel_of(tile, lane, v.tiles_x, v.height, v.width);
  const int pix = p.inside ? p.y * v.width + p.x : 0;
  const S px = S(p.x) + S(0.5);
  const S py = S(p.y) + S(0.5);
  S G[3] = {S(0), S(0), S(0)};
  S T_after = S(1);
  int last = 0;
  if (p.inside) {
    for (int ch = 0; ch < 3; ++ch) G[ch] = grad_image[3 * pix + ch];
    T_after = t_final[pix];
    last = last_contrib[pix];
  }
  S S_acc[3] = {S(0), S(0), S(0)};
  const int begin = v.tile_off[tile];
  const int end = v.tile_off[tile + 1];

  // Batches cover list positions [batch_lo, batch_hi), walked from the back.
  for (int batch_hi = end; batch_hi > begin; batch_hi -= kTilePixels) {
    const int batch_lo = batch_hi - kTilePixels > begin ? batch_hi - kTilePixels : begin;
    __syncthreads();
    if (batch_lo + lane < batch_hi) {
      const int id = v.tile_ids[batch_lo + lane];
      s_id[lane] = id;
      s_splat[lane] = load_splat(v, id);
    }
    __syncthreads();
    for (int j = batch_hi - batch_lo - 1; j >= 0; --j) {
      const int local = batch_lo - begin + j;
      if (local >= last) continue;  // also skips pixels outside the image (last == 0)
      const SplatParams<S>& sp = s_splat[j];
      const SplatEval<S> e = eval_splat(sp, px, py);
      if (e.alpha < S(kAlphaMin)) continue;
      S dL_dc[3];
      const S dL_dalpha = backprop_pixel_step(e, sp.color, G, T_after, S_acc, dL_dc);
      S g[kNumParams];
      chain_rule(sp, e, dL_dalpha, dL_dc, g);
      for (int k = 0; k < kNumParams; ++k) {
        atomicAdd(grad_slot(grads, s_id[j], k), g[k]);
      }
    }
  }
}

template <typename S>
void backward_pixel_cuda(const SceneView<S>& v, const S* grad_image, const S* t_final,
                         const int32_t* last_contrib, const Grads<S>& grads) {
  if (v.num_tiles == 0) return;
  const dim3 grid(static_cast<unsigned int>(v.num_tiles));
  const dim3 block(static_cast<unsigned int>(kTilePixels));
  SPLAT2D_LAUNCH(backward_pixel_kernel<S>, grid, block, 0)
  (v, grad_image, t_final, last_contrib, grads);
}

template void backward_pixel_cuda<float>(const SceneView<float>&, const float*, const float*,
                                         const int32_t*, const Grads<float>&);
template void backward_pixel_cuda<double>(const SceneView<double>&, const double*,
                                          const double*, const int32_t*, const Grads<double>&);

}  // namespace splat2d
