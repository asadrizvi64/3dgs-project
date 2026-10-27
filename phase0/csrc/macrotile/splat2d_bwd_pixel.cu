// splat2d_bwd_pixel.cu - backward (A): the 3DGS-style per-pixel baseline.
//
// One thread per pixel walks its tile list back to front from last_contrib, recovering
// T_j = T_{j+1} / (1 - alpha_j) from T_final, and issues one atomicAdd per
// (pixel, contributing splat, parameter): sum_p n_contrib(p) * 9 atomics in total.
#include "launch.h"
#include "splat2d_common.h"

namespace splat2d {

template <typename S>
__global__ void backward_pixel_kernel(SceneView<S> v, const S* grad_image, const S* t_final,
                                      const int32_t* last_contrib, Grads<S> grads) {
  const int tile = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const PixelCoord p = pixel_of(tile, lane, v.tiles_x, v.height, v.width);
  if (!p.inside) return;
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
    if (e.alpha < S(kAlphaMin)) continue;
    S dL_dc[3];
    const S dL_dalpha = backprop_pixel_step(e, v.colors + 3 * id, G, T_after, S_acc, dL_dc);
    S g[kNumParams];
    chain_rule(v, id, e, dL_dalpha, dL_dc, g);
    for (int k = 0; k < kNumParams; ++k) {
      atomicAdd(grad_slot(grads, id, k), g[k]);
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
