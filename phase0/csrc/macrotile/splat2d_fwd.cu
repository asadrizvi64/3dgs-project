// splat2d_fwd.cu - forward rasterizer, one thread per pixel, one 16x16 block per tile.
//
// The block walks its tile list in batches of 256 splats: each thread stages one splat's
// parameters in shared memory, then every thread composites the batch for its pixel.
// Saves what both backward passes need: the image, T_final, n_contrib, last_contrib, and
// a (C, T) checkpoint at the start of every 32-splat bucket (used by backward B).
#include "launch.h"
#include "splat2d_common.h"

namespace splat2d {

template <typename S>
__global__ void forward_kernel(SceneView<S> v, FwdOut<S> out) {
  __shared__ SplatParams<S> s_splat[kTilePixels];

  const int tile = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
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
  for (int batch = begin; batch < end; batch += kTilePixels) {
    __syncthreads();  // the previous batch is no longer in use
    if (batch + lane < end) s_splat[lane] = load_splat(v, v.tile_ids[batch + lane]);
    __syncthreads();
    const int nb = end - batch < kTilePixels ? end - batch : kTilePixels;
    for (int j = 0; j < nb; ++j) {
      const int local = batch - begin + j;
      if (local % kBucket == 0) {
        store_checkpoint(out.ckpt, bucket0 + local / kBucket, lane, C, T);
      }
      if (!p.inside || T < S(kTStop)) continue;
      const SplatEval<S> e = eval_splat(s_splat[j], px, py);
      if (e.alpha < S(kAlphaMin)) continue;
      composite_step(C, T, s_splat[j].color, e.alpha);
      ++count;
      last = local + 1;
    }
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

template <typename S>
void forward_cuda(const SceneView<S>& v, const FwdOut<S>& out) {
  if (v.num_tiles == 0) return;
  const dim3 grid(static_cast<unsigned int>(v.num_tiles));
  const dim3 block(static_cast<unsigned int>(kTilePixels));
  SPLAT2D_LAUNCH(forward_kernel<S>, grid, block, 0)(v, out);
}

template void forward_cuda<float>(const SceneView<float>&, const FwdOut<float>&);
template void forward_cuda<double>(const SceneView<double>&, const FwdOut<double>&);

}  // namespace splat2d
