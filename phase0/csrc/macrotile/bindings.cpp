// bindings.cpp - torch extension: forward, backward_pixel, backward_splat, get_counters,
// reset_counters. Dispatches on float/double; CUDA tensors go to the .cu launchers, which
// are only compiled and linked when the extension is built with -DWITH_CUDA.
// MSVC reports C4702 (unreachable code) inside torch's own headers at code-generation
// time, which /external:W0 cannot silence; suppress it for the torch include only.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <torch/extension.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <map>
#include <string>
#include <vector>

#include "splat2d_common.h"

namespace {

using torch::Tensor;

void check_tensor(const Tensor& t, const Tensor& like, const char* name) {
  TORCH_CHECK(t.device() == like.device(), name, " must be on the same device as means");
}

void check_device(const Tensor& means) {
#ifndef WITH_CUDA
  TORCH_CHECK(!means.is_cuda(), "splat2d was built without CUDA (rebuild with -DWITH_CUDA)");
#else
  (void)means;
#endif
}

struct SceneTensors {
  Tensor means;
  Tensor chol;
  Tensor opacity_logit;
  Tensor colors;
  Tensor tile_off;
  Tensor tile_ids;
  Tensor bucket_off;
  Tensor bucket_tile;
  int height;
  int width;
};

SceneTensors make_scene(const Tensor& means, const Tensor& chol, const Tensor& opacity_logit,
                        const Tensor& colors, const Tensor& tile_off, const Tensor& tile_ids,
                        const Tensor& bucket_off, const Tensor& bucket_tile, int64_t height,
                        int64_t width) {
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must be [N, 2]");
  const int64_t n = means.size(0);
  TORCH_CHECK(chol.sizes() == torch::IntArrayRef({n, 3}), "chol must be [N, 3]");
  TORCH_CHECK(opacity_logit.sizes() == torch::IntArrayRef({n}), "opacity_logit must be [N]");
  TORCH_CHECK(colors.sizes() == torch::IntArrayRef({n, 3}), "colors must be [N, 3]");
  TORCH_CHECK(chol.scalar_type() == means.scalar_type() &&
                  opacity_logit.scalar_type() == means.scalar_type() &&
                  colors.scalar_type() == means.scalar_type(),
              "all splat parameters must share one floating dtype");
  for (const Tensor* t : {&tile_off, &tile_ids, &bucket_off, &bucket_tile}) {
    TORCH_CHECK(t->scalar_type() == torch::kInt32, "tile/bucket tables must be int32");
  }
  TORCH_CHECK(height > 0 && width > 0, "image must be non-empty");
  check_tensor(chol, means, "chol");
  check_tensor(opacity_logit, means, "opacity_logit");
  check_tensor(colors, means, "colors");
  check_tensor(tile_off, means, "tile_offsets");
  check_tensor(tile_ids, means, "tile_ids");
  check_tensor(bucket_off, means, "bucket_offsets");
  check_tensor(bucket_tile, means, "bucket_tile");
  check_device(means);
  const int64_t tiles_x = (width + splat2d::kTile - 1) / splat2d::kTile;
  const int64_t tiles_y = (height + splat2d::kTile - 1) / splat2d::kTile;
  TORCH_CHECK(tile_off.numel() == tiles_x * tiles_y + 1, "tile_offsets has the wrong length");
  TORCH_CHECK(bucket_off.numel() == tiles_x * tiles_y + 1, "bucket_offsets has the wrong length");
  return SceneTensors{means.contiguous(),      chol.contiguous(),
                      opacity_logit.contiguous(), colors.contiguous(),
                      tile_off.contiguous(),   tile_ids.contiguous(),
                      bucket_off.contiguous(), bucket_tile.contiguous(),
                      static_cast<int>(height), static_cast<int>(width)};
}

template <typename S>
splat2d::SceneView<S> view_of(const SceneTensors& s) {
  splat2d::SceneView<S> v;
  v.means = s.means.data_ptr<S>();
  v.chol = s.chol.data_ptr<S>();
  v.opacity_logit = s.opacity_logit.data_ptr<S>();
  v.colors = s.colors.data_ptr<S>();
  v.tile_off = s.tile_off.data_ptr<int32_t>();
  v.tile_ids = s.tile_ids.data_ptr<int32_t>();
  v.bucket_off = s.bucket_off.data_ptr<int32_t>();
  v.bucket_tile = s.bucket_tile.data_ptr<int32_t>();
  v.height = s.height;
  v.width = s.width;
  v.tiles_x = (s.width + splat2d::kTile - 1) / splat2d::kTile;
  v.num_tiles = static_cast<int>(s.tile_off.numel() - 1);
  v.num_buckets = static_cast<int>(s.bucket_tile.numel());
  return v;
}

template <typename S>
splat2d::Grads<S> grads_of(const std::vector<Tensor>& g) {
  return splat2d::Grads<S>{g[0].data_ptr<S>(), g[1].data_ptr<S>(), g[2].data_ptr<S>(),
                           g[3].data_ptr<S>()};
}

std::vector<Tensor> zero_grads(const SceneTensors& s) {
  return {torch::zeros_like(s.means), torch::zeros_like(s.chol),
          torch::zeros_like(s.opacity_logit), torch::zeros_like(s.colors)};
}

// The CPU/CUDA branch lives in these helpers, not inside AT_DISPATCH: preprocessor
// directives inside a macro argument are undefined behaviour (MSVC rejects them).
template <typename S>
void run_forward(const SceneTensors& s, const splat2d::FwdOut<S>& out) {
  const splat2d::SceneView<S> v = view_of<S>(s);
#ifdef WITH_CUDA
  if (s.means.is_cuda()) {
    splat2d::forward_cuda<S>(v, out);
    return;
  }
#endif
  splat2d::forward_cpu<S>(v, out);
}

template <typename S>
void run_backward_pixel(const SceneTensors& s, const S* grad_image, const S* t_final,
                        const int32_t* last_contrib, const splat2d::Grads<S>& grads) {
  const splat2d::SceneView<S> v = view_of<S>(s);
#ifdef WITH_CUDA
  if (s.means.is_cuda()) {
    splat2d::backward_pixel_cuda<S>(v, grad_image, t_final, last_contrib, grads);
    return;
  }
#endif
  splat2d::backward_pixel_cpu<S>(v, grad_image, t_final, last_contrib, grads);
}

template <typename S>
void run_backward_splat(const SceneTensors& s, const S* grad_image, const S* image,
                        const S* ckpt, const splat2d::Grads<S>& grads) {
  const splat2d::SceneView<S> v = view_of<S>(s);
#ifdef WITH_CUDA
  if (s.means.is_cuda()) {
    splat2d::backward_splat_cuda<S>(v, grad_image, image, ckpt, grads);
    return;
  }
#endif
  splat2d::backward_splat_cpu<S>(v, grad_image, image, ckpt, grads);
}

std::vector<Tensor> forward(const Tensor& means, const Tensor& chol, const Tensor& opacity_logit,
                            const Tensor& colors, const Tensor& tile_off, const Tensor& tile_ids,
                            const Tensor& bucket_off, const Tensor& bucket_tile, int64_t height,
                            int64_t width) {
  const SceneTensors s = make_scene(means, chol, opacity_logit, colors, tile_off, tile_ids,
                                    bucket_off, bucket_tile, height, width);
  const auto opts = s.means.options();
  Tensor image = torch::zeros({height, width, 3}, opts);
  Tensor t_final = torch::ones({height, width}, opts);
  Tensor n_contrib = torch::zeros({height, width}, opts.dtype(torch::kInt32));
  Tensor last_contrib = torch::zeros({height, width}, opts.dtype(torch::kInt32));
  // No zero-fill: the forward writes every checkpoint slot (all 256 lanes of every bucket).
  Tensor ckpt =
      torch::empty({s.bucket_tile.numel(), splat2d::kTilePixels, splat2d::kCkptStride}, opts);
  AT_DISPATCH_FLOATING_TYPES(s.means.scalar_type(), "splat2d_forward", [&] {
    const splat2d::FwdOut<scalar_t> out{image.data_ptr<scalar_t>(), t_final.data_ptr<scalar_t>(),
                                        n_contrib.data_ptr<int32_t>(),
                                        last_contrib.data_ptr<int32_t>(),
                                        ckpt.data_ptr<scalar_t>()};
    run_forward<scalar_t>(s, out);
  });
  return {image, t_final, n_contrib, last_contrib, ckpt};
}

std::vector<Tensor> backward_pixel(const Tensor& grad_image, const Tensor& means,
                                   const Tensor& chol, const Tensor& opacity_logit,
                                   const Tensor& colors, const Tensor& tile_off,
                                   const Tensor& tile_ids, const Tensor& bucket_off,
                                   const Tensor& bucket_tile, const Tensor& t_final,
                                   const Tensor& last_contrib, int64_t height, int64_t width) {
  const SceneTensors s = make_scene(means, chol, opacity_logit, colors, tile_off, tile_ids,
                                    bucket_off, bucket_tile, height, width);
  const Tensor g_img = grad_image.contiguous().to(s.means.scalar_type());
  const Tensor tf = t_final.contiguous();
  const Tensor lc = last_contrib.contiguous();
  TORCH_CHECK(g_img.sizes() == torch::IntArrayRef({height, width, 3}), "grad_image must be [H, W, 3]");
  TORCH_CHECK(lc.scalar_type() == torch::kInt32, "last_contrib must be int32");
  std::vector<Tensor> g = zero_grads(s);
  AT_DISPATCH_FLOATING_TYPES(s.means.scalar_type(), "splat2d_backward_pixel", [&] {
    run_backward_pixel<scalar_t>(s, g_img.data_ptr<scalar_t>(), tf.data_ptr<scalar_t>(),
                                 lc.data_ptr<int32_t>(), grads_of<scalar_t>(g));
  });
  return g;
}

std::vector<Tensor> backward_splat(const Tensor& grad_image, const Tensor& means,
                                   const Tensor& chol, const Tensor& opacity_logit,
                                   const Tensor& colors, const Tensor& tile_off,
                                   const Tensor& tile_ids, const Tensor& bucket_off,
                                   const Tensor& bucket_tile, const Tensor& image,
                                   const Tensor& ckpt, int64_t height, int64_t width) {
  const SceneTensors s = make_scene(means, chol, opacity_logit, colors, tile_off, tile_ids,
                                    bucket_off, bucket_tile, height, width);
  const Tensor g_img = grad_image.contiguous().to(s.means.scalar_type());
  const Tensor img = image.contiguous();
  const Tensor ck = ckpt.contiguous();
  TORCH_CHECK(g_img.sizes() == torch::IntArrayRef({height, width, 3}), "grad_image must be [H, W, 3]");
  TORCH_CHECK(ck.size(0) == s.bucket_tile.numel(), "ckpt must have one slot per bucket");
  std::vector<Tensor> g = zero_grads(s);
  AT_DISPATCH_FLOATING_TYPES(s.means.scalar_type(), "splat2d_backward_splat", [&] {
    run_backward_splat<scalar_t>(s, g_img.data_ptr<scalar_t>(), img.data_ptr<scalar_t>(),
                                 ck.data_ptr<scalar_t>(), grads_of<scalar_t>(g));
  });
  return g;
}

std::map<std::string, int64_t> get_counters() {
  const splat2d::Counters& c = splat2d::counters();
  return {{"atomic_adds", c.atomic_adds},
          {"checkpoint_bytes", c.checkpoint_bytes},
          {"splat_pixel_evals", c.splat_pixel_evals}};
}

void reset_counters() { splat2d::counters() = splat2d::Counters(); }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "splat2d forward (image, T_final, n_contrib, last_contrib, ckpt)");
  m.def("backward_pixel", &backward_pixel, "backward (A): per-pixel atomics");
  m.def("backward_splat", &backward_splat, "backward (B): per-splat warp buckets");
  m.def("get_counters", &get_counters, "CPU-emulation op counters");
  m.def("reset_counters", &reset_counters, "zero the op counters");
}
