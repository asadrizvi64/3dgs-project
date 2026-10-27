// 2D Gaussian tile rasterizer: forward + two backward variants.
//
//   forward_kernel         one block per 16x16 tile, one thread per pixel (3DGS style).
//                          Optionally saves (T, C) at the start of every 32-splat bucket
//                          for the per-splat backward.
//   backward_pixel_kernel  one thread per pixel, walks splats back to front,
//                          recovers T by dividing by (1 - alpha), one atomicAdd per
//                          (pixel, splat) pair. The classic 3DGS backward.
//   backward_splat_kernel  one warp per (tile, 32-splat bucket), one lane per splat.
//                          Pixels flow through the warp front to back (lane k works on
//                          pixel t - k at step t and hands its (T, C) to lane k + 1).
//                          Gradients stay in registers: one atomicAdd per splat per
//                          bucket. The Taming-3DGS backward.
//
// Rendering rules must match gs2d/reference.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK_SIZE (TILE * TILE)
#define BUCKET 32
#define FULL_MASK 0xffffffffu

constexpr float ALPHA_MIN = 1.0f / 255.0f;
constexpr float ALPHA_MAX = 0.99f;
constexpr float T_MIN = 1e-4f;

struct Splat {
    float2 m;
    float3 con;  // A, B, C
    float o;
};

// Evaluates one splat at one pixel. Returns false if it does not contribute.
__device__ __forceinline__ bool eval_splat(const Splat& s, float px, float py,
                                           float& dx, float& dy, float& G, float& alpha, bool& clamped) {
    dx = s.m.x - px;
    dy = s.m.y - py;
    const float power = -0.5f * (s.con.x * dx * dx + s.con.z * dy * dy) - s.con.y * dx * dy;
    if (power > 0.0f) return false;
    G = expf(power);
    const float a = s.o * G;
    clamped = a > ALPHA_MAX;
    alpha = fminf(ALPHA_MAX, a);
    return alpha >= ALPHA_MIN;
}

// Chain rule from dL/dalpha down to mean, conic and opacity.
struct SplatGrad {
    float2 dm;
    float3 dcon;
    float3 dcol;
    float dop;
};

__device__ __forceinline__ void zero(SplatGrad& g) {
    g.dm = make_float2(0.f, 0.f);
    g.dcon = make_float3(0.f, 0.f, 0.f);
    g.dcol = make_float3(0.f, 0.f, 0.f);
    g.dop = 0.f;
}

__device__ __forceinline__ void alpha_chain(const Splat& s, float dx, float dy, float G, bool clamped,
                                            float dL_dalpha, SplatGrad& g) {
    if (clamped) return;  // min(0.99, oG) has zero gradient when clamped
    g.dop += G * dL_dalpha;
    const float dL_dpower = s.o * G * dL_dalpha;
    g.dm.x += (-s.con.x * dx - s.con.y * dy) * dL_dpower;
    g.dm.y += (-s.con.z * dy - s.con.y * dx) * dL_dpower;
    g.dcon.x += -0.5f * dx * dx * dL_dpower;
    g.dcon.y += -dx * dy * dL_dpower;
    g.dcon.z += -0.5f * dy * dy * dL_dpower;
}

__device__ __forceinline__ void atomic_add_grad(const SplatGrad& g, int id, float* dmeans, float* dconic,
                                                float* dcolor, float* dopacity) {
    atomicAdd(&dmeans[2 * id + 0], g.dm.x);
    atomicAdd(&dmeans[2 * id + 1], g.dm.y);
    atomicAdd(&dconic[3 * id + 0], g.dcon.x);
    atomicAdd(&dconic[3 * id + 1], g.dcon.y);
    atomicAdd(&dconic[3 * id + 2], g.dcon.z);
    atomicAdd(&dcolor[3 * id + 0], g.dcol.x);
    atomicAdd(&dcolor[3 * id + 1], g.dcol.y);
    atomicAdd(&dcolor[3 * id + 2], g.dcol.z);
    atomicAdd(&dopacity[id], g.dop);
}

__device__ __forceinline__ Splat load_splat(int id, const float2* means, const float* conic, const float* opacity) {
    Splat s;
    s.m = means[id];
    s.con = make_float3(conic[3 * id + 0], conic[3 * id + 1], conic[3 * id + 2]);
    s.o = opacity[id];
    return s;
}

__device__ __forceinline__ float3 load_color(int id, const float* color) {
    return make_float3(color[3 * id + 0], color[3 * id + 1], color[3 * id + 2]);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
__global__ void forward_kernel(const int2* __restrict__ ranges, const int* __restrict__ point_list,
                               const float2* __restrict__ means, const float* __restrict__ conic,
                               const float* __restrict__ color, const float* __restrict__ opacity, float3 bg,
                               int W, int H, int tiles_x,
                               float* __restrict__ out, float* __restrict__ final_T, int* __restrict__ n_contrib,
                               const int* __restrict__ bucket_offsets, float4* __restrict__ bucket_state) {
    const int tile = blockIdx.y * tiles_x + blockIdx.x;
    const int tid = threadIdx.y * TILE + threadIdx.x;
    const int px = blockIdx.x * TILE + threadIdx.x;
    const int py = blockIdx.y * TILE + threadIdx.y;
    const bool inside = px < W && py < H;
    bool done = !inside;

    const int2 range = ranges[tile];
    const int todo = range.y - range.x;
    const int rounds = (todo + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int bbase = bucket_state ? bucket_offsets[tile] : 0;

    __shared__ int s_id[BLOCK_SIZE];
    __shared__ Splat s_splat[BLOCK_SIZE];

    float T = 1.0f;
    float3 C = make_float3(0.f, 0.f, 0.f);
    int last = 0;

    for (int r = 0; r < rounds; r++) {
        // If every pixel is done, stop. Unwritten buckets stay zero; the backward
        // skips them because they start after every pixel's last contributor.
        if (__syncthreads_count(done) == BLOCK_SIZE) break;
        const int k = range.x + r * BLOCK_SIZE + tid;
        if (k < range.y) {
            const int id = point_list[k];
            s_id[tid] = id;
            s_splat[tid] = load_splat(id, means, conic, opacity);
        }
        __syncthreads();

        const int n = min(BLOCK_SIZE, todo - r * BLOCK_SIZE);
        for (int j = 0; j < n; j++) {
            const int idx = r * BLOCK_SIZE + j;  // position in this tile's list
            if (bucket_state && (idx % BUCKET) == 0)
                bucket_state[(size_t)(bbase + idx / BUCKET) * BLOCK_SIZE + tid] = make_float4(T, C.x, C.y, C.z);
            if (done) continue;

            float dx, dy, G, alpha;
            bool clamped;
            if (!eval_splat(s_splat[j], (float)px, (float)py, dx, dy, G, alpha, clamped)) continue;
            const float test_T = T * (1.0f - alpha);
            if (test_T < T_MIN) {
                done = true;
                continue;
            }
            const float3 c = load_color(s_id[j], color);
            const float w = alpha * T;
            C.x += c.x * w;
            C.y += c.y * w;
            C.z += c.z * w;
            T = test_T;
            last = idx + 1;
        }
    }

    if (inside) {
        const int pix = py * W + px;
        const int HW = H * W;
        out[0 * HW + pix] = C.x + T * bg.x;
        out[1 * HW + pix] = C.y + T * bg.y;
        out[2 * HW + pix] = C.z + T * bg.z;
        final_T[pix] = T;
        n_contrib[pix] = last;
    }
}

// ---------------------------------------------------------------------------
// Backward, per pixel (3DGS style)
// ---------------------------------------------------------------------------
__global__ void backward_pixel_kernel(const int2* __restrict__ ranges, const int* __restrict__ point_list,
                                      const float2* __restrict__ means, const float* __restrict__ conic,
                                      const float* __restrict__ color, const float* __restrict__ opacity, float3 bg,
                                      int W, int H, int tiles_x,
                                      const float* __restrict__ final_T, const int* __restrict__ n_contrib,
                                      const float* __restrict__ dL_dout,
                                      float* dmeans, float* dconic, float* dcolor, float* dopacity) {
    const int tile = blockIdx.y * tiles_x + blockIdx.x;
    const int tid = threadIdx.y * TILE + threadIdx.x;
    const int px = blockIdx.x * TILE + threadIdx.x;
    const int py = blockIdx.y * TILE + threadIdx.y;
    const bool inside = px < W && py < H;
    const int pix = py * W + px;
    const int HW = H * W;

    const int2 range = ranges[tile];
    const int todo = range.y - range.x;
    const int rounds = (todo + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int s_id[BLOCK_SIZE];
    __shared__ Splat s_splat[BLOCK_SIZE];
    __shared__ float3 s_col[BLOCK_SIZE];

    const float T_final = inside ? final_T[pix] : 0.f;
    const int last = inside ? n_contrib[pix] : 0;
    const float3 dpix = inside ? make_float3(dL_dout[pix], dL_dout[HW + pix], dL_dout[2 * HW + pix])
                               : make_float3(0.f, 0.f, 0.f);
    const float bg_dot = bg.x * dpix.x + bg.y * dpix.y + bg.z * dpix.z;

    float T = T_final;
    float3 acc = make_float3(0.f, 0.f, 0.f);  // colour behind the current splat, divided by T after it
    float last_alpha = 0.f;
    float3 last_color = make_float3(0.f, 0.f, 0.f);
    int contributor = todo;

    for (int r = 0; r < rounds; r++) {
        __syncthreads();  // previous round finished reading shared memory
        const int k = range.y - 1 - (r * BLOCK_SIZE + tid);
        if (k >= range.x) {
            const int id = point_list[k];
            s_id[tid] = id;
            s_splat[tid] = load_splat(id, means, conic, opacity);
            s_col[tid] = load_color(id, color);
        }
        __syncthreads();

        const int n = min(BLOCK_SIZE, todo - r * BLOCK_SIZE);
        for (int j = 0; j < n; j++) {
            contributor--;
            if (!inside || contributor >= last) continue;

            const Splat s = s_splat[j];
            float dx, dy, G, alpha;
            bool clamped;
            if (!eval_splat(s, (float)px, (float)py, dx, dy, G, alpha, clamped)) continue;

            const float one_minus = 1.0f - alpha;
            T = T / one_minus;  // T before this splat
            acc.x = last_alpha * last_color.x + (1.f - last_alpha) * acc.x;
            acc.y = last_alpha * last_color.y + (1.f - last_alpha) * acc.y;
            acc.z = last_alpha * last_color.z + (1.f - last_alpha) * acc.z;
            const float3 c = s_col[j];
            last_alpha = alpha;
            last_color = c;

            float dL_dalpha = ((c.x - acc.x) * dpix.x + (c.y - acc.y) * dpix.y + (c.z - acc.z) * dpix.z) * T;
            dL_dalpha += (-T_final / one_minus) * bg_dot;

            SplatGrad g;
            zero(g);
            const float w = alpha * T;
            g.dcol = make_float3(w * dpix.x, w * dpix.y, w * dpix.z);
            alpha_chain(s, dx, dy, G, clamped, dL_dalpha, g);
            atomic_add_grad(g, s_id[j], dmeans, dconic, dcolor, dopacity);
        }
    }
}

// ---------------------------------------------------------------------------
// Backward, per splat (Taming-3DGS style)
// ---------------------------------------------------------------------------
__global__ void backward_splat_kernel(int num_buckets, const int* __restrict__ bucket_to_tile,
                                      const int* __restrict__ bucket_offsets,
                                      const int2* __restrict__ ranges, const int* __restrict__ point_list,
                                      const float2* __restrict__ means, const float* __restrict__ conic,
                                      const float* __restrict__ color, const float* __restrict__ opacity,
                                      int W, int H, int tiles_x, const float* __restrict__ out,
                                      const int* __restrict__ n_contrib, const float* __restrict__ dL_dout,
                                      const float4* __restrict__ bucket_state,
                                      float* dmeans, float* dconic, float* dcolor, float* dopacity) {
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (b >= num_buckets) return;  // whole warp exits together

    const int tile = bucket_to_tile[b];
    const int2 range = ranges[tile];
    const int idx = (b - bucket_offsets[tile]) * BUCKET + lane;  // position in tile list
    const bool valid = range.x + idx < range.y;
    const int tile_x0 = (tile % tiles_x) * TILE;
    const int tile_y0 = (tile / tiles_x) * TILE;
    const int HW = H * W;

    int id = 0;
    Splat s;
    float3 c = make_float3(0.f, 0.f, 0.f);
    if (valid) {
        id = point_list[range.x + idx];
        s = load_splat(id, means, conic, opacity);
        c = load_color(id, color);
    }

    SplatGrad g;
    zero(g);

    // State of the pixel this lane is working on: T and accumulated colour
    // *before* this lane's splat.
    float T = 0.f;
    float3 C = make_float3(0.f, 0.f, 0.f);

    for (int t = 0; t < BLOCK_SIZE + BUCKET - 1; t++) {
        // Take the pixel state lane-1 finished on the previous step.
        const float T_in = __shfl_up_sync(FULL_MASK, T, 1);
        const float Cx_in = __shfl_up_sync(FULL_MASK, C.x, 1);
        const float Cy_in = __shfl_up_sync(FULL_MASK, C.y, 1);
        const float Cz_in = __shfl_up_sync(FULL_MASK, C.z, 1);
        const int p = t - lane;  // pixel index inside the tile
        if (lane == 0) {
            if (p < BLOCK_SIZE) {
                const float4 st = bucket_state[(size_t)b * BLOCK_SIZE + p];
                T = st.x;
                C = make_float3(st.y, st.z, st.w);
            }
        } else {
            T = T_in;
            C = make_float3(Cx_in, Cy_in, Cz_in);
        }

        if (!valid || p < 0 || p >= BLOCK_SIZE) continue;
        const int px = tile_x0 + p % TILE;
        const int py = tile_y0 + p / TILE;
        if (px >= W || py >= H) continue;
        const int pix = py * W + px;
        if (idx >= n_contrib[pix]) continue;

        float dx, dy, G, alpha;
        bool clamped;
        if (!eval_splat(s, (float)px, (float)py, dx, dy, G, alpha, clamped)) continue;

        const float3 dpix = make_float3(dL_dout[pix], dL_dout[HW + pix], dL_dout[2 * HW + pix]);
        const float w = alpha * T;
        const float3 C_after = make_float3(C.x + c.x * w, C.y + c.y * w, C.z + c.z * w);
        // Colour of everything behind this splat, background included.
        const float3 behind = make_float3(out[pix] - C_after.x, out[HW + pix] - C_after.y,
                                          out[2 * HW + pix] - C_after.z);
        const float inv = 1.0f / (1.0f - alpha);
        const float dL_dalpha = dpix.x * (T * c.x - behind.x * inv) + dpix.y * (T * c.y - behind.y * inv) +
                                dpix.z * (T * c.z - behind.z * inv);

        g.dcol.x += w * dpix.x;
        g.dcol.y += w * dpix.y;
        g.dcol.z += w * dpix.z;
        alpha_chain(s, dx, dy, G, clamped, dL_dalpha, g);

        C = C_after;
        T = T * (1.0f - alpha);
    }

    if (valid) atomic_add_grad(g, id, dmeans, dconic, dcolor, dopacity);
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
#define CHECK_INPUT(x, dt)                                                       \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                     \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                  \
    TORCH_CHECK((x).scalar_type() == (dt), #x " has the wrong dtype")

static float3 to_float3(const torch::Tensor& bg) {
    auto b = bg.to(torch::kCPU, torch::kFloat32).contiguous();
    const float* p = b.data_ptr<float>();
    return make_float3(p[0], p[1], p[2]);
}

std::vector<torch::Tensor> rasterize_forward(torch::Tensor means, torch::Tensor conic, torch::Tensor color,
                                             torch::Tensor opacity, torch::Tensor bg, torch::Tensor ranges,
                                             torch::Tensor point_list, torch::Tensor bucket_offsets,
                                             int64_t num_buckets, int64_t H, int64_t W, bool save_buckets) {
    CHECK_INPUT(means, torch::kFloat32);
    CHECK_INPUT(conic, torch::kFloat32);
    CHECK_INPUT(color, torch::kFloat32);
    CHECK_INPUT(opacity, torch::kFloat32);
    CHECK_INPUT(ranges, torch::kInt32);
    CHECK_INPUT(point_list, torch::kInt32);
    CHECK_INPUT(bucket_offsets, torch::kInt32);

    const int tiles_x = (W + TILE - 1) / TILE, tiles_y = (H + TILE - 1) / TILE;
    auto f = means.options();
    auto out = torch::zeros({3, H, W}, f);
    auto final_T = torch::zeros({H, W}, f);
    auto n_contrib = torch::zeros({H, W}, f.dtype(torch::kInt32));
    auto bucket_state = save_buckets ? torch::zeros({num_buckets, BLOCK_SIZE, 4}, f) : torch::empty({0}, f);

    dim3 grid(tiles_x, tiles_y), block(TILE, TILE);
    forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int2*>(ranges.data_ptr<int>()), point_list.data_ptr<int>(),
        reinterpret_cast<const float2*>(means.data_ptr<float>()), conic.data_ptr<float>(), color.data_ptr<float>(),
        opacity.data_ptr<float>(), to_float3(bg), (int)W, (int)H, tiles_x, out.data_ptr<float>(),
        final_T.data_ptr<float>(), n_contrib.data_ptr<int>(), bucket_offsets.data_ptr<int>(),
        (save_buckets && num_buckets > 0) ? reinterpret_cast<float4*>(bucket_state.data_ptr<float>()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, final_T, n_contrib, bucket_state};
}

std::vector<torch::Tensor> rasterize_backward_pixel(torch::Tensor means, torch::Tensor conic, torch::Tensor color,
                                                    torch::Tensor opacity, torch::Tensor bg, torch::Tensor ranges,
                                                    torch::Tensor point_list, torch::Tensor final_T,
                                                    torch::Tensor n_contrib, torch::Tensor dL_dout) {
    CHECK_INPUT(dL_dout, torch::kFloat32);
    CHECK_INPUT(final_T, torch::kFloat32);
    CHECK_INPUT(n_contrib, torch::kInt32);
    const int64_t H = final_T.size(0), W = final_T.size(1);
    const int tiles_x = (W + TILE - 1) / TILE, tiles_y = (H + TILE - 1) / TILE;
    auto dmeans = torch::zeros_like(means), dconic = torch::zeros_like(conic);
    auto dcolor = torch::zeros_like(color), dopacity = torch::zeros_like(opacity);

    dim3 grid(tiles_x, tiles_y), block(TILE, TILE);
    backward_pixel_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int2*>(ranges.data_ptr<int>()), point_list.data_ptr<int>(),
        reinterpret_cast<const float2*>(means.data_ptr<float>()), conic.data_ptr<float>(), color.data_ptr<float>(),
        opacity.data_ptr<float>(), to_float3(bg), (int)W, (int)H, tiles_x, final_T.data_ptr<float>(),
        n_contrib.data_ptr<int>(), dL_dout.data_ptr<float>(), dmeans.data_ptr<float>(), dconic.data_ptr<float>(),
        dcolor.data_ptr<float>(), dopacity.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dmeans, dconic, dcolor, dopacity};
}

std::vector<torch::Tensor> rasterize_backward_splat(torch::Tensor means, torch::Tensor conic, torch::Tensor color,
                                                    torch::Tensor opacity, torch::Tensor bg, torch::Tensor ranges,
                                                    torch::Tensor point_list, torch::Tensor bucket_offsets,
                                                    torch::Tensor bucket_to_tile, torch::Tensor out,
                                                    torch::Tensor final_T, torch::Tensor n_contrib,
                                                    torch::Tensor bucket_state, torch::Tensor dL_dout) {
    CHECK_INPUT(dL_dout, torch::kFloat32);
    CHECK_INPUT(out, torch::kFloat32);
    CHECK_INPUT(bucket_state, torch::kFloat32);
    CHECK_INPUT(bucket_to_tile, torch::kInt32);
    const int64_t H = final_T.size(0), W = final_T.size(1);
    const int tiles_x = (W + TILE - 1) / TILE;
    const int num_buckets = (int)bucket_to_tile.size(0);
    auto dmeans = torch::zeros_like(means), dconic = torch::zeros_like(conic);
    auto dcolor = torch::zeros_like(color), dopacity = torch::zeros_like(opacity);
    if (num_buckets == 0) return {dmeans, dconic, dcolor, dopacity};

    const int threads = 128;  // 4 buckets per block
    const int blocks = (int)(((int64_t)num_buckets * 32 + threads - 1) / threads);
    backward_splat_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_buckets, bucket_to_tile.data_ptr<int>(), bucket_offsets.data_ptr<int>(),
        reinterpret_cast<const int2*>(ranges.data_ptr<int>()), point_list.data_ptr<int>(),
        reinterpret_cast<const float2*>(means.data_ptr<float>()), conic.data_ptr<float>(), color.data_ptr<float>(),
        opacity.data_ptr<float>(), (int)W, (int)H, tiles_x, out.data_ptr<float>(),
        n_contrib.data_ptr<int>(), dL_dout.data_ptr<float>(),
        reinterpret_cast<const float4*>(bucket_state.data_ptr<float>()), dmeans.data_ptr<float>(),
        dconic.data_ptr<float>(), dcolor.data_ptr<float>(), dopacity.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dmeans, dconic, dcolor, dopacity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterize_forward, "2D Gaussian tile rasterizer forward");
    m.def("backward_pixel", &rasterize_backward_pixel, "per-pixel backward (atomics per pixel-splat pair)");
    m.def("backward_splat", &rasterize_backward_splat, "per-splat backward (warp per 32-splat bucket)");
}
