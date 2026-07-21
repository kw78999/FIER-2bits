#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_FLOATISH(x) TORCH_CHECK((x).scalar_type() == at::kFloat || (x).scalar_type() == at::kHalf || (x).scalar_type() == at::kBFloat16, #x " must be fp32/fp16/bf16")

template <typename scalar_t>
__device__ inline float load_as_float(const scalar_t* ptr, int64_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ inline float load_as_float<c10::Half>(const c10::Half* ptr, int64_t idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ inline float load_as_float<c10::BFloat16>(const c10::BFloat16* ptr, int64_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <typename scalar_t>
__global__ void popkv_pack_g32_kernel(
    const scalar_t* __restrict__ keys,
    int32_t* __restrict__ k_sign,
    int32_t* __restrict__ k_mag,
    __half* __restrict__ low_mean,
    __half* __restrict__ delta_mean,
    int64_t tokens,
    int64_t head_dim,
    int64_t words,
    int64_t num_groups,
    int64_t output_capacity,
    int64_t token_offset) {
    int lane = threadIdx.x;
    int64_t linear = blockIdx.x;
    int64_t word = linear % words;
    int64_t group = (linear / words) % num_groups;
    int64_t kvh = linear / (words * num_groups);
    int64_t d = word * 32 + lane;
    int64_t group_start = group * 32;
    int64_t group_end = min(group_start + 32, tokens);
    bool valid_dim = d < head_dim;
    float min_v = 0.0f, max_v = 0.0f;
    if (valid_dim && group_start < group_end) {
        min_v = max_v = load_as_float(keys, (kvh * tokens + group_start) * head_dim + d);
        for (int64_t t = group_start + 1; t < group_end; ++t) {
            float v = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
            min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
        }
    }
    for (int64_t t = group_start; t < group_end; ++t) {
        float v = valid_dim ? load_as_float(keys, (kvh * tokens + t) * head_dim + d) : 0.0f;
        bool sign = valid_dim && v >= 0.0f;
        bool mag = valid_dim && (sign ? (v > max_v * 0.5f) : (v < min_v * 0.5f));
        uint32_t sign_word = __ballot_sync(0xffffffffu, sign);
        uint32_t mag_word = __ballot_sync(0xffffffffu, mag);
        float low_sum = (!mag && valid_dim) ? fabsf(v) : 0.0f;
        float high_sum = (mag && valid_dim) ? fabsf(v) : 0.0f;
        int low_count = (!mag && valid_dim) ? 1 : 0;
        int high_count = (mag && valid_dim) ? 1 : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            low_sum += __shfl_down_sync(0xffffffffu, low_sum, offset);
            high_sum += __shfl_down_sync(0xffffffffu, high_sum, offset);
            low_count += __shfl_down_sync(0xffffffffu, low_count, offset);
            high_count += __shfl_down_sync(0xffffffffu, high_count, offset);
        }
        if (lane == 0) {
            float low, high;
            if (low_count == 0) {
                low = high_count ? high_sum / high_count : 0.0f; high = low;
            } else {
                low = low_sum / low_count;
                high = high_count ? high_sum / high_count : low;
            }
            int64_t out = (kvh * output_capacity + token_offset + t) * words + word;
            k_sign[out] = static_cast<int32_t>(sign_word);
            k_mag[out] = static_cast<int32_t>(mag_word);
            low_mean[out] = __float2half_rn(low);
            delta_mean[out] = __float2half_rn(high - low);
        }
    }
}

template <typename scalar_t>
__global__ void popkv_score_kernel(
    const scalar_t* __restrict__ queries,
    const int32_t* __restrict__ k_sign,
    const int32_t* __restrict__ k_mag,
    const __half* __restrict__ k_low_mean,
    const __half* __restrict__ k_delta_mean,
    const int64_t* __restrict__ head_to_kv,
    float* __restrict__ scores,
    int64_t batch,
    int64_t q_heads,
    int64_t token_capacity,
    int64_t output_tokens,
    int64_t words,
    int64_t head_dim) {
    int64_t row = blockIdx.x;
    if (row >= batch * q_heads) return;
    int64_t qh = row % q_heads;
    int64_t kvh = head_to_kv[qh];
    extern __shared__ unsigned char shared_raw[];
    uint32_t* q_sign = reinterpret_cast<uint32_t*>(shared_raw);
    uint32_t* q_mag = q_sign + words;
    float* q_low = reinterpret_cast<float*>(q_mag + words);
    float* q_delta = q_low + words;
    if (threadIdx.x == 0) {
        int64_t q_base = row * head_dim;
        float min_v = load_as_float(queries, q_base), max_v = min_v;
        for (int64_t d = 1; d < head_dim; ++d) {
            float v = load_as_float(queries, q_base + d);
            min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
        }
        float neg = min_v * 0.5f, pos = max_v * 0.5f;
        for (int64_t w = 0; w < words; ++w) {
            uint32_t sw = 0, mw = 0;
            float low_sum = 0.0f, high_sum = 0.0f;
            int low_count = 0, high_count = 0;
            int64_t start = w * 32, end = min(start + 32, head_dim);
            for (int64_t d = start; d < end; ++d) {
                float v = load_as_float(queries, q_base + d);
                bool sign = v >= 0.0f;
                bool mag = sign ? (v > pos) : (v < neg);
                uint32_t bit = uint32_t(1) << uint32_t(d - start);
                if (sign) sw |= bit;
                if (mag) { mw |= bit; high_sum += fabsf(v); ++high_count; }
                else { low_sum += fabsf(v); ++low_count; }
            }
            float low, high;
            if (low_count == 0) {
                low = high_count ? high_sum / high_count : 0.0f; high = low;
            } else {
                low = low_sum / low_count;
                high = high_count ? high_sum / high_count : low;
            }
            q_sign[w] = sw; q_mag[w] = mw;
            q_low[w] = low; q_delta[w] = high - low;
        }
    }
    __syncthreads();
    float scale = rsqrtf(static_cast<float>(head_dim));
    for (int64_t token = threadIdx.x; token < output_tokens; token += blockDim.x) {
        int64_t base = (kvh * token_capacity + token) * words;
        float score = 0.0f;
        for (int64_t w = 0; w < words; ++w) {
            uint32_t valid_mask = 0xffffffffu;
            if (w == words - 1 && (head_dim & 31))
                valid_mask = (uint32_t(1) << uint32_t(head_dim & 31)) - 1u;
            uint32_t qm = q_mag[w] & valid_mask;
            uint32_t km = static_cast<uint32_t>(k_mag[base + w]) & valid_mask;
            uint32_t match = ~(q_sign[w] ^ static_cast<uint32_t>(k_sign[base + w])) & valid_mask;
            uint32_t both = qm & km;
            int c_sign = 2 * __popc(match) - __popc(valid_mask);
            int c_q = 2 * __popc(match & qm) - __popc(qm);
            int c_k = 2 * __popc(match & km) - __popc(km);
            int c_qk = 2 * __popc(match & both) - __popc(both);
            float ql = q_low[w], dq = q_delta[w];
            float kl = __half2float(k_low_mean[base + w]);
            float dk = __half2float(k_delta_mean[base + w]);
            float group_score = fmaf(ql * kl, static_cast<float>(c_sign),
                fmaf(dq * kl, static_cast<float>(c_q),
                fmaf(ql * dk, static_cast<float>(c_k),
                     dq * dk * static_cast<float>(c_qk))));
            score += group_score;
        }
        scores[row * output_tokens + token] = score * scale;
    }
}

static void launch_popkv_pack(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset) {
    int64_t kv_heads = keys.size(0), tokens = keys.size(1), head_dim = keys.size(2);
    int64_t words = (head_dim + 31) / 32;
    int64_t capacity = k_sign.size(1), groups = (tokens + 31) / 32;
    int blocks = static_cast<int>(kv_heads * groups * words);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "popkv_pack_cuda", [&] {
        popkv_pack_g32_kernel<scalar_t><<<blocks, 32, 0, stream>>>(
            keys.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(low_mean.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(delta_mean.data_ptr<at::Half>()),
            tokens, head_dim, words, groups, capacity, token_offset);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> popkv_pack_cuda(torch::Tensor keys, int64_t token_capacity) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    TORCH_CHECK(keys.dim() == 3, "keys must be [KVH, T, D]");
    TORCH_CHECK(token_capacity >= keys.size(1), "token capacity is too small");
    c10::cuda::CUDAGuard guard(keys.device());
    int64_t words = (keys.size(2) + 31) / 32;
    auto shape = std::vector<int64_t>{keys.size(0), token_capacity, words};
    auto k_sign = torch::empty(shape, keys.options().dtype(torch::kInt32));
    auto k_mag = torch::empty(shape, keys.options().dtype(torch::kInt32));
    auto low_mean = torch::empty(shape, keys.options().dtype(torch::kFloat16));
    auto delta_mean = torch::empty(shape, keys.options().dtype(torch::kFloat16));
    launch_popkv_pack(keys, k_sign, k_mag, low_mean, delta_mean, 0);
    return {k_sign, k_mag, low_mean, delta_mean};
}

void popkv_pack_into_cuda(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(low_mean); CHECK_CUDA(delta_mean);
    CHECK_CONTIGUOUS(k_sign); CHECK_CONTIGUOUS(k_mag);
    CHECK_CONTIGUOUS(low_mean); CHECK_CONTIGUOUS(delta_mean);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag);
    TORCH_CHECK(low_mean.scalar_type() == at::kHalf && delta_mean.scalar_type() == at::kHalf,
                "Pop-KV representative metadata must be float16");
    TORCH_CHECK(token_offset >= 0 && token_offset + keys.size(1) <= k_sign.size(1),
                "Pop-KV cache capacity exceeded");
    c10::cuda::CUDAGuard guard(keys.device());
    launch_popkv_pack(keys, k_sign, k_mag, low_mean, delta_mean, token_offset);
}

torch::Tensor popkv_score_cuda(
    torch::Tensor queries, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean,
    torch::Tensor head_to_kv, int64_t tokens) {
    CHECK_CUDA(queries); CHECK_CONTIGUOUS(queries); CHECK_FLOATISH(queries);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(low_mean); CHECK_CUDA(delta_mean);
    CHECK_CUDA(head_to_kv); CHECK_CONTIGUOUS(head_to_kv);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag);
    TORCH_CHECK(head_to_kv.scalar_type() == at::kLong, "head mapping must be int64");
    c10::cuda::CUDAGuard guard(queries.device());
    int64_t batch = queries.size(0), q_heads = queries.size(1), head_dim = queries.size(2);
    int64_t capacity = k_sign.size(1), words = k_sign.size(2);
    TORCH_CHECK(tokens > 0 && tokens <= capacity, "invalid score token count");
    auto scores = torch::empty({batch, q_heads, tokens}, queries.options().dtype(torch::kFloat32));
    size_t shared = static_cast<size_t>(4 * words) * sizeof(uint32_t);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, queries.scalar_type(), "popkv_score_cuda", [&] {
        popkv_score_kernel<scalar_t><<<batch * q_heads, 256, shared, at::cuda::getCurrentCUDAStream()>>>(
            queries.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(low_mean.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(delta_mean.data_ptr<at::Half>()),
            head_to_kv.data_ptr<int64_t>(), scores.data_ptr<float>(),
            batch, q_heads, capacity, tokens, words, head_dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}
