#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

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
__global__ void quantize_query_pack_kernel(
    const scalar_t* __restrict__ query,
    int32_t* __restrict__ q_sign,
    int32_t* __restrict__ q_mag,
    int32_t* __restrict__ q_mag_count,
    int64_t n_heads,
    int64_t head_dim,
    int64_t words) {
    int64_t row = blockIdx.x;
    int64_t word = threadIdx.x;
    if (row >= n_heads || word >= words) {
        return;
    }

    const int64_t base = row * head_dim;
    float min_v = load_as_float(query, base);
    float max_v = min_v;
    for (int64_t d = 1; d < head_dim; ++d) {
        float v = load_as_float(query, base + d);
        min_v = fminf(min_v, v);
        max_v = fmaxf(max_v, v);
    }
    const float neg = min_v * 0.5f;
    const float pos = max_v * 0.5f;

    uint32_t sign_word = 0;
    uint32_t mag_word = 0;
    const int64_t start = word * 32;
    const int64_t end = min(start + 32, head_dim);
    for (int64_t d = start; d < end; ++d) {
        float v = load_as_float(query, base + d);
        bool sign = v >= 0.0f;
        bool mag = sign ? (v > pos) : (v < neg);
        uint32_t bit = uint32_t(1) << uint32_t(d - start);
        if (sign) {
            sign_word |= bit;
        }
        if (mag) {
            mag_word |= bit;
        }
    }
    q_sign[row * words + word] = static_cast<int32_t>(sign_word);
    q_mag[row * words + word] = static_cast<int32_t>(mag_word);

    atomicAdd(&q_mag_count[row], __popc(mag_word));
}

template <typename scalar_t>
__global__ void quantize_key_pack_kernel(
    const scalar_t* __restrict__ keys,
    int32_t* __restrict__ k_sign,
    int32_t* __restrict__ k_mag,
    int32_t* __restrict__ k_mag_count,
    int64_t kv_heads,
    int64_t tokens,
    int64_t head_dim,
    int64_t words,
    int64_t group_size) {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = kv_heads * tokens * words;
    if (linear >= total) {
        return;
    }
    int64_t word = linear % words;
    int64_t token = (linear / words) % tokens;
    int64_t kvh = linear / (tokens * words);
    int64_t group_start = (token / group_size) * group_size;
    int64_t group_end = min(group_start + group_size, tokens);
    int64_t start = word * 32;
    int64_t end = min(start + 32, head_dim);

    uint32_t sign_word = 0;
    uint32_t mag_word = 0;
    for (int64_t d = start; d < end; ++d) {
        int64_t channel_base = (kvh * tokens * head_dim) + d;
        float min_v = load_as_float(keys, channel_base + group_start * head_dim);
        float max_v = min_v;
        for (int64_t t = group_start + 1; t < group_end; ++t) {
            float candidate = load_as_float(keys, channel_base + t * head_dim);
            min_v = fminf(min_v, candidate);
            max_v = fmaxf(max_v, candidate);
        }
        float v = load_as_float(keys, channel_base + token * head_dim);
        bool sign = v >= 0.0f;
        bool mag = sign ? (v > max_v * 0.5f) : (v < min_v * 0.5f);
        uint32_t bit = uint32_t(1) << uint32_t(d - start);
        if (sign) {
            sign_word |= bit;
        }
        if (mag) {
            mag_word |= bit;
        }
    }
    k_sign[linear] = static_cast<int32_t>(sign_word);
    k_mag[linear] = static_cast<int32_t>(mag_word);
    atomicAdd(&k_mag_count[kvh * tokens + token], __popc(mag_word));
}

__global__ void score_packed_batched_kernel(
    const int32_t* __restrict__ q_sign,
    const int32_t* __restrict__ q_mag,
    const int32_t* __restrict__ q_mag_count,
    const int32_t* __restrict__ k_sign,
    const int32_t* __restrict__ k_mag,
    const int32_t* __restrict__ k_mag_count,
    const int64_t* __restrict__ head_to_kv,
    const int64_t* __restrict__ valid_tokens,
    int32_t* __restrict__ scores,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t max_tokens,
    int64_t words,
    int64_t head_dim,
    int64_t valid_tokens_numel) {
    int64_t token = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y;
    if (row >= batch * q_heads || token >= max_tokens) {
        return;
    }
    int64_t b = row / q_heads;
    int64_t qh = row - b * q_heads;
    int64_t valid = valid_tokens_numel == 1 ? valid_tokens[0] : valid_tokens[b];
    if (token >= valid) {
        scores[row * max_tokens + token] = INT32_MIN;
        return;
    }
    int64_t kvh = head_to_kv[qh];
    int64_t q_base = row * words;
    int64_t k_base = (kvh * max_tokens + token) * words;
    int score = static_cast<int>(head_dim) + q_mag_count[row] + k_mag_count[kvh * max_tokens + token];

    if (head_dim == 128 && words == 4) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t x = static_cast<uint32_t>(q_sign[q_base + w]) ^ static_cast<uint32_t>(k_sign[k_base + w]);
            uint32_t q = static_cast<uint32_t>(q_mag[q_base + w]);
            uint32_t k = static_cast<uint32_t>(k_mag[k_base + w]);
            score -= 2 * __popc(x);
            score -= 2 * __popc(x & q);
            score -= 2 * __popc(x & k);
        }
    } else {
        for (int64_t w = 0; w < words; ++w) {
            uint32_t x = static_cast<uint32_t>(q_sign[q_base + w]) ^ static_cast<uint32_t>(k_sign[k_base + w]);
            uint32_t q = static_cast<uint32_t>(q_mag[q_base + w]);
            uint32_t k = static_cast<uint32_t>(k_mag[k_base + w]);
            score -= 2 * __popc(x);
            score -= 2 * __popc(x & q);
            score -= 2 * __popc(x & k);
        }
    }
    scores[row * max_tokens + token] = score;
}

__global__ void histogram_topk_from_scores_kernel(
    const int32_t* __restrict__ scores,
    const int64_t* __restrict__ valid_tokens,
    int64_t batch,
    int64_t q_heads,
    int64_t max_tokens,
    int64_t budget,
    int64_t head_dim,
    int64_t bins,
    int64_t valid_tokens_numel,
    int64_t* __restrict__ indices) {
    int64_t row = blockIdx.x;
    if (row >= batch * q_heads) {
        return;
    }
    int64_t b = row / q_heads;
    int64_t valid = valid_tokens_numel == 1 ? valid_tokens[0] : valid_tokens[b];
    int64_t k = min(budget, valid);

    extern __shared__ int hist[];
    for (int64_t i = threadIdx.x; i < bins; i += blockDim.x) {
        hist[i] = 0;
    }
    __syncthreads();
    for (int64_t token = threadIdx.x; token < valid; token += blockDim.x) {
        int32_t score = scores[row * max_tokens + token];
        atomicAdd(&hist[score + 3 * head_dim], 1);
    }
    __syncthreads();

    __shared__ int threshold;
    __shared__ int higher_count;
    if (threadIdx.x == 0) {
        int cumulative = 0;
        threshold = -3 * static_cast<int>(head_dim);
        higher_count = 0;
        for (int64_t bin = bins - 1; bin >= 0; --bin) {
            int count = hist[bin];
            if (cumulative + count >= k) {
                threshold = static_cast<int>(bin - 3 * head_dim);
                higher_count = cumulative;
                break;
            }
            cumulative += count;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int out = 0;
        for (int score_value = 3 * static_cast<int>(head_dim); score_value > threshold; --score_value) {
            for (int64_t token = 0; token < valid; ++token) {
                int32_t score = scores[row * max_tokens + token];
                if (score == score_value) {
                    indices[row * budget + out] = token;
                    ++out;
                }
            }
        }
        int needed = static_cast<int>(k) - out;
        if (needed > 0) {
            for (int64_t token = 0; token < valid && needed > 0; ++token) {
                int32_t score = scores[row * max_tokens + token];
                if (score == threshold) {
                    indices[row * budget + out] = token;
                    ++out;
                    --needed;
                }
            }
        }
        for (; out < budget; ++out) {
            indices[row * budget + out] = valid > 0 ? valid - 1 : 0;
        }
    }
}

std::vector<torch::Tensor> quantize_query_pack_cuda(torch::Tensor query) {
    CHECK_CUDA(query);
    CHECK_CONTIGUOUS(query);
    CHECK_FLOATISH(query);
    TORCH_CHECK(query.dim() == 3, "query must be [B, QH, D]");
    c10::cuda::CUDAGuard guard(query.device());
    int64_t batch = query.size(0);
    int64_t q_heads = query.size(1);
    int64_t head_dim = query.size(2);
    int64_t words = (head_dim + 31) / 32;
    auto int_opts = query.options().dtype(torch::kInt32);
    auto q_sign = torch::empty({batch, q_heads, words}, int_opts);
    auto q_mag = torch::empty({batch, q_heads, words}, int_opts);
    auto q_mag_count = torch::zeros({batch, q_heads}, int_opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(), "quantize_query_pack_cuda", [&] {
        quantize_query_pack_kernel<scalar_t><<<batch * q_heads, words, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            q_sign.data_ptr<int32_t>(),
            q_mag.data_ptr<int32_t>(),
            q_mag_count.data_ptr<int32_t>(),
            batch * q_heads,
            head_dim,
            words);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_sign, q_mag, q_mag_count};
}

std::vector<torch::Tensor> quantize_key_pack_cuda(torch::Tensor keys, int64_t group_size) {
    CHECK_CUDA(keys);
    CHECK_CONTIGUOUS(keys);
    CHECK_FLOATISH(keys);
    TORCH_CHECK(keys.dim() == 3, "keys must be [KVH, T, D]");
    TORCH_CHECK(group_size > 0, "group_size must be positive");
    c10::cuda::CUDAGuard guard(keys.device());
    int64_t kv_heads = keys.size(0);
    int64_t tokens = keys.size(1);
    int64_t head_dim = keys.size(2);
    int64_t words = (head_dim + 31) / 32;
    auto int_opts = keys.options().dtype(torch::kInt32);
    auto k_sign = torch::empty({kv_heads, tokens, words}, int_opts);
    auto k_mag = torch::empty({kv_heads, tokens, words}, int_opts);
    auto k_mag_count = torch::zeros({kv_heads, tokens}, int_opts);
    int64_t total = kv_heads * tokens * words;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "quantize_key_pack_cuda", [&] {
        quantize_key_pack_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            keys.data_ptr<scalar_t>(),
            k_sign.data_ptr<int32_t>(),
            k_mag.data_ptr<int32_t>(),
            k_mag_count.data_ptr<int32_t>(),
            kv_heads,
            tokens,
            head_dim,
            words,
            group_size);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {k_sign, k_mag, k_mag_count};
}

torch::Tensor score_packed_batched_cuda(
    torch::Tensor q_sign,
    torch::Tensor q_mag,
    torch::Tensor q_mag_count,
    torch::Tensor k_sign,
    torch::Tensor k_mag,
    torch::Tensor k_mag_count,
    torch::Tensor head_to_kv,
    torch::Tensor valid_tokens,
    int64_t head_dim) {
    CHECK_CUDA(q_sign); CHECK_CUDA(q_mag); CHECK_CUDA(q_mag_count);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(k_mag_count);
    CHECK_CUDA(head_to_kv); CHECK_CUDA(valid_tokens);
    CHECK_CONTIGUOUS(q_sign); CHECK_CONTIGUOUS(q_mag); CHECK_CONTIGUOUS(q_mag_count);
    CHECK_CONTIGUOUS(k_sign); CHECK_CONTIGUOUS(k_mag); CHECK_CONTIGUOUS(k_mag_count);
    CHECK_CONTIGUOUS(head_to_kv); CHECK_CONTIGUOUS(valid_tokens);
    CHECK_INT32(q_sign); CHECK_INT32(q_mag); CHECK_INT32(q_mag_count);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag); CHECK_INT32(k_mag_count);
    TORCH_CHECK(head_to_kv.scalar_type() == at::kLong, "head_to_kv must be int64");
    TORCH_CHECK(valid_tokens.scalar_type() == at::kLong, "valid_tokens must be int64");
    TORCH_CHECK(q_sign.dim() == 3 && k_sign.dim() == 3, "packed tensors must be rank 3");
    c10::cuda::CUDAGuard guard(q_sign.device());
    int64_t batch = q_sign.size(0);
    int64_t q_heads = q_sign.size(1);
    int64_t words = q_sign.size(2);
    int64_t kv_heads = k_sign.size(0);
    int64_t tokens = k_sign.size(1);
    auto scores = torch::empty({batch, q_heads, tokens}, q_sign.options());
    dim3 block(128);
    dim3 grid((tokens + block.x - 1) / block.x, batch * q_heads);
    score_packed_batched_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_sign.data_ptr<int32_t>(),
        q_mag.data_ptr<int32_t>(),
        q_mag_count.data_ptr<int32_t>(),
        k_sign.data_ptr<int32_t>(),
        k_mag.data_ptr<int32_t>(),
        k_mag_count.data_ptr<int32_t>(),
        head_to_kv.data_ptr<int64_t>(),
        valid_tokens.data_ptr<int64_t>(),
        scores.data_ptr<int32_t>(),
        batch,
        q_heads,
        kv_heads,
        tokens,
        words,
        head_dim,
        valid_tokens.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

torch::Tensor histogram_topk_from_scores_cuda(
    torch::Tensor scores,
    torch::Tensor valid_tokens,
    int64_t budget,
    int64_t head_dim) {
    CHECK_CUDA(scores);
    CHECK_CUDA(valid_tokens);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(valid_tokens);
    CHECK_INT32(scores);
    TORCH_CHECK(valid_tokens.scalar_type() == at::kLong, "valid_tokens must be int64");
    TORCH_CHECK(scores.dim() == 3, "scores must be [B, QH, T]");
    TORCH_CHECK(budget > 0, "budget must be positive");
    c10::cuda::CUDAGuard guard(scores.device());
    int64_t batch = scores.size(0);
    int64_t q_heads = scores.size(1);
    int64_t tokens = scores.size(2);
    int64_t bins = 6 * head_dim + 1;
    TORCH_CHECK(bins <= 2048, "histogram implementation supports head_dim <= 341");
    auto indices = torch::empty({batch, q_heads, budget}, scores.options().dtype(torch::kInt64));
    int threads = 256;
    size_t shared = static_cast<size_t>(bins) * sizeof(int);
    histogram_topk_from_scores_kernel<<<batch * q_heads, threads, shared, at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<int32_t>(),
        valid_tokens.data_ptr<int64_t>(),
        batch,
        q_heads,
        tokens,
        budget,
        head_dim,
        bins,
        valid_tokens.numel(),
        indices.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return indices;
}
