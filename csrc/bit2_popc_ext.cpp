#include <torch/extension.h>

std::vector<torch::Tensor> quantize_query_pack_cuda(torch::Tensor query);
std::vector<torch::Tensor> quantize_key_pack_cuda(torch::Tensor keys, int64_t group_size);
torch::Tensor score_packed_batched_cuda(
    torch::Tensor q_sign,
    torch::Tensor q_mag,
    torch::Tensor q_mag_count,
    torch::Tensor k_sign,
    torch::Tensor k_mag,
    torch::Tensor k_mag_count,
    torch::Tensor head_to_kv,
    torch::Tensor valid_tokens,
    int64_t head_dim);
torch::Tensor histogram_topk_from_scores_cuda(
    torch::Tensor scores,
    torch::Tensor valid_tokens,
    int64_t budget,
    int64_t head_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_query_pack", &quantize_query_pack_cuda);
    m.def("quantize_key_pack", &quantize_key_pack_cuda);
    m.def("score_packed_batched", &score_packed_batched_cuda);
    m.def("histogram_topk_from_scores", &histogram_topk_from_scores_cuda);
}
