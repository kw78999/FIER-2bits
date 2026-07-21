#include <torch/extension.h>

std::vector<torch::Tensor> popkv_pack_cuda(torch::Tensor keys, int64_t token_capacity);
void popkv_pack_into_cuda(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset);
torch::Tensor popkv_score_cuda(
    torch::Tensor queries, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean,
    torch::Tensor head_to_kv, int64_t tokens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack", &popkv_pack_cuda);
    m.def("pack_into", &popkv_pack_into_cuda);
    m.def("score", &popkv_score_cuda);
}
