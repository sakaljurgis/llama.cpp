#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows of the gating matrix that topk_moe_cuda maps onto a single CUDA block.  Shared with
// ggml-cuda.cu, which may only allow the input/output aliasing exception for row counts that
// fit in one block.
#define GGML_CUDA_TOPK_MOE_ROWS_PER_BLOCK 4

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
