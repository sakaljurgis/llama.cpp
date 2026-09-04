#pragma once

#include "common.cuh"

// K-quant GEMV for sm_60 (GP100) built on HFMA2, widths 1..8 (see mmvq-f16-sm60.cu for the Q4_1 version).
// Handles Q4_K, Q5_K, IQ4_XS (widths 4..8 only) and Q6_K.
// Returns false when the shape or type does not qualify; the caller then falls back to plain MMVQ.
bool ggml_cuda_mmvq_k_f16_sm60_supported(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
    const void * fusion);

void ggml_cuda_mmvq_k_f16_sm60(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
