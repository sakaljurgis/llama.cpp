#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// A8, sm_60 (GP100) only: the largest ne11 that MUL_MAT still sends to MMVQ.  GP100 has no DP4A
// and therefore no MMQ, so a wider batch is dequantized to F16 and multiplied by cuBLAS, which
// re-reads the weight matrix twice and costs the same at 9 columns as at 128.  Above
// MMVQ_MAX_BATCH_SIZE the matvec is looped over column chunks instead; the ceiling per type is
// where that loop becomes the more expensive of the two.  GGML_CUDA_MMVQ_MAX_COLS_SM60 overrides
// it, and 8 or 0 there restores the upstream behaviour exactly.  Returns MMVQ_MAX_BATCH_SIZE on
// every other architecture.
int ggml_cuda_mmvq_max_cols_sm60(enum ggml_type type, int cc);

// A8: how many column chunks a MUL_MAT of ncols columns is split into, and the width of chunk i.
// ncols <= MMVQ_MAX_BATCH_SIZE always gives one chunk of ncols, so nothing below 9 columns moves.
int ggml_cuda_mmvq_cols_nchunks(int64_t ncols);

static inline int ggml_cuda_mmvq_cols_chunk(const int64_t ncols, const int nchunks, const int i) {
    return (int) (ncols/nchunks) + (i < (int) (ncols%nchunks) ? 1 : 0);
}

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
