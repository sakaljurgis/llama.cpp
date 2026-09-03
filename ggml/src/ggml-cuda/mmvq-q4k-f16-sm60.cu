// Q4_K GEMV for sm_60 (GP100) built on HFMA2, widths 1..8.
//
// Same idea as the Q4_1 kernel in mmvq-f16-sm60.cu: GP100 has no DP4A, the emulated int8 dot product
// (4x PTX vmad) runs at 0.3x the MAC rate of HFMA2, so the nibbles are expanded to half2 with LOP3
// magic constants (0x6400 = 1024.0 for a low nibble, 0x5400 = 64.0 for a high nibble, then one HSUB2)
// and accumulated with HFMA2.  The activations are converted to half once per matvec; the kernel
// itself has no int-to-float conversion.
//
// Lane mapping: 4 lanes per block_q4_K, lane g owns the 64-weight group g = sub-blocks 2g and 2g+1,
// i.e. qs[32g .. 32g+31] as two 16-byte loads.  A warp covers 8 consecutive blocks (2048 elements)
// per step and reads 1152 contiguous bytes per row.  In word i of the group the low nibble of byte k
// is element 4i+k of sub-block 2g and the high nibble element 4i+k of sub-block 2g+1, so LOP3 on the
// word and on word>>8 yields the pairs (4i, 4i+2) and (4i+1, 4i+3) of each sub-block; the quantize
// kernel stores the activations in that pair order.
//
// Two things the shape of the loop costs on GP100 (measured on 4096x14336, width 1, in a standalone
// harness): without a prefetch the DRAM latency of the weight loads is fully exposed, 314 -> 459 GB/s
// once the next step's header and qs sit in registers, and the activation slots have to be laid out so
// that slot i of the 32 lanes of a step is one contiguous 512 byte line (a per-lane 128 byte stride
// costs about 5%).  The activations themselves are not prefetched: they are L1/L2 hits and a second
// buffer costs 32*ncols_dst registers, which spills from width 4 up.  Prefetching pays up to width 2,
// above that the register pressure of the accumulators wins and the plain loop is faster.
//
// Math per sub-block j: a' = a/amax with amax per 256-element block, so |q*a'| <= 15 and a lane sums
// 16 products into one half2 without overflow.  sc_j and m_j (6 bit) become exact halves with the same
// magic trick, sc*sum and m*ysum and the d/dmin multiply happen in half, the subtraction of the two
// terms and the sum over blocks happen in FP32 (the two terms cancel often, half would lose bits).

#include "mmvq-q4k-f16-sm60.cuh"

#include <cstdio>
#include <cstdlib>

#define A16K_QK     256 // elements per block_q4_K
#define A16K_LPB      4 // lanes per block_q4_K
#define A16K_BPW      8 // blocks per warp step
#define A16K_QBLOCK 128 // quantize kernel block size, one block_q4_K per warp

static __device__ __forceinline__ half2 a16k_as_half2(const int v) {
    return *((const half2 *) &v);
}

// One warp per 256-element block of one activation row.  Lane l owns g = l/8, i = l%8:
// elements 64g+4i+{0..3} (sub-block 2g) and 64g+32+4i+{0..3} (sub-block 2g+1).
// Output per block: 32 int4 of half2 pairs, 4 half2 of sub-block sums (ys), one float amax (yd).
// The int4 of a group of 8 blocks are interleaved, index (kb/8)*256 + i*32 + (kb%8)*4 + g, so that the
// matvec warp reads slot i of its 8 blocks as one contiguous 512 byte load.  A group of 8 is complete
// even if nblocks is not a multiple of 8 (the buffer is allocated for the padded count); a lane of the
// matvec only ever reads the slots of its own block, so the pad slots are never read.
template <bool vec4>
static __global__ void quantize_a16k(
        const float * __restrict__ x, int4 * __restrict__ aq, half2 * __restrict__ ys, float * __restrict__ yd,
        const int s01, const int nblocks) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int kb   = blockIdx.x*(A16K_QBLOCK/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int col  = blockIdx.y;
    if (kb >= nblocks) {
        return;
    }
    const int g = lane >> 3;
    const int i = lane &  7;

    const float * xp = x + (size_t) col*s01 + kb*A16K_QK + 64*g + 4*i;

    float v[8];
    if (vec4) {
        const float4 t0 = *((const float4 *) xp);
        const float4 t1 = *((const float4 *) (xp + 32));
        v[0] = t0.x; v[1] = t0.y; v[2] = t0.z; v[3] = t0.w;
        v[4] = t1.x; v[5] = t1.y; v[6] = t1.z; v[7] = t1.w;
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k]     = xp[k];
            v[4 + k] = xp[32 + k];
        }
    }

    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        amax = fmaxf(amax, fabsf(v[k]));
    }
    amax = warp_reduce_max(amax);
    const float id = amax > 0.0f ? 1.0f/amax : 0.0f;

    // ysum is the sum of the rounded halves, so that a zero weight (d*sc*q == dmin*m) stays zero.
    half  h[8];
    float s0 = 0.0f;
    float s1 = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        h[k]     = __float2half(v[k]*id);
        h[4 + k] = __float2half(v[4 + k]*id);
        s0 += __half2float(h[k]);
        s1 += __half2float(h[4 + k]);
    }
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s0 += __shfl_xor_sync(0xffffffff, s0, off, WARP_SIZE);
        s1 += __shfl_xor_sync(0xffffffff, s1, off, WARP_SIZE);
    }

    int4 out;
    half2 * o = (half2 *) &out;
    o[0] = __halves2half2(h[0], h[2]);
    o[1] = __halves2half2(h[1], h[3]);
    o[2] = __halves2half2(h[4], h[6]);
    o[3] = __halves2half2(h[5], h[7]);
    const int nb_pad = ((nblocks + A16K_BPW - 1)/A16K_BPW)*A16K_BPW;
    aq[(size_t) col*nb_pad*32 + (kb/A16K_BPW)*(A16K_BPW*32) + i*WARP_SIZE + (kb%A16K_BPW)*A16K_LPB + g] = out;

    if (i == 0) {
        ys[(size_t) col*nblocks*4 + kb*4 + g] = __floats2half2_rn(s0, s1);
    }
    if (lane == 0) {
        yd[(size_t) col*nblocks + kb] = amax;
    }
}

// Header and the lane's 32 bytes of qs of block kb of every row: 3 x LDG.128 per row.
template <int nrows>
static __device__ __forceinline__ void a16k_load_x(
        const char * const (&xrow)[nrows], const int koff, const int goff, int4 (&hdr)[nrows], int (&qw)[nrows][8]) {
#pragma unroll
    for (int r = 0; r < nrows; ++r) {
        const char * p = xrow[r] + koff;
        hdr[r] = *((const int4 *) p);
        const int4 q0 = *((const int4 *) (p + goff));
        const int4 q1 = *((const int4 *) (p + goff + 16));
        qw[r][0] = q0.x; qw[r][1] = q0.y; qw[r][2] = q0.z; qw[r][3] = q0.w;
        qw[r][4] = q1.x; qw[r][5] = q1.y; qw[r][6] = q1.z; qw[r][7] = q1.w;
    }
}

template <int ncols_dst, int nrows, int nwarps, bool prefetch>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps)
mul_mat_vec_q4_K_a16(
        const void * __restrict__ vx, const int4 * __restrict__ aq, const half2 * __restrict__ ys,
        const float * __restrict__ yd, float * __restrict__ dst, const int nblocks, const int nrows_x,
        const int stride_row_x, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int lane = threadIdx.x;
    const int b    = lane / A16K_LPB;
    const int g    = lane % A16K_LPB;
    const int row0 = blockIdx.x*nrows;

    const int stride_col_aq = ((nblocks + A16K_BPW - 1)/A16K_BPW)*(A16K_BPW*32);
    const int stride_col_ys = nblocks*4;

    const half2 magic_lo = a16k_as_half2(0x64006400);
    const half2 magic_hi = a16k_as_half2(0x54005400);

    // Rows past the end read the last row and are not stored.
    const char * xrow[nrows];
#pragma unroll
    for (int r = 0; r < nrows; ++r) {
        const int row = min(row0 + r, nrows_x - 1);
        xrow[r] = (const char *) vx + (size_t) row*stride_row_x*sizeof(block_q4_K);
    }
    const int goff = 16 + 32*g;

    float sumf[ncols_dst][nrows] = {{0.0f}};

    const int kb0   = threadIdx.y*A16K_BPW + b;
    const int kstep = A16K_BPW*nwarps;

    int4 hdr[nrows], hdr_next[nrows];
    int  qw[nrows][8], qw_next[nrows][8];
    if constexpr (prefetch) {
        if (kb0 < nblocks) {
            a16k_load_x<nrows>(xrow, kb0*(int) sizeof(block_q4_K), goff, hdr, qw);
        }
    }

    for (int kb = kb0; kb < nblocks; kb += kstep) {
        const bool more = kb + kstep < nblocks;
        if constexpr (prefetch) {
            if (more) {
                a16k_load_x<nrows>(xrow, (kb + kstep)*(int) sizeof(block_q4_K), goff, hdr_next, qw_next);
            }
        } else {
            a16k_load_x<nrows>(xrow, kb*(int) sizeof(block_q4_K), goff, hdr, qw);
        }

        half2 Y[ncols_dst];
        float D[ncols_dst];
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
            Y[c] = ys[c*stride_col_ys + kb*4 + g];
            D[c] = yd[c*nblocks + kb];
        }

        half2 acc0[nrows][ncols_dst];
        half2 acc1[nrows][ncols_dst];
#pragma unroll
        for (int r = 0; r < nrows; ++r) {
#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                acc0[r][c] = make_half2(0.0f, 0.0f);
                acc1[r][c] = make_half2(0.0f, 0.0f);
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int4 a[ncols_dst];
#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                a[c] = aq[(size_t) c*stride_col_aq + (kb - b)*32 + i*WARP_SIZE + lane];
            }
#pragma unroll
            for (int r = 0; r < nrows; ++r) {
                const int w  = qw[r][i];
                const int ws = w >> 8;
                int t0, t1, t2, t3;
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t0) : "r"(w),  "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t1) : "r"(ws), "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t2) : "r"(w),  "n"(0x00f000f0), "n"(0x54005400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t3) : "r"(ws), "n"(0x00f000f0), "n"(0x54005400));
                const half2 w0 = __hsub2(a16k_as_half2(t0), magic_lo); // sub-block 2g,   (4i, 4i+2)
                const half2 w1 = __hsub2(a16k_as_half2(t1), magic_lo); // sub-block 2g,   (4i+1, 4i+3)
                const half2 w2 = __hsub2(a16k_as_half2(t2), magic_hi); // sub-block 2g+1, (4i, 4i+2)
                const half2 w3 = __hsub2(a16k_as_half2(t3), magic_hi); // sub-block 2g+1, (4i+1, 4i+3)
#pragma unroll
                for (int c = 0; c < ncols_dst; ++c) {
                    acc0[r][c] = __hfma2(w0, a16k_as_half2(a[c].x), acc0[r][c]);
                    acc0[r][c] = __hfma2(w1, a16k_as_half2(a[c].y), acc0[r][c]);
                    acc1[r][c] = __hfma2(w2, a16k_as_half2(a[c].z), acc1[r][c]);
                    acc1[r][c] = __hfma2(w3, a16k_as_half2(a[c].w), acc1[r][c]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < nrows; ++r) {
            // 6-bit scale/min pair of sub-blocks 2g, 2g+1 (get_scale_min_k4 without the branch):
            // g < 2 takes bytes 2g, 2g+1 of scales[0..3] (sc) and scales[4..7] (m) masked to 6 bits;
            // g >= 2 takes the nibbles of scales[8..11] plus the top 2 bits of the g-2 bytes.
            const unsigned int sh = 16*(g & 1);
            const unsigned int A  = ((unsigned int) hdr[r].y) >> sh;
            const unsigned int B  = ((unsigned int) hdr[r].z) >> sh;
            const unsigned int C  = ((unsigned int) hdr[r].w) >> sh;
            const bool         hi = g >= 2;
            const unsigned int X  = hi ? C : A;
            const unsigned int Z  = hi ? (C >> 4) : B;
            const unsigned int s2 = hi ? 2 : 0;
            const unsigned int sc = (X & 0x0f0f) | ((A >> s2) & 0x3030); // sc_{2g} | sc_{2g+1} << 8
            const unsigned int m  = (Z & 0x0f0f) | ((B >> s2) & 0x3030); //  m_{2g} |  m_{2g+1} << 8

            const half2 S0 = __hsub2(a16k_as_half2(__byte_perm(sc, 0, 0x4040) | 0x64006400), magic_lo); // (sc0, sc0)
            const half2 S1 = __hsub2(a16k_as_half2(__byte_perm(sc, 0, 0x4141) | 0x64006400), magic_lo); // (sc1, sc1)
            const half2 Mm = __hsub2(a16k_as_half2(__byte_perm(m,  0, 0x4140) | 0x64006400), magic_lo); // (m0, m1)
            const half2 dm = a16k_as_half2(hdr[r].x);

#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                const half2 T  = __hfma2(acc1[r][c], S1, __hmul2(acc0[r][c], S0));
                const half2 U  = __hmul2(Mm, Y[c]);
                const half  hs = __hadd(__low2half(T), __high2half(T));
                const half  hm = __hadd(__low2half(U), __high2half(U));
                const float2 f = __half22float2(__hmul2(__halves2half2(hs, hm), dm));
                sumf[c][r] += D[c]*(f.x - f.y);
            }
        }

        if constexpr (prefetch) {
            if (more) {
#pragma unroll
                for (int r = 0; r < nrows; ++r) {
                    hdr[r] = hdr_next[r];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        qw[r][i] = qw_next[r][i];
                    }
                }
            }
        }
    }

    if constexpr (nwarps == 1) {
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
#pragma unroll
            for (int r = 0; r < nrows; ++r) {
                const float v = warp_reduce_sum(sumf[c][r]);
                if (threadIdx.x == r && row0 + r < nrows_x) {
                    dst[c*stride_col_dst + row0 + r] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][nrows];

#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
#pragma unroll
            for (int r = 0; r < nrows; ++r) {
                const float v = warp_reduce_sum(sumf[c][r]);
                if (threadIdx.x == 0) {
                    tmp[threadIdx.y][c][r] = v;
                }
            }
        }

        __syncthreads();

        if (threadIdx.y != 0) {
            return;
        }

#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
#pragma unroll
            for (int r = 0; r < nrows; ++r) {
                if (threadIdx.x == r && row0 + r < nrows_x) {
                    float s = 0.0f;
#pragma unroll
                    for (int w = 0; w < nwarps; ++w) {
                        s += tmp[w][c][r];
                    }
                    dst[c*stride_col_dst + row0 + r] = s;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

static int a16k_env_int(const char * name, const int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

bool ggml_cuda_mmvq_q4k_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
        const void * fusion) {
    static const bool disabled = a16k_env_int("GGML_CUDA_DISABLE_MMVQ_F16_K", 0) != 0;
    if (disabled || fusion || ids) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // GP100 only: no DP4A but full-rate fp16.  sm_61 has DP4A and 1/64 rate fp16.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % A16K_QK != 0 || ((uintptr_t) src0->data) % 16 != 0 || src0->nb[1] % 16 != 0) {
        return false;
    }
    static const int ncols_min = a16k_env_int("GGML_A16K_NCOLS_MIN", 1);
    static const int ncols_max = a16k_env_int("GGML_A16K_NCOLS_MAX", 8);
    if (src1->ne[1] < ncols_min || src1->ne[1] > ncols_max || src1->ne[1] > 8) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

template <int ncols_dst, int nrows, bool prefetch>
static void launch_a16k_nw(
        const void * vx, const int4 * aq, const half2 * ys, const float * yd, float * dst,
        const int nblocks, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        const int nwarps, cudaStream_t stream) {
    const dim3 grid((nrows_x + nrows - 1)/nrows, 1, 1);
    switch (nwarps) {
        case 1:
            mul_mat_vec_q4_K_a16<ncols_dst, nrows, 1, prefetch><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
            break;
        case 2:
            mul_mat_vec_q4_K_a16<ncols_dst, nrows, 2, prefetch><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
            break;
        default:
            mul_mat_vec_q4_K_a16<ncols_dst, nrows, 4, prefetch><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
            break;
    }
}

// Only widths that can win with a prefetch get both instantiations.
template <int ncols_dst, int nrows>
static void launch_a16k_pf(
        const void * vx, const int4 * aq, const half2 * ys, const float * yd, float * dst,
        const int nblocks, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        const int nwarps, const bool prefetch, cudaStream_t stream) {
    if constexpr (ncols_dst <= 4) {
        if (prefetch) {
            launch_a16k_nw<ncols_dst, nrows, true>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, stream);
            return;
        }
    }
    launch_a16k_nw<ncols_dst, nrows, false>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, stream);
}

// Rows per block: 2 at width 1, 4 above.  At width 1 the prefetched step is the largest register block
// (95 registers at 2 rows against 148 at 4, i.e. 21 resident warps against 13, 70 us against 81 on
// 4096x14336), from width 2 up the activation loads of a step are shared by all rows of the block and the
// wider block pays for itself even where it spills a little (width 8: 224 us against 307 at 2 rows).
// Warps per block: 1 unless the grid is too small to fill the SMs.  All three overridable for tuning.
template <int ncols_dst>
static void launch_a16k(
        const void * vx, const int4 * aq, const half2 * ys, const float * yd, float * dst,
        const int nblocks, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        cudaStream_t stream) {
    static const int env_rows   = a16k_env_int("GGML_A16K_ROWS", 0);
    static const int env_nwarps = a16k_env_int("GGML_A16K_NWARPS", 0);
    static const int env_pf     = a16k_env_int("GGML_A16K_PF", -1);
    static const int small_grid = a16k_env_int("GGML_A16K_SMALL_GRID", 128);
    static const int log_level  = a16k_env_int("GGML_A16K_LOG", 0);

    bool prefetch = ncols_dst <= 2;
    if (env_pf == 0 || env_pf == 1) {
        prefetch = env_pf == 1;
    }

    int rows = ncols_dst == 1 ? 2 : 4;
    if (env_rows == 2 || env_rows == 4 || (env_rows == 8 && ncols_dst <= 4)) {
        rows = env_rows;
    }
    const int grid = (nrows_x + rows - 1)/rows;

    int nwarps = grid >= small_grid ? 1 : (grid >= small_grid/4 ? 2 : 4);
    if (env_nwarps == 1 || env_nwarps == 2 || env_nwarps == 4) {
        nwarps = env_nwarps;
    }
    while (nwarps > 1 && nblocks < A16K_BPW*nwarps) {
        nwarps /= 2;
    }

    if (log_level > 0) {
        static int nlog = 0;
        if (nlog < 32) {
            ++nlog;
            fprintf(stderr, "a16k: ne00=%d ne01=%d ncols=%d rows=%d nwarps=%d prefetch=%d grid=%d\n",
                    nblocks*A16K_QK, nrows_x, ncols_dst, rows, nwarps, (int) prefetch, grid);
        }
    }

    switch (rows) {
        case 2:
            launch_a16k_pf<ncols_dst, 2>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, prefetch, stream);
            break;
        case 8:
            if constexpr (ncols_dst <= 4) {
                launch_a16k_pf<ncols_dst, 8>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, prefetch, stream);
                break;
            }
            // fallthrough
        default:
            launch_a16k_pf<ncols_dst, 4>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, prefetch, stream);
            break;
    }
}

void ggml_cuda_mmvq_q4k_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    cudaStream_t stream = ctx.stream();

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];

    const int     nblocks = ne00/A16K_QK;
    const int64_t s11     = src1->nb[1]/ggml_type_size(src1->type);

    // One buffer: aq (512 B per block), then ys (16 B), then yd (4 B); every part stays 16 B aligned.
    // aq is padded to a whole number of 8 block groups, see the interleaving in quantize_a16k.
    const int    nb_pad   = ((nblocks + A16K_BPW - 1)/A16K_BPW)*A16K_BPW;
    const size_t aq_bytes = (size_t) ne11*nb_pad*(A16K_QK*sizeof(half));
    const size_t ys_bytes = (size_t) ne11*nblocks*(A16K_QK/32*sizeof(half));
    const size_t yd_bytes = (size_t) ne11*nblocks*sizeof(float);
    const size_t a_bytes  = aq_bytes + ys_bytes + yd_bytes;

    // Same single-slot cache as the q8_1 activations in ggml_cuda_mul_mat_vec_q: the gate and up
    // matvecs of an FFN share src1, so the second one reuses the first one's conversion.
    ggml_cuda_pool_alloc<char> scoped(ctx.pool());
    char * a_d = nullptr;
    {
        const bool cacheable = ctx.a16k_cache_stream == nullptr || ctx.a16k_cache_stream == stream;
        const bool hit = cacheable &&
                         ctx.a16k_cache_mem    != nullptr &&
                         ctx.a16k_cache_src1   == src1 &&
                         ctx.a16k_cache_data   == src1->data &&
                         ctx.a16k_cache_stream == stream &&
                         ctx.a16k_cache_size   == a_bytes &&
                         ctx.a16k_cache_s[0]   == ne00 &&
                         ctx.a16k_cache_s[1]   == s11;
        if (hit) {
            a_d = ctx.a16k_cache_mem;
        } else {
            if (!cacheable) {
                a_d = scoped.alloc(a_bytes);
            } else {
                if (a_bytes > ctx.a16k_cache_cap) {
                    ctx.a16k_cache_free();
                    ggml_cuda_set_device(ctx.device);
                    CUDA_CHECK(cudaMalloc((void **) &ctx.a16k_cache_mem, a_bytes));
                    ctx.a16k_cache_cap = a_bytes;
                }
                a_d = ctx.a16k_cache_mem;
                ctx.a16k_cache_src1   = src1;
                ctx.a16k_cache_data   = src1->data;
                ctx.a16k_cache_stream = stream;
                ctx.a16k_cache_size   = a_bytes;
                ctx.a16k_cache_s[0]   = ne00;
                ctx.a16k_cache_s[1]   = s11;
            }

            const dim3 grid((nblocks + A16K_QBLOCK/WARP_SIZE - 1)/(A16K_QBLOCK/WARP_SIZE), ne11, 1);
            const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
            if (vec4) {
                quantize_a16k<true><<<grid, A16K_QBLOCK, 0, stream>>>(
                    (const float *) src1->data, (int4 *) a_d, (half2 *) (a_d + aq_bytes), (float *) (a_d + aq_bytes + ys_bytes), s11, nblocks);
            } else {
                quantize_a16k<false><<<grid, A16K_QBLOCK, 0, stream>>>(
                    (const float *) src1->data, (int4 *) a_d, (half2 *) (a_d + aq_bytes), (float *) (a_d + aq_bytes + ys_bytes), s11, nblocks);
            }
        }
    }

    const int4  * aq = (const int4  *)  a_d;
    const half2 * ys = (const half2 *) (a_d + aq_bytes);
    const float * yd = (const float *) (a_d + aq_bytes + ys_bytes);

    const int stride_row_x   = src0->nb[1]/sizeof(block_q4_K);
    const int stride_col_dst = dst->nb[1]/ggml_type_size(dst->type);

#define A16K_CASE(N) case N: launch_a16k<N>(src0->data, aq, ys, yd, (float *) dst->data, nblocks, ne01, stride_row_x, stride_col_dst, stream); break;
    switch (ne11) {
        A16K_CASE(1) A16K_CASE(2) A16K_CASE(3) A16K_CASE(4)
        A16K_CASE(5) A16K_CASE(6) A16K_CASE(7) A16K_CASE(8)
        default: GGML_ABORT("unsupported ncols_dst");
    }
#undef A16K_CASE
}
