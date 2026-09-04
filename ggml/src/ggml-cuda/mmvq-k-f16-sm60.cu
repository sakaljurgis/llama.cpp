// Q4_K, Q5_K, IQ4_XS and Q6_K GEMV for sm_60 (GP100) built on HFMA2, widths 1..8.
//
// Same idea as the Q4_1 kernel in mmvq-f16-sm60.cu: GP100 has no DP4A, the emulated int8 dot product
// (4x PTX vmad) runs at 0.3x the MAC rate of HFMA2, so the nibbles are expanded to half2 with LOP3
// magic constants (0x6400 = 1024.0 for a low nibble, 0x5400 = 64.0 for a high nibble, then one HSUB2)
// and accumulated with HFMA2.  The activations are converted to half once per matvec; the kernel
// itself has no int-to-float conversion.
//
// Lane mapping: 4 lanes per block, lane g owns the 64-weight group g = sub-blocks 2g and 2g+1,
// i.e. qs[32g .. 32g+31] as two 16-byte loads.  A warp covers 8 consecutive blocks (2048 elements)
// per step and reads 1152 (Q4_K) or 1408 (Q5_K) contiguous bytes per row.  In word i of the group the
// low nibble of byte k is element 4i+k of sub-block 2g and the high nibble element 4i+k of sub-block
// 2g+1, so LOP3 on the word and on word>>8 yields the pairs (4i, 4i+2) and (4i+1, 4i+3) of each
// sub-block; the quantize kernel stores the activations in that pair order.  Both types share the
// header (d, dmin, 12 bytes of 6-bit scales and mins) and that qs order, so they share the kernel,
// the launcher and the activation buffer.
//
// Q5_K only: element l of sub-block s also has bit s of qh[l].  The 5-bit value still fits the magic
// mantissa (1024 + v is exact for v < 1024), so the bit only has to be placed at mantissa bit 4 of the
// low-nibble half or bit 8 of the high-nibble half.  One rotate of the qh word per pair does that for
// both halves at once (the two elements of a pair are 16 bit positions apart in the word and so are
// their qh bits), then LOP3 masks it and adds the magic constant.  All 4 lanes of a block read all 32
// qh bytes; they are the same two 16-byte lines for the whole quarter warp, so the L1 serves 3 of the
// 4.  The h expansion costs 8 more instructions per 8 weights, which is 24% of the width 1 kernel
// time (77 us against 102 us on 4096x14336 with the same loads).
//
// Two things the shape of the loop costs on GP100 (measured on 4096x14336, width 1, in a standalone
// harness): without a prefetch the DRAM latency of the weight loads is fully exposed, 314 -> 459 GB/s
// once the next step's header and qs sit in registers, and the activation slots have to be laid out so
// that slot i of the 32 lanes of a step is one contiguous 512 byte line (a per-lane 128 byte stride
// costs about 5%).  The activations themselves are not prefetched: they are L1/L2 hits and a second
// buffer costs 32*ncols_dst registers, which spills from width 4 up.  Prefetching pays up to width 2
// for Q4_K and Q5_K and up to width 4 for IQ4_XS and Q6_K; above that the register pressure of the
// accumulators wins and the plain loop is faster.
//
// What the prefetch buffer holds is a second trade (pf mode 1 against mode 2, see mul_mat_vec_k_a16).
// A warp is limited by how many weight bytes it has in flight, which is the size of that buffer times
// the number of resident warps, so taking the header out of it (it is read after the accumulation
// loop, not inside it) can go either way: 4 registers per row less against a third fewer bytes in
// flight.  Which side wins depends on the width and, at width 1, on the k range: a warp that runs
// only 2 or 3 of the 8 block steps (nblocks < 24, i.e. every k=5120 tensor of the target model) pays
// a fixed cost of 0.38 steps for the pipeline and prefers the smaller buffer, while at k=14336 the
// bigger buffer wins by 5-7% (B4c; the same measurement is why a shorter warp step does not pay).
//
// Math per sub-block j: a' = a/amax with amax per 256-element block, so |q*a'| <= 15 and a lane sums
// 16 products into one half2 without overflow.  sc_j and m_j (6 bit) become exact halves with the same
// magic trick, sc*sum and m*ysum and the d/dmin multiply happen in half, the subtraction of the two
// terms and the sum over blocks happen in FP32 (the two terms cancel often, half would lose bits).
//
// IQ4_XS: block_iq4_xs is only 8 byte aligned (136 = 8 mod 16), so header and qs are LDG.64.  The
// values come from kvalues_iq4nl, which no magic constant can produce, so the table is held biased by
// 128 in 4 registers and indexed with the __byte_perm trick of get_int_from_table_16 (vecdotq.cuh);
// the biased byte is exactly the mantissa of the magic form, so one more __byte_perm per pair inserts
// the constant high byte 0x64 and yields the half2.  A 16 byte run of qs is one sub-block (low nibbles
// are its elements 0..15, high nibbles 16..31) instead of two, so the two accumulators of a lane split
// the 8 words 4 + 4 and the activation order differs: layout 1 of the quantize kernel.  There is no min
// term and no dmin, and the sub-block scale is d*(ls - 32) with a 6 bit ls.  IQ4_XS is the one type
// whose int8 matvec is already close to the DRAM limit on Pascal (~400 GB/s at width 1), so this path
// only runs from width 4 up for it (a16k_ncols_min).
//
// Q6_K: block_q6_K is only 2 byte aligned (210 = 2 mod 4), so a 32 bit word of the block is two 4 byte
// aligned loads and one funnel shift; the shift is the same for every block of a row and is computed
// once per lane.  Lane g takes h = g>>1 (the 128 element half) and j = g&1 (16 of the 32 l values),
// which is the group ql[l], ql[l+32], qh[l] of the format, so the four lanes read ql and qh exactly
// once between them.  A lane then owns 4 sub-blocks of 16 elements, hence 4 accumulators and 2 rows
// per block at every width.  The 2 qh bits of an element sit next to each other, so one rotate of the
// qh word serves both the low and the high nibble half of a byte pair: 11 instructions per 8 weights.

#include "mmvq-k-f16-sm60.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define A16K_QK     256 // elements per block
#define A16K_LPB      4 // lanes per block
#define A16K_BPW      8 // blocks per warp step
#define A16K_QBLOCK 128 // quantize kernel block size, one block per warp

// Per-type block layout: offset of qs, whether there is a qh plane, whether the values come from
// kvalues_iq4nl instead of the nibble itself, whether there is a min term.  a16k_layout and
// a16k_align are the host side of the same table.
template <ggml_type type> struct a16k_traits;

template <> struct a16k_traits<GGML_TYPE_Q4_K> {
    typedef block_q4_K block_t;
    static constexpr int  qs_off  = 16;
    static constexpr bool has_qh  = false;
    static constexpr bool table   = false;
    static constexpr bool has_min = true;
    static constexpr bool q6      = false;
};

template <> struct a16k_traits<GGML_TYPE_Q5_K> {
    typedef block_q5_K block_t;
    static constexpr int  qs_off  = 48;
    static constexpr bool has_qh  = true;
    static constexpr bool table   = false;
    static constexpr bool has_min = true;
    static constexpr bool q6      = false;
};

template <> struct a16k_traits<GGML_TYPE_IQ4_XS> {
    typedef block_iq4_xs block_t;
    static constexpr int  qs_off  = 8;
    static constexpr bool has_qh  = false;
    static constexpr bool table   = true;
    static constexpr bool has_min = false;
    static constexpr bool q6      = false;
};

// Q6_K: ql, qh and the int8 scales instead of a header, and only 2 byte alignment (210 = 2 mod 16).
template <> struct a16k_traits<GGML_TYPE_Q6_K> {
    typedef block_q6_K block_t;
    static constexpr int  qs_off  = 0;
    static constexpr bool has_qh  = false;
    static constexpr bool table   = false;
    static constexpr bool has_min = false;
    static constexpr bool q6      = true;
};

static int a16k_layout(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS: return 1;
        case GGML_TYPE_Q6_K:   return 2;
        default:               return 0;
    }
}

// Widths at which the HFMA2 path is not faster than the int8 one stay on the int8 path.  IQ4_XS is
// the only type that loses: its int8 matvec already reaches ~400 GB/s at width 1 (78 us on
// 4096x14336 against 88 for the best HFMA2 config), and in the model it costs 4.8% tg at width 1.
static int a16k_ncols_min(const ggml_type type) {
    return type == GGML_TYPE_IQ4_XS ? 4 : 1;
}

// B4c: widths at which the header is loaded inside its own step (pf mode 2) instead of riding in the
// prefetch buffer.  Measured per type on the model shapes and on 4096x14336 against the B4b default
// of that width, which is mode 1 unless noted: Q4_K width 2 +8%, width 4 +3% (B4b runs mode 0 there);
// Q5_K width 2 +5%; Q6_K width 2 +3%, width 3 +3%; IQ4_XS width 3 +3%, width 4 +5%.  The widths left
// out lose: Q4_K and Q5_K at width 3 and Q5_K at width 4 are faster with no prefetch at all, IQ4_XS
// at width 2 loses 6% and Q6_K at width 4 is a wash.
// Width 1 is in the mask only for the types that win there with a short k range (the k condition is
// in launch_a16k), which is the one with the largest register block: Q5_K (125 registers, 16 resident
// warps) gains 2-5% at every k=5120 shape and at both row counts a shape can have, split over two
// GPUs or not, and IQ4_XS the same 3-8% (it runs the int8 path at width 1 anyway).  Q4_K's block is
// small enough (95) that it is a wash: +3.7% at 8704 rows but -2.2% at 5120, +0.1% in the model.
// Q6_K gains at most 1% and loses 1.1% on its 124160 row LM head, which is 77% of the type's bytes.
static int a16k_pf2_widths(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:   return (1 << 2) | (1 << 4);
        case GGML_TYPE_Q5_K:   return (1 << 1) | (1 << 2);
        case GGML_TYPE_Q6_K:   return (1 << 2) | (1 << 3);
        case GGML_TYPE_IQ4_XS: return (1 << 1) | (1 << 3) | (1 << 4);
        default:               return 0;
    }
}

static int a16k_align(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS: return 8;
        case GGML_TYPE_Q6_K:   return 2;
        default:               return 16;
    }
}

static __device__ __forceinline__ half2 a16k_as_half2(const int v) {
    return *((const half2 *) &v);
}

// One warp per 256-element block of one activation row.  Lane l owns g = l/8, i = l%8 and converts
// the 8 elements that the matvec lane of the same (g, i) needs, in the pair order it reads them.
// layout 0 (Q4_K, Q5_K): elements 64g+4i+{0..3} (sub-block 2g) and 64g+32+4i+{0..3} (sub-block 2g+1).
// layout 1 (IQ4_XS): the second run is 16 elements away, not 32, and the 8 slots of a lane split 4+4
// between the two sub-blocks, so the first run walks 32*(i/4) + 4*(i%4).
// layout 2 (Q6_K): 64 elements away, and the lane owns 16 element runs of four sub-blocks.
// Output per block: 32 int4 of half2 pairs, 4 half2 of sub-block sums (ys, layout 0 only, the min
// term needs them), one float amax (yd).
// The int4 of a group of 8 blocks are interleaved, index (kb/8)*256 + i*32 + (kb%8)*4 + g, so that the
// matvec warp reads slot i of its 8 blocks as one contiguous 512 byte load.  A group of 8 is complete
// even if nblocks is not a multiple of 8 (the buffer is allocated for the padded count); a lane of the
// matvec only ever reads the slots of its own block, so the pad slots are never read.
template <bool vec4, int layout>
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

    // layout 2 (Q6_K): lane g takes the 16 element run 16*(g&1) of the half 128*(g>>1); the runs of the
    // four positions of an element group are 32 and 64 elements apart.
    const int p0   = layout == 0 ? 64*g + 4*i :
                     layout == 1 ? 64*g + 32*(i >> 2) + 4*(i & 3)
                                 : 128*(g >> 1) + 16*(g & 1) + 32*(i >> 2) + 4*(i & 3);
    const int step = layout == 0 ? 32 : layout == 1 ? 16 : 64;

    const float * xp = x + (size_t) col*s01 + kb*A16K_QK + p0;

    float v[8];
    if (vec4) {
        const float4 t0 = *((const float4 *) xp);
        const float4 t1 = *((const float4 *) (xp + step));
        v[0] = t0.x; v[1] = t0.y; v[2] = t0.z; v[3] = t0.w;
        v[4] = t1.x; v[5] = t1.y; v[6] = t1.z; v[7] = t1.w;
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k]     = xp[k];
            v[4 + k] = xp[step + k];
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
    if (layout == 0) {
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s0 += __shfl_xor_sync(0xffffffff, s0, off, WARP_SIZE);
            s1 += __shfl_xor_sync(0xffffffff, s1, off, WARP_SIZE);
        }
    }

    int4 out;
    half2 * o = (half2 *) &out;
    o[0] = __halves2half2(h[0], h[2]);
    o[1] = __halves2half2(h[1], h[3]);
    o[2] = __halves2half2(h[4], h[6]);
    o[3] = __halves2half2(h[5], h[7]);
    const int nb_pad = ((nblocks + A16K_BPW - 1)/A16K_BPW)*A16K_BPW;
    aq[(size_t) col*nb_pad*32 + (kb/A16K_BPW)*(A16K_BPW*32) + i*WARP_SIZE + (kb%A16K_BPW)*A16K_LPB + g] = out;

    if (layout == 0 && i == 0) {
        ys[(size_t) col*nblocks*4 + kb*4 + g] = __floats2half2_rn(s0, s1);
    }
    if (lane == 0) {
        yd[(size_t) col*nblocks + kb] = amax;
    }
}

// Header, qh (Q5_K) and the lane's 32 bytes of qs of block kb of every row: 3 (Q4_K) or 5 (Q5_K)
// x LDG.128 per row.
// Q6_K only: a block starts at 0 or 2 mod 4, so a 32 bit word at block offset X comes from two 4 byte
// aligned loads.  The offset is the same for every block of a row (210*8 is a multiple of 16), so sh8
// is a per row constant and there is no divergence.
#define A16K_Q6_W(q, X, sh8) __funnelshift_r((q)[(X) >> 2], (q)[((X) >> 2) + 1], (sh8))

// do_hdr and do_qs select the two halves of a block's load.  They are separate because the header
// (the 6 bit scales and mins, or the int8 scales and d of Q6_K) is only read after the 8 word
// accumulation loop of its own step, so it does not have to travel through the prefetch buffer:
// pf mode 2 loads it at the top of its own step, which is 4 registers per row less than mode 1.
template <ggml_type type, int nrows, int nqh, bool do_hdr, bool do_qs>
static __device__ __forceinline__ void a16k_load_x(
        const char * const (&xrow)[nrows], const int koff, const int goff,
        int4 (&hdr)[nrows], int (&qh)[nrows][nqh], int (&qw)[nrows][8], const int (&sh8)[nrows], const int gq6) {
#pragma unroll
    for (int r = 0; r < nrows; ++r) {
        const char * p = xrow[r] + koff;
        if constexpr (a16k_traits<type>::q6) {
            if constexpr (do_qs) {
                // gq6 packs the lane's block offsets: run A of ql, run B, the qh run and the scale byte.
                const int * q = (const int *) ((uintptr_t) p & ~(uintptr_t) 3);
                const int rA = (gq6 >> 0) & 0xff, rB = rA + 32, rC = (gq6 >> 8) & 0xff;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    qw[r][i]     = A16K_Q6_W(q, rA + 4*i, sh8[r]);
                    qw[r][4 + i] = A16K_Q6_W(q, rB + 4*i, sh8[r]);
                    qh[r][i]     = A16K_Q6_W(q, rC + 4*i, sh8[r]);
                }
            }
            if constexpr (do_hdr) {
                // 4 int8 scales at stride 2 and the fp16 d: byte and short loads, both always in bounds.
                // Kept as halves in the order of the accumulators: (m0, m2) and (m1, m3).
                const signed char    * sc = (const signed char    *) (p + 192 + ((gq6 >> 16) & 0xff));
                const unsigned short * dp = (const unsigned short *) (p + 208);
                hdr[r].x = (int) __half_as_ushort(__int2half_rn(sc[0])) | (((int) __half_as_ushort(__int2half_rn(sc[4]))) << 16);
                hdr[r].y = (int) __half_as_ushort(__int2half_rn(sc[2])) | (((int) __half_as_ushort(__int2half_rn(sc[6]))) << 16);
                hdr[r].z = (int) dp[0];
            }
            continue;
        }
        if constexpr (a16k_traits<type>::table) {
            // block_iq4_xs is 8 byte aligned only: header (d, scales_h, scales_l) is one LDG.64 and
            // the lane's 32 bytes of qs are four more.
            if constexpr (do_hdr) {
                const int2 h = *((const int2 *) p);
                hdr[r].x = h.x;
                hdr[r].y = h.y;
            }
            if constexpr (do_qs) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int2 q = *((const int2 *) (p + goff + 8*j));
                    qw[r][2*j + 0] = q.x;
                    qw[r][2*j + 1] = q.y;
                }
            }
            continue;
        }
        if constexpr (do_hdr) {
            hdr[r] = *((const int4 *) p);
        }
        if constexpr (do_qs) {
            if constexpr (a16k_traits<type>::has_qh) {
                const int4 h0 = *((const int4 *) (p + 16));
                const int4 h1 = *((const int4 *) (p + 32));
                qh[r][0] = h0.x; qh[r][1] = h0.y; qh[r][2] = h0.z; qh[r][3] = h0.w;
                qh[r][4] = h1.x; qh[r][5] = h1.y; qh[r][6] = h1.z; qh[r][7] = h1.w;
            }
            const int4 q0 = *((const int4 *) (p + goff));
            const int4 q1 = *((const int4 *) (p + goff + 16));
            qw[r][0] = q0.x; qw[r][1] = q0.y; qw[r][2] = q0.z; qw[r][3] = q0.w;
            qw[r][4] = q1.x; qw[r][5] = q1.y; qw[r][6] = q1.z; qw[r][7] = q1.w;
        }
    }
}

// IQ4_XS: one qs word -> 4 half2 in magic form, pairs (4i, 4i+2) and (4i+1, 4i+3) of the low nibble
// half of the sub-block, then the same two pairs of its high nibble half.
// tab holds kvalues_iq4nl biased by 128, so a table byte is the mantissa of the magic form and the
// high byte 0x64 is a constant; the 4 bit index is resolved as in get_int_from_table_16 (vecdotq.cuh).
static __device__ __forceinline__ void a16k_expand_iq4(const int w, const int (&tab)[4], int (&t)[4]) {
    const int sel = 0x32103210 | ((w & 0x88888888) >> 1);
    int tmp[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int lo = __byte_perm(tab[0], tab[1], w >> (16*j));
        const int hi = __byte_perm(tab[2], tab[3], w >> (16*j));
        tmp[j] = __byte_perm(lo, hi, sel >> (16*j));
    }
    const int x = __byte_perm(tmp[0], tmp[1], 0x6420); // values of the low nibbles,  bytes 0..3
    const int y = __byte_perm(tmp[0], tmp[1], 0x7531); // values of the high nibbles, bytes 0..3
    t[0] = __byte_perm(x, 0x64006400, 0x5250);
    t[1] = __byte_perm(x, 0x64006400, 0x5351);
    t[2] = __byte_perm(y, 0x64006400, 0x5250);
    t[3] = __byte_perm(y, 0x64006400, 0x5351);
}

// Q6_K: one ql word plus the qh word -> 4 half2 in magic form.  run 0 is the ql run at block offset
// 64h+16j (positions m0 and m2 of the element group), run 1 the one 32 bytes later (m1 and m3).  The
// 2 qh bits of a position sit next to each other, so one rotate of the qh word serves both the low
// nibble half (target mantissa bits 4,5) and the high nibble half (bits 8,9) of the same byte pair.
template <int run>
static __device__ __forceinline__ void a16k_expand_q6(const int w, const int hw, int (&t)[4]) {
    constexpr int r0 = run == 0 ? 28 : 30; // byte pair (0, 2) of the word
    constexpr int r1 = run == 0 ?  4 :  6; // byte pair (1, 3)
    const int ws = w >> 8;
    const int h0 = __funnelshift_r(hw, hw, r0);
    const int h1 = __funnelshift_r(hw, hw, r1);
    int c0, c1, c2, c3;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c0) : "r"(h0), "n"(0x00300030), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c1) : "r"(h1), "n"(0x00300030), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c2) : "r"(h0), "n"(0x03000300), "n"(0x54005400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c3) : "r"(h1), "n"(0x03000300), "n"(0x54005400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[0]) : "r"(w),  "n"(0x000f000f), "r"(c0));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[1]) : "r"(ws), "n"(0x000f000f), "r"(c1));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[2]) : "r"(w),  "n"(0x00f000f0), "r"(c2));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[3]) : "r"(ws), "n"(0x00f000f0), "r"(c3));
}

// One qs word -> 4 half2 in magic form: sub-block 2g pairs (4i, 4i+2) and (4i+1, 4i+3), then the same
// two pairs of sub-block 2g+1.  rot holds the per-lane rotate amounts of the qh word (Q5_K only).
template <ggml_type type>
static __device__ __forceinline__ void a16k_expand(
        const int w, const int hw, const int (&rot)[4], int (&t)[4]) {
    const int ws = w >> 8;
    if constexpr (a16k_traits<type>::has_qh) {
        const int h0 = __funnelshift_r(hw, hw, rot[0]);
        const int h1 = __funnelshift_r(hw, hw, rot[1]);
        const int h2 = __funnelshift_r(hw, hw, rot[2]);
        const int h3 = __funnelshift_r(hw, hw, rot[3]);
        int c0, c1, c2, c3;
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c0) : "r"(h0), "n"(0x00100010), "n"(0x64006400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c1) : "r"(h1), "n"(0x00100010), "n"(0x64006400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c2) : "r"(h2), "n"(0x01000100), "n"(0x54005400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(c3) : "r"(h3), "n"(0x01000100), "n"(0x54005400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[0]) : "r"(w),  "n"(0x000f000f), "r"(c0));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[1]) : "r"(ws), "n"(0x000f000f), "r"(c1));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[2]) : "r"(w),  "n"(0x00f000f0), "r"(c2));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[3]) : "r"(ws), "n"(0x00f000f0), "r"(c3));
    } else {
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[0]) : "r"(w),  "n"(0x000f000f), "n"(0x64006400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[1]) : "r"(ws), "n"(0x000f000f), "n"(0x64006400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[2]) : "r"(w),  "n"(0x00f000f0), "n"(0x54005400));
        asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[3]) : "r"(ws), "n"(0x00f000f0), "n"(0x54005400));
    }
}

// pfmode 0: no prefetch, the whole block is loaded at the top of its own step.
// pfmode 1: the next step's header, qh and qs are prefetched into registers (B4/B4b).
// pfmode 2: the same, except that the header is not prefetched but loaded at the top of its own step
//           and read after the accumulation loop.  That is 4 registers per row less than mode 1, so
//           the wider register blocks get resident warps back (Q4_K width 2: 166 against 168, Q5_K
//           width 1: 122 against 125), at the price of a load latency that only the 8 words of the
//           step itself cover.
template <ggml_type type, int ncols_dst, int nrows, int nwarps, int pfmode>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps)
mul_mat_vec_k_a16(
        const void * __restrict__ vx, const int4 * __restrict__ aq, const half2 * __restrict__ ys,
        const float * __restrict__ yd, float * __restrict__ dst, const int nblocks, const int nrows_x,
        const int stride_row_x, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    typedef typename a16k_traits<type>::block_t block_t;
    constexpr bool q6t  = a16k_traits<type>::q6;
    constexpr int  nqh  = a16k_traits<type>::has_qh ? 8 : (q6t ? 4 : 1);
    constexpr int  nacc = q6t ? 4 : 2; // Q6_K: 4 sub-blocks per lane, the others 2

    const int lane = threadIdx.x;
    const int b    = lane / A16K_LPB;
    const int g    = lane % A16K_LPB;
    const int row0 = blockIdx.x*nrows;

    const int stride_col_aq = ((nblocks + A16K_BPW - 1)/A16K_BPW)*(A16K_BPW*32);
    const int stride_col_ys = nblocks*4;

    const half2 magic_lo = a16k_as_half2(0x64006400);
    const half2 magic_hi = a16k_as_half2(0x54005400);

    // kvalues_iq4nl biased by 128, four bytes per register (IQ4_XS only).
    int tab[4] = { 0, 0, 0, 0 };
    if constexpr (a16k_traits<type>::table) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            tab[k] = ((const int *) kvalues_iq4nl)[k] ^ 0x80808080;
        }
    }

    // Rows past the end read the last row and are not stored.
    const char * xrow[nrows];
#pragma unroll
    for (int r = 0; r < nrows; ++r) {
        const int row = min(row0 + r, nrows_x - 1);
        xrow[r] = (const char *) vx + (size_t) row*stride_row_x*sizeof(block_t);
    }
    const int goff = a16k_traits<type>::qs_off + 32*g;

    // Q6_K lane offsets: h = g>>1 picks the 128 element half, j = g&1 the 16 element run.
    // Packed as run A | qh run << 8 | scale byte << 16 to keep the load helper's signature short.
    const int gq6 = q6t ? ((64*(g >> 1) + 16*(g & 1)) | ((128 + 32*(g >> 1) + 16*(g & 1)) << 8) |
                           ((8*(g >> 1) + (g & 1)) << 16)) : 0;

    // Q6_K: bit offset of the block inside its 4 byte aligned word, constant over the k loop.
    int sh8[nrows];
#pragma unroll
    for (int r = 0; r < nrows; ++r) {
        sh8[r] = q6t ? 8*((int) (((uintptr_t) xrow[r] + (size_t) (threadIdx.y*A16K_BPW + b)*sizeof(block_t)) & 3)) : 0;
    }

    // Rotates that put qh bit 2g of the pair into mantissa bit 4 (low nibble, one for each of the byte
    // pairs (0, 2) and (1, 3) of the word) and bit 2g+1 into mantissa bit 8 (high nibble).
    int rot[4];
    rot[0] = (2*g + 28) & 31;
    rot[1] = (2*g +  4) & 31;
    rot[2] = (2*g + 25) & 31;
    rot[3] = (2*g +  1) & 31;

    float sumf[ncols_dst][nrows] = {{0.0f}};

    const int kb0   = threadIdx.y*A16K_BPW + b;
    const int kstep = A16K_BPW*nwarps;

    constexpr bool pf_hdr = pfmode == 1; // the header travels through the prefetch buffer

    int4 hdr[nrows], hdr_next[nrows];
    int  qh[nrows][nqh] = {{0}}, qh_next[nrows][nqh] = {{0}}; // unused for a type without a qh plane
    int  qw[nrows][8], qw_next[nrows][8];
    if constexpr (pfmode != 0) {
        if (kb0 < nblocks) {
            a16k_load_x<type, nrows, nqh, pf_hdr, true>(xrow, kb0*(int) sizeof(block_t), goff, hdr, qh, qw, sh8, gq6);
        }
    }

    for (int kb = kb0; kb < nblocks; kb += kstep) {
        const bool more = kb + kstep < nblocks;
        if constexpr (pfmode != 0) {
            if constexpr (pfmode == 2) {
                a16k_load_x<type, nrows, nqh, true, false>(xrow, kb*(int) sizeof(block_t), goff, hdr, qh, qw, sh8, gq6);
            }
            if (more) {
                a16k_load_x<type, nrows, nqh, pf_hdr, true>(xrow, (kb + kstep)*(int) sizeof(block_t), goff, hdr_next, qh_next, qw_next, sh8, gq6);
            }
        } else {
            a16k_load_x<type, nrows, nqh, true, true>(xrow, kb*(int) sizeof(block_t), goff, hdr, qh, qw, sh8, gq6);
        }

        half2 Y[ncols_dst];
        float D[ncols_dst];
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
            if constexpr (a16k_traits<type>::has_min) {
                Y[c] = ys[c*stride_col_ys + kb*4 + g];
            }
            D[c] = yd[c*nblocks + kb];
        }

        // acc[0] and acc[1] are the two sub-blocks of the lane.  Q4_K and Q5_K put the low nibbles of
        // every word in acc[0] and the high nibbles in acc[1]; IQ4_XS has both halves of one sub-block
        // in the same word, so the first four words go to acc[0] and the last four to acc[1].
        half2 acc[nacc][nrows][ncols_dst];
#pragma unroll
        for (int u = 0; u < nacc; ++u) {
#pragma unroll
            for (int r = 0; r < nrows; ++r) {
#pragma unroll
                for (int c = 0; c < ncols_dst; ++c) {
                    acc[u][r][c] = make_half2(0.0f, 0.0f);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            constexpr bool tab_t = a16k_traits<type>::table;
            const int sa = tab_t ? (i >> 2) : (q6t ? 2*(i >> 2) : 0);
            const int sb = sa + (tab_t ? 0 : 1);

            int4 a[ncols_dst];
#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                a[c] = aq[(size_t) c*stride_col_aq + (kb - b)*32 + i*WARP_SIZE + lane];
            }
#pragma unroll
            for (int r = 0; r < nrows; ++r) {
                int t[4];
                half2 w0, w1, w2, w3;
                if constexpr (q6t) {
                    const half2 magic_q6l = a16k_as_half2(0x64206420); // 1024 + 32
                    const half2 magic_q6h = a16k_as_half2(0x56005600); //   64 + 32
                    if (i < 4) {
                        a16k_expand_q6<0>(qw[r][i], qh[r][i & 3], t);
                    } else {
                        a16k_expand_q6<1>(qw[r][i], qh[r][i & 3], t);
                    }
                    w0 = __hsub2(a16k_as_half2(t[0]), magic_q6l);
                    w1 = __hsub2(a16k_as_half2(t[1]), magic_q6l);
                    w2 = __hsub2(a16k_as_half2(t[2]), magic_q6h);
                    w3 = __hsub2(a16k_as_half2(t[3]), magic_q6h);
                } else if constexpr (tab_t) {
                    const half2 magic_tab = a16k_as_half2(0x64806480); // table bytes are biased by 128
                    a16k_expand_iq4(qw[r][i], tab, t);
                    w0 = __hsub2(a16k_as_half2(t[0]), magic_tab);
                    w1 = __hsub2(a16k_as_half2(t[1]), magic_tab);
                    w2 = __hsub2(a16k_as_half2(t[2]), magic_tab);
                    w3 = __hsub2(a16k_as_half2(t[3]), magic_tab);
                } else {
                    a16k_expand<type>(qw[r][i], qh[r][a16k_traits<type>::has_qh ? i : 0], rot, t);
                    w0 = __hsub2(a16k_as_half2(t[0]), magic_lo); // sub-block 2g,   (4i, 4i+2)
                    w1 = __hsub2(a16k_as_half2(t[1]), magic_lo); // sub-block 2g,   (4i+1, 4i+3)
                    w2 = __hsub2(a16k_as_half2(t[2]), magic_hi); // sub-block 2g+1, (4i, 4i+2)
                    w3 = __hsub2(a16k_as_half2(t[3]), magic_hi); // sub-block 2g+1, (4i+1, 4i+3)
                }
#pragma unroll
                for (int c = 0; c < ncols_dst; ++c) {
                    acc[sa][r][c] = __hfma2(w0, a16k_as_half2(a[c].x), acc[sa][r][c]);
                    acc[sa][r][c] = __hfma2(w1, a16k_as_half2(a[c].y), acc[sa][r][c]);
                    acc[sb][r][c] = __hfma2(w2, a16k_as_half2(a[c].z), acc[sb][r][c]);
                    acc[sb][r][c] = __hfma2(w3, a16k_as_half2(a[c].w), acc[sb][r][c]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < nrows; ++r) {
            if constexpr (q6t) {
                // one int8 scale per sub-block, no min term; the -32 is already in the magic constant
                const half2 S01 = a16k_as_half2(hdr[r].x); // (m0, m2)
                const half2 S23 = a16k_as_half2(hdr[r].y); // (m1, m3)
                const half2 S0 = __low2half2(S01),  S1 = __high2half2(S01);
                const half2 S2 = __low2half2(S23),  S3 = __high2half2(S23);
                const half  d  = __ushort_as_half((unsigned short) hdr[r].z);
#pragma unroll
                for (int c = 0; c < ncols_dst; ++c) {
                    half2 T = __hmul2(acc[0][r][c], S0);
                    T = __hfma2(acc[1][r][c], S1, T);
                    T = __hfma2(acc[2][r][c], S2, T);
                    T = __hfma2(acc[3][r][c], S3, T);
                    const half hs = __hadd(__low2half(T), __high2half(T));
                    sumf[c][r] += D[c]*__half2float(__hmul(hs, d));
                }
                continue;
            }
            if constexpr (a16k_traits<type>::table) {
                // IQ4_XS: scale of sub-block j is d*(ls - 32) with ls 6 bit, 4 low bits in scales_l
                // byte g and the 2 high bits in scales_h.  No min term.
                const unsigned int sh2 = ((unsigned int) hdr[r].x) >> 16;
                const unsigned int sl  = ((unsigned int) hdr[r].y) >> (8*g);
                const unsigned int l0  = ( sl       & 0x0f) | (((sh2 >> (4*g    )) & 3) << 4);
                const unsigned int l1  = ((sl >> 4) & 0x0f) | (((sh2 >> (4*g + 2)) & 3) << 4);
                const unsigned int ls  = l0 | (l1 << 8);

                const half2 magic_ls = a16k_as_half2(0x64206420); // 1024 + 32, so the result is ls - 32
                const half2 S0 = __hsub2(a16k_as_half2(__byte_perm(ls, 0, 0x4040) | 0x64006400), magic_ls);
                const half2 S1 = __hsub2(a16k_as_half2(__byte_perm(ls, 0, 0x4141) | 0x64006400), magic_ls);
                const half  d  = __ushort_as_half((unsigned short) (((unsigned int) hdr[r].x) & 0xffff));

#pragma unroll
                for (int c = 0; c < ncols_dst; ++c) {
                    const half2 T  = __hfma2(acc[1][r][c], S1, __hmul2(acc[0][r][c], S0));
                    const half  hs = __hadd(__low2half(T), __high2half(T));
                    sumf[c][r] += D[c]*__half2float(__hmul(hs, d));
                }
                continue;
            }
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
                const half2 T  = __hfma2(acc[1][r][c], S1, __hmul2(acc[0][r][c], S0));
                const half2 U  = __hmul2(Mm, Y[c]);
                const half  hs = __hadd(__low2half(T), __high2half(T));
                const half  hm = __hadd(__low2half(U), __high2half(U));
                const float2 f = __half22float2(__hmul2(__halves2half2(hs, hm), dm));
                sumf[c][r] += D[c]*(f.x - f.y);
            }
        }

        if constexpr (pfmode != 0) {
            if (more) {
#pragma unroll
                for (int r = 0; r < nrows; ++r) {
                    if constexpr (pf_hdr) {
                        hdr[r] = hdr_next[r];
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        qw[r][i] = qw_next[r][i];
                    }
                    if constexpr (a16k_traits<type>::has_qh || q6t) {
#pragma unroll
                        for (int i = 0; i < nqh; ++i) {
                            qh[r][i] = qh_next[r][i];
                        }
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

// GGML_CUDA_DISABLE_MMVQ_F16_K_TYPES is a comma separated list of ggml_type_name values, so one type
// can go back to the int8 path without a rebuild.
static bool a16k_type_disabled(const ggml_type type) {
    static const char * list = getenv("GGML_CUDA_DISABLE_MMVQ_F16_K_TYPES");
    if (list == nullptr) {
        return false;
    }
    const char * name = ggml_type_name(type);
    const size_t len  = strlen(name);
    for (const char * p = list; *p != '\0'; ) {
        const char * end = strchr(p, ',');
        const size_t n   = end ? (size_t) (end - p) : strlen(p);
        if (n == len && strncmp(p, name, n) == 0) {
            return true;
        }
        if (!end) {
            break;
        }
        p = end + 1;
    }
    return false;
}

static bool a16k_type_supported(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS ||
           type == GGML_TYPE_Q6_K;
}

bool ggml_cuda_mmvq_k_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
        const void * fusion) {
    static const bool disabled = a16k_env_int("GGML_CUDA_DISABLE_MMVQ_F16_K", 0) != 0;
    if (disabled || fusion || ids) {
        return false;
    }
    if (!a16k_type_supported(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (a16k_type_disabled(src0->type)) {
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
    const int align = a16k_align(src0->type);
    if (src0->ne[0] % A16K_QK != 0 || ((uintptr_t) src0->data) % align != 0 || src0->nb[1] % align != 0) {
        return false;
    }
    static const int env_ncols_min = a16k_env_int("GGML_A16K_NCOLS_MIN", 0);
    static const int ncols_max     = a16k_env_int("GGML_A16K_NCOLS_MAX", 8);
    const int ncols_min = env_ncols_min > 0 ? env_ncols_min : a16k_ncols_min(src0->type);
    if (src1->ne[1] < ncols_min || src1->ne[1] > ncols_max || src1->ne[1] > 8) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

template <ggml_type type, int ncols_dst, int nrows, int pfmode>
static void launch_a16k_nw(
        const void * vx, const int4 * aq, const half2 * ys, const float * yd, float * dst,
        const int nblocks, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        const int nwarps, cudaStream_t stream) {
    const dim3 grid((nrows_x + nrows - 1)/nrows, 1, 1);
    switch (nwarps) {
        case 1:
            mul_mat_vec_k_a16<type, ncols_dst, nrows, 1, pfmode><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
            break;
        case 2:
            mul_mat_vec_k_a16<type, ncols_dst, nrows, 2, pfmode><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
            break;
        default:
            mul_mat_vec_k_a16<type, ncols_dst, nrows, 4, pfmode><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst);
            break;
    }
}

// Only widths that can win with a prefetch get the prefetching instantiations.
template <ggml_type type, int ncols_dst, int nrows>
static void launch_a16k_pf(
        const void * vx, const int4 * aq, const half2 * ys, const float * yd, float * dst,
        const int nblocks, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        const int nwarps, const int pfmode, cudaStream_t stream) {
    if constexpr (ncols_dst <= 4) {
        if (pfmode == 2) {
            launch_a16k_nw<type, ncols_dst, nrows, 2>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, stream);
            return;
        }
        if (pfmode == 1) {
            launch_a16k_nw<type, ncols_dst, nrows, 1>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, stream);
            return;
        }
    }
    launch_a16k_nw<type, ncols_dst, nrows, 0>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, stream);
}

// Rows per block and prefetch, measured per type on 4096x14336 (test-backend-ops perf, both knobs
// swept against each other at every width).  Q4_K: 2 rows at width 1, 4 above; at width 1 the
// prefetched step is the largest register block (95 registers at 2 rows against 148 at 4, i.e. 21
// resident warps against 13, 70 us against 81), from width 2 up the activation loads of a step are
// shared by all rows of the block and the wider block pays for itself even where it spills a little
// (width 8: 224 us against 307 at 2 rows).  Q5_K adds the qh registers, so width 8 wants 2 rows
// (292 us against 316).  Q6_K holds 4 sub-block accumulators per lane, so 4 rows spill at width 4 with
// the prefetch and at width 8: 2 rows there (215 against 235, 342 against 377).  The prefetch pays to
// width 2 for Q4_K and Q5_K and to width 4 for IQ4_XS and Q6_K, whose per-row register blocks are
// smaller (IQ4_XS 148 us against 154 at width 4, Q6_K 215 against 302).
// Warps per block: 1 unless the grid is too small to fill the SMs.  2 warps look 6-7% better for Q6_K
// at widths 1-3 on the 56 block test shape but lose 0.4% tg in the model, where k = 5120 leaves only
// 20 blocks to split between them.  All three overridable for tuning.
// The prefetch mode (B4c) is chosen per type and width by a16k_pf2_widths, plus the k range at
// width 1; rows and nwarps are unchanged by B4c, a sweep of both at k = 5120 and 6144 found nothing
// above 3% and nothing that held for two shapes of the same type.
template <ggml_type type, int ncols_dst>
static void launch_a16k(
        const void * vx, const int4 * aq, const half2 * ys, const float * yd, float * dst,
        const int nblocks, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        cudaStream_t stream) {
    static const int env_rows   = a16k_env_int("GGML_A16K_ROWS", 0);
    static const int env_nwarps = a16k_env_int("GGML_A16K_NWARPS", 0);
    static const int env_pf     = a16k_env_int("GGML_A16K_PF", -1);
    static const int small_grid = a16k_env_int("GGML_A16K_SMALL_GRID", 128);
    static const int shortk     = a16k_env_int("GGML_A16K_SHORTK", 1);
    static const int shortk_nb  = a16k_env_int("GGML_A16K_SHORTK_NB", 24);
    static const int log_level  = a16k_env_int("GGML_A16K_LOG", 0);

    constexpr bool wide_pf = type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q6_K;
    int pfmode = ncols_dst <= (wide_pf ? 4 : 2) ? 1 : 0;

    int rows = ncols_dst == 1 ? 2 : 4;
    if (type == GGML_TYPE_Q5_K && ncols_dst >= 8) {
        rows = 2;
    }
    if (type == GGML_TYPE_IQ4_XS && ncols_dst <= 3) {
        rows = 2;
    }
    if (type == GGML_TYPE_Q6_K && (ncols_dst == 4 || ncols_dst >= 8)) {
        rows = 2;
    }

    // B4c.  At width 1 the header trade is decided by the k range: pf mode 2 wins while a warp runs
    // only 2 to 3 of the 8 block steps (nblocks < 24, which is every k=5120 tensor of the model) and
    // loses 5-7% at k=14336, where the pipeline has enough steps to pay for the wider buffer.  At the
    // other widths it is the width that decides, per type (a16k_pf2_widths).
    // GGML_A16K_SHORTK=0 restores the B4b configuration exactly.
    if (shortk != 0 && ((a16k_pf2_widths(type) >> ncols_dst) & 1) != 0 &&
        (ncols_dst != 1 || nblocks < shortk_nb)) {
        pfmode = 2;
    }

    if (env_pf >= 0 && env_pf <= 2) {
        pfmode = env_pf;
    }
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
            fprintf(stderr, "a16k: type=%s ne00=%d ne01=%d ncols=%d rows=%d nwarps=%d pf=%d grid=%d\n",
                    ggml_type_name(type), nblocks*A16K_QK, nrows_x, ncols_dst, rows, nwarps, pfmode, grid);
        }
    }

    switch (rows) {
        case 2:
            launch_a16k_pf<type, ncols_dst, 2>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, pfmode, stream);
            break;
        case 8:
            if constexpr (ncols_dst <= 4) {
                launch_a16k_pf<type, ncols_dst, 8>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, pfmode, stream);
                break;
            }
            // fallthrough
        default:
            launch_a16k_pf<type, ncols_dst, 4>(vx, aq, ys, yd, dst, nblocks, nrows_x, stride_row_x, stride_col_dst, nwarps, pfmode, stream);
            break;
    }
}

void ggml_cuda_mmvq_k_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    cudaStream_t stream = ctx.stream();

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];

    const int     nblocks = ne00/A16K_QK;
    const int64_t s11     = src1->nb[1]/ggml_type_size(src1->type);
    const int     layout  = a16k_layout(src0->type); // types with a different element order do not share the cache

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
                         ctx.a16k_cache_s[1]   == s11 &&
                         ctx.a16k_cache_s[2]   == layout;
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
                ctx.a16k_cache_s[2]   = layout;
            }

            const dim3 grid((nblocks + A16K_QBLOCK/WARP_SIZE - 1)/(A16K_QBLOCK/WARP_SIZE), ne11, 1);
            const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
            float * const  yd_d = (float *) (a_d + aq_bytes + ys_bytes);
            half2 * const  ys_d = (half2 *) (a_d + aq_bytes);
            const float * x_d = (const float *) src1->data;
#define A16K_QCASE(L)                                                                                       \
            if (vec4) {                                                                                     \
                quantize_a16k<true,  L><<<grid, A16K_QBLOCK, 0, stream>>>(x_d, (int4 *) a_d, ys_d, yd_d, s11, nblocks); \
            } else {                                                                                        \
                quantize_a16k<false, L><<<grid, A16K_QBLOCK, 0, stream>>>(x_d, (int4 *) a_d, ys_d, yd_d, s11, nblocks); \
            }
            switch (layout) {
                case 0:  A16K_QCASE(0) break;
                case 1:  A16K_QCASE(1) break;
                default: A16K_QCASE(2) break;
            }
#undef A16K_QCASE
        }
    }

    const int4  * aq = (const int4  *)  a_d;
    const half2 * ys = (const half2 *) (a_d + aq_bytes);
    const float * yd = (const float *) (a_d + aq_bytes + ys_bytes);

    const int stride_row_x   = src0->nb[1]/ggml_type_size(src0->type);
    const int stride_col_dst = dst->nb[1]/ggml_type_size(dst->type);

#define A16K_CASE(T, N) case N: launch_a16k<T, N>(src0->data, aq, ys, yd, (float *) dst->data, nblocks, ne01, stride_row_x, stride_col_dst, stream); break;
#define A16K_CASES(T)                                             \
    switch (ne11) {                                               \
        A16K_CASE(T, 1) A16K_CASE(T, 2) A16K_CASE(T, 3)           \
        A16K_CASE(T, 4) A16K_CASE(T, 5) A16K_CASE(T, 6)           \
        A16K_CASE(T, 7) A16K_CASE(T, 8)                           \
        default: GGML_ABORT("unsupported ncols_dst");              \
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_K: A16K_CASES(GGML_TYPE_Q4_K) break;
        case GGML_TYPE_Q5_K: A16K_CASES(GGML_TYPE_Q5_K) break;
        case GGML_TYPE_IQ4_XS: A16K_CASES(GGML_TYPE_IQ4_XS) break;
        case GGML_TYPE_Q6_K: A16K_CASES(GGML_TYPE_Q6_K) break;
        default: GGML_ABORT("unsupported type");
    }
#undef A16K_CASES
#undef A16K_CASE
}
