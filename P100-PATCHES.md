# P100 branch

This branch is upstream `master` plus the Tesla P100 (sm_60) performance patch set and a
meta-backend fix for `-sm tensor`. It is a private fork branch, not meant for upstream PRs.

Target machine: 2x Tesla P100-PCIE-16GB, CUDA 12.x (CUDA 13 dropped Pascal), `-sm tensor`.
Models in use: Qwen3.8-27B (arch `qwen35`, Q4_K_M / Q6_K), planned Gemma-4 31B (Q4_K_M / Q4_K_XL).

## Sources

| What | Where | Generated against |
|---|---|---|
| Patches 01-30 | https://github.com/shinbunbun/llama-cpp-p100-patches (commit `acd5897`) | tag `b10133` |
| Meta backend graph reuse fix | https://gist.github.com/philpax/5144e0f9417a83e866513de0d7d900a1 | ~`b10000` (2026-07-12) |

Patch rationale, measurements and kill switches: `docs/patches.md` in the patch repo.
Every patch also carries its reasoning in the comments it adds to the source.

## Branch layout

- Base: upstream `master` at `b81c99b47` (`b10758` + 1 commit, master of 2026-09-02).
- One commit per patch, subject `p100: NN-name`, in the original numeric order.
  Order matters: several patches touch the same lines and later ones build on earlier ones.
- Commits after the patch series: the meta backend gist, this document, and local patches
  written for this machine with subject `p100x: NN-name`, numbered from 31 so they never collide
  with the upstream patch repo's 01-30 (see "Local patches" below). 09, 22 and 28 are not on the
  branch (deferred, see below); 03 and 06 were dropped (superseded upstream, see below).
- One branch per upstream base, named `p100-b<build>` (`p100-b10133`, `p100-b10630`,
  `p100-b10758`). A rebase starts a new branch and leaves the old one untouched, so every build
  stays available for recovery. This document describes `p100-b10758`.

## Status per patch (against b81c99b47)

Scope tags are from the patch repo. "Fires" says whether the patch does anything for the
models listed above; inert patches are kept to stay close to the upstream patch set.

| # | Patch | Scope | Status | Fires for our models |
|---:|---|---|---|---|
| 01 | vmad-dp4a-sm60 | sm_60 | clean | yes (all quantized matvec) |
| 02 | mmvq-rows-per-block-sm60 | pre-Turing | clean | yes |
| 04 | concat-non-cont-flat | CUDA | clean | qwen35 (delta-net) |
| 05 | mmvf-f32-pascal | pre-Turing | clean | yes (batch 2-8 F32 matvec) |
| 07 | mmvq-moe-rows-sm60 | all archs | clean | MoE only |
| 08 | mmvq-mmid-batch-sm60 | pre-Volta | clean | MoE only |
| 09 | mmvq-nwarps-small-k-sm60 | pre-Turing | **deferred** (see below) | MoE only |
| 10 | mmvq-q8-1-activation-cache | CUDA | clean | yes |
| 11 | penalties-direct | host | clean | yes (CPU sampling with penalties) |
| 12 | mmvq-f16-sm60 | sm_60 | clean | Q4_1 only, inert for K-quants |
| 13 | sampler-prefilter | host | fixed (see below) | yes (CPU sampling) |
| 14 | getrows-narrow-rows | CUDA | clean | with 15 |
| 15 | mtp-draft-vocab | model | clean | off unless `LLAMA_MTP_DRAFT_VOCAB` is set |
| 16 | cpy-fastdiv | CUDA | clean | yes |
| 17 | norm-register-cache | CUDA | clean | yes |
| 18 | fuse-sibling-nodes | CUDA | clean | yes |
| 19 | fuse-pre-add-rms-norm | CUDA | clean | yes |
| 20 | fuse-add-unary-mul | CUDA (delta-net) | clean | qwen35 |
| 21 | sched-reset-lazy | host | clean | yes |
| 22 | decode-sched-slots | host | **deferred** (see below) | speculative decoding only |
| 23 | fuse-gdn-beta-sigmoid | CUDA (delta-net) | clean | qwen35 |
| 24 | fuse-gdn-state-gather | CUDA (delta-net) | clean | qwen35 |
| 25 | gdn-gather-single-snapshot | CUDA (delta-net) | clean | qwen35 |
| 26 | cpy-fused-rows | CUDA | clean | yes |
| 27 | fuse-concat-gather | CUDA (delta-net) | clean | qwen35 |
| 28 | top-k-partial | CUDA | **deferred** (see below) | GPU sampling only |
| 29 | mmvq-iq3xxs-grid-smem | CUDA | clean | IQ3_XXS only |
| 30 | mmvq-ksigns-smem | CUDA | clean | IQ2/IQ3 only |

"clean" = applied by `git rebase` without conflict. That is not the same as compiled or
measured; see the test checklist.

The Qwen3.8-27B GGUF ships the MTP head; it is loaded and unused without speculative
decoding. 14, 15 and 22 go live the moment MTP speculative decoding is switched on, so
re-measure then (and revisit 22).

### 09 mmvq-nwarps-small-k-sm60

Conflicts with upstream #26843 (`25ae3a9b3`, "MMVQ nwarps=8 for bs=1 on DGX Spark"), which
rewrote `calc_nwarps`, `calc_launch_params` and the `ncols_dst == 1` launch path that 09 edits.
Re-implementing it means adding a warp-count override for `cc < VOLTA` on top of the new
`launch(small_k_tag, halve_iters_tag)` structure. Measured +1.29% on MoE decode, +0.12% on
dense. Deferred 2026-08-26: no MoE model in use, so nothing to gain. Revisit when one is.

### 13 sampler-prefilter

Applied without conflict but did not compile: upstream #26276 removed
`common_params_sampling::has_logit_bias()` and now builds the logit-bias sampler from
`params.logit_bias` merged with the vocab's suppress tokens. `common_sampler_prefilter_nkeep`
takes the vocab now and repeats that condition, so the prefilter stays off whenever a
logit-bias sampler is in the chain. The clone initializer also copies `pf_nkeep`.

### 22 decode-sched-slots

Upstream renamed the sampling copy helpers (`copy_tensor_async_ints/floats/candidates` ->
`copy_tensor_async_rows`, #25532). The conflict is the 4 calls in `llama_context::decode`;
resolution is `sched.get()` -> `sched_active()` on the new lines.

Deferred 2026-08-26. The patch only pays off when the decode graph shape changes every step
(MTP speculative decoding: draft graph, then a verify batch of n_draft+1), where upstream's
single cached graph is reused ~14% of the time. Without speculative decoding every decode
step has the same shape and upstream reuse already hits, so the measured +0.97% (4 slots,
+72 MiB VRAM) does not apply. It also adds four `ggml_backend_sched` instances with their own
compute buffers, which is more surface for `-sm tensor`. Revisit when MTP is switched on, and
then measure `LLAMA_DEC_SLOTS=0` vs `4` on the real workload instead of trusting the number.

### 03 topk-moe-multirow and 06 mmq-mul-mat-id-sm60 (dropped)

Both superseded upstream between b10630 and b81c99b47, so the commits were dropped on
2026-09-02:

- 03: #27621 (`41ef91f7c`) made the fused topk-moe kernel multi-row (`TOPK_MOE_ROWS_PER_BLOCK`)
  and extended the MoE glu and router fusion past one token, which is what 03 did.
- 06: #26264 (`fc35562ba`) adds the same rule to `ggml_cuda_should_use_mmq` (MMQ for `MUL_MAT_ID`
  only, on `cc >= PASCAL` without native DP4A) plus a separate MMQ tile config for non-DP4A Pascal
  (`mmq-config-pascal-older.cuh`, lower occupancy for Q2_K/Q4_K/Q5_K/Q6_K). The dense `MUL_MAT`
  path is unchanged: dequantize-to-F16 + cuBLAS above 8 columns.

Both were MoE-only and did not fire for the models above. Upstream's versions have not been
measured here.

### 23 fuse-gdn-beta-sigmoid, 24 fuse-gdn-state-gather

Both insert a fusion rule at the top of `ggml_cuda_try_fuse`, where upstream #25952
(`3466812d1`) inserted its MoE weighted-reduction rule (`node->op == GGML_OP_MUL`). Resolved by
keeping upstream's rule first and the patch rule after it; the rules match different ops, so
the order does not matter. Patch content is unchanged (`git range-diff` against the b10630
series shows context-only differences for 23, 24 and 29).

### 28 top-k-partial

Conflicts with upstream #27466 (`f8dbcd618`, "ROCm: add radix TOP_K for long rows"), which
rewrote most of `top-k.cu`; the conflict is one block covering the whole region 28 edits. Only
used with GPU (backend) sampling, which is off here, so deferred on 2026-09-02 instead of
re-implemented. Revisit if GPU sampling is switched on.

## Local patches (p100x)

Written against this branch from measurements on the 2x P100 server (2026-09-03, see
`P100-OPTIMIZATION-PLAN.md` for the plan and `~/p100-opt/P100-OPTIMIZATION-LOG.md` on the server
for the numbers). Same rules as the upstream set: one commit each, a kill switch each.

| # | Patch | Scope | Fires for our models |
|---:|---|---|---|
| 31 | server-ckpt-adopt | host (server) | yes (hybrid/recurrent models, every follow-up request) |
| 32 | mmvq-q4k-hfma2 | sm_60 (CUDA) | yes (every Q4_K matvec at decode widths 1-8) |
| 33 | mmvq-k-hfma2 | sm_60 (CUDA) | yes (Q5_K and Q6_K matvec at widths 1-8, IQ4_XS at 4-8) |
| 34 | mmvq-k-shortk | sm_60 (CUDA) | yes (prefetch mode of the same kernels per type/width and k) |
| 35 | mmvq-cols-sm60 | sm_60 (CUDA) | yes (every quantized MUL_MAT of 9-64 columns: follow-up re-decodes, speculative verify above 8) |
| 36 | fattn-tile-p100 | sm_60 (CUDA) | yes (every flash-attention launch of the model: retuned tg entry for D=256; GQA 6 packing at 1-2 Q columns above n_kv 16384) |
| 37 | server-ckpt-save | host (server) | yes (hybrid/recurrent models, every context checkpoint save on a follow-up) |
| 38 | fattn-pb-tiebreak | sm_60 (CUDA) | yes (every one-Q-column flash-attention launch; GQA 6 packing now from n_kv 2560) |
| 39 | convert-vec | CUDA (all archs, measured on sm_60) | yes (every dequant + cuBLAS matmul: pp at every ubatch, the LM head above 64 columns, the BF16 allreduce wire) |

### 31 server-ckpt-adopt

`tools/server/server-context.cpp`. For recurrent and hybrid models the server keeps context
checkpoints (copies of the recurrent state, 149.6 MiB for Qwen3.8-27B) so a later request can
roll back. Upstream (PR 20288) breaks the prompt batch at the start of the last user message and
at 4 and 4+n_ubatch tokens before the end, creating a checkpoint at each break regardless of
`--checkpoint-min-step`, and re-creates a checkpoint at the restore point after every restore.
On a 25k-token chat every 32-token follow-up therefore paid 1 restore, 3 erases, 3 x 150 MiB
checkpoint saves and 3 separate small decodes: 1302 ms prompt phase, 1.98 s wall.

The patch (a) makes `create_checkpoint` adopt the newest checkpoint when it already holds the
state at the batch start (the one just restored), instead of erasing and copying it again, and
(b) drops the forced break and the min-step bypass at the last user message; the periodic
user-message breaks and the two near-end breaks stay. Measured: prompt phase 1302 -> 732 ms,
follow-up wall 1.98 -> 1.41 s; regenerate and exact-repeat cases unchanged; editing the last
user message re-decodes 4 + the previous rendered reply extra (~10 tokens here). Not a math
change, but a follow-up is now decoded as one batch of 28 instead of 10 + 18, so greedy tokens on
follow-ups can differ from a legacy run at near-tied positions (batch-shape numerics of the
cuBLAS path). `LLAMA_SERVER_CKPT_LEGACY=1` restores upstream placement exactly.

### 32 mmvq-q4k-hfma2

`ggml/src/ggml-cuda/mmvq-q4k-f16-sm60.cu/.cuh`, a hook in `ggml_cuda_mul_mat_vec_q` next to the
patch 12 hook, and an activation cache in `common.cuh`. GP100 has no DP4A: patch 01 emulates
the int8 dot product with 4 `vmad` per 4 MACs, while `__hfma2` does 2 MACs per instruction at
full rate. Tier 0 showed tg is matvec-bound (`mul_mat_vec_q` 77% of a token at 315 GB/s of a
603 GB/s ceiling), so a Q4_K matvec on HFMA2 is the lever; patch 12 does the same for Q4_1,
which no K-quant model uses. Patch 33 renamed these files to `mmvq-k-f16-sm60.cu/.cuh` and
generalized them to Q5_K, IQ4_XS and Q6_K; the Q4_K path is unchanged.

The kernel expands the nibbles to half2 with LOP3 magic constants, accumulates one half2 per
sub-block, applies the 6-bit scales and mins in half and sums the blocks in FP32. The
activations are converted once per matvec to half scaled by the amax of each 256-element
block (plus per-sub-block sums for the min term), stored so that a warp step reads one
contiguous 512 B line per slot, and the next step's header and nibbles are prefetched into
registers at widths 1 and 2 (the register budget does not allow it above). Widths 1-8 in one
template, 2 rows per block at width 1 and 4 above; the same single-slot activation cache as
the q8_1 path, so gate and up share one conversion.

Measured on 4096x14336 (us per call, new vs the int8 path, same binary): width 1 70 vs 104
(1.49x, 471 GB/s of weights), 2 96 vs 167, 3 105 vs 232, 4 127 vs 296, 5 136 vs 389, 8 224
vs 632. Model tg on Qwen3.8-27B-UD-Q4_K_M: +2.5% (30.0 vs 29.3 t/s at d0, 28.9 vs 28.3 at
d16384). That file is a mixed quant in which Q4_K is only 17.7% of the tg kernel time (Q5_K
24.7%, IQ4_XS 19.4%, Q6_K 9.0%, Q3_K 3.5%), and the model's k=5120 shapes reach 373 GB/s
against 459 at k=14336, so the kernel is 1.21x in-model. The same treatment for Q5_K, IQ4_XS
and Q6_K is the follow-up that carries the tg gain.

Math change (half accumulation within a block). Perplexity (wiki.test.raw, `-c 2048 -b 2048`,
decoded at `-ub 1` over 4 chunks and `-ub 8` over 10 chunks so that the matvec path is what runs;
the plan's `-ub 2048` run never touches it): ub 1 new 6.2352 vs int8 path 6.2431 (-0.13%), ub 8 new 5.4006 vs 5.3942 (+0.12%); the int8
path equals the stock build to the last digit in both. Greedy (`--temp 0 --seed 1`, 256 tokens):
the int8 path is byte-identical to stock; the new path matches for the first 76 words and then
takes a different continuation at a near-tied token.
Kill switch `GGML_CUDA_DISABLE_MMVQ_F16_K=1` (restores the int8 path exactly); tuning knobs
`GGML_A16K_ROWS`, `GGML_A16K_NWARPS`, `GGML_A16K_PF`, `GGML_A16K_SMALL_GRID`,
`GGML_A16K_NCOLS_MIN`, `GGML_A16K_NCOLS_MAX`, `GGML_A16K_LOG=1`.

### 33 mmvq-k-hfma2

`ggml/src/ggml-cuda/mmvq-k-f16-sm60.cu/.cuh` (patch 32's `mmvq-q4k-f16-sm60.cu/.cuh` renamed and
generalized), the hook in `ggml_cuda_mul_mat_vec_q` and the `a16k` activation cache in `common.cuh`
(its key gains an activation layout id). Extends the HFMA2 matvec of patch 32 to the other quant
types of the UD-Q4_K_M: Q5_K (24.7% of its tg kernel time), IQ4_XS (19.4%) and Q6_K (9.0%, mostly
`output.weight`, the 1 GB LM head at k=5120). Kernel, launcher and quantizer take a `ggml_type`
template parameter with a small per-type traits struct; the Q4_K arm keeps patch 32's exact
instruction sequence and its numbers to 0.1 us.

Per type. Q5_K has the Q4_K header and qs order plus a 32 B high-bit plane: every lane loads the
plane (the L1 serves 3 of the 4 lanes of a block), one funnel-shift rotate per pair puts the bit at
mantissa bit 4 (low nibble) or 8 (high nibble) and the existing LOP3 folds it in, 8 instructions more
per 8 weights (24% of the width-1 kernel time). IQ4_XS blocks are 8 B aligned (136 B), so header and
qs are LDG.64; each 16 B run of qs is one 32-element sub-block (low nibbles 0-15, high 16-31), so the
quantizer has a second activation layout and the two accumulators split the words 4+4; the values
come from `kvalues_iq4nl`, held biased by 128 in 4 registers and resolved with the `__byte_perm`
trick of `get_int_from_table_16`, then one more `__byte_perm` inserts the 0x64 high byte (16
instructions per 8 weights against 5 for Q4_K). Q6_K blocks are 2 B aligned (210 B), so every
32-bit word is two aligned loads and a funnel shift by a per-row constant (20 load instructions per
block against 3 for Q4_K); lane g takes the format's natural group (`ql[l]`, `ql[l+32]`, `qh[l]` of
half g>>1, l in the 16-run g&1), so the four lanes read ql and qh once between them, holds 4
sub-block accumulators and applies the int8 scales in half; the -32 sits in the magic constant.

Per-type defaults from a rows x prefetch sweep at every width: prefetch to width 2 for Q4_K/Q5_K
and to width 4 for IQ4_XS/Q6_K; rows 2 at width 1 and 4 above, except Q5_K at width 8, IQ4_XS at
widths 2-3 and Q6_K at widths 4 and 8 (2 rows). Two warps per block win 6-7% for Q6_K at widths 1-3
on the 56-block test shape but lose 0.4% tg in the model (k=5120 leaves 20 blocks to split), so the
grid rule of patch 32 stays. IQ4_XS runs the new path only from width 4 up: its int8 matvec already
reaches 399 GB/s at width 1 on GP100 (78 us on 4096x14336, 70% of the measured DRAM ceiling,
against 317 GB/s for int8 Q4_K), the best HFMA2 config is 6% slower there and enabling it costs
4.8% tg in the model. Beating it needs a different kernel structure, not a better table.

Measured (test-backend-ops perf, 4096x14336, us per call, new vs the same binary with that type on
the int8 path): Q5_K width 1 96.9 vs 131.4 (1.36x, 417 GB/s of weights), 2 121.4 vs 199.4, 3 144.4
vs 264.7, 4 148.3 vs 318.8, 5 166.4 vs 412.1, 8 291.9 vs 659.7 (up to 2.48x); Q6_K width 1 147.5 vs
206.3 (1.40x, 327 GB/s), 2 164.2 vs 221.6, 3 180.8 vs 248.9, 4 215.1 vs 290.9, 5 200.6 vs 365.7, 8
342.3 vs 499.1 (up to 1.82x); IQ4_XS width 4 148.1 vs 162.4, 5 158.0 vs 259.0, 8 228.6 vs 434.5
(1.10x-1.90x). Model tg on Qwen3.8-27B-UD-Q4_K_M (`-sm tensor -fa on`, tg64, two alternating passes):
31.57 t/s at d0 and 30.28 at d16384, against 30.02 / 28.77 with patch 32 alone and 29.30 / 28.23 on
the int8 path: +5.2% over patch 32 at both depths, +7.8% / +7.3% over the int8 path. Per type at d0:
Q5_K +3.0%, Q6_K +2.1%, IQ4_XS 0 at width 1 by design. nsys over 64 tokens: the Q5_K matvec 1061 ->
931 ms (1.14x in-model, the k=5120/6144 shapes again), Q6_K 385 -> 295 ms (1.30x), total tg kernel
time 4308 ms before patch 32 -> 3983 ms (-7.5%); IQ4_XS is now the largest kernel (841 ms, 21.1%).

Math change as in 32. Perplexity (wiki.test.raw, `-c 2048 -b 2048`, `-ub 1` over 4 chunks and
`-ub 8` over 10): new 6.2364 / 5.4074, int8 path 6.2431 / 5.3942 = stock to the last digit, i.e.
-0.11% / +0.25%. Greedy (256 tokens): the kill-switch path is byte-identical to stock; the new path
diverges after the same 76 words as patch 32's Q4_K-only path. Full `test-backend-ops` 14675/14675
on CUDA0 and on CUDA1. No spills in the default configs (Q6_K width 7 with 4 rows spills 48 B and is
not in the perf set; the width 6-7 defaults are interpolated).

Kill switches `GGML_CUDA_DISABLE_MMVQ_F16_K=1` (all types, exact int8 path) and
`GGML_CUDA_DISABLE_MMVQ_F16_K_TYPES=q5_K,iq4_xs,q6_K,q4_K` (comma separated `ggml_type_name`
values, per type). Knobs as in 32; `GGML_A16K_NCOLS_MIN` now overrides the per-type minimum
(`GGML_A16K_NCOLS_MIN=1` forces IQ4_XS on at width 1), `GGML_A16K_LOG=1` prints the type.
Left for later: a layer with an IQ4_XS gate and a Q5_K up converts its activations twice at widths
4-8 (different layouts; a two-slot cache would fix it, ceiling ~1.7% of kernel time), and the
short-k gap (plan item B4c) now costs all three types.

### 34 mmvq-k-shortk

`ggml/src/ggml-cuda/mmvq-k-f16-sm60.cu` only (+141/-60). Plan item B4c: the HFMA2 K-quant matvec
ran the model's k=5120 and 6144 tensors (20 and 24 blocks) at 0.74-0.81 of its k=14336 rate.
Measured first with a new timing tool that runs the in-tree kernel at any (type, k, rows, width)
(`~/p100-opt/b4c/shape.cpp` on the server; `test-backend-ops perf` only has m=4096 k=14336): the
gap is a width-1 effect (at widths 4 and 8 the short shapes are as fast as the long ones), it is the
k loop and not the grid (a 69632-row grid at k=5120 still saturates 24% below k=14336), and a warp
pays a fixed 0.38 of an 8-block step while the streaming part already runs at 566 GB/s, within 3% of
the DRAM ceiling. So only the fixed part is addressable, and the existing knobs (rows, nwarps,
prefetch swept at k=5120/6144) gave at most 3% and never the same setting for two shapes of one type.

The change is a third prefetch mode. The header of a step (the 6-bit scales and mins, or Q6_K's
int8 scales and d) is only read after the step's accumulation loop, so it can be loaded at the top
of its own step instead of riding in the prefetch buffer (pf mode 2: 4 registers per row less than
mode 1, at the price of a load latency covered only by the step's own 8 words). Which side wins is
decided per type and width by a bitmask (`a16k_pf2_widths`: Q4_K widths 2 and 4, Q5_K 1 and 2,
Q6_K 2 and 3, IQ4_XS 3 and 4) and at width 1 by the k range (`nblocks < 24`, i.e. the k=5120
tensors; at k=14336 mode 1 wins by 5-7%). Rows and nwarps are unchanged. Width 1 stays on mode 1
for Q4_K (a wash: +3.7% at 8704 rows, -2.2% at 5120, +0.1% in the model) and Q6_K (+1% on the short
shapes, -1.3% on the LM head, which is 77% of the type's bytes). One trap worth recording:
`-sm tensor` halves the rows each GPU sees, so a rule tuned on full-row shapes can flip sign in the
model; the width-1 rule was re-measured at per-GPU row counts (`b4c/split.sh`) before it was
narrowed to Q5_K.

Measured (us per call, new vs `GGML_A16K_SHORTK=0`): at m=4096 k=14336 Q4_K width 2 88.2 vs 95.9
(+8.1%) and width 4 121.0 vs 127.0 (+4.7%), IQ4_XS width 4 142.2 vs 148.3 (+4.1%), Q6_K width 2
160.7 vs 164.2 and width 3 178.0 vs 180.7, everything else within noise; at the model's k=5120
shapes Q5_K width 1 +2.2-4.9%, Q4_K width 2 +6-8%. Model tg 31.72 vs 31.57 t/s at d0 (+0.5%) and
30.46 vs 30.30 at d16384 (+0.5%), +8.2% / +7.7% cumulative over the int8 path; speculative verify
widths (`-p 2,4,8 -n 0`) +4.0% at 2 (noisy), +2.1% at 4, unchanged at 8. nsys: Q5_K matvec 928 ->
898 ms (-3.3%), Q4_K and Q6_K unchanged by design, total tg kernel time -0.5%; the LM head is
untouched at 1.43 ms per call.

No math change: greedy output is byte-identical to `GGML_A16K_SHORTK=0`, perplexity equals patch
33's to the last digit (6.2364 / 5.4074), the kill-switch path stays byte-identical to stock. Full
`test-backend-ops` 14675/14675 on CUDA0 and CUDA1. No spills in the new instantiations.

Kill switch `GGML_A16K_SHORTK=0` (restores the patch 33 configuration exactly); knobs
`GGML_A16K_SHORTK_NB` (block count below which width 1 takes mode 2, default 24) and `GGML_A16K_PF`
now accepts 2; `GGML_A16K_LOG=1` prints `pf=`. Not done, with numbers: a half-step pipeline (4 blocks
per warp step) lost 18-26% because halving the bytes in flight per warp costs more than the shorter
pipeline drain saves, which is also the argument against the plan's 4-blocks-per-step candidate;
`__launch_bounds__` min-blocks lost 1-7% and made Q5_K spill; the short-k gap itself stands at
0.74-0.81 of the long-k rate for everything except Q5_K.

### 35 mmvq-cols-sm60

`ggml/src/ggml-cuda/mmvq.cu`, `mmvq.cuh` and `mmvq-k-f16-sm60.cu` (+142/-15). Plan item A8,
dispatch only: no kernel is changed. GP100 has no DP4A, so `ggml_cuda_should_use_mmq` refuses
every dense quantized `MUL_MAT` and everything wider than `MMVQ_MAX_BATCH_SIZE` (8) is dequantized
to F16 and multiplied by cuBLAS. Measured first at the model's per-GPU shapes with a copy of the
B4c shape tool (`~/p100-opt/a8/`): the step at 9 columns is 8.0x (Q4_K 5120x8704: 176 us at width 8,
1409 us at width 9, for 12.5% more work), and from there cuBLAS is nearly flat to 64 columns,
because what it spends is the dequantization of the whole weight matrix plus the F16 round trip and
not the GEMM. The matvec costs 8-64 us per column depending on type and shape, so a loop of column
chunks stays cheaper until the chunk count catches up; the crossover is 105-133 columns for Q4_K,
89-113 for Q5_K, 94-120 for IQ4_XS, 60-104 for Q6_K and 43-51 for the int8 path (Q3_K).

The change is a loop over column chunks in the two matvec host entries plus one branch in the gate.
`ggml_cuda_should_use_mmvq` gets an sm_60 arm (`cc == GGML_CUDA_CC_PASCAL`, 600 exactly, so sm_61
with its DP4A and MMQ is untouched) returning `ne11 <= ggml_cuda_mmvq_max_cols_sm60(type)`, a
per-type ceiling. `ggml_cuda_mul_mat_vec_q` quantizes the activations once for all `ne11` columns
as before and then calls `mul_mat_vec_q_switch_type` once per chunk with the src1 and dst column
offsets; `ggml_cuda_mmvq_k_f16_sm60` does the same around its width `switch`, offsetting the three
parts of the a16k activation cache. `MMVQ_MAX_BATCH_SIZE` itself, `get_mmvq_mmid_max_batch`, the
`ids` asserts and every other architecture are unchanged, and `mul_mat_id` never enters the loop.

The split is `ceil(ncols/7)` nearly equal chunks, not 8 columns until the remainder: the kernel's
cost per column has its minimum at width 5-7 because the width-8 configuration needs the most
registers per row, and 28 columns as 7+7+7+7 cost 552 us against 624 us as 8+8+8+4 (Q4_K
5120x8704). It has a floor, though: when that split would put a chunk below 6 columns the fewest
chunks are taken instead, because a narrow chunk costs more in the model than the per-call time of
the kernels suggests (16 columns as 8+8 spend 8% less matvec kernel time in nsys than as 6+5+5,
and 8% more pp; the per-call table has 6+5+5 0.3% ahead; the mechanism is not explained). The
floor moves only n = 15, 16, 22, 23 and 29.

Ceilings (measured, per type): Q4_K 64, Q5_K 48, Q6_K 32, IQ4_XS 64, everything else 32. Each is
the largest width at which the loop still won on every per-GPU shape of the model, cross-checked
with `llama-bench`, which is slightly more favourable to the loop than the per-call table (Q3_K at
32 is a wash per call but 5% faster than cuBLAS in the model, so 32 and not 28).

Measured (`llama-bench -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -b 2048 -ub 2048 -p <w> -n 0 -r 5`,
pp t/s, new against `GGML_CUDA_MMVQ_MAX_COLS_SM60=8`):

    width      9    12    15    16    22    23    24    28    29    32    48    64    96   128
    old    23.5  30.7  37.2  39.6  44.2  46.1  48.0  55.9  57.3  62.8  90.5 117.2 130.9 217.0
    new    90.7 110.2 107.2 112.6 116.3 114.6 119.8 128.3 121.6 121.3 129.2 131.6 130.9 218.8
    x      3.87  3.59  2.88  2.84  2.63  2.49  2.50  2.30  2.12  1.93  1.43  1.12  1.00  1.01

tg and pp2048 do not move (tg64 31.54-31.72 t/s in both arms, pp2048 420.5-421.3 in both, pp8
unchanged). Per call the loop is 3.1-7.6x at 9 columns, 2.0-3.4x at 28 and 0.7-1.5x at 64
depending on the type, Q4_K best and the int8 path worst. nsys at pp28: total GPU kernel time
3999 -> 1768 ms, the 2457 ms of `maxwell_hgemm_128x64_tn` and the 1268 ms of `dequantize_block_*`
replaced by 1329 ms of `mul_mat_vec_k_a16` and 198 ms of `mul_mat_vec_q`. End to end (the J4
harness, a 23570-token chat and 10 follow-ups, each a 32-token re-decode): prompt phase 709.7 ->
434.3 ms (1.63x), whole request 1.344 -> 1.074 s (-20%), the 16 generated tokens unchanged at 528
vs 531 ms, the first 23570-token request unchanged at 66.3 vs 66.0 s. At these widths the Q6_K
LM head no longer allocates its 1.27 GB F16 copy per call (the small-batch half of plan item A4).

The result differs numerically from cuBLAS F16 at widths 9 to the ceiling (F16 or int8 activations
instead of an F16 GEMM with F32 accumulation), the same trade patches 32-34 make at widths 1-8.
Perplexity (wiki.test.raw, `-c 2048 --chunks 10`): -ub 16 5.4074 against stock 5.4076 (-0.004%),
-ub 32 5.4069 against 5.4267 (-0.36%), -ub 64 5.4292 against 5.4247 (+0.083%); the kill switch at
-ub 32 gives 5.4267, stock to four decimals. 256 greedy tokens are byte-identical between new and
old at -ub 512 and at -ub 16, so A8 does not move the sampled tokens on that prompt at all; the
branch's divergence from stock at word 76 is patch 32/33 and disappears with
`GGML_CUDA_DISABLE_MMVQ_F16_K=1` on top of the kill switch. Full `test-backend-ops` 14675/14675 on
CUDA0 and CUDA1.

Kill switch `GGML_CUDA_MMVQ_MAX_COLS_SM60=8` (or `=0`): every type goes back to 8, the a16k width
gate is the pre-A8 one and both loops run exactly once with no offset, i.e. the launches of
27d6d7876. Values below 8 are clamped to 8. Knobs: `GGML_CUDA_MMVQ_COLS_CHUNK=N` (2-8, default 7)
sets the target chunk width and with it the floor (N-1), so 8 means "fewest chunks";
`GGML_CUDA_MMVQ_COLS_LOG=1` prints the first 32 chunked calls. Not done, with numbers: a per-type
chunk width (Q6_K and Q3_K want 8, the others 7, worth 3-17% on their share of the bytes, left as
one global rule); a ceiling above the values above, which loses at 96 and 128 columns (a uniform
1024 costs 38% of pp128); and Q4_1, whose dedicated HFMA2 kernel still stops at 8 columns, so its
wider batches take the int8 chunk loop.

### 36 fattn-tile-p100

`ggml/src/ggml-cuda/fattn-tile.cuh`, `fattn-common.cuh` and `fattn.cu` (+251/-42). Plan item C2,
with C3 and C4 folded in as measurements. The Pascal FP16 flash-attention tile table has carried a
"TODO optimize kernel parameters for FP16 NVIDIA (P100)" since it was written, and only GP100 uses
it: sm_61 takes the FP32 table (slow FP16), Volta and newer take the MMA kernel. Two things were
wrong for Qwen3.8-27B, which is D=256 with 24 q heads and 4 kv heads (GQA 6; 12 and 2 per GPU
under `-sm tensor`), not the D=128 the plan assumed. Every FLASH_ATTN_EXT of the model picks the
tile kernel (`GGML_CUDA_FATTN_LOG=1` log of every launch in `~/p100-opt/log/C2-41.log`).

First, `occupancy` in that table is only the `__launch_bounds__` min-blocks hint, i.e. the register
budget `65536/(nthreads*occupancy)`. At the tg entry (`ncols 2`, 64 threads, occupancy 2) that
budget is 512 registers per thread, above the hardware maximum of 255, so ptxas is unconstrained
and spends all 255, and exactly 4 blocks of 2 warps fit per SM: 8 of the SM's 64 warp slots. The
entry is not "occupancy 2", it is "no register limit", the least occupied entry of the table.
`(128, 4, 64, 64)` gives ptxas a real 128-register budget and two warps per KV tile: +1.7% at
n_kv 32768 and +9.6% at 131072 per launch, nothing lost at 4096. Sweeping the other parameters says
`nbatch_K` must stay 64 (every 32 loses 8-19% at depth: 64 contiguous bytes per K row per load
instead of 128, and eight KQ steps per tile instead of four) and that the `ncols 32` (prompt) entry
is already optimal: 14 candidates, none faster, the best 1% slower.

Second, and this is where the time is: with the power-of-two `ncols2` chain a GQA ratio of 6 lands
on `ncols2 = 2`, so three CUDA blocks per kv head each stream the whole K/V. Forcing `ncols2 = 1`
(six streams) costs 2.3x the time at every depth, i.e. the re-read is paid in DRAM time, so the
patch adds `ncols2 = 6`: a `(256, 256, 6)` table entry (192 threads = 6 warps, one Q column each,
occupancy 4, `nbatch_fa 64`, `nbatch_K 64`), a `(256, 256, 12)` entry for two Q columns, and a
branch before the `% 2` case that takes them. `nwarps` need not be a power of two (6 warps), but
`KQ_cs = min(cpw, 4)` must be, which rules out 2 and 4 warps for `ncols 6`. The branch is gated on
`K->ne[1] >= 16384 && K->ne[2] >= 2 && gqa_ratio % 8 != 0` and on at most 2 Q columns, all four
clauses measured: below n_kv 16384 the L2 already absorbs most of the re-read (the shipped path
runs at a nominal 855 GB/s there, above the 603 GB/s DRAM ceiling) while the coarser grid, one
output tile per kv head instead of three, costs 5-11%; with one kv head the grid is a single output
tile and it costs up to 50%; multiples of 8 pack more per block on `ncols2 = 8`; and
`cols_per_block 24` for 3-8 Q columns loses 14-17% at 4k for a 10% gain at 32k. The gate sat at 16384
in this patch because the curve was non-monotonic between 2k and 8k: the `parallel_blocks` search
ignores the ragged last KV tile (`ntiles_KV % parallel_blocks`), so at 8192 16 of 224 blocks do two KV
tiles and the wave waits for them. Patch 38 (item C2b) fixes that in `launch_fattn` and lowers the
gate to 2560.

Measured per launch (`~/p100-opt/c2/fa-shape.cpp`, one FLASH_ATTN_EXT op at the model's per-GPU
shape, new vs `GGML_CUDA_FATTN_TILE_LEGACY=1`): tg 1.12x at n_kv 16384, 1.31x at 32768, 1.46x at
65536, 1.70x at 131072 (485 us against 824, 553 GB/s of K/V read once, 92% of the card's read
ceiling); two Q columns 1.20x / 1.43x / 1.60x at 16384 / 32768 / 131072; everything else unchanged
to three digits. Model (`llama-bench -p 0 -n 64`, median of two alternating passes): tg 31.90 vs
31.70 t/s at d0 (+0.7%), 30.93 vs 30.47 at d16384 (+1.6%), 30.02 vs 29.06 at d32768 (+3.3%),
28.37 vs 26.31 at d65536 (+7.9%); pp2048 419 vs 419 t/s at d0 and 333 vs 335 at d16384
(unchanged); speculative verify widths 2, 4 and 8 unchanged. Server harness at a 23.6k context
(`tmp/j4run.sh` style): the generation part of a follow-up 517.7 vs 528.5 ms (+2.1% tg),
`prompt_ms` and the 23570-token first request unchanged. Prompt processing is untouched by design.
Extrapolated to the production `-c 160000` (about 30% attention share at 130-160k, kernel 1.70x):
about +12% tg, not measured.

Correctness: `test-backend-ops` 14675/14675 on CUDA0 and CUDA1 (and 2942/2942 of
`-o FLASH_ATTN_EXT` with and without the kill switch), but the suite has no small-batch GQA 6 case
(its only `hsk 256, nr23 {6,1}` cases are `nb = 512`, which take `ncols2 = 2` as before), so the
new path is covered by `~/p100-opt/c2/check6.sh`: 14 shapes (n_kv 256 to 65536, 1 to 4 kv heads,
GQA 4, 6 and 12, 1 and 2 Q columns) against the CPU backend, NMSE 1.2e-6 to 1.8e-4, within a few
percent of the legacy path's figure. Run it after every rebase that touches `fattn-tile.cuh`.
Perplexity (`-c 2048 --chunks 20` and `-c 32768 --chunks 4`) identical to stock to the last digit
in all arms, as expected: perplexity is prompt processing and never runs the changed paths.
Greedy 128 tokens after a 22054-token prompt (`-c 24576`, the `ncols2 = 6` kernel runs during
generation): new, old and stock byte-identical.

Kill switch `GGML_CUDA_FATTN_TILE_LEGACY=1` restores the b10758 kernels exactly (the changed
`ncols 2` entry is instantiated a second time with its old values behind a template `cfg` index,
and the `ncols2 = 6` branch is skipped); the legacy per-launch times reproduce the pre-patch
measurements to 0.1%. Knob `GGML_CUDA_FATTN_LOG=1` prints one line per distinct flash-attention
launch tuple (kernel kind, ncols1/ncols2, config, blocks per SM, parallel_blocks, grid); off by
default. Compile time of `fattn-tile-instance-dkq256-dv256.cu` 73.3 s against 65.5 s. The
`(256, 256, 6)` and `(256, 256, 12)` rows added to the FP32 and the two AMD tables exist so the
kernel compiles for those targets; they are never reached (the branch is GP100 only) and were not
measured.

Not done, with numbers: the `ncols 32` prompt entry (14 candidates, best 1% slower than shipped);
the `ncols 4/8/16` speculative-verify entries (every candidate that gains 5-16% at depth loses
1-13% at n_kv 4096, and they are unreachable for a GQA-6 model at 1-2 columns anyway);
`cols_per_block 24` for 3-8 Q columns (above); the vector kernel for single-column tg (plan item
C3: 1.4-1.8x slower than the tile kernel at every KV length, no threshold wins); forcing
`parallel_blocks` above what the search picks (plan item C4: 2x costs 6-21%, 4x up to 83%; the
search already fills exactly one wave, `ntiles_dst 6` x `pb 37` = 222 of 224 block slots).
Rebase note: all three files are upstream-churned; the table edits are one-line entries, the
launcher edit is the `launch_fattn_tile_cfg` indirection in every `cols_per_block` case of
`launch_fattn_tile_switch_ncols1`.

### 37 server-ckpt-save

`tools/server/server-context.cpp` (+49/-2). Plan item J4b. `create_checkpoint` drops at least one old
checkpoint and then appends a new one on every follow-up request, and the new one allocated its own
storage: for Qwen3.8-27B that is a fresh 149.6 MiB `std::vector` per request, 72.8 ms to allocate and
first-touch (page-fault bound at 2.2 GB/s) plus 8.5 ms to give the old mapping back to the kernel,
against 30.4 ms for the device-to-host copy that fills it. The patch moves `data_tgt` (and `data_dft`
when a draft context exists) out of every checkpoint the two eviction loops drop into a local spare
and into the new checkpoint right after `emplace_back()`, so `update_tgt` resizes a vector that
already has exactly that size, neither an allocation nor a zero-fill, and the state copy overwrites
every byte. The `created context checkpoint` trace line now also reports how long the save took.

The item's premise in the plan ("150 MiB DtoH in 200 ms vs 37 ms restore") was wrong twice, measured
first with scratch timers on the real server path (`~/p100-opt/log/J4b-notes.md` section 1): the
768 shard copies of the recurrent state take 30.5 ms at 5.14 GB/s, faster than the 36.9 ms HtoD
restore; and `llama_state_seq_get_data_ext` calls `ctx->synchronize()` before it reads anything, so
the allocation that ran before it was hiding about 66 ms of decode still in flight from the preceding
28-token prompt batch. Removing the allocation exposes that wait, which is why the gain is 16 ms and
not 80.

Measured on a 23.5k-token chat, 32-token follow-ups (`~/p100-opt/j4b/chain1.sh`, `chain2.sh`, two
passes with the arms alternated): the create window (erase to "created") 112.4 -> 96.4 ms, prompt
phase 433.3 -> 416.9 ms (-3.8%), token generation unchanged (517.5 vs 518.1 ms per 16 tokens), client
wall 1.060 -> 1.057 s (inside the +-15 ms scatter of the end-to-end measurement), RSS 2.98 GB and
locked memory 0 in every arm. The kill switch reproduces the unpatched build exactly (433.3 / 112.4 ms
against 432.8 / 112.2 ms). Edit and regenerate requests (the harness cases c, c2, b1, b2, a2) do not
improve: their old checkpoint is dropped by the "erased invalidated context checkpoint" loop in
`update_slots`, outside `create_checkpoint`, so there is nothing to recycle; covering them needs a
spare that outlives one call, i.e. a permanent 149.6 MiB per slot, not taken for 16 ms.

Correctness: in a scratch build every checkpoint was saved a second time into a fresh buffer and
compared byte for byte, 17/17 identical with the patch on and 17/17 with the kill switch on;
`prompt_n` identical across arms in every case. `test-save-load-state -c 4096 -n 64` passes on both
builds; `llama-bench` tg64 31.90 vs 31.92 t/s and pp2048 420.87 vs 420.87 t/s (control: nothing below
the server is touched). `test-backend-ops` not needed. Kill switch `LLAMA_SERVER_CKPT_SAVE_LEGACY=1`
(read once in `load_model` next to patch 31's switch, logged as a warning) restores a fresh buffer per
checkpoint.

Not done: a pinned (`ggml_backend_dev_host_buffer_type`) checkpoint buffer, which the microbenchmark
(`~/p100-opt/j4b/copy-bench.cu`) prices at 30.4 -> 16.2 ms for the save and 35.1 -> 18.1 ms for the
restore, because it needs a pinned allocator for `common_prompt_checkpoint` (with copy semantics for
`server_prompt::clone()`), a fallback when the allocation fails, and up to `--ctx-checkpoints`
(default 32 on this branch) x 149.6 MiB of locked memory per slot. Batching the copies asynchronously
is worth nothing without it (pageable DtoH is synchronous in the driver: 27.7 vs 28.3 ms) and would
need a segmented path in `ggml_backend_meta_get_tensor_async`, which asserts `n_segments == 1`; the
real tensors split into 3 (ssm) and 5 (conv) pieces per layer on two GPUs.

### 38 fattn-pb-tiebreak

`ggml/src/ggml-cuda/fattn-common.cuh` and `fattn-tile.cuh` (+58/-13). Plan item C2b, the follow-up
patch 36 asked for. The `parallel_blocks` search in `launch_fattn` (the non-stream-k path, i.e.
everything on Pascal) maximises how full a wave of blocks is and never looks at
`ntiles_KV % parallel_blocks`. Block `y` walks the KV tiles `y, y+pb, y+2pb, ...`, so a block does
`ceil(ntiles_KV/pb)` tiles and its neighbours one less, and the launch ends when the longest block
does. At the model's tg shape the search lands on `pb = 112` (`ntiles_dst 2` x 112 = 224 = one full
wave of 56 SMs x 4 blocks) for every n_kv from 8192 up, so at n_kv 8192 exactly 16 of 112 blocks per
output tile do a second KV tile: one round of 224 blocks, then a round of 32. Timing the rounds
apart (launch overhead about 15 us) gives 28 us for the 224-block round and 18 us for the 32-block
one, and a 128-block round also costs 18 us: below about 128 blocks this kernel is pure latency and
a ragged tail throws away a whole round.

The patch adds, after the existing search and only when `ncols1 == 1` (one Q column per block) on
Pascal: `rounds = ceil(ntiles_KV/parallel_blocks)`, then
`parallel_blocks = max(pb_start, ceil(ntiles_KV/rounds))`. That is by construction the smallest
`parallel_blocks` with the same number of rounds, so it never adds a round, never adds a wave, only
lowers `parallel_blocks` (less combine work), and reduces to today's choice whenever the search's
value is already the smallest with that round count: a tie-break, not a re-optimisation. Two Q
columns are excluded because the measurement says the opposite there: that kernel does twice the
arithmetic per K/V byte, is not bandwidth bound, and the concurrency of the larger `parallel_blocks`
beats the even split by 4-7% on every row. Rejected with numbers (`~/p100-opt/log/C2b-notes.md`
3.2): "then the larger wave efficiency" (changes nothing at 8192, all of 64..112 have two rounds
and 112 the best wave fill), an acceptance band around the best wave efficiency (the useful
candidate sits at 57% wave fill, no band reaches it), a thin-tail test (keeps 8192, drops 12288 and
32768).

Per launch, tg (1 Q column, D=256, 2 kv heads, GQA 6), packed `ncols2 = 6` path: 61.51 -> 52.15 us
at n_kv 8192 (1.18x), 71.99 -> 68.17 at 12288, 90.54 -> 87.54 at 16384, 146.12 -> 143.71 at 32768;
unpacked `ncols2 = 2` path 36.87 -> 34.70 at 4096 (1.06x, `pb` 37 -> 32) at the price of 0.7% at
8192 and 2.0% at 12288 (`pb` 37 -> 32 there gives up 15% of the blocks to save one sixth of a
round; the model does not reach that path above 2560 any more). Everything with more than one Q
column is unchanged to 0.3%: 2, 4, 8, 32, 512 and 2048 columns at n_kv 4096/16384/32768 all measure
1.000. A `parallel_blocks` sweep over 32..224 (`log/C2b-pbsweep.log`) says the rule picks the
measured optimum at n_kv 8192, 12288 and 32768 and is 1% off it at 16384; the predictions written
before the build matched the measurement to under 1% on every row.

With the curve monotonic the GQA 6 gate of patch 36 comes down from `K->ne[1] >= 16384` to 2560 for
one Q column (`GGML_CUDA_FATTN_GQA6_MIN_KV_DEFAULT`): the packing is 0.96x at n_kv 2304, 1.10x at
2560 and 1.06x or better at every measured point above it (1.13x at 8192, 1.21x at 12288, 1.30x at
32768), i.e. it now runs from a context of about 2.3k instead of 16k. Against patch 36 as committed
the shipped launch is 1.21x at 2560, 1.12x at 4096, 1.21x at 5120, 1.12x at 8192, 1.19x at 12288,
1.04x at 16384, 1.02x at 32768. Two Q columns keep the search's `parallel_blocks`, so their curve
still dips (0.86x at n_kv 5632, where 4 of 84 blocks do the second KV round); their gate is
`max(knob, 7680)` (`GGML_CUDA_FATTN_GQA6_MIN_KV_NCOLS2`), above which every measured point is 1.13x
or better. With 4 kv heads (a single card) the packing goes from 0.88x to 1.14x at n_kv 4096 (the
tie-break takes that launch from 67.08 to 49.69 us) and reaches 1.41x at 32768; with GQA 12 from
0.77x to 1.00x at 4096 and 1.20x at 32768. The `K->ne[2] >= 2` clause is kept: with one kv head
the packing is 0.76x at 4096 but 1.09x at 16384 and 1.28x at 65536, so it wants a threshold of its
own (measured, not applied: a model would need a single kv head per GPU).

Model level: unchanged, and that is the arithmetic. `llama-bench -p 0 -n 64 -d 0,4096,8192,16384
-r 5` gives 31.88/31.91/31.54/30.92 t/s against 31.91/31.94/31.54/30.91 for patch 36, and
`-p 2048 -d 0,4096` 420.8/397.2 against 419.7/396.5, all inside the +-0.3% the two passes
reproduce. The model runs about 16 flash-attention launches per token (the token grows 0.61 ms
between d8192 and d16384 while the launch grows 38 us), so the 5.7 us saved per launch at n_kv 8448
is 0.29% of a 31.7 ms token. The value of the patch is the per-launch curve and the gate it
unlocks, not this benchmark; `pp` and the speculative widths are untouched by construction
(`ncols1` 16, 2, 4, 8).

Correctness: `test-backend-ops -o FLASH_ATTN_EXT` 2942/2942 in all three arms (new, patch 36, legacy),
full suite 14675/14675 on CUDA1 and 14674/14675 on CUDA0, where the one failure is
`MUL_MAT(type_a=iq1_s,m=16,n=1,k=4096)` at ERR 6.6e-4 against a 5.0e-4 tolerance, unrelated to
flash attention and flaky by construction (the suite seeds its tensors from `std::random_device`);
it passes on two reruns and on the other card with the same binary. `~/p100-opt/c2b/check6.sh` (22
shapes against the CPU backend, including the ones at and just above the new gate): NMSE 1.9e-6 to
2.1e-4, within a few percent of the legacy path's and usually slightly smaller, since the combine
now sums fewer and more even partial results. Perplexity `-c 2048 --chunks 10` identical (5.4367).
Greedy 128 tokens after a 5391-token prompt at `-c 8192`, which runs the packed kernel with the new
`parallel_blocks` throughout the generation (n_kv 5632): byte-identical to stock, to patch 36 and
to the legacy kernels.

Caveats: the rule fires on every one-Q-column `launch_fattn` call on Pascal (tile at any head size,
VEC) but was measured at D=256 only; Gemma 4's D=512 GQA 8 layers and D=128 models take it
unmeasured (C6), and the kill switch below is the answer if one of them regresses. The gate scan has
256-token steps up to 4096 and 4096-token steps above 8192, so a narrow dip between two measured
points is not excluded. `test-backend-ops` still has no small-batch GQA 6 case; `c2b/check6.sh`
covers the packed path.

Kill switches: `GGML_CUDA_FATTN_PB_TIEBREAK=0` restores the b10758 search; `GGML_CUDA_FATTN_TILE_LEGACY=1`
does that too and restores the b10758 kernels; `GGML_CUDA_FATTN_GQA6_MIN_KV=<n>` moves the gate
(16384 = patch 36, 0 = pack always, also with one kv head). Not done, with numbers: a tie-break for
two Q columns (that launch loses up to 14% between n_kv 4608 and 6656 but the naive rule costs 4-7%
there; needs a rule that knows how far the kernel is from bandwidth saturation); the one-kv-head
clause above 16384 (one line, its own protocol run); `nbatch_fa 32` as the other lever on the tail
granularity (5-8% slower per launch in C2, never measured together with `pb`). Rebase note: the
tie-break is one block at the end of the `else` (non-stream-k) branch of `launch_fattn`, and the
gate is the `gqa6_ok` lines of `launch_fattn_tile_switch_ncols2`; both files are upstream-churned,
and 36 and 38 touch the same lines (rebase them as a pair).

### 39 convert-vec

`ggml/src/ggml-cuda/convert.cu` (+76/-15), `dequantize.cuh` (+36/-8) and `common.cuh` (+10/-0). Plan
items A3' and A2. Two kernels around every cuBLAS call on GP100 ran at a third of the copy ceiling.
nsys at `-ub 2048` (Qwen3.8-27B UD-Q4_K_M, per graph evaluation per card, 4910 ms of kernel time):
`convert_unary` 463.6 ms in 1248 launches (9.4% of pp) moving 76.5 GB at 165 GB/s, and
`dequantize_block_*` 159.0 ms in 496 launches (3.2%) moving 31.0 GB at 195 GB/s, against the
560 GB/s device copy ceiling of `P100-HARDWARE.md`.

A3' replaces the one-element-per-thread kernel behind the contiguous entry point
(`convert_unary_cont_cuda`) with `convert_unary_vec<src_t, dst_t, nelem, cpy>`: `nelem` contiguous
elements per thread moved with `cpy`-byte transfers through the new `ggml_cuda_memcpy_n` in
`common.cuh` (the loop that `ggml_cuda_memcpy_1`'s own assert message asks for), a scalar tail loop
for `k % nelem` in the same launch, and a grid that covers the data instead of one block per 256
elements. The element expression is the unchanged `ggml_cuda_cast<dst_t>(x[i])`, so no rounding
moves. 4 elements per thread with 8-byte transfers won the sweep below; the host gate falls back to
the old kernel when either pointer is not 8-byte aligned (pool buffers always are). The
non-contiguous `convert_unary_cuda` and the `_nc` variants are untouched. All six instantiations of
the entry point take the new kernel, so the BF16 allreduce wire (`ggml-cuda.cu`, 99.5 ms per
evaluation per card) is covered for free.

A2 turned out not to be the item the plan described. One super block per warp with 8 warps per
block, byte-identical by construction (`threadIdx.y` selects the super block, every thread keeps
its element mapping), is a no-op on its own: q4_K 151 -> 153 GB/s, q5_K 214 -> 214, q6_K 329 -> 331,
iq4_xs 177 -> 192, q3_K 189 -> 176. What costs is the store: `dequantize_q4_K` has each of 32 threads
write `y[l]` and `y[l+32]` for l = 0..3 as eight 2-byte stores, so one store instruction writes
2 bytes every 8 across the warp. Issuing each contiguous group of 4 as one 8-byte transfer (same
values, same addresses; new `bool vec_store` template parameter on `dequantize_q4_K` and
`dequantize_iq4_xs`, default `false` so `getrows.cu` is unchanged) gives q4_K 151 -> 377 GB/s and
iq4_xs 177 -> 323, and the block packing then adds the rest: 460 and 341 GB/s, 3.04x and 1.93x per
call. q5_K only has 2 contiguous elements per thread and gains 1.04x, q6_K writes singles at a
stride of 32 and is already at 329 GB/s, q3_K loses from packing; all three keep their launchers.

Per-call sweep (standalone harness on the in-tree `convert.cu`, k = 10485760 = 5120x2048, GB/s
counting both directions):

    config                     f32->f16  f32->bf16  f16->f32  bf16->f32
    old, 1 element per thread    173         162        192       192
    nelem 4, cpy 8               544         536        494       494    <- shipped
    nelem 8, cpy 8               522         399        285       247
    nelem 8, cpy 16              544         540        417       418
    nelem 16, cpy 8              407         157          -         -
    nelem 4, cpy 8, 448 blocks   540         525        440       440

16-byte transfers (Pascal has `LDG.128`; `ggml_cuda_get_max_cpy_bytes()` returns 8 on sm_60) tie in
one direction and lose 15% in the other, so the tree's 8-byte convention stands and plan item C5
gets a data point. A bounded grid with a grid-stride loop, the usual shape for a streaming kernel,
loses 1-11%: the P100 retires 10240 blocks without trouble; what it could not do was retire 40960
blocks that move 1.5 KB each. Block size is flat from 64 to 1024 threads.

In the model (Qwen3.8-27B UD-Q4_K_M, 2x P100, `-sm tensor -fa on`, arms alternated, median of `-r 5`,
the f80d6db61 binaries as the base arm):

    pp2048 -ub 2048   base 420.82 (420.34-421.69) -> new 449.36 (449.17-449.70) t/s   +6.8%
                      A3' alone +5.55% (new against GGML_CUDA_DISABLE_CONVERT_VEC=1, 425.75)
                      A2  alone +1.22% (new against GGML_CUDA_DISABLE_DEQUANT_VEC=1, 443.93)
                      both switches set on the new binary: 420.06, i.e. base within the spread
    pp2048 -ub 512    base 235.06 (234.92-235.17) -> new 248.30 (248.19-248.34) t/s   +5.6%
                      A3' alone +3.18%, A2 alone +2.92% (the dequant work does not shrink with the ubatch)
    tg64 d0 / d16384  unchanged (31.92 -> 31.91 and 30.92 -> 30.91 with patch 40 off in both arms)

nsys per graph evaluation per card: kernel time 4909.8 -> 4560.1 ms (-7.1%); `convert_unary`
463.6 ms at 165 GB/s -> `convert_unary_vec` 147.2 ms at 520 GB/s (29% -> 93% of the ceiling,
547-551 GB/s = 98% on the f32 -> f16 direction); dequant 159.0 -> 109.9 ms at 195 -> 282 GB/s, which
is q4_K 458.3 -> 167.5 us per launch (168 -> 461 GB/s, 2.74x) and iq4_xs 457.9 -> 293.4 us
(212 -> 330 GB/s, 1.56x) with q5_K, q6_K, q3_K and iq4_nl unchanged as designed. The old
`convert_unary` does not appear in the capture at all, so the alignment gate passes on every call
of this model.

Perplexity 5.4367 in every arm (new, both kill switches, base and stock: identical to the last
digit) and 256 greedy tokens byte-identical to base at `-ub 2048` and `-ub 512`. Full
`test-backend-ops` 14675/14675 on CUDA0 and on CUDA1 (and on CUDA0 with both switches set); the
harness memcmps both new kernels against the old ones over 25 widths (every tail case), 6 type
pairs and 5 vector configurations plus the misaligned fallback: 0 mismatches. No new registers and
no spills (`convert_unary_vec` 17-29 registers against 23-24; `dequantize_block_sb` 22-30 against
23-28). Compute buffer sizes unchanged.

The plan's A3 one-off, measured the same day: `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (F32 compute, no
conversion pass, no F16 dequant target) gives pp2048 420.90 -> 272.13 t/s (-35.3%) and perplexity
5.4094 against 5.4367, so the halved HFMA2 rate costs far more than the passes it removes, and F16
accumulation in the shipped path prices at 0.50% perplexity.

Kill switches: `GGML_CUDA_DISABLE_CONVERT_VEC=1` (the old `convert_unary` for the contiguous entry)
and `GGML_CUDA_DISABLE_DEQUANT_VEC=1` (per-element stores and one super block per block, i.e. the
launches of b10758). Not done, with numbers: q5_K needs a different index assignment inside
`dequantize_q5_K` for a 4-element store group, about 0.5% of pp2048 left there; `getrows.cu` could
ask for `vec_store = true` and get the same 1.9-3.0x on quantized get_rows, which this model does
not use. Rebase note: one kernel and one launcher template in `convert.cu`, two `if constexpr`
branches in `dequantize.cuh`; both files are upstream's with modest churn.

### Meta backend gist

Only used with `-sm tensor` (`ggml/src/ggml-backend-meta.cpp`). Replaces the buffer-global
rotating `stc_compute[2]` shard containers with containers owned by the backend instance
that runs the graph, plus identity-validated scratch pools for graph-external `set/get_tensor`.
Fixes `GGML_ASSERT(bcj.nodes[i]) failed` in `ggml_backend_meta_graph_compute` after a graph
rebuild. Applied as commit `p100: meta backend graph reuse fix`.

Crash signature on stock upstream (b10133 through b10615, `ggml-backend-meta.cpp:1836`):
fires on the first decode of a request with a near-exact prefix cache hit (`sim_best`
0.997-1.000) over a large context, before any prompt-processing line is logged. Not a size
threshold: crashed at 48.9k / 63.2k / 65.6k tokens while 42k, 43k and 62k went through, and
high similarity alone is not sufficient either. The child aborts, the router respawns it and
the next request reprocesses the whole conversation (minutes at ~280 t/s pp) - that
reprocessing is the visible symptom. Noise in the same log, not causes:
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`, the NCCL `libnccl-*.so`
plugin-not-found lines, the Qwen-VL image-token warning.

With the gist the equivalent assert is the `GGML_ASSERT(bcj.nodes[i])` right after
`ggml_backend_meta_simple_tensor_ensure(backend_ctx->stc_compute[...], node, j)`. If it ever
fires again, `LLAMA_GRAPH_REUSE_DISABLE=1` is the escape hatch (see knobs; -27% tg here).

Validated 2026-08-26 on the same workload: single instance to 89,139 tokens,
`graphs reused = 39203`, no assert; passed every old crash point and dozens of
`f_sim_best = 1.000` / 0.998-0.999 requests at 65.8k-87.5k. Both cards at 15,472 MiB from
one PID, nothing on CPU (no partial-offload regression).

Adaptations made against b10630:

- Upstream #27574 changed the PARTIAL-split branch of `ggml_backend_meta_buffer_set_tensor`
  (contributor mask). Upstream's logic is kept, only the shard lookup is swapped for
  `scratch.get(tensor, j)`.
- `ggml_backend_meta_buffer_memset_tensor` was added upstream after the gist and still used the
  static-only lookup, which returns `nullptr` for compute tensors and views under the gist's
  design. It now uses a scratch pool like set/get_tensor.
- `ggml_backend_meta_buffer_init_tensor` is a no-op with the gist, but upstream #27586's
  `ggml_backend_buffer_init_tensor(simple_buf, t_ij)` inside `init_tensor_impl` is kept, so
  simple backends still see init for every shard that gets registered.
- Dead code after the merge removed: `params_compute`/`compute_headroom` in
  `ggml_backend_meta_alloc_ctx_tensors_from_buft`, `params` in `..._buffer_type_alloc_buffer`,
  `#include <set>`.

Upstream did not touch `ggml-backend-meta.cpp` between b10630 and b81c99b47; the gist commit
re-applied without conflict.

When rebasing, any new function in `ggml-backend-meta.cpp` that calls
`ggml_backend_meta_buffer_simple_tensor()` on non-static tensors needs the same treatment.
`grep -n "ggml_backend_meta_buffer_simple_tensor(" ggml/src/ggml-backend-meta.cpp` should
list only its definition and the call inside `ggml_backend_meta_simple_tensor_ensure`.

## Runtime knobs added by the patches

| Variable | Patch | Effect |
|---|---|---|
| `GGML_CUDA_DISABLE_FUSE_CPY`, `GGML_CUDA_DISABLE_FUSE_L2_NORM` | 18 | kill switches |
| `GGML_CUDA_FUSE_LOG=1|2` | 18 | log which fusion rules fired / why not |
| `GGML_CUDA_DISABLE_FUSE_PRE_ADD` | 19 | kill switch |
| `GGML_CUDA_DISABLE_FUSE_ADD_UNARY_MUL` | 20 | kill switch |
| `GGML_CUDA_DISABLE_FUSE_GDN_BETA`, `GGML_CUDA_DISABLE_FUSE_GDN_GATHER` | 23, 24 | kill switches |
| `GGML_CUDA_DISABLE_CPY_ROWS` | 26 | kill switch |
| `GGML_CUDA_DISABLE_CONCAT_ROWS`, `GGML_CUDA_DISABLE_FUSE_CONCAT_GATHER` | 27 | kill switches |
| `GGML_CUDA_DISABLE_TOP_K_PARTIAL` | 28 | kill switch (not on the branch) |
| `LLAMA_SAMPLER_PREFILTER=0` | 13 | kill switch |
| `LLAMA_DEC_SLOTS=N` (default 4, 0 = off), `LLAMA_DEC_MAX_TOK` (default 4) | 22 | decode slots (not on the branch) |
| `LLAMA_MTP_DRAFT_VOCAB=<file>` | 15 | enable draft vocab subset |
| `GGML_A16_*` | 12 | Q4_1 HFMA2 kernel tuning |
| `GGML_CUDA_DISABLE_FUSION=1` | upstream | disables all CUDA fusion, including the patched rules of 18, 19, 20, 23, 24, 27 |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | upstream | disables graph reuse (28.9 -> 18.1 t/s tg here, -27%; pp unchanged) |
| `LLAMA_SERVER_CKPT_LEGACY=1` | 31 | upstream checkpoint placement (3 checkpoints + 3 decodes per follow-up) |
| `GGML_CUDA_DISABLE_MMVQ_F16_K=1` | 32, 33 | kill switch (HFMA2 K-quant matvec: Q4_K, Q5_K, IQ4_XS, Q6_K; back to the int8 path) |
| `GGML_CUDA_DISABLE_MMVQ_F16_K_TYPES=q5_K,iq4_xs,...` | 33 | per-type kill switch (comma separated `ggml_type_name` values) |
| `GGML_A16K_*` | 32, 33 | HFMA2 K-quant kernel tuning (`ROWS`, `NWARPS`, `PF`, `SMALL_GRID`, `NCOLS_MIN/MAX`, `LOG`); `NCOLS_MIN` overrides the per-type minimum (IQ4_XS default 4) |
| `GGML_A16K_SHORTK=0` | 34 | kill switch (prefetch mode 2 off, patch 33 configuration) |
| `GGML_A16K_SHORTK_NB=N` (default 24), `GGML_A16K_PF=2` | 34 | width-1 block-count threshold for mode 2; force prefetch mode 2 |
| `GGML_CUDA_MMVQ_MAX_COLS_SM60=8` | 35 | kill switch (per-type MMVQ column ceiling on GP100; 8 or 0 = upstream `ne11 <= 8`, values below 8 clamped) |
| `GGML_CUDA_MMVQ_COLS_CHUNK=N` (2-8, default 7), `GGML_CUDA_MMVQ_COLS_LOG=1` | 35 | target chunk width of the column split (floor N-1; 8 = fewest chunks); log the first 32 chunked calls |
| `GGML_CUDA_FATTN_TILE_LEGACY=1` | 36 | kill switch (b10758 flash-attention tile kernels: old `(256, 256, 2)` entry, no GQA 6 packing) |
| `GGML_CUDA_FATTN_LOG=1` | 36 | one stderr line per distinct flash-attention launch tuple (kernel kind, ncols1/ncols2, config, blocks per SM, parallel_blocks, grid) |
| `LLAMA_SERVER_CKPT_SAVE_LEGACY=1` | 37 | kill switch (a freshly allocated buffer for every context checkpoint) |
| `GGML_CUDA_FATTN_PB_TIEBREAK=0` | 38 | kill switch (b10758 `parallel_blocks` search; `GGML_CUDA_FATTN_TILE_LEGACY=1` implies it) |
| `GGML_CUDA_FATTN_GQA6_MIN_KV=<n>` | 38 | smallest n_kv for the GQA 6 packing: default 2560 (7680 at 2 Q columns), 16384 = patch 36, 0 = always, also with one kv head |
| `GGML_CUDA_DISABLE_CONVERT_VEC=1` | 39 | kill switch (contiguous F16/BF16 <-> F32 conversion back to the one-element-per-thread `convert_unary`) |
| `GGML_CUDA_DISABLE_DEQUANT_VEC=1` | 39 | kill switch (Q4_K and IQ4_XS dequant back to per-element stores and one super block per CUDA block) |

No kill switch: 01, 02, 04, 05, 10, 11, 14, 16, 17, 21, 25, 29, 30 (and the MoE-only 07, 08).
To bisect one of those, build with the commit dropped (`git rebase -i` or
`git revert`). 21 is a scheduler patch (`ggml-backend.cpp`, lazy `ggml_backend_sched_reset`)
next to where the meta backend crash lived; it was on the branch for the 89k validation run
above, so it is cleared for `-sm tensor`, but it is the first one to drop if scheduler-side
symptoms come back.

Graph reuse is a CPU-side optimization (skips graph rebuild and the meta backend's re-split
across devices); it has nothing to do with KV/prompt caching, which is `cache_prompt` plus
LCP slot matching in the server.

## Runtime notes (2x P100, `-sm tensor`)

- Batches wider than 8 columns run quantized matmuls through dequantize-to-F16 + cuBLAS on
  sm_60 (MMQ is excluded below DP4A for dense `MUL_MAT`, see `ggml_cuda_should_use_mmq`; since
  #26264 upstream allows it for `MUL_MAT_ID` only). The
  F16 copy lives in the CUDA temp pool for the call. Prompt processing sizes the pool for
  the layer matrices, but the LM head only ever runs at width 1 there (logits for the last
  token), so the first time more than 8 positions need logits - a speculative verify batch
  with a draft longer than 7 tokens - `output.weight` (~1.3B params on Qwen3.8-27B) gets a
  ~2.5 GB F16 copy, ~1.3 GB per card. With the cards at ~94% that is
  `CUDA error: out of memory` in `ggml_cuda_mul_mat_cublas_impl<F16>` on the first draft.
  Either keep drafts <= 7 (`--spec-draft-n-max 7`, `--spec-ngram-mod-n-max 7`; MTP with
  `n-max 4` is fine) or free ~1.3 GB per card by lowering `--ctx-size`.
- `draft-mtp` on this model (head is in the main GGUF, no sidecar needed): acceptance
  0.78-1.00, but with `--draft-p-min 0.75` the verify width changes every step, graph reuse
  drops to ~10% and each miss is a meta-backend re-split (15-20 ms). Measured 2026-08-26:
  short replies 30-38 t/s, long reasoning outputs 23-25 t/s at 15-38k context vs ~27
  without MTP; prompt processing ~25% slower and ~1 s fixed cost per request. Untested
  fixes: `--draft-p-min 0` (constant width 5, reuse should recover), patch 22.
- `ngram-mod` costs nothing when it does not draft; see the first note for the draft length
  limit. Measured 2026-08-26 with `n-max 7 n-min 4` at 200k ctx: tg 27.1-27.4 t/s flat at
  25-27k context (= no-speculation baseline), drafts fire on ~1% of reasoning tokens
  (acceptance 0.64-1.0 when they do). Harmless; only pays on repetitive output.
- The fixed ~1 s prompt phase per request on short follow-ups (`929 ms / 55 tokens`) was the
  server's checkpoint logic for recurrent models, not the meta-backend re-splits and not
  `--cache-ram` (measured identical at 16384 and 0; with `--parallel 1` the slot is always
  re-selected by prefix similarity, so the host KV save never runs). Patch 31 removes most of it;
  what remains (~730 ms) is one 150 MiB checkpoint save (~200 ms) and a 28-token decode through
  the dequant + cuBLAS path (~390 ms), see `P100-OPTIMIZATION-PLAN.md` items J4b and A8.
- Measured 2026-09-03 with the standard protocol (`P100-OPTIMIZATION-PLAN.md` section 4): branch
  pp2048 421 t/s, tg 29.3 t/s at depth 0 (stock 417 / 21.6); tg is matvec-bound, `mul_mat_vec_q`
  is 77% of the step at 315 GB/s of the 603 GB/s the card delivers; GPU busy 97%, so launch
  overhead and CUDA graphs are not a lever on Pascal. `NCCL_P2P_LEVEL=SYS` is worth +1.5% pp,
  +0.9% tg (NCCL otherwise bounces through host memory). `--load-mode dio` loads in 70 s from the
  HDD vs 4 s for mmap; keep mmap.
- 2026-09-04, same protocol: with patches 32-34 tg is 31.7 t/s at depth 0 and 30.5 at depth
  16384 (int8 path 29.3 / 28.2). The Q4_K, Q5_K and Q6_K matvecs run on HFMA2; IQ4_XS stays on
  the int8 path at widths 1-3 (already ~400 GB/s there) and is now the largest tg kernel (21%).

## Updating to a new upstream

The branch is a linear commit series, so updating is one rebase:

```sh
git fetch origin --tags
git branch p100-b<new-build> p100-b<old-build>             # old branch stays untouched
git rebase --onto <new-tag> <old-base> p100-b<new-build>   # old-base: the "Base:" commit above
git range-diff <old-base>..p100-b<old-build> <new-tag>..p100-b<new-build>   # '=' except resolved patches
```

`git fetch origin` over HTTPS fails on the workstation (`could not read Username for
'https://github.com'`, git 2.34, sandboxed or not) while SSH works, so fetch upstream with:

```sh
git fetch git@github.com:ggml-org/llama.cpp.git \
    '+refs/heads/master:refs/remotes/origin/master' 'refs/tags/*:refs/tags/*'
```

Turn on `git rerere` once (`git config rerere.enabled true`) so a conflict resolved once is
replayed automatically on later rebases.

For each conflicting commit decide:

1. **Fixed or superseded upstream** - drop the commit (`git rebase --skip`), remove its row
   above and say so here.
2. **Moved** - resolve, keep the commit, note anything non-obvious in the patch section above.
3. **Rewritten upstream** (like 09) - reimplement on the new structure or drop.

Then update the "Base:" line and the status table, and run the test checklist. Also check
whether upstream retuned values a patch overrides (02, 05, 07, 08 change constants that
upstream tunes for newer hardware).

To regenerate patch files in the layout of the source repo:

```sh
git format-patch --no-numbered --zero-commit -o patches/ <base>..p100-b<build>
```

## Test checklist (on the GPU server)

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/test-backend-ops                       # patch repo reports 13327 tests, 0 failures
./build/bin/test-llama-archs                       # runs every arch through -sm tensor on the GPUs (meta backend)
./build/bin/llama-bench -m <model> -ngl 99 -sm tensor -p 512 -n 64   # compare with stock master
```

`test-llama-archs` is the only test that exercises `ggml-backend-meta.cpp`; it skips the meta
configuration on CPU-only machines, so it has to run on the server.

If `test-backend-ops` reports a single MUL_MAT failure, rerun the full unfiltered suite two
or three times rather than the filtered single case: the `-p` filter changes the random
draw, so "passes alone" only says the case is data dependent, not that it is harmless.
Reproducible in the full run but not alone = unlucky draw; intermittent in the full run =
something real.

Then the real workload: `llama-server` with the usual flags. The shape that used to crash
is a large context (50k+) followed by a request with a near-exact prefix cache hit
(`sim_best` 0.997-1.000 in the log), i.e. a long chat that keeps going. Watch for the
assert and for the router respawning the child; `GGML_META_DEBUG=1` logs the meta backend's
rebuilds if that needs pinning down.

Output vs. stock: the patches that change numbers are 12 (Q4_1 weights, n >= 2), 15 (off
unless `LLAMA_MTP_DRAFT_VOCAB` is set) and 32 (every Q4_K matvec at decode widths 1-8, on by
default). Run the greedy comparison (`--temp 0 --seed 1`, same prompt, this build vs a stock
build of the same base commit) with `GGML_CUDA_DISABLE_MMVQ_F16_K=1`: with the switch set the
output is expected to be identical to the last token, not just similar, and any divergence is
a bug; bisect at runtime with the kill switches above (start with `GGML_CUDA_DISABLE_FUSION=1`,
then `LLAMA_SAMPLER_PREFILTER=0`) before rebuilding anything. Without the switch the Q4_K path
of 32 may take a different token at a near-tied position (2026-09-03: identical for the first
76 words of a 256-token run, then a different continuation); its check is the perplexity one
in the section on 32.

Since the 2026-09-02 rebase upstream forces the fused GDN ops on (#27877 set `auto_fgdn = false`,
`fused_gdn_ar/ch = true`); b10630 probed the backend first. With CUDA and the meta backend the
probe should already have picked fused, so no change is expected for `qwen35`. A tg difference
on `qwen35` that the kill switches above do not explain points here first.

## Change log

- 2026-08-26: branch created on `b10630`. Patches 01-08, 10-21, 23-30 applied via rebase from
  `b10133` (28 without conflict). 09 not applied (upstream rewrite), 22 pending (trivial
  conflict). 13 needed a source fix (`has_logit_bias` removed upstream). Host-side patches
  (11, 13, 15, 21) compile-checked with a CPU-only build and `test-sampling`; CUDA build not
  yet verified.
- 2026-08-26: step 2, philpax meta backend gist applied (1 conflict hunk, memset_tensor
  adapted, dead code removed). `ggml-base` compiles clean; not yet run on the GPUs.
- 2026-08-26: first server run (2x P100, CUDA 12.6, NCCL): CUDA build OK, `test-llama-archs`
  all OK including `qwen35` on Meta, `test-backend-ops` 13567/13568. The one failure,
  `MUL_MAT(type_a=q4_1,type_b=f32,m=16,n=1,k=32,...)`, passes when rerun alone on either GPU
  (`-o MUL_MAT -p 'type_a=q4_1,type_b=f32,m=16,n=1,k=32'`). No patch is Q4_1-specific on the
  n=1 path (12 needs n >= 2), so this is taken as the fp16 `m*s` term of upstream's Q4_1
  vec_dot on sm_60 going borderline for an unlucky random draw at one block of K. Not a real
  inference shape; treat a recurrence as known unless it becomes deterministic.
- 2026-08-26: crash test on the server (Qwen3.8-27B Q4_K_M, `-sm tensor -fa on -c 200000
  -b 2048 -ub 2048 -np 1`, router mode). Stock upstream crashed at 48.9k / 63.2k / 65.6k
  tokens on prefix-cache-hit decodes; this branch ran one instance to 89,139 tokens with
  39,203 graph reuses and no assert. pp 475 t/s at 3.7k -> ~270 t/s at 89k, tg 28.9 -> 22.8
  t/s (normal attention scaling). `LLAMA_GRAPH_REUSE_DISABLE=1`: tg 18.1 t/s flat, pp
  unchanged. VRAM is preallocated at load (~94% per card), no headroom for more context or a
  second slot. Conclusion: the fault was upstream's meta backend, not the patch set; the
  gist is what fixed it (a plain version bump would not have - the only meta-backend commits
  between b10133 and b10615 are #26502 and its revert #27433, plus #27574 which is a
  different mechanism).
- 2026-09-02: rebased onto `b81c99b47` (`b10758` + 1; 128 upstream commits, 2026-08-26 to
  2026-09-02). 03 and 06 dropped (superseded by #27621 and #26264), 28 deferred (`top-k.cu`
  rewritten by #27466). 23 and 24 conflicted with #25952 at the top of `ggml_cuda_try_fuse`,
  both rules kept; 25 and 27 only conflicted in a dry run that had skipped 23. The gist and every
  other patch applied without conflict. Upstream did not retune the constants of 02, 05, 07, 08
  (its mmvq/mmvf changes are SWIGLU_CLAMP fusion plumbing) and did not touch
  `ggml-backend-meta.cpp`. Old tip kept as branch `p100-b10630`. Host side compile-checked with
  the CPU-only build (`llama`, `llama-common`, `test-sampling` build and `test-sampling` passes;
  the full `build-cpu` target set now fails on upstream's `test-chat` including server headers
  that need `mtmd.h` with `LLAMA_BUILD_TOOLS=OFF`, unrelated to the patches). CUDA build and the
  test checklist not yet run on the server. The patch repo moved to a `v0.2.0` base on 2026-08-29
  (`dc4740d`); `v0.2.0` predates b10630, so its patch files are not newer than this branch.
- 2026-09-03: Tier 0 of `P100-OPTIMIZATION-PLAN.md` measured on the server (hardware fact sheet,
  baselines branch vs stock, nsys breakdowns, NCCL transport, server per-request timeline). First
  local patch `p100x: 31-server-ckpt-adopt` (follow-up prompt phase 1302 -> 732 ms).
- 2026-09-02: branch `p100` renamed to `p100-b10758`; one branch per upstream base from now on
  (see Branch layout). The fork's `p100` (`e9b087580`) is two doc commits behind `p100-b10630`.
- 2026-09-03: local patch 32 (`p100x: 32-mmvq-q4k-hfma2`): HFMA2 Q4_K matvec for sm_60, widths
  1-8, 1.49x-2.85x the int8 path per call, tg +2.5% on the mixed UD-Q4_K_M (Q4_K is 17.7% of its
  tg kernel time; Q5_K/IQ4_XS/Q6_K next). Perplexity at `-ub 1`/`-ub 8` within 0.13% of stock (the int8 path equals stock); greedy
  identical to stock with the kill switch, diverges after 76 words without it. Full `test-backend-ops`
  14675/14675 on CUDA1 and 14674/14675 on CUDA0 (the known q4_1 width-1 tolerance case, passes
  alone and on stock). Kill switch `GGML_CUDA_DISABLE_MMVQ_F16_K=1`.
- 2026-09-04: local patch 33 (`p100x: 33-mmvq-k-hfma2`): the HFMA2 matvec of 32 generalized to
  Q5_K, IQ4_XS (widths 4-8 only, its int8 path already runs at 400 GB/s) and Q6_K; files renamed to
  `mmvq-k-f16-sm60.cu/.cuh`. tg 30.0 -> 31.6 t/s at d0 (+5.2%; +7.8% over the int8 path), 28.8 ->
  30.3 at d16384. Full `test-backend-ops` 14675/14675 on both cards; perplexity within 0.25% of stock
  at `-ub 1`/`-ub 8` (the int8 path equals stock); greedy identical to stock with the kill switch.
  New per-type kill switch `GGML_CUDA_DISABLE_MMVQ_F16_K_TYPES`.
- 2026-09-04: local patch 34 (`p100x: 34-mmvq-k-shortk`): prefetch mode 2 (header loaded in its own
  step) per type and width in the HFMA2 K-quant matvec, chosen with a new any-shape timing tool; tg
  +0.5% (31.7 t/s at d0, +8.2% over the int8 path), speculative verify widths 2/4 +4% / +2%; no math
  change (greedy byte-identical to the switch-off path); full `test-backend-ops` 14675/14675 on both
  cards. Kill switch `GGML_A16K_SHORTK=0`. The short-k gap itself stands (0.74-0.81 of the long-k rate)
  and a shorter warp step is measured out (-18-26%).
- 2026-09-05: local patch 35 (`p100x: 35-mmvq-cols-sm60`): MMVQ column-chunk loop for 9-64 columns on
  GP100 (dispatch only, no kernel change): balanced chunks of about 7 columns over the activations
  quantized once, per-type ceiling Q4_K 64, Q5_K 48, Q6_K 32, IQ4_XS 64, int8 32, above which dequant +
  cuBLAS stays. pp9 3.87x, pp16 2.84x, pp28 2.30x, pp32 1.93x, pp48 1.43x, pp64 1.12x, unchanged at 96+;
  the 25k-chat follow-up prompt phase 710 -> 434 ms (request 1.34 -> 1.07 s); tg and pp2048 unchanged.
  Perplexity within 0.36% of stock at `-ub 16/32/64`; greedy byte-identical new vs old; full
  `test-backend-ops` 14675/14675 on both cards. Kill switch `GGML_CUDA_MMVQ_MAX_COLS_SM60=8`.
- 2026-09-06: local patch 36 (`p100x: 36-fattn-tile-p100`, measured 2026-09-05): Pascal FP16
  flash-attention tile table for D=256 (the model is D=256 GQA 6, not D=128): the tg entry had no
  register budget (255 registers, 8 warps per SM) and GQA 6 fell onto `ncols2 = 2` with a 3x K/V
  re-read; `(128, 4, 64, 64)` for `ncols 2` plus `ncols2 = 6` packing above n_kv 16384 at 1-2 Q
  columns. Attention kernel 1.31x at 32k, 1.70x at 128k; tg +0.7% at d0, +1.6% at d16384, +3.3% at
  d32768, +7.9% at d65536; pp unchanged (its entry is already optimal). Full `test-backend-ops`
  14675/14675 on both cards plus `c2/check6.sh` for the GQA 6 path (the suite has no such case);
  perplexity identical to stock; greedy byte-identical after a 22k prompt. C3 (VEC at tg) and C4
  (forced `parallel_blocks`) measured out. Kill switch `GGML_CUDA_FATTN_TILE_LEGACY=1`.
- 2026-09-06: local patch 37 (`p100x: 37-server-ckpt-save`): recycle the storage of an erased context
  checkpoint instead of allocating a fresh 149.6 MiB vector per save; checkpoint save 112.4 -> 96.4 ms,
  follow-up prompt phase 433.3 -> 416.9 ms (-3.8%), RSS unchanged, checkpoints byte-identical (17/17
  against a second save of the same state). The plan's premise was wrong: the DtoH copy is 30.5 ms at
  5.14 GB/s (faster than the 36.9 ms restore); the rest of the old window was the allocation, which was
  hiding ~66 ms of decode in flight. Kill switch `LLAMA_SERVER_CKPT_SAVE_LEGACY=1`.
- 2026-09-06: local patch 38 (`p100x: 38-fattn-pb-tiebreak`): `parallel_blocks` tie-break in `launch_fattn`
  (smallest value with the same number of KV rounds; one Q column, Pascal only) and the GQA 6 packing gate
  of 36 lowered from n_kv 16384 to 2560 (7680 at 2 Q columns), now the knob `GGML_CUDA_FATTN_GQA6_MIN_KV`.
  Per launch 1.10-1.21x from n_kv 2560 to 12288 against 36, 1.18x at 8192 from the tie-break alone; model
  tg unchanged within 0.3% (about 16 FA launches per token). FA suite 2942/2942 in all arms, full suite
  clean on CUDA1 and one unrelated flaky iq1_s case on CUDA0; check6 22 shapes; perplexity identical;
  greedy byte-identical to stock with the packed kernel running. Kill switches
  `GGML_CUDA_FATTN_PB_TIEBREAK=0`, `GGML_CUDA_FATTN_TILE_LEGACY=1`.
- 2026-09-07: local patch 39 (`p100x: 39-convert-vec`): the contiguous F16/BF16 <-> F32 conversion around
  every cuBLAS call moves 4 elements per thread in 8-byte transfers (`convert_unary` 165 -> 520 GB/s, 29% -> 93%
  of the copy ceiling), and the Q4_K / IQ4_XS dequant-to-F16 kernels store each contiguous group of 4 as one
  8-byte transfer with 8 super blocks per block (q4_K 168 -> 461 GB/s, iq4_xs 212 -> 330). pp2048 420.8 -> 449.4
  t/s at `-ub 2048` (+6.8%; conversion +5.55%, dequant +1.22%), 235.1 -> 248.3 at `-ub 512` (+5.6%); tg unchanged.
  Output byte-identical: perplexity 5.4367 in every arm including stock, greedy identical, full suite
  14675/14675 on both cards. The plan's A2 premise was wrong: block packing alone is a no-op, these kernels are
  store bound. Kill switches `GGML_CUDA_DISABLE_CONVERT_VEC=1`, `GGML_CUDA_DISABLE_DEQUANT_VEC=1`.
