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
