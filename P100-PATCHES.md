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

- Base: upstream `master` at `5d5cb4c3a` (`b10630`).
- One commit per patch, subject `p100: NN-name`, in the original numeric order.
  Order matters: several patches touch the same lines and later ones build on earlier ones.
- Commits after the patch series: the meta backend gist and this document. 09 and 22 are
  not on the branch (deferred, see below).

## Status per patch (against b10630)

Scope tags are from the patch repo. "Fires" says whether the patch does anything for the
models listed above; inert patches are kept to stay close to the upstream patch set.

| # | Patch | Scope | Status | Fires for our models |
|---:|---|---|---|---|
| 01 | vmad-dp4a-sm60 | sm_60 | clean | yes (all quantized matvec) |
| 02 | mmvq-rows-per-block-sm60 | pre-Turing | clean | yes |
| 03 | topk-moe-multirow | CUDA | clean | MoE only |
| 04 | concat-non-cont-flat | CUDA | clean | qwen35 (delta-net) |
| 05 | mmvf-f32-pascal | pre-Turing | clean | yes (batch 2-8 F32 matvec) |
| 06 | mmq-mul-mat-id-sm60 | sm_60 | clean | MoE only |
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
| 28 | top-k-partial | CUDA | clean | GPU sampling only |
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
| `GGML_CUDA_DISABLE_TOP_K_PARTIAL` | 28 | kill switch |
| `LLAMA_SAMPLER_PREFILTER=0` | 13 | kill switch |
| `LLAMA_DEC_SLOTS=N` (default 4, 0 = off), `LLAMA_DEC_MAX_TOK` (default 4) | 22 | decode slots (not on the branch) |
| `LLAMA_MTP_DRAFT_VOCAB=<file>` | 15 | enable draft vocab subset |
| `GGML_A16_*` | 12 | Q4_1 HFMA2 kernel tuning |
| `GGML_CUDA_DISABLE_FUSION=1` | upstream | disables all CUDA fusion, including the patched rules of 18, 19, 20, 23, 24, 27 |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | upstream | disables graph reuse (28.9 -> 18.1 t/s tg here, -27%; pp unchanged) |

No kill switch: 01, 02, 04, 05, 10, 11, 14, 16, 17, 21, 25, 29, 30 (and the MoE-only 03,
06-08). To bisect one of those, build with the commit dropped (`git rebase -i` or
`git revert`). 21 is a scheduler patch (`ggml-backend.cpp`, lazy `ggml_backend_sched_reset`)
next to where the meta backend crash lived; it was on the branch for the 89k validation run
above, so it is cleared for `-sm tensor`, but it is the first one to drop if scheduler-side
symptoms come back.

Graph reuse is a CPU-side optimization (skips graph rebuild and the meta backend's re-split
across devices); it has nothing to do with KV/prompt caching, which is `cache_prompt` plus
LCP slot matching in the server.

## Updating to a new upstream

The branch is a linear commit series, so updating is one rebase:

```sh
git fetch origin --tags
git rebase --onto <new-tag> <old-base> p100     # old-base: the "Base:" commit above
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
git format-patch --no-numbered --zero-commit -o patches/ <base>..p100
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

Output vs. stock: the only patches that change numbers are 03 (MoE, above one row), 12
(Q4_1 weights, n >= 2) and 15 (off unless `LLAMA_MTP_DRAFT_VOCAB` is set). None of them fire
for Q4_K_M/Q6_K `qwen35` without speculative decoding, so a greedy run (`--temp 0 --seed 1`)
of the same prompt through this build and a stock build of the same base commit is expected
to be identical to the last token, not just similar. Any divergence is a bug; bisect at
runtime with the kill switches above (start with `GGML_CUDA_DISABLE_FUSION=1`, then
`LLAMA_SAMPLER_PREFILTER=0`) before rebuilding anything.

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
