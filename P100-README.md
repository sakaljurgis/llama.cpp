# P100 quick reference (2x Tesla P100 on krk-lab)

Branch `p100-b11436` = upstream master at 6753a033f (b11436 + 3) + the shinbunbun P100 patch set
(01-30, three dropped as superseded) + the philpax meta-backend gist + local patches `p100x` 31-45.
Measured on the box against a stock build of the same base: Qwen3.8-27B UD-Q4_K_M tg 32.45 vs 22.12 t/s
(1.47x), pp2048 453 vs 445 t/s at `-ub 2048`, MTP chat 47-64 t/s; Gemma 4 31B UD-Q4_K_XL tg 29.3 vs
18.0 t/s (1.62x). The previous branch `p100-b10758` is kept unchanged as the fallback.

Docs: `P100-PATCHES.md` (what every patch does, knobs, rebase recipe, test checklist),
`P100-OPTIMIZATION-PLAN.md` (measurements; section 0.1 machines and paths, section 7 the recommended
production configuration, section 5 the gotchas, roadmap "Deferred code items" for what is left).
`~/p100-opt/` on the server is agent scratch space and can be deleted; the repo holds everything.

## 1. Build production from this branch (one time)

```sh
cd /home/krk/llama.cpp
git status                                     # clean first; stash or commit anything local
git fetch origin p100-b11436:p100-b11436       # origin = github.com/sakaljurgis/llama.cpp (https)
git switch p100-b11436
mv build build-p100-b10758                     # keep the old binaries as a fallback
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_NCCL=ON -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j14
```

Same flags as `~/llama-build.txt` plus an explicit Release. cmake 3.22 needs the explicit arch 60
(no `native`). NCCL is required (`libnccl-dev` is installed). About 10-15 minutes at `-j14`.
Fallback at any time: `mv build build-p100-b11436 && mv build-p100-b10758 build`.

## 2. Smoke test before serving (5 minutes)

```sh
export CUDA_VISIBLE_DEVICES=GPU-caf732cd-6831-4ec5-61b0-2e6fc172b1ee,GPU-b86ac28c-57b4-7387-c86f-79fc17819a1a
export NCCL_P2P_LEVEL=SYS                      # as in ~/llama-serve.sh; without it pp512 drops ~6%
M=/mnt/hdd/gguf/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf
./build/bin/llama-bench -m $M -ngl 99 -sm tensor -fa on -p 512 -n 64 -r 2
GGML_A16K_CHECK=1 ./build/bin/llama-completion -m $M -p "Hello, who are you?" -n 64 -no-cnv \
    -ngl 99 -sm tensor -fa on -c 4096 --temp 0 -s 1
```

Expected: tg64 about 32.3 t/s (tg128 measured 32.45), sensible text, and no `a16k_check:` line on stderr (that line means a
non-finite matvec output). `llama-completion` needs `-c 4096` or it takes the 262k training context
and runs out of memory. The pp512 number here is at `-ub 512` and is not the production pp figure.
`NCCL_P2P_LEVEL=SYS` matters here: on 2026-10-06 the production build gave pp512 256.6 t/s with it and
241.3 without (tg64 32.1 either way), so a pp512 near 241 means the variable is missing, not a bad build.

## 3. Serve

`~/llama-serve.sh` points at `/home/krk/llama.cpp/build/bin/llama`, which is the new build after
step 1, so nothing in it has to change for the update. As read on 2026-10-06 it runs `llama serve`
(router, `--models-dir`) with `NCCL_P2P_LEVEL=SYS --split-mode tensor -fa on -b 2048 -ub 2048
-c 149000 --parallel 1 --cache-ram 16384 --jinja`, and its MTP lines are comments. The recommended
configuration (plan section 7) adds `--spec-type draft-mtp --draft-p-min 0` at the default draft width
3 (J5: +49% on chat follow-ups). Stale comment lines in the script: draft width 4 and `ngram-mod` were
measured and rejected; `-ctk/-ctv q8_0` is not supported on the tile attention path.

What b11436 changes at the serving level (P100-PATCHES.md, rebase section): `preserve_reasoning` is on
by default (earlier turns' reasoning stays in the prompt; `--no-reasoning-preserve` for the old
behaviour); context checkpoints are kept until the list is full, so a long chat holds up to
`--ctx-checkpoints` (default 32) x 150 MB of host memory per slot, 5.6 GB measured; `--mmap`,
`--no-mmap`, `--mlock` and `-dio` no longer exist (`-lm/--load-mode`), the script uses none of them.

No GPU or host state changes are needed or useful: no power limit (tg draws ~155 W per card and is
flat down to a 175 W cap; only pp is clipped), persistence mode is a no-op on this host, stock
governor and C-states. Do not set compute mode EXCLUSIVE_PROCESS: the router runs one child per model.

## 4. New model checklist

Before serving a GGUF the branch has not run (plan gotcha 31): run the `GGML_A16K_CHECK=1` sample from
step 2 with that model (`--jinja` instead of `-no-cnv` for chat models with a strict template), then
compare a greedy chat reply against the same command with `GGML_CUDA_DISABLE_MMVQ_F16_K=1
GGML_CUDA_FATTN_TILE_LEGACY=1` (stock math). The two replies should read the same; byte-identical is not required (F16 accumulation). If
anything looks off, serve with that variable set (costs ~8% tg) and note it.

## 5. Kill switches worth knowing (full table in P100-PATCHES.md, "Runtime knobs")

| symptom | set |
|---|---|
| garbage or nan output on a model | `GGML_CUDA_DISABLE_MMVQ_F16_K=1` (HFMA2 K-quant matvec off, exact int8 path) |
| attention wrong at long context | `GGML_CUDA_FATTN_TILE_LEGACY=1` (patches 36/38 off) |
| follow-up requests misbehave | `LLAMA_SERVER_CKPT_LEGACY=1`, `LLAMA_SERVER_CKPT_SAVE_LEGACY=1` (patches 31/37 off) |
| abort inside cuBLAS on a wide batch | `GGML_CUDA_CUBLAS_CHUNK_MB=0` (patch 44 off) or a smaller value |
| sampler oddities with grammars or tools | `LLAMA_SAMPLER_PREFILTER=0` |
| any fusion suspicion | `GGML_CUDA_DISABLE_FUSION=1` (all CUDA fusion rules, upstream and patched) |
| the meta-backend assert returns | `LLAMA_GRAPH_REUSE_DISABLE=1` (escape hatch only: -27% tg) |

## 6. Updating to a newer upstream

`P100-PATCHES.md`, "Updating to a new upstream": the branch is a linear series, so it is one
`git rebase` onto the new master, then the test checklist there. Rebase patches 17/19/40/42 as a
group (one kernel); 23/24 sit in `ggml_cuda_try_fuse`, which upstream reworks now and then. After
every rebase check the `launch_fattn` call of patch 36 against upstream's parameter list (b11436 added
`use_sparse` in front of `warp_size` and a clean merge compiled into a wrong call), and run
`test-llama-archs` several times (patch 45's bug crashed only some runs).

## 7. What is left

Plan roadmap "Deferred code items": E2 (0.4% tg), D4 (0.4%), the delta-net conv chain (0.8%), F9
(unverified), and the Tier 3 research items. Everything above 1% that remains sits inside the matvec
kernels and NCCL. Rejected with numbers so nobody repeats them: B2, E7, E8, G2/G3, F5, C3, C4.
