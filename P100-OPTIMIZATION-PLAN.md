# P100 optimization plan

Plan for an implementing agent (Claude Opus / Sonnet class) working on branch `p100-b<build>`
of this fork. Two work packages:

1. Research the Tesla P100 hardware and the host machine, and write the results down as a fact
   sheet that later kernel work can rely on.
2. Walk the llama.cpp code paths that run on this machine, find every place where prompt
   processing (pp) or token generation (tg) can be made faster, including the PCIe traffic between
   the two cards, and turn the findings into measured, kill-switchable, one-commit-per-change
   patches.

Read `AGENTS.md`, `P100-PATCHES.md` and this file before touching anything. `P100-PATCHES.md`
describes what is already on the branch (25 upstream-independent P100 patches + a meta backend fix);
do not redo that work and do not fight it.

## 0. How to work

### 0.1 Machines

Two machines are involved. Every step in this plan is tagged:

- `[WS]` workstation: this checkout, no GPU, no nvcc. Reading code, editing, host-only builds
  (`build-cpu`), git work.
- `[SRV]` GPU server: HP Z440, 1x Xeon E5-2690 v4 (14C/28T, Broadwell-EP, AVX2, no AVX-512),
  32 GB DDR4 in 4 channels, 2x Tesla P100-PCIE-16GB (sm_60), CUDA 12.6, NCCL present. All CUDA
  builds, tests and measurements run here.

Filled in 2026-09-03 (confirmed with the user; re-check paths if a hash changes):

```
SRV_SSH="ssh -o BatchMode=yes krk-lab"        # 192.168.1.142, user krk, key auth; agents run on the workstation and ssh in
SRV_ROOT=~/p100-opt                            # the ONLY place agents may write on the server
SRV_REPO=~/p100-opt/llama.cpp                  # branch p100-b10758 (pushed from the workstation over the LAN; GitHub anonymous clones are rate-limited there)
SRV_STOCK=~/p100-opt/llama.cpp-stock           # worktree at the base commit b81c99b47, build in build/
SRV_BUILD=~/p100-opt/llama.cpp/build           # built by ~/p100-opt/build-all.sh, log in ~/p100-opt/build.log
CUDA=/usr/local/cuda-12.6                      # export PATH=/usr/local/cuda/bin:$PATH; cmake 3.22 (no arch "native"), gcc 11.4, nsys 2024.5.1, nvprof, NCCL 2.30.7, driver 580.173.02
GPUS="CUDA_VISIBLE_DEVICES=GPU-caf732cd-6831-4ec5-61b0-2e6fc172b1ee,GPU-b86ac28c-57b4-7387-c86f-79fc17819a1a"   # the two P100s; index 2 is a Quadro K2200 display card and must stay excluded
M=/mnt/hdd/gguf                                # HF cache layout (LLAMA_CACHE), files are symlinks into blobs
MODEL_Q4=$M/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf     # 16.5 GB, production
MODEL_Q6=$M/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q6_K.gguf       # 22.0 GB
MODEL_IQ4=$M/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf    # 14.3 GB (patches 29/30 territory)
MODEL_GEMMA=$M/models--unsloth--gemma-4-31B-it-GGUF/snapshots/c1ac76e99d5513b141e8adde7288b85c3f9c32ec/gemma-4-31B-it-UD-Q4_K_XL.gguf  # 18.8 GB
MODEL_MOE=$M/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf  # 22.1 GB MoE (07/08/09 live)
SERVER_FLAGS: router `llama serve --host 0.0.0.0 --port 8080 --models-dir /mnt/hdd/gguf --tools all --split-mode tensor --cache-ram 16384 --jinja -ngl 99 -c 160000 --parallel 1 -fa on -b 2048 -ub 2048 --no-mmproj-offload --spec-type draft-mtp --draft-p-min 0`
              started by ~/llama-serve.sh (also sets LLAMA_CACHE=/mnt/hdd/gguf NCCL_DEBUG=INFO and the GPUS line above); child per model adds --ctx-size 160000 --cache-ram 16384 --batch-size 2048 --ubatch-size 2048
              production binary: /home/krk/llama.cpp/build/bin/llama (branch `p100`, old b10630-based tip; do not touch)
SUDO=no                                        # no clock/power/persistence/IOMMU changes possible; record and recommend only
PROD_PAUSE=not-needed                          # user 2026-09-03: production is not in use; agents may stop a `llama serve` process if one appears and must NOT restart it (no `llama-serve.sh` runs by agents)
NUMERICS=perplexity                            # user 2026-09-03: changes may alter low-order bits if llama-perplexity stays within 0.5% of stock and P100-PATCHES.md documents it
MODELS=Qwen3.8-27B Q4_K_M (primary), Gemma-4 31B UD-Q4_K_XL, Qwen3.6-35B-A3B MoE (secondary)
```

Server rule from the user: do not change settings or files on krk-lab outside `~/p100-opt/`
(no nvidia-smi state changes, no sysfs, no edits to `~/llama.cpp` or the scripts). Ask first.

### 0.2 Rules

- One change per commit, subject `p100x: NN-short-name` (numbering continues from 31 so the
  numbers never collide with the upstream patch repo's 01-30). Every commit note goes into
  `P100-PATCHES.md` (status table + knobs table + change log) in the same commit.
- Never `git push`, never open PRs or issues, never write commit messages for the user without
  being asked; the user commits or approves each commit (see `AGENTS.md`). Ask before each commit.
- Every new fusion rule, kernel or heuristic gets a kill switch env var (`GGML_CUDA_DISABLE_...`
  or `LLAMA_..._DISABLE`), the same way patches 18-27 do. No exceptions: that is how regressions
  are bisected at runtime without rebuilding.
- ASCII only in code, comments and docs (no unicode dashes, arrows, ellipses). Short comments.
- A change is "done" only when: `test-backend-ops` passes, the greedy-determinism check
  (section 4.2) passes, `llama-bench` A/B shows the gain with the repetitions given in section
  4.1, and the doc is updated. Numbers without the protocol in section 4 do not count.
- Measure first, change second. Most items below start with a measurement that decides whether
  the code change is worth doing. Skip a code change whose measurement says "no gain"; record
  the negative result in `P100-PATCHES.md` so nobody repeats it.
- Prefer small, local changes. Do not add new subsystems. If a change needs more than ~300
  lines, stop and present the design to the user first.
- Stock reference: keep a stock build of the same upstream base (`build-stock/` from
  `origin/master` at the branch base commit) on the server for A/B and determinism checks.

### 0.3 Glossary

- pp: prompt processing, batched forward pass, compute bound (GEMM). Measured in tokens/s
  at a given batch and context depth.
- tg: token generation, one token per step, memory-bandwidth bound (matvec) plus per-step
  fixed costs (kernel launch, host overhead, inter-GPU sync).
- ubatch: micro batch (`-ub`), the number of tokens one graph evaluation processes.
- `-sm tensor`: tensor split via the meta backend (`ggml/src/ggml-backend-meta.cpp`), both GPUs
  work on every layer. `-sm layer`: layers split between GPUs, pipeline style. `-sm row`: the
  older row split in `ggml-cuda.cu`.
- MMQ: quantized GEMM kernel (int8 dot products). MMVQ: quantized matvec kernel (tg). MMVF:
  float matvec. FA: flash attention. GDN: gated delta net (the recurrent layers of `qwen35`).

### 0.4 Deliverables

1. `P100-HARDWARE.md`: the fact sheet from Part 1 (measured numbers, not just datasheet values).
2. `P100-OPTIMIZATION-LOG.md`: one entry per experiment (date, commit or flags, command lines,
   numbers, verdict). Negative results included.
3. Commits `p100x: 31-...` onward on the branch, each documented in `P100-PATCHES.md`.
4. An updated recommended `llama-server` command line for the user's workload.

## 1. Part 1: hardware research

Goal: know exactly what the two cards and the host can do, with measured numbers, so that kernel
work in Part 2 targets real limits instead of guesses. Write everything into `P100-HARDWARE.md`.
Every value below marked "verify" must be confirmed on the server; datasheet values are
starting points only.

### 1.1 Tesla P100 PCIe 16 GB: what it is

Datasheet values (verify each with `deviceQuery` / `nvidia-smi -q`):

| Item | Value | Why it matters here |
|---|---|---|
| GPU | GP100, compute capability 6.0 (`sm_60`) | the only discrete Pascal with fast FP16; NO DP4A (that is sm_61: GP102/104/106) |
| SMs / cores | 56 SMs x 64 FP32 cores = 3584 | 56 SMs: grid sizes of ~56*k blocks fill the chip; small matvec grids under-fill it |
| Clocks | measured on krk-lab: default application clock 1189 MHz, max 1328 MHz (memory 715 MHz) | passive card; power capping has fired briefly, thermal slowdown never (see 1.5) |
| FP32 | 9.5 TFLOPS at 1328 MHz (3584 x 2 x 1.328e9) | ceiling for F32 GEMM / F32 matvec math |
| FP16 (HFMA2) | 19.0 TFLOPS at 1328 MHz | 2x FP32, only via `half2` ops (`__hfma2`); scalar `half` math gets no speedup |
| FP64 | 4.7 TFLOPS | irrelevant |
| INT8 | no DP4A/DP2A | quantized dot products need emulation (patch 01 `vmad`), int GEMM is not competitive |
| Tensor cores / MMA | none | `mmf.cu`, `fattn-mma-f16.cuh`, `fattn-wmma` never run here |
| Memory | 16 GB HBM2, 4096-bit, 732 GB/s | tg ceiling: bytes of weights read per token / 732 GB/s per card |
| ECC | always on, native HBM2 ECC | no bandwidth or capacity penalty; cannot be turned off |
| L2 | 4 MB | small; activation reuse across kernels is not cached across a 27B layer |
| Shared memory | 64 KB per SM, 48 KB max per block, no opt-in above 48 KB | kernels tuned for 96-228 KB smem (Volta+) cannot run; tile sizes are limited |
| Registers | 64K 32-bit per SM (256 KB), 255 max per thread | occupancy: 2048 threads/SM only if <= 32 regs/thread |
| Threads | 2048 threads/SM, 32 blocks/SM, 64 warps/SM, warp 32 | |
| L1/texture | per-SM unified L1/tex (verify size) | `__ldg` / read-only path helps dequant kernels |
| Copy engines | verify `asyncEngineCount` (expect 2) | needed for compute/copy overlap of P2P and H2D traffic |
| PCIe | Gen3 x16, ~15.75 GB/s raw, 11-13 GB/s achievable per direction | inter-GPU and host-GPU ceiling |
| NVLink | none on the PCIe card | all inter-GPU traffic is PCIe |
| BAR1 | 16 GB in compute mode (P100 PCIe product brief); needs "Above 4G decoding" in BIOS | full-size BAR1 lets peers map all of GPU memory for P2P; 256 MB means the BIOS setting is off |
| Power | 250 W TDP, 8-pin EPS (CPU-style) connector | 2 cards + CPU vs the Z440 700 W PSU; power capping shows as clock drops |
| Cooling | passive | needs chassis airflow; thermal slowdown is the most likely silent perf killer |
| Unified memory | page migration engine, 49-bit VA | do not use for inference; on-demand paging over PCIe is catastrophically slow |
| Compute preemption | yes | irrelevant |

Instruction-set facts the kernel work must respect (verify the ones marked with a
microbenchmark in 1.8 if a kernel design depends on them):

- `__hfma2`/`__hadd2`/`__hmul2` on `half2` run at 2x FP32 rate. Conversions `__float2half2_rn`
  and `__half22float2` cost ALU slots; a kernel that converts every operand loses the 2x.
- No `__dp4a`, no `__dp2a`. `ggml_cuda_dp4a` in `common.cuh` is emulated on sm_60 (patch 01
  replaced upstream's emulation; read that commit). Cost per emulated dp4a: measure (1.8).
- 32-bit integer multiply is not full rate on Pascal (XMAD based), integer shifts/logic
  (`LOP3`, funnel shift `__funnelshift_*`, `__byte_perm`) are; bit unpacking should favor
  shifts, masks and `__byte_perm` over multiplies.
- Warp shuffles (`__shfl_*_sync`), warp votes, shared-memory atomics (native since Maxwell),
  `atomicAdd` on `half2` and `double` are available.
- No independent thread scheduling (Volta+): divergent code inside a warp serializes fully;
  `__syncwarp` is a no-op-ish hint. Do not write code that relies on Volta forward progress.
- No `cp.async` (sm_80), no `ldmatrix` (sm_75), no `mma`/`wmma` (sm_70+). `cp_async_available`
  style helpers in `common.cuh` are false here.
- Cooperative groups grid sync (`cudaLaunchCooperativeKernel`) is supported on cc 6.0.
- Max 48 KB dynamic shared memory per block; `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize)`
  above 48 KB fails.

### 1.2 Host: HP Z440 + E5-2690 v4 + 32 GB

| Item | Value | Why it matters |
|---|---|---|
| CPU | E5-2690 v4, 14C/28T, 2.6/3.5 GHz, 35 MB L3, AVX2+FMA3, no AVX-512 | CPU-side sampling and tokenization; `GGML_NATIVE=ON` gives AVX2 host kernels |
| Memory | 4x DDR4-2400 (verify populated channels and speed), 76.8 GB/s theoretical | model load, host prompt cache (`--cache-ram`), pinned staging |
| RAM size | 32 GB | 27B Q4_K_M file ~16.5 GB + page cache + host caches; do not run `--no-mmap` and a large `--cache-ram` together |
| PCIe lanes | 40 lanes Gen3 from the CPU; one root complex (single socket) | both cards share one root complex: P2P is possible, both x16 only if they sit in the two x16 slots |
| Z440 slots | slot 1 PCIe2 x1, slot 2 PCIe3 x16 (primary graphics), slot 3 PCIe2 x4, slot 4 PCIe3 x8 (x16 connector), slot 5 PCIe3 x16 (secondary graphics), slot 6 legacy PCI (HP Z440 QuickSpecs / service guide; confirm with `dmidecode -t slot`) | both cards must be in slots 2 and 5; a card in slot 4 runs at x8 |
| PSU | 700 W | 2x250 W + 135 W CPU + rest is close to the limit; power capping may be active |
| NUMA | one node | `--numa` flags are irrelevant |

### 1.3 Software stack facts (verify versions on the server)

- CUDA 12.x is the last toolkit line with sm_60 (CUDA 13.0 removed Maxwell/Pascal/Volta). Stay
  on 12.6-12.9. Later 12.x minors print a deprecation warning for `sm_60`; harmless.
- Driver: the R580 branch is the last driver branch that supports Maxwell/Pascal/Volta (NVIDIA
  data center driver release notes; Phoronix 2025-07). R580 itself runs CUDA 12.x toolkits fine.
  Record the `nvidia-smi` driver version. Do not upgrade past 580.x.
- Profilers: Nsight Compute (`ncu`) supports Turing and newer only (its GPU support page lists
  Maxwell, Pascal and Volta as unsupported) and will NOT profile the P100. Use `nsys` (Nsight
  Systems still traces Pascal) for timelines and kernel durations, and the legacy `nvprof`
  (verify `which nvprof`; it ships with CUDA 12.x toolkits and supports Pascal) for per-kernel
  hardware counters (achieved occupancy, dram throughput, `--metrics`). Build with
  `-DCMAKE_CUDA_FLAGS=-lineinfo` for source correlation.
- cuBLAS on sm_60: `cublasGemmEx` with `CUBLAS_COMPUTE_16F` uses HFMA2 (18.7 TFLOPS class);
  `CUBLAS_COMPUTE_32F` on F16 inputs runs at FP32 rate. Which one llama.cpp picks decides pp
  speed (Part 2, item A1). cuBLASLt availability on sm_60: verify with a tiny test if item A
  needs it.
- NCCL: present on the server (log shows `libnccl-*.so` plugin lookups). Record version. Whether
  the meta backend uses it is a Part 2 question (item F1).
- cuda-samples (`deviceQuery`, `bandwidthTest`, `p2pBandwidthLatencyTest`, `simpleP2P`) are
  needed for 1.4-1.8. Clone github.com/NVIDIA/cuda-samples at a tag matching CUDA 12.x and build
  only those four (each has its own CMakeLists / Makefile).

### 1.4 PCIe and P2P topology (the inter-card link)

Run and record `[SRV]`:

```sh
nvidia-smi -q                                    # full dump, keep it
nvidia-smi topo -m                               # expect PHB between GPU0 and GPU1 (same root complex)
nvidia-smi topo -p2p r; nvidia-smi topo -p2p w; nvidia-smi topo -p2p n   # P2P read/write/NVLink capability matrix
nvidia-smi -q -d MEMORY                          # BAR1 size (want 16384 MiB), FB usage
lspci -vv -s <bdf of each GPU> | grep -E 'LnkCap|LnkSta|Slot'   # want "Speed 8GT/s, Width x16" in LnkSta, both cards
lspci -tv                                        # tree: which root port each card hangs on
sudo dmidecode -t slot                           # which physical slot each card is in, slot lane width
cat /proc/cmdline                                # iommu settings
dmesg | grep -iE 'iommu|dmar|acs|nvidia'         # IOMMU state, NVIDIA driver messages
lstopo --of txt 2>/dev/null || lstopo-no-graphics
```

Then measure:

```sh
./p2pBandwidthLatencyTest          # P2P enabled vs disabled bandwidth + latency matrices; keep the whole output
./bandwidthTest --memory=pinned --mode=range --start=1024 --end=67108864 --increment=1048576 --device=0
./bandwidthTest --memory=pageable --mode=quick --device=0
./simpleP2P                        # confirms cudaDeviceCanAccessPeer
```

What to look for and the gotchas:

- `LnkSta` at x8 or 5GT/s on either card: wrong slot or a degraded link. Fix the slot before any
  software work; it is worth more than any kernel patch for `-sm tensor`.
- `nvidia-smi topo -p2p r/w` showing `NS` (not supported) between the two cards while the topology
  is PHB: usually IOMMU/ACS. Try `intel_iommu=off` (or `iommu=pt`) in the kernel command line; on a
  single-socket Z440 IOMMU off is fine unless VMs are needed. Re-measure.
- Expected P2P numbers over one Gen3 root complex: 10-12 GB/s unidirectional, ~20 GB/s
  bidirectional aggregate, latency 2-5 us with P2P on, 10+ us with P2P off (bounce through host).
  Record the exact numbers; Part 2 item F uses them to estimate the communication share per token.
- With P2P disabled by the driver, `cudaMemcpyPeer` falls back to a host bounce (two PCIe
  transfers). That halves bandwidth and adds latency; the tg impact under `-sm tensor` is item F2.
- BAR1 smaller than 16 GB (e.g. 256 MB) means "Above 4G decoding" is off in the BIOS; P2P then only
  maps a window and the driver may refuse peer access. Enable it in BIOS.
- Pinned host memory: `bandwidthTest` pinned vs pageable gap tells whether host staging matters
  (model load, logits copies). Expect ~12 GB/s pinned, 3-6 GB/s pageable.

### 1.5 Power, thermals, clocks

The P100 is passive. In a Z440 it depends on the chassis fans. Do this before any benchmark:

```sh
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE     # current clocks, limits, throttle reasons
nvidia-smi -q -d SUPPORTED_CLOCKS | head -40             # app clocks available (P100: mem 715 MHz, gfx up to 1328)
sudo nvidia-smi -pm 1                                    # persistence mode: no driver reload latency
sudo nvidia-smi -ac 715,1328                             # application clocks at max (if supported on this SKU; else skip)
nvidia-smi dmon -s pucvmet -d 1                          # during a benchmark: power, util, clocks, violations, mem, temp
```

Gotchas:

- "Clocks Throttle Reasons": `SW Thermal Slowdown`, `HW Slowdown`, `SW Power Cap` during a run
  mean every benchmark number is a function of chassis temperature or power, not of the code.
  Measured 2026-09-03 (`P100-HARDWARE.md`): thermal slowdown has never fired on this box (idle
  34-39 C vs 82 C threshold), SW power capping has (about 0.5 s accumulated per card). So log
  power and clocks with every measurement and warm up 60 s; cooling is not the problem here.
- Two cards at 250 W each on a 700 W PSU: if `SW Power Cap` appears with `-sm tensor` pp, consider
  `nvidia-smi -pl 200` on both cards. tg is bandwidth bound and barely notices; pp loses some. Measure.
- `nvidia-smi -lgc` (lock clocks) is Volta+ only; on Pascal use application clocks (`-ac`).
- Memory clock is fixed at 715 MHz on HBM2; only the graphics clock moves.
- Record idle and load temperatures. P100 slowdown threshold is in the `TEMPERATURE` query.

### 1.6 Host settings to record and fix

```sh
lscpu; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # want performance
sudo dmidecode -t memory | grep -E 'Size|Speed|Locator'             # 4 sticks, 2400 MT/s, one per channel
free -g; cat /sys/kernel/mm/transparent_hugepage/enabled
nproc; ulimit -l                                                    # for --mlock
```

- Governor `performance` for benchmarks (`cpupower frequency-set -g performance`); krk-lab runs
  `schedutil` and has no passwordless sudo, so this is a recommendation to the user.
- `ulimit -l` is 3.9 GiB on krk-lab: `--mlock` / `--load-mode mlock` on a 16.5 GB file will fail
  unless the user raises the limit (`/etc/security/limits.conf`). Do not plan on it.
- If only 2 channels are populated the theoretical host bandwidth halves; only matters for
  model load and `--cache-ram`.

### 1.7 Hardware fact sheet template (`P100-HARDWARE.md`)

Sections: (1) identification: driver, CUDA, NCCL, kernel, BIOS version; (2) `deviceQuery`
dump; (3) PCIe topology and link state per card, slot numbers; (4) P2P and H2D/D2H bandwidth and
latency tables (from 1.4); (5) clocks/power/thermal behavior under a 5 minute `llama-bench` run
(min/max clock, throttle reasons seen); (6) host memory config; (7) microbenchmark results (1.8);
(8) a "limits" summary: per-card bytes/s and FLOP/s the code can hope for, PCIe bytes/s and
us-latency per transfer, and derived ceilings for the user's model: tg ceiling =
2 cards * 732 GB/s / bytes-of-weights-per-token; pp ceiling = 2 * 18.7 TFLOPS / FLOPs-per-token.

### 1.8 Microbenchmarks (only the ones a Part 2 item depends on)

Write them as single-file `.cu` programs in a scratch dir on the server (not in the repo):

1. HFMA2 vs FFMA throughput and the cost of `half2 <-> float2` conversion (decides how a
   K-quant HFMA2 matvec should be structured, item B3).
2. `ggml_cuda_dp4a` emulation cost on sm_60 vs a 4x `fmaf` on converted values vs `__hfma2` on
   packed halves (decides whether an F16-accumulating vec_dot beats the int8 emulation, item B3).
3. Memory copy kernel bandwidth: how close a simple `float4` copy gets to 732 GB/s at grid sizes
   of 56, 112, 224, 448 blocks (calibrates the tg matvec target).
4. P2P: `cudaMemcpyPeerAsync` for 32 KB - 4 MB messages vs a kernel that reads peer memory
   through a mapped pointer and reduces in place (decides item F3: allreduce design).
5. Small-transfer latency: 10k back-to-back `cudaMemcpyPeerAsync` of 20 KB with an event wait
   each (this is the pattern of per-layer tensor-parallel reductions at tg).
6. Kernel launch overhead: 10k empty launches on one stream with and without a CUDA graph
   (calibrates the per-token fixed cost, item G1). Expect ~3-5 us per launch, <1 us in a graph.

## 2. Part 2: codebase walk and optimization catalogue

### 2.0 Where the time goes today (working model, to be confirmed by the Tier 0 measurements)

Numbers from `P100-PATCHES.md` (Qwen3.8-27B Q4_K_M, 2x P100, `-sm tensor -fa on -ub 2048`):
pp 475 t/s at 3.7k ctx falling to ~270 t/s at 89k; tg 28.9 t/s falling to 22.8 t/s; graph reuse
off costs -27% tg; a fixed ~1 s prompt phase on short follow-up requests.

Token generation, 34.6 ms per token at short context:

| Component | Estimate | Basis |
|---|---|---|
| Weight streaming floor | ~11 ms | 8.3 GB per card per token at 732 GB/s |
| Cross-GPU reductions | 2.5-5 ms | 2 per layer x 64 layers = 128 collectives of ~20 KB, 20-40 us each with NCCL |
| Kernel launch and host overhead | unknown, likely 10+ ms | CUDA graphs are disabled below Volta (area G); ~1000 launches per card per token from one host thread; graph reuse off costs 9 ms which shows how host-bound the step is |
| Attention (4k) + delta-net + glue | remainder | grows to ~6 ms extra at 89k |

Prompt processing, 4.3 s per 2048 tokens at short context: cuBLAS F16 GEMM already at ~68% of the
HFMA2 peak; the losses are dequant round trips, the F16 -> F32 output pass, collectives of 42 MB
each, and at long context the flash-attention tile kernel running at roughly 40% of peak.

Measured 2026-09-03 on krk-lab (`P100-HARDWARE.md`, sections 7-8), which corrects the table above:
read-only device bandwidth tops out at 603 GB/s (82% of 732), so the weight floor is ~13.8 ms per
token, not 11; a plain kernel launch costs 1.57 us and 1.18 us inside a CUDA graph, so ~1000
launches are ~1.6 ms and graphs alone cannot explain the gap; a 20 KB peer transfer with event
sync is 3.4 us with P2P on vs 13.3 us off; `__hfma2` delivers 3.4x the MAC rate of the emulated
`dp4a`. Consequence: the llama-level `nsys` breakdown (B1/G1) decides, with MMVQ achieved
bandwidth (B3) the first suspect for the missing ~20 ms, ahead of launch overhead.

So the three big levers, in order of expected payoff (revise after Tier 0):

1. Launch and host overhead at tg (area G, then E/D fusions): CUDA graphs on Pascal.
2. Attention kernel tuning for long contexts (area C): untuned tile table, VEC vs TILE, parallel blocks.
3. Per-request overhead (area J): the KV serialization to host RAM on each new task is the
   prime suspect for the ~1 s.

Then the medium levers: communication transport and tuning (F), HFMA2 K-quant matvec (B4),
dequant kernels (A2), memory headroom (A4/H).

### 2.1 Code map

| Area | Files (entry points) | What runs there on this machine |
|---|---|---|
| A dense matmul (pp) | `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat`, `ggml_cuda_mul_mat_cublas_impl`), `convert.cu`, `mmq.cu` (`ggml_cuda_should_use_mmq`) | dequant-to-F16 + cuBLAS 16F for every quantized GEMM with 9+ columns |
| B matvec (tg) | `mmvq.cu`, `vecdotq.cuh`, `mmvf.cu`, `mmvq-f16-sm60.cu`, `quantize.cu` | MMVQ with emulated dp4a for 1-8 columns; patch 12 HFMA2 path for Q4_1 only |
| C attention / KV | `fattn.cu`, `fattn-tile.cuh`, `fattn-vec.cuh`, `fattn-common.cuh`, `src/llama-kv-cache*.cpp` | TILE kernel for pp and tg (F16 KV + GQA), untuned Pascal FP16 table |
| D delta-net | `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`, `gated_delta_net.cu`, `ssm-conv.cu` | 48 of 64 layers; fused GDN forced on |
| E fusion / elementwise | `ggml-cuda.cu` (`ggml_cuda_can_fuse`, `ggml_cuda_try_fuse`), `norm.cu`, `rope.cu`, `binbcast.cu`, `unary.cu`, `cpy.cu`, `concat.cu`, `getrows.cu` | upstream rules + patches 18-27 |
| F multi-GPU comms | `ggml/src/ggml-backend-meta.cpp`, `ggml-cuda.cu` (comm init, NCCL allreduce, P2P init), `allreduce.cu`, `ggml/src/ggml-backend.cpp` | NCCL allreduce (default), butterfly fallback, P2P opt-in, sync split-input copies |
| G graphs / host | `ggml-cuda.cu` (graph capture), `common.cuh` (`ggml_cuda_graph`), `src/llama-context.cpp`, `ggml-backend.cpp` (scheduler) | CUDA graphs off (arch gate), llama graph reuse on, one sched |
| H memory | `ggml-cuda.cu` (pools), `src/llama-context.cpp` (reserve), `src/llama-model.cpp` (split policy) | VMM pool high-water mark, 94% VRAM at load |
| I sampling / CPU | `src/llama-sampler.cpp`, `common/sampling.cpp`, `common/arg.cpp` | CPU sampling only (backend sampling refused under tensor split), patches 11/13 |
| J server | `tools/server/server-context.cpp`, `server-task.cpp`, `common/speculative.cpp` | prompt cache, host KV cache, checkpoints, speculative decoding |
| K loading | `src/llama-model-loader.cpp`, `src/llama-mmap.cpp` | mmap by default; pinned async upload only without mmap |
| L build | `ggml/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt`, `docs/build.md`, `docs/multi-gpu.md` | arch 60 must be explicit; NCCL on by default |

### 2.2 Tier 0: measurements before any code (`[SRV]`, 1-2 days)

Do all of these first and write the results to `P100-OPTIMIZATION-LOG.md`. They are referenced
by the items below.

1. Part 1 fact sheet (1.4-1.8), especially P2P bandwidth/latency and throttle behavior.
2. Standard benchmark matrix (4.1) on the branch build and the stock build.
3. `nsys` breakdowns: tg 64 tokens at 4k and 64k (G1, B1, F2, E1), pp 2048 at `-ub 512` and
   `-ub 2048` (A1). Note: CUDA tracing needs some free GPU memory; at 94% VRAM use a smaller
   `-c` for profiling runs.
4. NCCL transport check (F1) and the env sweeps F3/F4, including `GGML_CUDA_P2P=1`.
5. Fusion census: `GGML_CUDA_FUSE_LOG=2` on one decode step (D1, E2, E3).
6. Server per-request timeline for the ~1 s (J1), including the `--cache-ram 0` A/B.
7. Host settings: `-t`/`--poll` sweep (I4), `CUDA_SCALE_LAUNCH_QUEUES=4x` (G5), governor.
8. Load-mode timing (K1).

Only then pick code items, by the tiers in section 3.

### Area A: dense matmul for prompt processing (`ne11 >= 9`)

Files: `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat` ~:1820, `ggml_cuda_mul_mat_cublas`
~:1627, `ggml_cuda_mul_mat_cublas_impl` ~:1414), `ggml/src/ggml-cuda/convert.cu` (dequantize to
F16), `ggml/src/ggml-cuda/mmq.cu` (`ggml_cuda_should_use_mmq` ~:259), `ggml/src/ggml-cuda/common.cuh`
(arch helpers ~:298-360).

Current behavior on sm_60 (verified in code, 2026-09-03; re-check line numbers after a rebase):

- The dispatch order in `ggml_cuda_mul_mat` is: cuBLAS for non-F32 src1/dst, then MMVF, then MMF
  (never on sm_60: needs MMA), then MMVQ (`ne11 <= 8`), then MMQ, then cuBLAS. MMQ is refused for
  dense `MUL_MAT` on sm_60 (`highest_compiled_arch < DP4A` returns `n_experts > 0`), and
  `GGML_CUDA_FORCE_MMQ` is checked after that early return, so it cannot override it.
- So every quantized GEMM with 9+ columns is: dequantize the whole weight matrix to an F16
  temporary in the pool (`ggml_nelements(src0)` halves, per call), `cublasGemmEx` with
  `CUBLAS_COMPUTE_16F` (HFMA2, the fast path; `fast_fp16_hardware_available(600)` is true), F16
  output into a second temporary, then a `to_fp32` pass into dst (`prefer_f32_output` is false on
  sm_60). `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` switches to F32 compute (half the FLOPS).
- Dequantizers: only `Q8_0 -> F16` is vectorized (`dequantize_block_q8_0_f16`); every other type
  runs the generic `dequantize_block` with 2 elements per thread and scalar 2-byte stores, or
  64-thread K-quant blocks (one super block per block).
- `-use_fast_math` is already in the nvcc flags. The default arch list never contains `60`; the
  build must pass `-DCMAKE_CUDA_ARCHITECTURES=60` (or `native` on the server), otherwise
  `ggml_cuda_highest_compiled_arch(600)` maps to 500 and `fp16_available` turns false, which
  silently selects slower paths. Check the build log for `sm_60` / `compute_60`.

Ceiling: at 475 t/s (Qwen3.8-27B, 3.7k ctx) each card does ~12.8 TFLOP/s on GEMM, ~68% of the
18.7 TFLOPS HFMA2 peak at 1303 MHz and ~81% if the clock has sagged to 1100 MHz. The GEMM itself
has little headroom. The headroom is in everything around it and in the attention kernels at long
context (pp 475 -> 270 t/s from 3.7k to 89k is attention, area C).

Items:

- A1 `[SRV]` Measure the pp time budget with `nsys` at `-ub 512` and `-ub 2048`, 4k and 64k
  context: cuBLAS GEMM kernels vs `dequantize_block*` vs `to_fp32` (`convert_unary`) vs fattn vs
  NCCL/allreduce vs everything else. This decides A2-A5. Expected: dequant + conversions 5-15%
  at `-ub 512`, 2-4% at `-ub 2048`; attention grows with context.
- A2 Vectorize the dequant-to-F16 kernels for the types in use (Q4_K, Q6_K, Q5_K, Q8_0 already
  done; later Q4_K_XL types). Pattern: the existing `dequantize_block_q8_0_f16` (shared memory
  staging, `half2` stores). Target: one super block per warp, 8-byte stores (`ggml_cuda_get_max_cpy_bytes`
  is 8 on sm_60). Gain: bounded by A1's dequant share. Effort: small. Risk: low; `test-backend-ops`
  covers `CPY`/`MUL_MAT` for every type. Kill switch: `GGML_CUDA_DISABLE_DEQUANT_VEC`.
- A3 Skip the F16 -> F32 output pass when the consumer is the next GEMM or a fusable op: not
  possible without a graph rewrite; instead measure `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (F32
  compute, F32 output directly, no conversion pass, but half the FLOPS). Expected: slower; do it
  once to have the number and to see whether F16 accumulation costs accuracy (perplexity check,
  section 4.2). No code.
- A4 Bounded-memory GEMM for huge matrices (fixes the LM head OOM documented in
  `P100-PATCHES.md` runtime notes: the ~2.5 GB F16 copy of `output.weight` on the first 9+ column
  verify batch). In `ggml_cuda_mul_mat_cublas_impl`, when `ggml_nelements(src0) * 2` exceeds a
  threshold (env `GGML_CUDA_CUBLAS_CHUNK_MB`, default e.g. 512), loop over row chunks of src0:
  dequantize chunk, GEMM chunk into the matching rows of dst. Same math, same output. Gain: not
  speed, it removes the OOM and frees ~1.3 GB per card of pool headroom that today has to be
  held back via `--ctx-size`. That headroom is worth context or a second slot. Effort: medium.
  Risk: medium (batched GEMM variants `ne12/ne13 > 1` must keep the non-chunked path). Kill
  switch: the env var at 0.
- A5 Keep the weight F16 copy across the ubatches of one prompt: with `-b 2048 -ub 512` the same
  layer weights are dequantized 4 times per batch. Only pays at small `-ub`; the user runs
  `-ub 2048`, so first check with A1 whether smaller `-ub` is ever better (it frees pool memory
  and may raise the clock by lowering power). Effort: medium (cache keyed like patch 10's q8_1
  cache). Likely skip.
- A6 Experiment only: allow MMQ for dense on sm_60 by editing the rule in
  `ggml_cuda_should_use_mmq` (`n_experts > 0` -> `true`) in a scratch build and measure pp at
  `-ub 512/2048`. Expected: slower than cuBLAS F16 (int8 dot via 4x `vmad` per `dp4a`, ~0.6 MAC per
  instruction vs 2 MAC per HFMA2), but it uses no F16 temporary, so record it as the memory-saving
  fallback number. No commit unless it wins.
- A8 Small batches (9-64 columns) without the full dequant: measured 2026-09-03 (J4 notes), a
  28-token follow-up decode costs ~390 ms on the cuBLAS path (every weight matrix dequantized to
  F16 per call), while 8 columns through MMVQ cost ~35 ms. Loop MMVQ over column chunks of 8 for
  `ne11` 9..N on sm_60 (or raise the sm_60 MMVQ ceiling if the kernel allows) and find the
  crossover N where cuBLAS wins again with `llama-bench -p 8,16,24,32,48,64 -n 0`. Pays for every
  follow-up request (area J), for speculative verify widths above 8 (J5) and removes the LM head
  F16 copy at those widths (A4). Env: `GGML_CUDA_MMVQ_MAX_COLS_SM60`. Small change in the dispatch
  of `ggml_cuda_mul_mat` / `ggml_cuda_should_use_mmvq`; the kernel itself is unchanged.
- A7 Research (tier 3): a Pascal HFMA2 tiled GEMM that dequantizes Q4_K/Q6_K tiles into `half2`
  registers/shared memory and multiplies with `__hfma2`, replacing dequant + cuBLAS for `ne11`
  9..256. It removes the F16 round trip through HBM and the per-call temporaries. The bar is
  cuBLAS at ~68-80% of peak; a hand-written Pascal GEMM reaching that is a multi-week project.
  Only start it after A1 shows the round trip is a double-digit share of pp at the `-ub` the user
  actually runs. Design must respect 48 KB shared memory per block and 8-byte vector loads.

### Area B: matvec for token generation (`ne11 <= 8`)

Files: `ggml/src/ggml-cuda/mmvq.cu` (+ `mmvq.cuh`, `vecdotq.cuh`), `ggml/src/ggml-cuda/mmvf.cu`,
`ggml/src/ggml-cuda/mmvq-f16-sm60.cu` (patch 12), `ggml/src/ggml-cuda/quantize.cu` (q8_1 of the
activations), `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_should_fuse_mul_mat_vec_q` ~:1803).

Current behavior on sm_60:

- Quantized weights with `ne11 <= 8` go to MMVQ (`MMVQ_MAX_BATCH_SIZE 8`), generic parameter table
  (the tuned tables are Turing+). Warps: 4 for 1-4 columns, 2 for 5-8. Patch 02 raised rows per
  block to 4 for 1-4 columns (L2 re-read of the q8_1 activation was the limit, +23% tg). Patches
  29/30 stage IQ tables in shared memory (inert for K-quants). Patch 10 caches the q8_1
  activation so gate/up quantize once.
- The int8 dot product is emulated: `ggml_cuda_dp4a` on sm_60 is 4x PTX `vmad` (patch 01).
- Patch 12 gives Q4_1 a dedicated HFMA2 GEMV for widths 2-8; it is the only non-int8 quantized
  matvec on the branch and inert for the user's K-quant models.
- Upstream disables MMVQ gate/up fusion for `cc <= 600` ("not universally faster on Pascal");
  MMVF fusion has no such gate. Patch 05 lets F32 matvec (ssm alpha/beta, routers) stay in MMVF up
  to 8 columns instead of cuBLAS.
- Small-K widening (`should_use_small_k`) is disabled on `cc < VOLTA` for IQ3_S/Q2_K/Q3_K.

Ceiling: tg is bound by weight bytes per token. Qwen3.8-27B Q4_K_M is ~16.5 GB of weights -> ~8.3
GB per card per token -> 732 GB/s gives ~11 ms -> ~88 t/s if nothing else cost anything. Measured
28.9 t/s (34.6 ms per token). So 2/3 of each tg step is NOT weight streaming: kernel launch gaps,
small kernels, attention, delta-net, cross-GPU reductions, host overhead. The `nsys` breakdown in
G1 is the first thing to do for tg; matvec tuning alone cannot get more than the matvec's own share.

Items:

- B1 `[SRV]` `nsys` tg breakdown at 4k and 64k context: total per-token time; sum of
  `mul_mat_vec_q*` kernel time and its achieved bandwidth (bytes of weights / kernel time; target
  > 600 GB/s per card); attention kernels; GDN kernels; NCCL/allreduce kernels and memcpys; host
  gaps (time with no kernel running on either GPU). Write the table into the log. Decides B2-B5 vs
  F vs G.
- B2 Re-measure MMVQ gate/up fusion on sm_60: build with the `cc <= PASCAL` gate in
  `ggml_cuda_should_fuse_mul_mat_vec_q` removed, `llama-bench -n 64` A/B. Upstream's "not
  universally faster" predates patches 02/07/10 (rows per block and the activation cache change
  the economics). Effort: trivial. Kill switch: `GGML_CUDA_DISABLE_FUSE_MMVQ` if it stays in.
- B3 `[SRV]` MMVQ achieved bandwidth per type with `test-backend-ops -o MUL_MAT perf` at the
  model's real shapes (m = 5120/13824-ish rows, k = 5120, n = 1..8, types q4_K, q6_K, q5_K, q8_0):
  compare against microbenchmark 1.8.3 (copy kernel bandwidth). If MMVQ for Q4_K/Q6_K is under
  ~75% of the copy bandwidth at n = 1, the int8 emulation is compute bound and B4 is justified.
- B4 Extend the patch 12 approach (HFMA2 GEMV, nibble expansion with LOP3/`__byte_perm`, half2
  activations) from Q4_1 to Q4_K and Q6_K, widths 1-8. This is the single kernel project most
  likely to pay for tg on GP100, because HFMA2 gives 2 MACs per instruction where the emulated
  dp4a gives ~0.6. It changes numerics (F16 accumulation over a block, F32 across blocks; patch 12
  documents the same). Effort: large (Q4_K has 6-bit scales/mins per sub block, Q6_K has 8 sub
  blocks with 6-bit weights split across `ql`/`qh`). Validate with `test-backend-ops` (already
  covers Q4_K/Q6_K MUL_MAT at n = 1..8), then perplexity. Env knobs like `GGML_A16_*`. Do B3 first.
- B5 Patch 09 (`mmvq-nwarps-small-k-sm60`, deferred; +0.12% dense) stays deferred unless an MoE
  model shows up. Recheck the GENERIC `calc_nwarps` values 4/2 once with `GGML_A16`-style env
  overrides in a scratch build (nwarps 2/4/8 for n = 1 and n = 5-8) at the model's shapes; upstream
  tuned these for Turing+ and patch 02 only touched rows per block.
- B6 Activation quantization: `quantize_row_q8_1_cuda` runs once per matvec input (patch 10 dedups
  gate/up). Count its launches per token in B1; if it is > 3% of the step, fuse it into the
  producer (norm output) as upstream does for some paths, or widen the cache to the attention
  q/k/v inputs. Small gain, small effort.
- B7 Batch 2-8 (speculative verify shapes) vs 9+: `llama-bench -p 2,4,8,9,16` (section 4.1) shows
  the cliff at 9 where cuBLAS takes over with the full-matrix dequant. If MTP drafting is switched
  on with `n-max 4` (verify width 5), everything stays in MMVQ; make sure any draft setting keeps
  the verify width <= 8, or A4 is a prerequisite.
### Area C: attention and KV cache

Files: `ggml/src/ggml-cuda/fattn.cu` (`ggml_cuda_get_best_fattn_kernel` ~:358, Pascal branch
~:520-533), `ggml/src/ggml-cuda/fattn-tile.cuh` (config tables ~:21-75, `launch_fattn_tile_switch_*`
~:1148-1319), `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/fattn-common.cuh`
(`launch_fattn`, `parallel_blocks` search ~:1113-1175, KV dequant ~:1022-1085), `src/llama-kv-cache.cpp`,
`src/llama-context.cpp` (FA/KV-type gates ~:3672-3719).

Current behavior on sm_60 (no `p100:` commit touches `fattn*`; this is stock upstream):

- Only TILE and VEC kernels are reachable (no MMA). For F16 KV with GQA (both `qwen35` full
  attention layers and Gemma 4), even single-token decode picks TILE, because VEC is chosen only
  when `Q->ne[1] == 1 && !gqa_opt_applies`, and `gqa_opt_applies` is true whenever there is a mask,
  no ALiBi and `n_kv % 256 == 0` (the cache pads `n_kv` to 256). VEC is reached at tg only with
  quantized KV (`Q->ne[1] <= 2`).
- Tile configs come from `ggml_cuda_fattn_tile_get_config_nvidia_fp16`; `fattn-tile.cuh:7` says
  "TODO optimize kernel parameters for FP16 NVIDIA (P100)". For D=128 and D=256: ncols 2 ->
  64 threads, 4 -> 128, 8/16/32 -> 256 threads, occupancy 2, `nbatch_fa 64`, `nbatch_K 64`
  everywhere. GQA packing (`ncols2`) is clamped on Pascal to `Q->ne[1] <= 16` (`gqa_limit = 16`).
  Static shared memory is ~28.5 KB at D=256/ncols 32 and ~9.7 KB at ncols 2, far below 48 KB.
  Occupancy 2 x 256 threads = 512 threads per SM = 25% of the SM's thread capacity.
- Pascal loads K/V with 8-byte copies (`ggml_cuda_get_max_cpy_bytes() == 8`), halving lanes per
  value vs Volta+ in both kernels (`nthreads_KQ = nthreads_V = 16` in VEC).
- Stream-k and the fixup kernels are inert on sm_60; long-KV parallelism comes only from
  `parallel_blocks` + `flash_attn_combine_results`. The `parallel_blocks` search probes occupancy
  with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and stops at 95% efficiency.
- `-sm tensor` requires FA on and refuses quantized KV entirely ("simultaneous use of
  SPLIT_MODE_TENSOR and KV cache quantization not implemented"). Under `-sm layer`, quantized KV
  at batch > 2 falls to TILE, which dequantizes the whole K and V cache to F16 into the pool on
  every call: a per-step bandwidth cost that grows with context. So on this machine KV
  quantization is a memory tool, not a speed tool, and it is unavailable in the user's mode.
- The KV cache is sharded across the two cards under `-sm tensor` (axis 0, same segmentation as
  `attn_output.weight`), so attention itself moves nothing over PCIe.
- No defrag exists any more; `llama_kv_cache::update` only does cross-stream copies and the RoPE
  K-shift graph.

Why it matters: measured pp falls 475 -> 270 t/s and tg 28.9 -> 22.8 t/s between 3.7k and 89k
context on Qwen3.8-27B; that delta is attention. Back-of-envelope: at 89k the tile kernel runs
at roughly 35-45% of the card's FLOP or bandwidth ceiling in both modes, while cuBLAS reaches
~68%. Attention is the largest untuned surface on this device.

Items:

- C1 `[SRV]` Baseline the kernels in isolation: `test-backend-ops -o FLASH_ATTN_EXT perf -b CUDA0`
  with the model's shapes (qwen35: D=128, GQA ratio = n_head/n_head_kv from the GGUF metadata, KV
  lengths 4096/32768/65536, `nb` = 1 for tg and 512/2048 for pp; gemma4: D=256, sliding window
  1024 for local layers). Record us and derived GB/s (tg) or TFLOPS (pp). Note which kernel ran
  (`nsys` kernel names: `flash_attn_tile`, `flash_attn_ext_vec`, `flash_attn_combine_results`).
- C2 Tune the Pascal FP16 tile table (`ggml_cuda_fattn_tile_get_config_nvidia_fp16`) for D=128
  and D=256: sweep `nthreads` (128/256), `occupancy` (2/3/4), `nbatch_fa` (32/64/128), `nbatch_K`
  (32/64/128) at ncols 2/8/16/32 with the C1 harness. The table is a plain constexpr switch, so a
  sweep is a rebuild per point (or make it env-overridable in a scratch build:
  `GGML_CUDA_FATTN_TILE_CFG=nthreads,occ,nbfa,nbk`). Hard limits: 48 KB smem per block, 64K
  registers per SM (check `-Xptxas -v` for spills at higher occupancy). Expected gain: 1.2-2x on
  the attention kernel, i.e. +5-15% tg and pp at 32k+ context, nothing at 4k. Kill switch: none
  needed if the table is just retuned (document old values); add `GGML_CUDA_FATTN_TILE_LEGACY=1`
  if the change is structural.
- C3 Try VEC for tg with GQA on Pascal: in a scratch build, return VEC for
  `Q->ne[1] == 1` regardless of `gqa_opt_applies` on the Pascal branch and compare C1 numbers at
  KV 4k/32k/64k. VEC processes one q column per block with 128 threads; TILE with `ncols2` GQA
  packing shares K/V loads across the GQA group, which is why upstream prefers it. Measure
  instead of guessing; keep whichever wins per KV length (a KV-length threshold is a fine rule).
- C4 `parallel_blocks` on sm_60: log the chosen `parallel_blocks` (add a one-line `GGML_CUDA_FATTN_LOG`
  debug print in a scratch build) for tg at 4k/32k/64k. 56 SMs with occupancy 2 means 112 blocks
  fill the card; with `ncols2` GQA packing and 8 KV heads a single-token decode has few blocks
  unless `parallel_blocks` is large. If the search stops early (95% rule) at long KV, force
  higher values and measure. Small code change, potentially large tg gain at long context.
- C5 8-byte vs 16-byte loads: `ggml_cuda_get_max_cpy_bytes` returns 8 on sm_60 deliberately
  (Pascal has 16-byte vector loads, `LDG.128`, but the upstream author chose 8; find the commit
  and its reason with `git log -S"ggml_cuda_get_max_cpy_bytes"`). Test 16 in a scratch build with
  `test-backend-ops` (correctness) and C1 (speed). Alignment asserts may fire; that is the answer
  then.
- C6 Gemma 4 prep: D=256 tile configs are in the same table (C2 covers them); logit softcap only
  selects a template. iSWA layers use a small cache (`n_swa`), so their attention cost is flat.
  Verify `-sm tensor` supports the gemma4 arch (allow-list in `src/llama-model.cpp` ~:351) before
  planning anything else for it; `test-llama-archs` runs every arch through the meta backend.
- C7 KV quantization under `-sm tensor` (research, tier 3): implementing quantized K/V shards in
  the meta backend would halve KV memory (more context or a second slot) but cost tg through the
  TILE dequant-per-call behavior above. Only worth it after C2/C3 if VEC wins for tg (VEC reads
  quantized KV natively). Do not start before then.
- C8 Non-FA path check: with `-fa auto` a `NONE` verdict silently disables FA (`-fa on` errors
  instead). Always run the server with `-fa on` so a future kernel-selection regression is a
  startup error, not a silent 2x slowdown.

### Area D: delta-net (GDN) layers of `qwen35`

Files: `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`, `ggml/src/ggml-cuda/gated_delta_net.cu`,
`ggml/src/ggml-cuda/ssm-conv.cu`, `ggml/src/ggml-cuda/norm.cu` (L2 norm), the fusion rules in
`ggml-cuda.cu` (`ggml_cuda_try_fuse` ~:3706-3960).

Current state: 3 of every 4 layers are delta-net layers (`full_attention_interval` 4). Per
delta-net layer the graph emits: qkvz `MUL_MAT`, beta `MUL_MAT` + `SIGMOID`, gate `MUL_MAT` +
`ADD` + `SOFTPLUS` + `MUL`, conv-state `GET_ROWS` + `TRANSPOSE` + `CONCAT` + `CPY` writeback,
`SSM_CONV`, `SILU`, three `VIEW`s, two `L2_NORM`, SSM-state `GET_ROWS`, `GATED_DELTA_NET` (fused,
forced on since #27877), `CPY` state snapshots, gated `RMS_NORM` + `SILU` + `MUL`, out `MUL_MAT`.
Patches 04, 18, 19, 20, 23, 24, 25, 26, 27 already fuse or speed up most of the glue. The
`GATED_DELTA_NET` kernel is `__launch_bounds__(min(warp, S_v) * 4, 2)` with grid `(H, n_seqs,
ceil(S_v / num_warps))`: at tg that is a small grid on a 56-SM part.

Items:

- D1 `[SRV]` `GGML_CUDA_FUSE_LOG=2` on one decode step: list every node that did NOT fuse and
  why. Then `nsys` kernel count per delta-net layer at tg. Target: < 10 launches per layer. Each
  remaining launch costs ~3-5 us on a device without CUDA graphs (see G1); 48 delta-net layers x
  1 saved launch = ~0.2 ms/token = 0.6% tg per launch removed.
- D2 `GATED_DELTA_NET` kernel occupancy at tg: `nvprof --metrics achieved_occupancy,sm_efficiency`
  on it. If it fills < 50% of SMs, split S_v further across blocks or process 2 heads per block.
  Requires understanding the recurrence (read `gated_delta_net.cu` fully first). Medium effort.
  Kill switch via a template/env `GGML_CUDA_GDN_LEGACY`.
- D3 `SSM_CONV` + `SILU` are already fusable upstream (`{SSM_CONV,(ADD),UNARY(SILU)}`); confirm it
  fires here (D1). If not, find the shape condition that blocks it.
- D4 The `L2_NORM` pair (q and k) fuses as a sibling run (patch 18); confirm. The `RMS_NORM +
  SILU + MUL` gated norm at the layer output may have no rule: check D1's log, and if it is 3
  launches, add a rule modeled on patch 20 (`ADD -> UNARY -> MUL`), `GGML_CUDA_DISABLE_FUSE_RMS_GATE`.
- D5 State writeback copies (`ggml_cpy` into cache views) fuse up to 8 per launch (patch 26). With
  `n_rs_seq > 0` (rollback slots) it is `n_rs_seq + 1` copies per layer; the server's checkpoint
  settings decide `n_rs_seq` (see area J). Fewer checkpoints = fewer copies.

### Area E: elementwise, norm, rope, cpy and the fusion machinery

Files: `ggml-cuda.cu` (`ggml_cuda_can_fuse` ~:3177, `ggml_cuda_try_fuse` ~:3706, `GGML_CUDA_FUSE_LOG`),
`norm.cu`, `rope.cu`, `binbcast.cu`, `unary.cu`, `cpy.cu`, `getrows.cu`, `concat.cu`.

Upstream rules present: `{RMS_NORM,MUL,ROPE,VIEW,SET_ROWS}`, `{RMS_NORM,MUL,ROPE}`, `{RMS_NORM,MUL,ADD}`,
`{RMS_NORM,MUL}`, `{SSM_CONV,(ADD),UNARY(SILU)}`, `{UNARY,MUL}`, mul_mat + GLU, topk-moe; the MMVQ
gate/up fusion is disabled for Pascal (area B2). Patch rules: 18, 19, 20, 23, 24, 27 + GDN cache
fusion, each with a kill switch.

Items:

- E1 `[SRV]` Per-token launch census: `nsys stats --report cuda_gpu_kern_sum` on 64 decode tokens;
  divide counts by 64. Produce a table kernel -> launches per token -> total us per token. Sort by
  total time and by count. Everything under ~3 us per launch is launch-bound and a fusion or
  CUDA-graph candidate (G1); everything over 50 us is a kernel-tuning candidate.
- E2 For the attention layers: confirm `RMS_NORM + MUL + ROPE` fusion fires for q and k (MRoPE
  `ggml_rope_multi` may block the rule: check the op-params condition in `ggml_cuda_can_fuse`).
  Confirm the gate `SIGMOID + MUL` after attention fuses (`{UNARY,MUL}` rule needs same shapes).
- E3 `binbcast` residual `ADD` + the next `RMS_NORM`: patch 19 folds the pre-add; verify it fires
  for both the attention and the ffn residual in every layer (count in E1 should be 2 x 64 fused
  launches, not 4 x 64).
- E4 `rope.cu` uses `dim3(1, CUDA_ROPE_BLOCK_SIZE, 1)` blocks with `n_blocks_x = ceil(ne00 / (2*BLOCK))`;
  at tg a single token means very few blocks. Only relevant if E1 shows rope > 2% of the step;
  else skip.
- E5 `rms_norm_f32<1024>` register cache (patch 17) applies when `1024 < ncols <= 4096`; Qwen3.8's
  `n_embd` decides whether it fires (5120 would not: > 4096). Check `n_embd` in the GGUF and, if
  it is 5120, extend `max_cache` to 5 or 6 in a scratch build and measure `test-backend-ops -o
  RMS_NORM perf` at that width. Trivial change if it helps.
- E6 Anything else E1 surfaces with count >= 64 per token and < 5 us each: candidates for a
  sibling-run fusion (patch 18 infrastructure `ggml_cuda_collect_same_op`), with a kill switch.

### Area F: multi-GPU communication over PCIe

Files: `ggml/src/ggml-backend-meta.cpp` (subgraph split ~:2268, allreduce call ~:2519, butterfly
fallback ~:2394-2508), `ggml/src/ggml-cuda/ggml-cuda.cu` (comm init ~:1210-1258, NCCL allreduce
~:1005-1077, P2P init ~:394-407, `cpy_tensor_async` ~:2482-2539), `ggml/src/ggml-cuda/allreduce.cu`
(internal allreduce, Volta+ only), `ggml/src/ggml-backend.cpp` (split input copies ~:1789-1797),
`src/llama-context.cpp` (pipeline parallel condition ~:428-452), `docs/multi-gpu.md`.

Current behavior:

- `-sm row` is gone on CUDA (the split buffer type is no longer exported; it errors at load).
  Live modes: `none`, `layer`, `tensor`.
- `-sm tensor`: dense weights are sharded (q/k/v/up/gate axis 1, `attn_output`/`ffn_down` axis 0,
  `output.weight` axis 1, KV cache axis 0). Two PARTIAL nodes per layer (`attn_output`, `ffn_down`)
  close a subgraph each and trigger one AllReduce of `n_tokens * n_embd * 4` bytes. For 64 layers:
  128 collectives per decode step, each ~20 KB at tg (n_embd 5120) and ~42 MB at `-ub 2048`.
  The logits are vocab-sharded and gathered only on `get_tensor`.
- AllReduce selection at init: `GGML_CUDA_ALLREDUCE=nccl|internal|none`; default NCCL on Linux
  (build option `GGML_CUDA_NCCL` ON). The internal allreduce refuses to initialize on `cc < VOLTA`
  (it spins with `__nanosleep`), so on P100 the choice is NCCL or the meta backend's butterfly
  fallback (one `cudaMemcpyPeerAsync` into a temp + a 1-node `ADD` graph per device per
  collective, no BF16 wire compression). NCCL uses F32 on the wire for small messages (`ne < 32768`)
  and BF16 above.
- Peer access is OFF by default. `GGML_CUDA_P2P=1` enables `cudaDeviceEnablePeerAccess` for all
  pairs at init (and `cuMemSetAccess` for the VMM pool). Without it every `cudaMemcpyPeerAsync`
  (butterfly path, scheduler cross-backend copies) is staged through host memory by the driver.
  NCCL does its own transport selection (P2P vs shared host memory) independent of this flag.
- The meta backend exposes no events and no async copy, so llama runs the scheduler with
  `n_copies = 1` for `-sm tensor`; pipeline parallelism (4 copies) exists only for `-sm layer`.
- Scheduler split inputs flagged INPUT (token embedding, positions, out_ids, kq_mask, k/v idxs)
  are copied synchronously after a backend sync: ~6 small H2D copies per step, x2 devices
  (MIRRORED). `tok_embd` always lives on the CPU by policy, so the embedding row crosses PCIe every
  step (tiny at tg, 42 MB per 2048-token ubatch at pp).
- Logits: `n_vocab * 4` bytes D2H per output token into the pinned output buffer. Backend
  sampling is refused under `-sm tensor`.

Items (in order):

- F1 `[SRV]` Establish which transport NCCL actually uses: run the server once with
  `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P` and read the lines for the 0<->1 channel:
  "via P2P/direct pointer" (PCIe P2P), "via P2P/IPC", or "via SHM" (host memory bounce). Also
  confirm the absence of "NCCL is unavailable" in the llama log. Record NCCL version. If SHM: try
  `NCCL_P2P_LEVEL=SYS` (or `PHB`) and re-check; if P2P is refused by the driver, fix IOMMU/ACS
  (Part 1, 1.4) first. This single setting can be worth several ms per token.
- F2 `[SRV]` Quantify the collective cost: `nsys` on 64 tg tokens, sum of NCCL kernels
  (`ncclDevKernel_AllReduce*`) and any `Memcpy PtoP/DtoH/HtoD` per token, plus the gaps that
  bracket them. Expected order of magnitude: 128 collectives x 20-40 us = 2.5-5 ms per token =
  7-15% of a 34.6 ms step. If it is much higher, NCCL is on SHM or the GPUs wait on each other.
- F3 NCCL tuning for tiny 2-GPU messages (env only, no code): `NCCL_PROTO=LL` (low-latency
  protocol for small messages), `NCCL_ALGO=Ring` vs `Tree`, `NCCL_NTHREADS=64/128`, `NCCL_BUFFSIZE`,
  `NCCL_MAX_NCHANNELS=1/2` (fewer channels for 20 KB messages), `NCCL_CHECK_POINTERS=0`. Sweep
  with `llama-bench -n 64` and keep the winners in the recommended command line. Also try
  `GGML_CUDA_P2P=1` together with each (it changes the VMM pool's peer mapping).
- F4 `GGML_CUDA_ALLREDUCE=none` vs `nccl` A/B, with and without `GGML_CUDA_P2P=1`: the butterfly
  with real P2P (one `cudaMemcpyPeerAsync` over PCIe + one tiny ADD kernel) might beat NCCL's
  kernel-based allreduce for 20 KB at tg. Measure both tg and pp (pp moves 42 MB per collective;
  there NCCL should win). If the butterfly wins tg and NCCL wins pp, item F6.
- F5 Make the internal allreduce run on sm_60: `ggml_cuda_ar_pipeline_init` rejects `cc < VOLTA`
  only because of `__nanosleep`. Replace the wait loop with a `clock64()`-based spin (or
  `__threadfence_system()` + volatile poll without sleep) under `#if __CUDA_ARCH__ < 700`, remove
  the gate, rebuild, `test-llama-archs`, then A/B against NCCL at tg. The internal path is a
  host-bounce design (mapped pinned memory, BF16 on the wire by default via
  `GGML_CUDA_AR_BF16_THRESHOLD`), so it does not need P2P; its 2-slot pool forces a host
  `cudaEventSynchronize` every other call, which may hurt at 128 collectives per token. Effort:
  small. Kill switch: `GGML_CUDA_ALLREDUCE` already selects the implementation. Note BF16 on the
  wire changes numerics (document; compare perplexity).
- F6 Size-dependent transport choice: if F4/F5 show different winners for tg (20 KB) and pp
  (40 MB), add a byte threshold in `ggml_backend_cuda_comm_allreduce_tensor` that routes small
  messages to the cheaper path (`GGML_CUDA_AR_NCCL_MIN_BYTES`). Small change.
- F7 P2P-direct allreduce kernel for exactly 2 GPUs (research, tier 3): with `GGML_CUDA_P2P`, each
  GPU can read the peer's shard through a mapped pointer; a single kernel per device that waits
  on a flag in peer memory, reads the peer's partial and writes the sum removes the memcpy and
  the second launch. Microbenchmarks 1.8.4/1.8.5 give the achievable latency (target < 10 us per
  collective vs NCCL's 20-40). Requires stable P2P (F1) and careful fences (`__threadfence_system`);
  Pascal has no `__nanosleep`, use `clock64()` backoff. Only after F1-F6.
- F8 Fewer collectives: the two reductions per layer are inherent to Megatron-style sharding.
  Alternatives that would halve them (sequence-parallel or sharding `ffn_down` along the token
  axis) are model-graph rewrites in the meta backend's policy (`src/llama-model.cpp` ~:404-591
  and `handle_mul_mat`); out of scope unless F2 shows collectives > 25% of the step.
- F9 Split inputs: 6 synchronous H2D copies per step, each preceded by a sync. Measure their gap
  share in F2's timeline (look for `Memcpy HtoD` with idle GPU before them). If visible (> 0.5
  ms/token), batch them: the meta backend could accept a single pinned staging buffer for all
  MIRRORED inputs, or llama could pack pos/out_ids/mask into one tensor. Medium effort; only if
  the timeline shows it.
- F10 `-sm layer` re-evaluation with pipeline parallelism: the standard matrix (4.1) plus
  `CUDA_SCALE_LAUNCH_QUEUES=4x` (documented in `docs/build.md` as beneficial for multi-GPU
  pipeline parallelism). Expect layer to lose tg (each card idles while the other computes) and
  to compete on pp. Record it; the user's mode stays `tensor` unless the numbers say otherwise.
- F11 `GGML_CUDA_PEER_MAX_BATCH_SIZE` and `GGML_CUDA_NO_PEER_COPY` are dead or irrelevant for the
  CUDA backend in this tree; do not spend time on them. Update `P100-PATCHES.md` knobs table to
  say so.

### Area G: CUDA graphs, launch overhead and host-side per-token cost

Files: `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_graph_set_enabled` ~:5006-5019,
`ggml_backend_cuda_graph_compute` ~:4897-4980, `ggml_cuda_graph_update_required` ~:2592-2630),
`ggml/src/ggml-cuda/common.cuh` (`ggml_cuda_graph` ~:1240-1272), `ggml/src/ggml-backend-meta.cpp`
(`graph_compute` ~:2050, per-device `graph_compute_async` ~:2513), `src/llama-context.cpp` (graph
reuse ~:279, decode), `ggml/src/ggml-backend.cpp` (scheduler, patch 21).

Current behavior:

- CUDA graphs are permanently OFF on the P100: `ggml_cuda_graph_set_enabled` sets
  `disable_due_to_gpu_arch` for `cc < GGML_CUDA_CC_VOLTA`. Not because of `-sm tensor` (the meta
  backend forwards each per-device subgraph to the real CUDA `graph_compute`), and not because of
  a build option (`GGML_CUDA_GRAPHS` default is on in presets; check the build). Every kernel of
  every token is launched individually from the host: for a 64-layer model that is on the order
  of 1000+ launches per device per token, driven by one host thread that alternates between the
  two devices. Launch cost ~3-5 us each on the host side, plus GPU-side gaps when the host falls
  behind. This is the most likely explanation for the gap between the ~11 ms weight-streaming
  floor and the measured 34.6 ms per token, and for the -27% when graph reuse (a host-side
  saving) is disabled.
- Graph capture in this tree works with a property snapshot + `cudaGraphExecUpdate` (no per-node
  kernel-param patching), two-call warm-up, and `cgraph->uid` short-circuit. Under `-sm tensor`
  each decode step submits ~128 subgraphs per device (one per PARTIAL boundary); the CUDA backend
  keeps one `cuda_graph` per backend context, so 128 different cgraphs per token would defeat a
  single cached graph (constant re-capture, slower than no graphs). `GGML_CUDA_GRAPH_OPT`
  reordering is single-device only.
- llama-side graph reuse (`LLAMA_GRAPH_REUSE_DISABLE`) already saves the ggml graph rebuild and
  the meta backend re-split (15-20 ms per miss); it hits every step when the ubatch shape is
  constant. Patch 21 trimmed `ggml_backend_sched_reset`.

Items (this area has the largest expected tg gain; do G1 before any kernel work):

- G1 `[SRV]` Measure the launch-bound share: `nsys` on 64 tg tokens, then (a) total GPU busy time
  per token per device vs wall time per token, (b) count of kernel launches per token per device,
  (c) CPU thread timeline: is the llama thread 100% busy during decode? If GPU busy is < 60% of
  wall and the host thread is saturated, the step is host/launch bound and G2-G4 are the top
  priority. Also run tg with `-sm none` on one card and a smaller model (or `-ngl 40`) to see the
  single-device launch-bound share without collectives.
- G2 Enable CUDA graphs on Pascal in a scratch build (remove the `cc < VOLTA` gate) and test
  with `-sm none` first (single device, one cgraph per token): `test-backend-ops` unaffected, run
  `llama-bench -n 128` and the greedy determinism check. CUDA graphs are a driver feature and
  work on Pascal; the upstream gate is a heuristic, find the commit and its reason with
  `git log -S"disable_due_to_gpu_arch"`. Expected on a single card: +10-30% tg if G1 shows launch
  binding. Kill switch exists (`GGML_CUDA_DISABLE_GRAPHS`); add `GGML_CUDA_GRAPHS_PASCAL=1` as the
  opt-in if upstream's reason turns out to be real for some ops.
- G3 CUDA graphs under `-sm tensor`: needs a small cache of CUDA graphs keyed by `cgraph->uid`
  (or subgraph index) in `ggml_backend_cuda_context` instead of a single `cuda_graph`, so the
  ~128 per-device subgraphs per token each keep their own captured graph across decode steps.
  The meta backend already keeps subgraph identity stable across steps (gist fix, `needs_rebuild`
  check on `cgraph->uid`). Design: `std::unordered_map<uid, std::unique_ptr<ggml_cuda_graph>>`
  with an LRU cap (`GGML_CUDA_GRAPH_CACHE_MAX`, default 512); everything else in the capture
  path stays. Each captured graph replaces ~10-20 launches with 1 `cudaGraphLaunch`. This is a
  medium-size change (~150 lines) in a hot path: present the design to the user before coding.
  Expected: the bulk of the launch-bound share found in G1. Correctness: greedy determinism,
  soak test (section 4.2 step 5), and `GGML_CUDA_DISABLE_GRAPHS=1` as the kill switch.
- G4 Alternatively or additionally, reduce the number of subgraphs: NCCL allreduce is itself a
  stream operation; if the meta backend issued it on the device stream without closing the
  ggml subgraph, one CUDA graph per device per token would suffice. That is a deeper change in
  `ggml_backend_meta_graph_compute` (the subgraph boundary is where the allreduce is scheduled);
  read it fully before judging. Tier 3.
- G5 Host thread: with graphs off, the host enqueue is the bottleneck candidate. Check
  `CUDA_SCALE_LAUNCH_QUEUES=4x` (bigger command buffer), CPU governor `performance`, and that the
  server process is not sharing the core with the router or the tokenizer thread (`taskset`,
  `--prio`, `-t`). Cheap to test with `llama-bench -n 128`.
- G6 Patch 22 (`decode-sched-slots`, deferred) targets graph-reuse misses when the ubatch shape
  alternates (MTP drafting). Re-evaluate only together with MTP (area J); with a constant shape
  upstream reuse already hits every step.
- G7 Scheduler cost per step: `ggml_backend_sched_reset`/`alloc_graph`/`compute_splits` with a
  64-layer graph of ~3000 nodes; patch 21 measured ~100 us saved. Profile the host side with
  `perf record -g` on the server during 64 tg tokens (`perf` supports the CPU part; the GPU part
  is nsys). Anything above 1 ms/token in ggml/llama host code is a target; expect
  `ggml_backend_sched_alloc_graph` (galloc) and the meta backend split bookkeeping to top the
  list if graph reuse misses. With reuse hitting, expect the launch calls themselves to dominate,
  which loops back to G2/G3.
### Area H: memory headroom

Files: `ggml-cuda.cu` (VMM pool ~:538-676, legacy pool ~:421-509), `src/llama-context.cpp`
(`sched_reserve` ~:581-717, output buffer ~:2066-2132), `src/llama-model.cpp` (split policy).

Current state: both cards at ~94% after load with `-c 200000 -np 1`. The VMM pool reserves a 32 GB
virtual range per device and grows physical backing as a high-water mark that is never returned;
one wide LM-head cuBLAS call (~1.3 GB F16 per card) raises it permanently, which is the OOM in the
runtime notes. Compute buffers are sized by the pp reserve at `min(n_ctx, n_ubatch)` tokens.
Quantized KV is unavailable under `-sm tensor`. Patch 10's q8_1 cache is a raw `cudaMalloc` that
grows and never shrinks (small).

Items:

- H1 `[SRV]` Memory breakdown: run with `--verbose` and copy the `memory breakdown` table (model
  weights per device, KV, compute buffer, pool) into the log. Then `nvidia-smi` after a 2048-token
  prompt and after a 9+ column verify batch (if speculative decoding is on) to see the pool
  high-water mark move.
- H2 A4 (chunked cuBLAS) is the memory fix with the largest payoff: it removes the need to hold
  back ~1.3 GB per card. Do it early.
- H3 `-ub` vs compute buffer size: record the compute buffer at `-ub 512/1024/2048` (H1); if
  `-ub 1024` costs < 3% pp (A1) and frees hundreds of MB, that memory buys context or a second
  slot. Decide with numbers.
- H4 Second slot without doubling KV: `-np 2 --kv-unified` shares one `-c` budget across slots
  (verify it is supported for the hybrid memory: `llama_memory_hybrid` with unified KV; run
  `llama-server --help` and a smoke test). Only if the user wants concurrent requests; tg per
  request drops because the step is launch-bound, not because of bandwidth.
- H5 If the F16 KV size at 200k is the dominant consumer (H1 will show it: 16 attention layers x
  2 x n_head_kv x 128 x 2 bytes per token), the only reductions on this branch are a smaller
  `-c` or C7 (quantized KV in the meta backend, tier 3).

### Area I: sampling and the CPU side

Files: `src/llama-sampler.cpp`, `common/sampling.cpp` (chain, prefilter patch 13 ~:112-190),
`common/arg.cpp` (threads ~:1514-1576, `-bs` ~:2303), `src/llama-context.cpp` (backend sampling
refused under tensor split ~:1224-1235).

Current state: all sampling is on the CPU under `-sm tensor` (backend sampling is refused because
the logits are vocab-sharded across the two cards). Patches 11 and 13 cut the two biggest costs
(penalties scan, candidate array build). The prefilter disables itself when a logit-bias sampler
is in the chain (any per-request `logit_bias`, or vocab suppress tokens), when a non-penalties
sampler precedes top-k, when `top_k <= 0`, or when `8 * (top_k + penalty_last_n) >= n_vocab`.
The default thread count is the number of physical cores (14) with `--poll 50`; with everything
offloaded those threads mostly spin.

Items:

- I1 `[SRV]` Confirm the prefilter is live for the production sampler settings: temporarily log
  `pf_nkeep` (or run once with `LLAMA_SAMPLER_PREFILTER=0` and compare server tg). Check whether
  the Qwen3.8 vocab carries suppress tokens (`gguf-py` dump of the tokenizer metadata); if it does,
  the logit-bias sampler is always in the chain and the prefilter never runs, and patch 13's
  condition should be relaxed to "no user biases" (the suppress list is tiny and static).
- I2 `[SRV]` Sampling share: server `predicted_per_second` vs `llama-bench -n` tg at the same
  context. llama-bench does not sample, so the difference is sampling + server bookkeeping. If it
  exceeds ~1.5 ms per token (4%), profile the sampler chain with `perf record` and look at the
  order of samplers in the production request (moving top-k first keeps the prefilter on).
- I3 Backend sampling under tensor split (research, tier 3): per-shard top-k on each card, gather
  2 x k candidates to one device, finish there. Removes the ~1 MB logits D2H per token and the CPU
  sort. Needs a meta-backend aware path in `llama_context`; only after G and C.
- I4 `[SRV]` Threads and polling: `llama-bench -t 2,4,8,14 --poll 0,50` (or server A/B) at tg.
  Fewer spinning threads free the launch thread's core and reduce host power (which matters on a
  700 W PSU shared with two 250 W cards). Expect `-t 4 --poll 0` or similar to be neutral or better;
  put the winner in the recommended command line. Also `--prio high` for the server process.
- I5 Governor `performance` (1.6). Cheap; check it is persistent across reboots.
- I6 CPU swap option: the user has a spare E5-1650 v4 (6C/12T, 3.6/4.0 GHz vs the E5-2690 v4's
  2.6/3.5 GHz; same Broadwell-EP, same 40 lanes, same quad-channel DDR4). It only helps the
  single-thread host share of the tg step (launches, meta bookkeeping, sampling): at most ~14-19%
  on that share, i.e. a few % tg if G1 shows the step is host-bound, and ~nothing once G2/G3
  (CUDA graphs) work. Decide after G1 and G2: swap only if the host thread is saturated and graphs
  cannot be enabled. I4 (fewer spinning threads) recovers part of the same clock headroom for free.

### Area J: server and request-level behavior

Files: `tools/server/server-context.cpp` (slot selection ~:1550-1591, host prompt cache save
~:1636 and ~:2409-2416, prefix match ~:3190-3262, checkpoints ~:2327-2358, `[TAG_PROMPT_LOGITS]`
~:3375-3380), `tools/server/server-task.cpp` (`server_prompt_cache::load` ~:1793-1830, metrics),
`common/speculative.cpp`, `common/arg.cpp` (`--cache-ram`, `--cache-idle-slots`, `--ctx-checkpoints`,
`--checkpoint-min-step`, `--cache-reuse`, `--spec-*`).

Current state and the ~1 s suspect list (ordered by plausibility):

1. `--cache-idle-slots` (default on) with `--cache-ram` (default 8192 MiB): when a slot gets a new
   task, its whole KV state is serialized to host RAM (`llama_state_seq_get_data_ext`) before the
   new prompt is processed. For this model that is on the order of 64 KB per token of context
   (16 attention layers x K+V x n_head_kv x 128 x 2 bytes; verify with the GGUF head count), so
   50k tokens of context is ~3 GB moved D2H into pageable host memory on the request's critical
   path: several hundred ms to a second. This matches "fixed, independent of prompt length,
   present with and without a draft model".
2. Retokenization and re-rendering of the full chat history per request (tens to a few hundred
   ms at 30k+ tokens on this CPU).
3. Two graph rebuilds per request (prompt ubatch shape, then tg shape), ~20 ms each with the
   meta re-split.
4. Sampler chain construction per request (`common_sampler_init`): with 250k vocab and suppress
   tokens it builds a logit-bias sampler; small unless it materializes vocab-sized arrays.
5. Recurrent-state checkpoint save/restore for the hybrid model (~100 MB each, ~10-20 ms).
6. `n_past--` forcing a 1-token decode on exact cache hits (~35 ms).

Items:

- J1 `[SRV]` Per-request timeline: run the server with `-v` (timestamps) and
  `LLAMA_SERVER_SLOTS_DEBUG=1` on the fixed 20-request follow-up script (4.1) at ~30k context.
  Measure wall time from request receipt to first prompt-processing log line, prompt phase,
  first token. Then A/B `--cache-ram 0` (disables the host cache and the per-task KV save). If
  the ~1 s disappears, keep `--cache-ram 0` for the single-slot production setup (the host cache
  only pays when slots are evicted, which never happens with `-np 1` and one conversation) and
  note it in `P100-PATCHES.md`. If it stays, walk the remaining suspects with the timestamps.
- J2 If the host cache is wanted anyway (multi-conversation use), make the save asynchronous
  or skip it when the new prompt extends the slot's prompt (prefix match >= old length): a
  server-side change in `server-context.cpp` around the `cache_idle_slots` save. Medium effort,
  server code only, kill switch = the existing flags.
- J3 `--cache-reuse N`: requires `llama_memory_can_shift`, which is false for recurrent/hybrid
  memory; confirm in the startup log and do not spend time on it for `qwen35`. Relevant for Gemma 4.
- J4 Checkpoints: `--ctx-checkpoints` (default 8) and `--checkpoint-min-step` decide how far a
  hybrid model must reprocess when the common prefix ends before the slot's end (e.g. when the
  retokenized assistant turn differs from the generated tokens). Watch for "restored context
  checkpoint" / "created context checkpoint" lines in J1; if reprocessing from a checkpoint shows
  up on follow-ups, lower `--checkpoint-min-step` (more checkpoints, each costing a ~100 MB
  state copy during pp) and measure.
- J5 Speculative decoding (revisit after A4 and G):
  - MTP (`draft-mtp`): the untested `--draft-p-min 0` (constant verify width 5 keeps graph reuse
    at ~100%; measured today ~10% reuse with p-min 0.75 and 15-20 ms per miss). Keep the verify
    width <= 8 (`--spec-draft-n-max 7` or less) until A4 lands, or the LM head F16 copy OOMs.
  - ngram-mod is free when idle; keep `n-max 7` for the same reason.
  - Patch 22 (`decode-sched-slots`) only pays with alternating shapes; measure with MTP on,
    `LLAMA_DEC_SLOTS=0` vs `4`, as `P100-PATCHES.md` says.
- J6 Request hygiene that keeps the fast paths on: no per-request `logit_bias` (kills the
  prefilter), `n_probs = 0`, `-fa on` explicitly, `--temp` etc. fixed in the preset rather than
  per request (a changed sampler set does not break graph reuse, but a changed `n_outputs` does).
- J7 Router respawn: a child crash loses the host prompt cache and reprocesses the whole
  conversation. The gist fixed the known crash; keep the soak test (4.2 step 5) in the routine
  and keep `P100-PATCHES.md`'s crash-signature notes current.

### Area K: model loading (secondary; matters after respawns)

Files: `src/llama-model-loader.cpp` (upload path ~:1505-1717), `src/llama-mmap.cpp`, `common/arg.cpp`
(`--load-mode`).

Current state: default `auto` = mmap; the pinned 4-buffer async upload runs only when mmap is
off (`--load-mode dio` or the deprecated `--no-mmap`). `mlock` pins ~17 GB of the 32 GB. With
mmap, the model file competes with `--cache-ram` (8 GB default) and the process for page cache.

Items:

- K1 `[SRV]` Time `llama-server` startup to "model loaded" for `--load-mode mmap`, `dio`, `mlock`,
  cold (`sync; echo 3 > /proc/sys/vm/drop_caches`, needs root) and warm. Expect `dio` to win cold
  loads (direct I/O + pinned staging, no page cache pass) and mmap to win warm loads. Pick per the
  respawn scenario (cold after a crash is the case that hurts). `GGML_CUDA_REGISTER_HOST=1`
  (registers the mmap region) is a third variant to time.
- K2 Host RAM budget: 17 GB model (page cache or pinned) + `--cache-ram` + process + OS on 32 GB.
  If J1 keeps `--cache-ram 0`, mmap or mlock both fit; if not, lower `--cache-ram` to the measured
  need. Never combine `mlock` with the 8 GB default cache on this box.

### Area L: build configuration

Files: `ggml/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt`, `docs/build.md`.

- `-DCMAKE_CUDA_ARCHITECTURES=60` (or `native` on the server) is mandatory; the default list has
  no `60`, and a non-native build silently degrades (`fp16_available(600)` false, slow Q8_0
  dequant, wrong MMQ config). Verify in the build log that only `sm_60`/`compute_60` is compiled;
  a single arch also halves build time.
- `-DGGML_CUDA_NCCL=ON` is the default; confirm `find_package(NCCL)` succeeded in the CMake log,
  otherwise `-sm tensor` uses the butterfly fallback and logs "NCCL is unavailable".
- `GGML_NATIVE=ON` (default) gives AVX2 host kernels for the CPU-side ops.
- `GGML_CUDA_GRAPHS`: irrelevant until G2 removes the arch gate; then it must be on.
- `GGML_CUDA_FA_ALL_QUANTS`, `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`,
  `GGML_CUDA_NO_PEER_COPY`, `GGML_CUDA_NO_VMM`: no benefit here (reasons in areas A, C, F, H). Leave
  defaults.
- Profiling build: add `-DCMAKE_CUDA_FLAGS="-lineinfo"` (no perf cost) and once per big kernel
  change `-Xptxas -v` to catch register spills (`sm_60` has 64K registers per SM; spills at
  occupancy 2 x 256 threads mean > 128 regs per thread).
- Compiler: CUDA 12.6 supports gcc up to 13; record `gcc --version` and `nvcc --version` in the
  fact sheet. `-DCMAKE_BUILD_TYPE=Release`; `GGML_LTO` optional (host side only).
- Keep `build-stock/` (upstream base, same flags) next to `build/` for every A/B and determinism
  check.

## 3. Roadmap: what to do in which order

Every item name refers to Part 2. Dependencies are in parentheses. Do not start a tier before the
Tier 0 numbers exist; they decide which items survive.

### Tier 0: measure and configure, no code (`[SRV]`, ~2 days)

| Item | What | Decides |
|---|---|---|
| Part 1 | hardware fact sheet, P2P/IOMMU state, throttle behavior, slot/link check | everything in F; benchmark validity |
| 4.1 | baseline matrix, branch vs stock, both split modes | the reference for every A/B |
| G1 B1 A1 E1 F2 | `nsys` breakdowns for tg (4k, 64k) and pp (`-ub 512`, `2048`) | which of G / C / F / B is worth most |
| F1 F3 F4 | NCCL transport check, NCCL env sweep, `GGML_CUDA_P2P=1`, `GGML_CUDA_ALLREDUCE=none` A/B | comms transport, recommended env |
| J1 | per-request timeline, `--cache-ram 0` A/B | the ~1 s per request |
| D1 E2 E3 | `GGML_CUDA_FUSE_LOG=2` census | remaining fusion gaps |
| I4 G5 I5 | `-t`/`--poll`, `CUDA_SCALE_LAUNCH_QUEUES=4x`, governor | recommended command line |
| K1 H1 | load-mode timing, memory breakdown | respawn cost, memory budget |
| C1 B3 | `test-backend-ops perf` for FA and MMVQ at the model's shapes | C2/C3 and B4 go/no-go |

Deliverable: `P100-HARDWARE.md`, the first `P100-OPTIMIZATION-LOG.md` entries, and a revised
recommended `llama-server` command line with the env vars that measured better.

### Tier 0 results (2026-09-03, krk-lab, details in `~/p100-opt/P100-OPTIMIZATION-LOG.md`)

Baseline Qwen3.8-27B Q4_K_M, `-sm tensor -fa on -ub 2048`: branch pp2048 421 t/s, tg 29.3 t/s at
d0 (stock 417 / 21.6); at d16384 branch 338 / 28.4. Clocks held 1328 MHz, no throttling.

tg step (34.1 ms, 1620 launches per card, GPU busy 97.4%): `mul_mat_vec_q` 26.2 ms (77%, 315 GB/s
= 52% of the 603 GB/s ceiling; q6_K 233 GB/s, q8_0 384), NCCL 2.3 ms (6.6%), rms_norm 1.6 ms (209
launches), quantize_q8_1 0.6 ms (257 launches), flash_attn_tile 0.6 ms at 4k / 3.3 ms at 32k,
915 glue launches 3.5 ms (10%), idle 2.6%. pp at `-ub 2048`: GEMM 66% at 17.0 TFLOP/s (98% of
HFMA2 peak), `convert_unary` (F16 -> F32 output pass) 9.4%, NCCL 9.4%, GDN 6.1%, dequant 3.3%,
FA 1.8%; `-ub 512` is 45% slower. NCCL default transport is SHM; `NCCL_P2P_LEVEL=SYS` gives P2P
(+1.5% pp, +0.9% tg). Host knobs (-t, --poll, launch queues) are noise.

Verdicts: tg is matvec-bound. KILLED: G2/G3 (graphs), F5 (internal allreduce), I4/G5/I6 (host),
A4-as-speed, `-ub` reduction. SUPPORTED: B4 (HFMA2 Q4_K/Q6_K GEMV, the only lever above 3 t/s:
80% of ceiling would give ~40 t/s), A2/A3' (vectorize `convert_unary` and dequant: up to ~10% pp),
E5 (rms_norm at 7.5 us per launch is slow for 5120 floats), B6 (quantize launches), C2 (long
context only), D2 (pp only), F3 as env. N1 (n_kv-sized DtoH copies) RETRACTED after diagnosis:
they are llama-bench's `llama_state_seq_get_data` of the `-d` prefill state, a one-time burst
between prefill and decode, outside the measured window and absent in the server. The long-context
tg falloff is attention growth (C2), not PCIe.

Per-request cost (J1, measured with the branch server, 25k-token chat, 10 follow-ups of 32 new
tokens): median 1.98 s wall, 1.30 s prompt phase, identical with `--cache-ram 16384` and `0`
(the slot is re-selected by LCP similarity, so the host KV save never runs; J2 is dead, but
`--cache-ram 0` still frees 16 GiB of host RAM with `--parallel 1`). The cost sits inside the
prompt phase: 1 checkpoint restore (37 ms), 3 erases (24 ms), 3 x create_checkpoint of the
149.6 MiB recurrent state (467 ms) and 3 separate small decodes of 10/18/4 tokens (~772 ms),
because `tools/server/server-context.cpp` ~:3523-3548 (upstream PR 20288) breaks the prompt batch
at the user-message start and at 4 and 4+n_ubatch tokens before the end, bypassing
`--checkpoint-min-step` (~:3598). J4 is therefore the item: one checkpoint per prompt phase and
one ubatch, upper bound 1.98 -> ~1.1 s per follow-up, context independent. Design question to
settle first: which of the three checkpoints the `n_past--` re-decode and a mid-turn prefix
mismatch actually need. K1: mmap loads in 4 s warm; `--load-mode dio` takes 70 s on the HDD.
Keep mmap.

### Tier 1: small, high-confidence code changes (each < 1 day, one commit, kill switch)

| Item | Change | Expected | Depends on |
|---|---|---|---|
| G2 | CUDA graphs on Pascal (remove the `cc < VOLTA` gate), validate with `-sm none` first | +10-30% tg single card if launch-bound | G1 |
| A4 | chunked dequant + GEMM for huge src0 (LM head) | removes the OOM, frees ~1.3 GB per card | A1 (optional) |
| F5 | internal allreduce on sm_60 (`__nanosleep` replacement) | A/B vs NCCL at tg; may win 1-3 ms per token | F1 F2 |
| B2 | re-enable MMVQ gate/up fusion on Pascal | 0-3% tg | B1 |
| C4 | `parallel_blocks` logging + forced values for long KV | up to +10% tg at 64k | C1 |
| E5 | `rms_norm` register cache width for `n_embd` 5120 | < 1% | E1 |
| D4 E6 | one or two extra fusion rules from the census | ~0.5% per launch removed | D1 E1 |
| I1 | prefilter condition vs suppress tokens | up to 2% tg if the prefilter was off | I1 measurement |
| A2 | vectorized dequant for Q4_K/Q6_K | 2-10% pp at small `-ub`, ~1% at 2048 | A1 |
| J4 | DONE 2026-09-03 (`p100x: 31-server-ckpt-adopt`, worktree wt-j4): adopt the just-restored checkpoint, no forced break at the last user message; `LLAMA_SERVER_CKPT_LEGACY=1` restores upstream | follow-up 1.98 -> 1.41 s wall, prompt phase 1302 -> 732 ms | - |
| A8 | MMVQ column-chunk loop for 9-64 columns on sm_60 | 28-token decode ~390 -> ~150 ms per follow-up; speculative widths > 8 | J4 numbers |
| J4b | make the 150 MiB checkpoint save faster (288 shard copies + sync each, 200 ms vs 37 ms restore) | ~150 ms per follow-up | J4 |
| B4c | DONE 2026-09-04 as `p100x: 34-mmvq-k-shortk`: prefetch mode 2 (header loaded in its own step) per type and width, at width 1 only for Q5_K below 24 blocks; tg +0.5% (31.7 t/s at d0), speculative verify widths 2/4 +4% / +2%; measured with a new any-shape timing tool (`~/p100-opt/b4c/shape.cpp`) that Area B work should use from now on | the gap itself stands: k=5120 runs at 0.74-0.81 of the k=14336 rate, the streaming part is within 3% of the DRAM ceiling and only a fixed 0.38 step per warp is addressable; a shorter warp step is measured out (see What not to do); the LM head was already at its ceiling | B4 B4b |

### Tier 2: medium changes (design note to the user first, 2-5 days each)

| Item | Change | Expected | Depends on |
|---|---|---|---|
| B4b | DONE 2026-09-04 as `p100x: 33-mmvq-k-hfma2`: Q5_K 1.36x and Q6_K 1.40x at width 1 per call (1.14x / 1.30x in-model, short k again), IQ4_XS only from width 4 (its int8 path already runs at 400 GB/s, see What not to do); tg 30.0 -> 31.6 t/s at d0 (+5.2%; +7.8% over the int8 path), 28.8 -> 30.3 at d16384 | left: two-slot `a16k` cache for mixed-type gate/up pairs at widths 4-8 (~1.7% of kernel time), width 6-7 defaults interpolated | B4 |
| G3 | per-subgraph CUDA graph cache so `-sm tensor` can use graphs | the bulk of the launch-bound share found in G1 | G2 success |
| C2 | retune the Pascal FP16 tile table (D=128, D=256) | +5-15% tg and pp at 32k+ | C1 |
| C3 | VEC for GQA tg on Pascal with a KV-length threshold | measured per KV length | C1 |
| F6 | size-dependent allreduce transport | tg wins from F4/F5 without losing pp | F4 F5 |
| F9 | batch the synchronous split-input copies | < 1 ms per token, only if the timeline shows them | F2 |
| D2 | GDN kernel occupancy at tg | a few % of the delta-net share | D1 + nvprof |
| H3 H4 | `-ub` vs compute buffer, unified KV second slot | memory for context or slots | H1 A1 |

### Tier 3: research (weeks, only with Tier 0-2 numbers behind them)

| Item | Change | Why it might pay | Why it might not |
|---|---|---|---|
| B4 | DONE 2026-09-03 as `p100x: 32-mmvq-q4k-hfma2` (Q4_K only): 1.49x at width 1 (471 GB/s), up to 2.85x at widths 5-8 per call; tg +2.5% | Q4_K is only 17.7% of the tg kernel time on the UD-Q4_K_M (Q5_K 24.7%, IQ4_XS 19.4%, Q6_K 9.0%): B4b carries the rest | the k=5120 shapes reach 373 GB/s against 459 at k=14336 (B4c) |
| A7 | HFMA2 tiled GEMM with in-register dequant | removes the F16 round trip and temporaries | cuBLAS is already at ~68-80% of peak |
| F7 | P2P direct allreduce kernel for 2 GPUs | < 10 us per collective vs 20-40 | needs stable P2P; Pascal has no `__nanosleep` |
| G4 | fewer subgraph boundaries in the meta backend | 1 CUDA graph per device per token | deep change in `ggml_backend_meta_graph_compute` |
| C7 | quantized KV shards in the meta backend | context or slots | TILE dequantizes the whole cache per call; only with VEC |
| I3 | backend sampling with vocab-sharded logits | removes 1 MB D2H + CPU sort per token | small share; needs meta-aware gather |
| F8 | fewer collectives per layer (sequence parallel) | halves PCIe traffic | model-graph rewrite |

### What not to do

- Do not touch `ggml-backend-meta.cpp` beyond what F/G need; upstream churns it and the gist
  already carries a 236-line rewrite. Every extra line there is rebase cost.
- Do not retune MMQ for dense on sm_60 (A6 is a one-off number, not a project).
- Do not try quantized KV, `-sm row`, `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`,
  or unified memory; the reasons are in Part 2.
- Do not add tests under `tests/`; use `test-backend-ops` cases and the protocol in section 4.
- Do not retry the IQ4_XS HFMA2 matvec at widths 1-3 with a better table lookup (B4b, 2026-09-04):
  its int8 path already runs at ~400 GB/s on GP100 (70% of the DRAM ceiling, 78 us on 4096x14336),
  the best HFMA2 config (16 instructions per 8 weights) is 6% slower there and costs 4.8% tg in the
  model. A win needs a different kernel structure, not a better table.
- Do not shorten the HFMA2 K-quant warp step (4 blocks per step, half-step pipeline) to fix short k
  (B4c, 2026-09-04): halving the bytes in flight per warp lost 18-26% per call; the kernel's streaming
  part is already within 3% of the DRAM ceiling and only the fixed pipeline cost (0.38 of a step per
  warp) remains. Measure Area B changes with `~/p100-opt/b4c/shape.cpp` at the model's shapes and at
  per-GPU row counts (`-sm tensor` halves the rows a GPU sees), not only on the 4096x14336 test shape.

## 4. Measurement and validation protocol

Nothing in this plan is accepted on a single number. The P100s are passive and the machine
is power-limited, so the noise floor is higher than on a normal box.

### 4.1 Benchmark protocol (`[SRV]`)

Before every session:

```sh
sudo nvidia-smi -pm 1
nvidia-smi -q -d PERFORMANCE | grep -A12 'Clocks Throttle'     # must show all "Not Active" at idle
nvidia-smi dmon -s pucvmet -d 2 > dmon.$(date +%s).log &        # keep running during the whole session
```

Warm-up: run the first `llama-bench` line twice and discard the first run. The cards heat up
within ~60 s and the clock they settle at is the one that matters.

Standard matrix (both builds, `build/` = branch, `build-stock/` = upstream base):

```sh
B=./build/bin/llama-bench
# tg and pp at several context depths; -r 5 repetitions; markdown output
$B -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -p 512,2048 -n 64 -d 0,8192,32768,65536 -r 5 -o md
# batch sweep for pp (pp is what -ub changes)
$B -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -p 2048 -n 0 -b 2048 -ub 128,256,512,1024,2048 -r 3 -o md
# split mode comparison (same everything, only -sm changes); -sm row may not support all ops
$B -m $MODEL_Q4 -ngl 99 -sm layer  -fa on -p 2048 -n 64 -d 0,32768 -r 5 -o md
$B -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -p 2048 -n 64 -d 0,32768 -r 5 -o md
# small-batch tg (speculative verify shape): batch 2..8 through mmvq, 9+ through cuBLAS
$B -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -n 0 -p 2,4,8,9,16,32 -r 5 -o md
# single card for reference (fits only smaller quant or lower ctx; use Q4 with -c small)
CUDA_VISIBLE_DEVICES=0 $B -m $MODEL_Q4 -ngl 99 -fa on -p 512 -n 64 -r 5 -o md   # may OOM at 27B, then use -ngl 40
```

Rules:

- Report median of `-r 5`, plus the min/max spread. A change smaller than the spread is not a
  result; raise `-r` or fix cooling.
- Always paste the `nvidia-smi dmon` clock column range for the run next to the number.
- A/B runs alternate (A, B, A, B) instead of AAAAA BBBBB, so drift affects both equally.
- Keep the exact command line with every number in `P100-OPTIMIZATION-LOG.md`.
- `scripts/compare-llama-bench.py` compares two `-o sql`/json outputs; use it for the big matrix.

Server-level numbers (the real workload) come from `llama-server --metrics` and the `timings`
object in responses (`prompt_ms`, `predicted_ms`, `prompt_per_second`, `predicted_per_second`).
For the "fixed ~1 s per request" item use a fixed script of 20 follow-up requests over a 30k
context and record per-request `prompt_ms` and wall time.

### 4.2 Correctness protocol

1. `./build/bin/test-backend-ops -b CUDA0` and `-b CUDA1` after every kernel change (the full
   unfiltered suite; see `P100-PATCHES.md` on how to treat a single MUL_MAT failure).
2. `./build/bin/test-backend-ops -o <OP> perf -b CUDA0` for the op that was changed; keep the
   before/after table.
3. Greedy determinism: same prompt, `--temp 0 --seed 1`, 256 tokens, branch build vs stock
   build. Any patch that does not intend to change math must give identical output to the last
   token. Patches that do change math (F16 accumulation paths) must document it and pass a
   perplexity check: `./build/bin/llama-perplexity -m $MODEL_Q4 -f wiki.test.raw -c 2048 --chunks 20`
   within 0.5% of stock.
4. `./build/bin/test-llama-archs` for anything touching `ggml-backend-meta.cpp` or the scheduler.
5. Soak: the shape that used to crash (`P100-PATCHES.md`, test checklist): 50k+ context, then a
   near-exact prefix-hit request. Run once before declaring a scheduler/meta/graph change done.

### 4.3 Profiling recipes for Pascal

```sh
# timeline + per-kernel time summary (works on Pascal)
nsys profile -t cuda,nvtx -o tg.$(date +%s) --stats=true ./build/bin/llama-bench -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -p 0 -n 128 -r 1
nsys stats --report cuda_gpu_kern_sum tg.*.nsys-rep      # kernel name, total time, count, avg
nsys stats --report cuda_gpu_mem_time_sum tg.*.nsys-rep  # memcpy/memset time incl. P2P and H2D

# per-kernel hardware counters on Pascal: nvprof (ncu does not support cc 6.0)
nvprof --metrics achieved_occupancy,dram_read_throughput,dram_write_throughput,gld_efficiency,sm_efficiency \
       --kernels "mul_mat_vec_q" ./build/bin/llama-bench -m $MODEL_Q4 -ngl 99 -p 0 -n 16 -r 1
nvprof --print-gpu-trace ...                              # per-launch durations and grid sizes

# register/spill report at compile time
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v" && cmake --build build -j 2>&1 | tee ptxas.log
grep -B1 -A2 'spill' ptxas.log | grep -v '0 bytes spill' | head    # any non-zero spill on sm_60 is worth a look
```

A useful per-op breakdown without a profiler: `GGML_SCHED_DEBUG=2` (prints the split/graph
assignment per node), `GGML_CUDA_FUSE_LOG=2` (patch 18: which fusion rules fired or why not),
`GGML_META_DEBUG=1` (meta backend rebuilds). With `nsys` the per-token time budget for tg is:
sum of kernel time + memcpy time + gaps. The gaps are host overhead and cross-GPU waits; that
split decides whether items G (graphs/host) or F (PCIe) are worth more than kernel work.

### 4.4 What a finished item looks like in `P100-OPTIMIZATION-LOG.md`

```
## 2026-09-xx  G2 CUDA graphs on Pascal (numbers below are placeholders showing the format)
Commit: p100x: 31-cuda-graphs-pascal (or: no commit, measurement only)
Setup: Qwen3.8-27B Q4_K_M, -sm none on CUDA0 -ngl 40, clocks 1240-1290 MHz, no throttle flags
Command: llama-bench ... -n 128 -r 5 -o md
Result: tg = 31.0 (30.6-31.4) -> 36.2 (35.9-36.5) t/s, +17%; pp unchanged
Correctness: test-backend-ops CUDA0/CUDA1 pass; greedy identical to stock for 256 tokens
Kill switch: GGML_CUDA_DISABLE_GRAPHS=1
Verdict: keep; next G3 for -sm tensor
```
## 5. Consolidated gotchas

Hardware and system:

1. GP100 has fast FP16 (HFMA2) but no DP4A; consumer Pascal (sm_61) is the reverse. Code paths
   gated on `cc >= DP4A` are off here, and int8 dot products are emulated (4x `vmad`).
2. The cards are passive; clocks sag 1328 -> ~1100 MHz over a long run (-17%), larger than most
   effects under test. Warm up, log clocks, alternate A/B arms.
3. Two 250 W cards on a 700 W PSU: `SW Power Cap` is plausible under pp; consider `-pl 200`.
4. Nsight Compute does not support Pascal; `nsys` and `nvprof` do. `ncu` output claiming to profile
   this GPU is wrong by construction.
5. `nsys` CUDA tracing needs free GPU memory; at 94% VRAM use a smaller `-c` for profiling runs.
6. P2P over a Broadwell root complex works but is off by default in both the driver (needs the
   right IOMMU/ACS/BIOS state) and llama.cpp (`GGML_CUDA_P2P` opt-in). NCCL picks its transport on
   its own; read `NCCL_DEBUG=INFO`.
7. R580 is the last driver branch for Pascal; CUDA 12.x is the last toolkit line. Never upgrade
   either past that on this box.
8. Z440 slot 4 is x8 electrical; the cards belong in slots 2 and 5. Check `LnkSta` after any
   hardware change.

Build:

9. `-DCMAKE_CUDA_ARCHITECTURES=60` (or `native`) is mandatory; the default arch list has no `60`
   and a non-native build silently picks slower paths.
10. Confirm NCCL was found at configure time; otherwise `-sm tensor` runs the slow butterfly.

llama.cpp behavior on this hardware:

11. CUDA graphs are hard-disabled below Volta; every kernel is launched individually. This is
    the single most important fact for tg on this machine.
12. `-sm row` no longer exists on CUDA; `-sm tensor` needs `-fa on` and F16/BF16/F32 KV only.
13. Backend (GPU) sampling is refused under `-sm tensor`; CPU sampling patches 11/13 are the relief.
14. The sampler prefilter (patch 13) turns itself off with any logit-bias sampler in the chain,
    including vocab suppress tokens; verify it is live.
15. FA at tg with F16 KV and GQA uses the TILE kernel, not VEC, and the Pascal FP16 tile table is
    an upstream TODO. Quantized KV at batch > 2 dequantizes the whole cache per call.
16. `GGML_CUDA_FORCE_MMQ` cannot enable MMQ for dense matmul on sm_60; the DP4A rule returns first.
17. cuBLAS temporaries are per-call pool allocations of the whole weight matrix in F16; the VMM
    pool never shrinks, so one wide LM-head call permanently raises the high-water mark (the OOM).
18. The meta backend closes a subgraph at every PARTIAL node: 128 subgraphs per device per token
    for 64 layers. Any CUDA-graph work under `-sm tensor` must cache per subgraph (G3).
19. `-sm tensor` runs the scheduler with one copy and no events; pipeline parallelism only exists
    for `-sm layer`.
20. Split inputs (embedding row, positions, out_ids, mask, idxs) are copied synchronously after a
    sync, twice (mirrored). `tok_embd` always lives on the CPU by policy.
21. Hybrid/recurrent memory forces equal, sequential ubatch splits and bounds tokens per sequence
    by `n_ubatch / n_seqs`; with `-np 2` an effective ubatch above half of `-ub` is unreachable.
22. The server saves the slot's whole KV to host RAM on every new task by default
    (`--cache-idle-slots`, `--cache-ram 8192`); at long context that is gigabytes over PCIe per
    request. `--cache-ram 0` disables it.
23. `--cache-reuse` needs a shiftable memory; recurrent/hybrid models do not qualify.
24. Speculative verify batches wider than 8 leave MMVQ for cuBLAS and trigger the LM-head F16
    copy; keep drafts <= 7 until A4 lands.
25. Graph reuse breaks on any ubatch shape change (MTP with variable draft length), costing a
    rebuild plus a 15-20 ms meta re-split; `--draft-p-min 0` keeps the shape constant.
26. The pinned async model upload exists only without mmap (`--load-mode dio`); mmap and the 8 GB
    host cache compete for page cache on 32 GB.
27. Greedy output must be bit-identical to stock for every patch that does not change math;
    F16-accumulating changes (B4, F5's BF16 wire, cuBLAS compute type) must document it and pass
    the perplexity check.
28. Upstream churns `ggml-cuda.cu` (fusion region), `mmvq.cu`, `common/sampling.cpp` and
    `ggml-backend-meta.cpp`; every new patch in those files is rebase cost. Keep patches small
    and behind kill switches.

## 6. Open questions for the user (answered 2026-09-03 where marked)

Answered: 1 (ssh krk-lab, paths in 0.1), 2 (no sudo), 3 (production flags in 0.1), 4 (drift allowed
with the perplexity check), 5 (single stream; `-np 1` in production), 6 (Gemma-4 31B is present;
`-sm tensor` arch support still to verify), 8 (router mode with `--models-dir` is the production
setup, respawn/load time matters). Open: 7 (P2P/NCCL env in production after tests).

Original list:

1. How does the implementing agent reach the server: runs on it, or ssh from the workstation?
   Paths for the checkout, build dirs and the GGUF files (section 0.1 fill-in block).
2. Does the agent get sudo on the server (clocks, power limit, persistence mode, IOMMU kernel
   parameter, drop_caches)? If not, which of those may the user run on request?
3. The exact production `llama-server` command line and preset (flags, `-c`, `-np`, `--cache-ram`,
   speculative settings, router mode). The plan's J1/J5/J6 items depend on it.
4. Is F16 accumulation acceptable where it changes numerics slightly (B4 HFMA2 matvec, F5 BF16 on
   the wire, cuBLAS compute type), given a perplexity check within 0.5%? Or must every change stay
   bit-identical to stock?
5. Priority between tg latency and throughput: is a second concurrent slot (`-np 2`, area H4)
   wanted, or is everything about single-stream speed?
6. Is Gemma 4 31B available now (C6 needs the GGUF) and is `-sm tensor` confirmed to accept its
   arch on this build?
7. May the agent enable `GGML_CUDA_P2P=1` and change NCCL env vars in production after the tests,
   given the documented risk of instability on some boards (P100-PATCHES.md crash history)?
8. Is the router-mode multi-model setup part of the target (respawn/load times, area K), or is a
   single long-lived instance the only case that matters?

## Appendix A: environment variables worth knowing (this tree)

| Variable | Meaning |
|---|---|
| `GGML_CUDA_P2P=1` | enable peer access for all GPU pairs at init (off by default) |
| `GGML_CUDA_ALLREDUCE=nccl|internal|none` | pick the tensor-split reduction path (internal refuses sm_60 today) |
| `GGML_CUDA_AR_COPY_THRESHOLD`, `GGML_CUDA_AR_COPY_CHUNK_BYTES`, `GGML_CUDA_AR_BF16_THRESHOLD` | internal allreduce tuning (after F5) |
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P`, `NCCL_P2P_LEVEL`, `NCCL_PROTO`, `NCCL_ALGO`, `NCCL_MAX_NCHANNELS`, `NCCL_NTHREADS`, `NCCL_BUFFSIZE` | NCCL transport and tuning (F1, F3) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16|f32|bf16|auto` | cuBLAS compute type override (A3) |
| `GGML_CUDA_DISABLE_GRAPHS=1` | CUDA graphs off (relevant after G2) |
| `GGML_CUDA_GRAPH_OPT=1` | graph reordering, single device only |
| `GGML_CUDA_DISABLE_FUSION=1` | all CUDA fusion off (upstream + patch rules) |
| `GGML_CUDA_FUSE_LOG=1|2` | patch 18: fusion diagnostics |
| `GGML_CUDA_DISABLE_FUSE_*`, `GGML_CUDA_DISABLE_CPY_ROWS`, `GGML_CUDA_DISABLE_CONCAT_ROWS` | per-patch kill switches (table in `P100-PATCHES.md`) |
| `GGML_A16_*` | patch 12 Q4_1 HFMA2 kernel tuning |
| `GGML_CUDA_NO_PINNED=1` | pinned host memory off (never for production) |
| `GGML_CUDA_REGISTER_HOST=1` | register the mmap region with CUDA (K1) |
| `GGML_OP_OFFLOAD_MIN_BATCH` | min batch to offload a host-weight op (default 32) |
| `GGML_META_DEBUG=1` | meta backend rebuild/split logging |
| `GGML_SCHED_DEBUG=1|2`, `GGML_SCHED_DEBUG_REALLOC=1` | scheduler assignments, unexpected reallocs |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | llama graph reuse off (-27% tg here; diagnostic only) |
| `LLAMA_SAMPLER_PREFILTER=0` | patch 13 off |
| `LLAMA_MTP_DRAFT_VOCAB=<file>` | patch 15 draft vocab subset |
| `LLAMA_SERVER_SLOTS_DEBUG=1`, `LLAMA_TRACE=1` | server slot/trace logging (J1) |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` | larger CUDA command buffer (G5) |
| `CUDA_VISIBLE_DEVICES` | single-card runs |

## Appendix B: reading order for a new agent

1. `AGENTS.md`, then `P100-PATCHES.md` end to end.
2. This file: sections 0, 2.0, 2.1, 3, 5.
3. `docs/multi-gpu.md` and the "Runtime CUDA environmental variables" and "Performance Tuning"
   sections of `docs/build.md`.
4. Before touching an area, read the files in its "Files" line completely, plus
   `git log --oneline -20 -- <file>` to see what upstream changed recently and which `p100:`
   commits are there.
5. Before writing a kernel, read the existing kernel it replaces and one comparable tuned kernel
   (e.g. `dequantize_block_q8_0_f16` for dequant, `mmvq-f16-sm60.cu` for HFMA2 matvec,
   `fattn-tile.cuh` config tables for FA).
