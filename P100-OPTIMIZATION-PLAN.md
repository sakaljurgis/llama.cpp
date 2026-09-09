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

State found 2026-09-09 (read-only, both P100s): persistence mode Disabled (the `nvidia-persistenced`
service is active but does not manage these GPUs), compute mode Default, application clock 1189 MHz
default / 1328 max (steady-state benchmarks already average 1319, so `-ac 715,1328` only shortens the
ramp after idle), ECC on (native HBM2, keep), PCIe Gen3 x16 on both, IOMMU off (0 groups, nothing to
do for P2P), ASPM policy `default` (BIOS setting; link L1 state needs root to read), CPU governor
`schedutil`, C-states C3/C6 disabled and C1/C1E enabled, THP madvise. Do NOT set compute mode
EXCLUSIVE_PROCESS: the router spawns one child process per model, and exclusive mode allows one
context per GPU. What is worth doing sits in the Final roadmap table as PL1-PL3.

Update 2026-09-09 afternoon: the user enabled persistence mode permanently (the daemon unit's
`--no-persistence-mode` removed) and installed the root helper `/usr/local/sbin/p100-root.sh` (draft and
install notes in `~/p100-opt/root/`; sudoers `/etc/sudoers.d/p100-root`, to be removed after the session).
`p100-root.sh link` read the P100 links as root: `ASPM not supported`, `ASPM Disabled` on both, so the
ASPM knob of PL3 is moot and only the governor and the C1E hold remain. Answer to open question 9: the
helper script.

Item PL1 (added 2026-09-09 at the user's request; scheduled LAST, after every code item of Tiers 1-3):
power-limit sweep to find the sweet spot for pp and tg. The cards report `Min Power Limit` 125 W,
`Default/Max` 250 W (2026-09-09), and a full pp run sits at 165-231 W per card on a 700 W PSU with a
135 W CPU. tg is bandwidth-bound (HBM2 stays at 715 MHz whatever the limit) so it should barely move;
pp is HFMA2-bound and follows the graphics clock, so it should fall with the limit; the production
MTP verify step (width 4) sits between. Protocol (`[SRV]`, no code, ~2 h of GPU time):

- Arms: 250 (default), 225, 200, 175, 150, 125 W on BOTH cards, run 250 first and again last (drift
  check). `nvidia-smi -pl` needs root and agents may not change GPU state (0.1, `SUDO=no`), so either
  the user runs `sudo nvidia-smi -i <uuid> -pl N` between arms on the agent's request, or grants a
  sudoers line limited to `/usr/bin/nvidia-smi -pl *` for that session (open question 9). The default
  250 W is restored at the end in either case; with persistence mode off the setting also resets when
  the driver unloads, so re-check `Current Power Limit` before every arm.
- Per arm, after a 60 s warm-up: `llama-bench -m $MODEL_Q4 -ngl 99 -sm tensor -fa on -b 2048 -ub 2048`
  with `-p 2048 -n 0 -r 3` (pp), `-p 0 -n 64 -d 0,16384 -r 5` (tg), `-p 4 -n 0 -r 5` (verify width),
  then one production-shape run with the J5 harness (`~/p100-opt/j5`, MTP `n_max` 3, 25k chat, 512
  tokens) for the real t/s. `nvidia-smi dmon -s pucvmet -d 1` for the whole sweep; per arm record the
  mean and max board power per card, mean graphics clock, `SW Power Cap` violation time, temperature.
- Table: limit W | pp2048 t/s | tg d0 | tg d16384 | pp4 | MTP chat t/s | mean W both cards | mean MHz |
  cap time | tokens per joule for tg and pp (t/s divided by the summed board power). Sweet spot = the
  lowest limit where tg and the MTP chat lose < 1% against 250 W, with the pp loss stated next to it;
  the user picks. Deliverable: the table in the log and a one-line recommendation for
  `~/llama-serve.sh` (a `sudo nvidia-smi -pl N` step needs root at boot: recommend, do not install).

Results 2026-09-09 (PL1-PL3, `~/p100-opt/pl/PL-report.md`, Final roadmap table): the two P100s never reach
250 W. tg draws ~155 W per card (peak 168) and is flat down to a 175 W limit (32.04 t/s, MTP chat 39.5 at
every limit from 250 to 175); pp2048 peaks at 237 W per card and follows the clock once capped (-0.9% at
225 W, -3.5% at 200, -7.3% at 175, -11.8% at 150, -22% at 125). No thermal slowdown at any limit (max
64 C). Persistence mode is a no-op here (the display card keeps the driver loaded), governor and C1E hold
do nothing, ASPM is not supported on the links. Production recommendation: no power limit, all defaults;
see section 7. The user's wall meter agrees: no change in system draw until 175 W, -50 W at 150, -100 W at 125.

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
- A2 DONE 2026-09-07 (`p100x: 39-convert-vec`, together with A3'; numbers in the Tier 1 table and
  P100-PATCHES.md 39). The premise below was wrong: block packing alone is a no-op, the kernels are
  store bound, and the fix is the 8-byte store of each contiguous group (Q4_K, IQ4_XS). Original item:
  Vectorize the dequant-to-F16 kernels for the types in use (Q4_K, Q6_K, Q5_K, Q8_0 already
  done; later Q4_K_XL types). Pattern: the existing `dequantize_block_q8_0_f16` (shared memory
  staging, `half2` stores). Target: one super block per warp, 8-byte stores (`ggml_cuda_get_max_cpy_bytes`
  is 8 on sm_60). Gain: bounded by A1's dequant share. Effort: small. Risk: low; `test-backend-ops`
  covers `CPY`/`MUL_MAT` for every type. Kill switch: `GGML_CUDA_DISABLE_DEQUANT_VEC`.
- A3 Measured 2026-09-07: `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` gives pp2048 420.9 -> 272.1 t/s (-35%),
  perplexity 5.4094 against 5.4367; dead as written. What paid is A3' = vectorize the pass itself
  (DONE as `p100x: 39-convert-vec`: `convert_unary` 165 -> 520 GB/s, +5.55% pp2048). Original item:
  Skip the F16 -> F32 output pass when the consumer is the next GEMM or a fusable op: not
  possible without a graph rewrite; instead measure `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (F32
  compute, F32 output directly, no conversion pass, but half the FLOPS). Expected: slower; do it
  once to have the number and to see whether F16 accumulation costs accuracy (perplexity check,
  section 4.2). No code.
- A4 DONE 2026-09-08 (`p100x: 44-cublas-src0-chunk`, numbers in the Tier 1 table and P100-PATCHES.md 44).
  Original item: Bounded-memory GEMM for huge matrices (fixes the LM head OOM documented in
  `P100-PATCHES.md` runtime notes: the ~2.5 GB F16 copy of `output.weight` on the first 9+ column
  verify batch). In `ggml_cuda_mul_mat_cublas_impl`, when `ggml_nelements(src0) * 2` exceeds a
  threshold (env `GGML_CUDA_CUBLAS_CHUNK_MB`, default e.g. 512), loop over row chunks of src0:
  dequantize chunk, GEMM chunk into the matching rows of dst. Same math, same output. Gain: not
  speed, it removes the OOM and frees ~1.3 GB per card of pool headroom that today has to be
  held back via `--ctx-size`. That headroom is worth context or a second slot. Effort: medium.
  Note 2026-09-05: at 9-64 columns A8 (patch 35) routes the LM head through the matvec loop, so
  the F16 copy is no longer allocated there; A4 still matters at `-ub 2048`.
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
- A8 DONE 2026-09-05 (`p100x: 35-mmvq-cols-sm60`, numbers in the Tier 1 table and P100-PATCHES.md 35).
  Original item: Small batches (9-64 columns) without the full dequant: measured 2026-09-03 (J4 notes), a
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
  Correction 2026-09-07: not trivial any more, see the Tier 1 row: `ggml_cuda_mmvq_k_f16_sm60_supported`
  returns false when `fusion` is set, so the fused pairs would leave the HFMA2 kernel. Needs a fused
  GLU epilogue in that kernel first.
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

Correction (2026-09-05, C2 measurements): the primary model is D=256 with GQA 6 (24 q heads, 4 kv
heads; 12 and 2 per GPU under `-sm tensor`), not D=128; the Tier 0 C1 numbers at GQA 4 and 8 do not
bracket it, because the launcher packs by powers of two and 6 landed on `ncols2 = 2` (three blocks
per kv head each streaming the whole K/V); and only the tile kernel is ever selected for this model
(every launch is in `~/p100-opt/log/C2-41.log`). The table's `occupancy` field is a register budget
(`65536/(nthreads*occupancy)`), not a block count: every `ncols 2/4` FP16 entry at 64/128 threads
and occupancy 2 has no effective register limit. Patch 36 fixes the D=256 entries; the D=128 ones
are probably the same one-line fix (unmeasured).

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
- C2 DONE 2026-09-05 (`p100x: 36-fattn-tile-p100`, numbers in the Tier 2 table and P100-PATCHES.md 36).
  Original item: Tune the Pascal FP16 tile table (`ggml_cuda_fattn_tile_get_config_nvidia_fp16`) for D=128
  and D=256: sweep `nthreads` (128/256), `occupancy` (2/3/4), `nbatch_fa` (32/64/128), `nbatch_K`
  (32/64/128) at ncols 2/8/16/32 with the C1 harness. The table is a plain constexpr switch, so a
  sweep is a rebuild per point (or make it env-overridable in a scratch build:
  `GGML_CUDA_FATTN_TILE_CFG=nthreads,occ,nbfa,nbk`). Hard limits: 48 KB smem per block, 64K
  registers per SM (check `-Xptxas -v` for spills at higher occupancy). Expected gain: 1.2-2x on
  the attention kernel, i.e. +5-15% tg and pp at 32k+ context, nothing at 4k. Kill switch: none
  needed if the table is just retuned (document old values); add `GGML_CUDA_FATTN_TILE_LEGACY=1`
  if the change is structural.
- C3 NO (2026-09-05, measured in C2: 1.4-1.8x slower at every KV length, see the Tier 2 row).
  Original item: Try VEC for tg with GQA on Pascal: in a scratch build, return VEC for
  `Q->ne[1] == 1` regardless of `gqa_opt_applies` on the Pascal branch and compare C1 numbers at
  KV 4k/32k/64k. VEC processes one q column per block with 128 threads; TILE with `ncols2` GQA
  packing shares K/V loads across the GQA group, which is why upstream prefers it. Measure
  instead of guessing; keep whichever wins per KV length (a KV-length threshold is a fine rule).
- C4 NO (2026-09-05, measured in C2: the search fills one wave, forcing more costs 6-83%, see the Tier 1
  row; the ragged-last-tile tie-break is C2b). Original item: `parallel_blocks` on sm_60: log the chosen `parallel_blocks` (add a one-line `GGML_CUDA_FATTN_LOG`
  debug print in a scratch build) for tg at 4k/32k/64k. 56 SMs with occupancy 2 means 112 blocks
  fill the card; with `ncols2` GQA packing and 8 KV heads a single-token decode has few blocks
  unless `parallel_blocks` is large. If the search stops early (95% rule) at long KV, force
  higher values and measure. Small code change, potentially large tg gain at long context.
- C5 8-byte vs 16-byte loads: `ggml_cuda_get_max_cpy_bytes` returns 8 on sm_60 deliberately
  (Pascal has 16-byte vector loads, `LDG.128`, but the upstream author chose 8; find the commit
  and its reason with `git log -S"ggml_cuda_get_max_cpy_bytes"`). Test 16 in a scratch build with
  `test-backend-ops` (correctness) and C1 (speed). Alignment asserts may fire; that is the answer
  then.
- C6 DONE 2026-09-07 (measurement only; `~/p100-opt/log/C6-report.md`, `C6-notes.md`, LOG entry):
  `-sm tensor` is accepted: `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp` ~1116, called from
  `llama-model.cpp` ~351) is a deny-list and gemma4 is not in it. Confirmed: 50 iSWA layers D=256
  GQA 2 on a flat n_kv 1280 cache, 10 global layers D=512 GQA 8, n_embd 5376, n_ff 21504, Q4_K 68% /
  Q6_K 17% / Q5_K 15% by bytes; the softcap is a final logit softcap, no FA launch uses the softcap
  template; the retuned `(256, 256, 2)` entry is 1.003x at the shipped n_kv 1280 (the 1.13x below was
  measured at GQA 6). No FA regression on 11 launch shapes; the C2b tie-break gains 1.094x on the
  previously unmeasured D=512 GQA 8 launch at n_kv 4096. tg 1.58-1.65x stock (17.5 -> 28.8 t/s at
  d0), pp2048 1.06x. Found the nan logits at matvec widths on this model (perplexity nan at `-ub 1`, chat output
  degenerates): a half range overflow in the Q6_K epilogue of patches 32/33, fixed the same day by
  B4d (`p100x: 41-mmvq-k-f16-range`); chat output is now byte-identical to stock at 1.63x stock tg.
  Also: patch 40's norm cache stopped at 5120 columns; E5b (`p100x: 42-norm-cache-5376`) covers
  5376 (+1.6% tg on this model). Original item: Gemma 4 prep: D=256 tile configs
  are in the same table (C2 covers them; note 2026-09-05: its SWA layers are D=256 GQA 2 and take
  the retuned `(256, 256, 2)` entry, 1.13x per launch at n_kv 1024; the global layers are D=512
  GQA 8, untouched); logit softcap only selects a template. iSWA layers use a small cache
  (`n_swa`), so their attention cost is flat. Verify `-sm tensor` supports the gemma4 arch before
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
  Census 2026-09-09: fires. One `ssm_conv_f32` per delta-net layer (48 per token per card, 3.0 us), no
  standalone silu kernel anywhere in the profile.
- D4 The `L2_NORM` pair (q and k) fuses as a sibling run (patch 18); confirm. The `RMS_NORM +
  SILU + MUL` gated norm at the layer output may have no rule: check D1's log, and if it is 3
  launches, add a rule modeled on patch 20 (`ADD -> UNARY -> MUL`), `GGML_CUDA_DISABLE_FUSE_RMS_GATE`.
  Census 2026-09-09: the L2_NORM pair fuses (patch 18, `l2_norm_f32_fused` 48 per token). The gated norm
  is 2 launches, not 3: `{RMS_NORM,MUL}` -> `rms_norm_f32` (24,1,1) 3.5 us and `{UNARY(SILU),MUL}` ->
  `unary_gated_op_kernel` (12,1,1) 2.6 us both fire upstream. 2 -> 1 is worth 0.12 ms per token (0.4%);
  the blocker is the z MUL_MAT (node #51) between #50 and #53, which makes `ggml_cuda_collect_ops` bail,
  so it needs a patch 23 style defer, not a chain rule. Ranked 5th in the census; below E7 and E8.
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
  Census 2026-09-09: `RMS_NORM + MUL + ROPE` does NOT fire. The node order already matches (the K side even
  matches the 5-op `VIEW + SET_ROWS` form); `ggml_cuda_should_fuse_rms_norm_mul_rope` rejects
  `mode != NORMAL && mode != NEOX` at ggml-cuda.cu ~2811 and qwen35 uses `ggml_rope_multi` (MROPE = 8).
  32 `rope_multi` + 32 `k_set_rows` per token; fixing it needs a 4-section MRoPE variant of the fused
  kernel, worth 0.12 ms (0.4%). The `SIGMOID + MUL` gate fuses; the `ggml_cont_2d` in front of it does
  not (`cpy_scalar_fastdiv` (48,1,1), 16 per token, 3.2 us).
- E3 `binbcast` residual `ADD` + the next `RMS_NORM`: patch 19 folds the pre-add; verify it fires
  for both the attention and the ffn residual in every layer (count in E1 should be 2 x 64 fused
  launches, not 4 x 64).
  Census 2026-09-09: fires in every layer (128 pre-add `rms_norm_f32` launches per token per card).
- E4 `rope.cu` uses `dim3(1, CUDA_ROPE_BLOCK_SIZE, 1)` blocks with `n_blocks_x = ceil(ne00 / (2*BLOCK))`;
  at tg a single token means very few blocks. Only relevant if E1 shows rope > 2% of the step;
  else skip.
- E5 DONE 2026-09-07 (`p100x: 40-norm-cache-5120`): `n_embd` is 5120, the cache never fired; `max_cache`
  5 gives 1.63x per launch in isolation, 1.16x in the model, tg +0.6%. Original item:
  `rms_norm_f32<1024>` register cache (patch 17) applies when `1024 < ncols <= 4096`; Qwen3.8's
  `n_embd` decides whether it fires (5120 would not: > 4096). Check `n_embd` in the GGUF and, if
  it is 5120, extend `max_cache` to 5 or 6 in a scratch build and measure `test-backend-ops -o
  RMS_NORM perf` at that width. Trivial change if it helps.
- E6 Anything else E1 surfaces with count >= 64 per token and < 5 us each: candidates for a
  sibling-run fusion (patch 18 infrastructure `ggml_cuda_collect_same_op`), with a kill switch.
  Census 2026-09-09: everything at count >= 64 and < 5 us is `quantize_a16k` 197, `quantize_q8_1` 138,
  `unary_gated_op_kernel` 128, `cpy_scalar_fastdiv` 64 and the narrow rms_norms (80). None is an unfused
  graph-node pattern; the quantize helpers are not graph nodes at all (gotcha 36). Outcome: E7 and E8.
- E7 Multi-block rms_norm for the single-row tg case (from the D4/E6 census, 2026-09-09; approved).
  `rms_norm_f32_cuda` maps one row to one block, grid (nrows, nchannels, nsamples), so at tg the 129
  5120-wide norms per token (attn_norm, post_attn_norm, output norm) each run as one 1024-thread block on
  1 of 56 SMs: 8.84 us in the model, 1.14 ms per token = 3.6% of the step, against 3.89 us in isolation.
  Design: split a cached row across ceil(ncols/1024) blocks with a redundant reduction (every block
  computes the identical sum of squares over the whole row, then stores only its chunk; the pre-add form
  writes `pre_dst` for its chunk only), so the result is bit-identical; split only when the total block
  count is small so pp (nrows 2048) is untouched. Env `GGML_CUDA_NORM_SPLIT` (0 = off, N = block-count
  threshold). Expected 0.4-0.7 ms per token (1.3-2.2% tg). Files: `norm.cu` only (patches 17/19/40/42
  live in the same kernel; rebase them as one group). Brief: `~/p100-opt/e7/BRIEF.md`.
  Result 2026-09-09: NEGATIVE, see the Tier 1 row and gotchas 37/38. The pre-add form aliases its output
  with an input (any multi-block norm that reads the whole row races), and the redundant split is 1.22x
  slower per launch because the per-block latency chain is unchanged. Diff kept in `~/p100-opt/e7/`.
- E8 Producer writes the matvec's quantized activation (from the D4/E6 census, 2026-09-09; approved as
  the item after E7). 256 of the 335 quantize launches per token directly follow the kernel that produced
  their input: `unary_gated_op_kernel` (128: FFN SwiGLU 64, delta-net gated norm 48, attention gate 16)
  and `rms_norm_f32` (128: attn_norm, post_attn_norm). Let the producer emit the consumer's block format
  (a16k for Q4_K/Q5_K/Q6_K and IQ4_XS at width >= 4, q8_1 otherwise; both when a mixed-type gate/up pair
  follows) into the existing single-slot activation cache and set the cache key, so the consumer's lookup
  hits and its quantize launch disappears. The producer learns the consumer type from the graph in
  `ggml_cuda_try_fuse` (the GLU node's consumers). q8_1 needs a per-32 amax (one warp), a16k a per-256
  amax plus per-32 sums (one 256-thread block), both fit the GLU kernel's block shape. Bit-exact. Kill
  switch `GGML_CUDA_DISABLE_FUSE_QUANT_GLU`; the rms_norm half is a follow-up as `..._QUANT_NORM`.
  Expected 0.33 ms per token (1.0% tg) for the GLU half. Files: `unary.cu` (GLU dispatch at
  ggml-cuda.cu ~4738), `quantize.cu`, `mmvq.cu` / `mmvq-k-f16-sm60.cu` cache lookups, `ggml-cuda.cu`.
  Result 2026-09-09: NEGATIVE after two iterations (shared-memory epilogue, then register-only on the
  quantizer's grid). Correct and bit-exact, 335 -> 207 quantize launches, Qwen tg +0.4-0.55% but Gemma 4
  -0.4% because its a16k layout 2 halves the transaction size of the elementwise part (gotcha 40). See
  the Tier 1 row. Diff and report kept in `~/p100-opt/e8/`, tree reverted, nothing committed.

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
- H2 DONE 2026-09-08 with A4 (patch 44): the pool high-water at a wide LM-head call drops 956 MiB
  per card on Qwen3.8-27B and 2432 MiB on Gemma 4 31B, and `-c` can go 190000 -> 215000. Left in this
  area: Gemma's tied `token_embd` LM head is not row-split by the meta backend (each card converts the
  whole 2688 MiB copy), and a wide all-logits pass still allocates the full F16 `dst_temp`.
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
Before patch 43 it also stayed off per token whenever a grammar was in the chain, i.e. for every
request that carries `tools` (I1, 2026-09-07).
The default thread count is the number of physical cores (14) with `--poll 50`; with everything
offloaded those threads mostly spin.

Items:

- I1 DONE 2026-09-07 (`p100x: 43-sampler-prefilter-grammar`; numbers in the Tier 1 table and
  P100-PATCHES.md 43). The premise below was wrong: no GGUF has suppress tokens, the prefilter is live
  for plain chat and was off for every request that carries `tools` (lazy grammar, every token outside
  the thinking block). The fix gates it on `grammar_first && grammar_should_apply()`, which the server
  never sets. Original item: `[SRV]` Confirm the prefilter is live for the production sampler settings: temporarily log
  `pf_nkeep` (or run once with `LLAMA_SAMPLER_PREFILTER=0` and compare server tg). Check whether
  the Qwen3.8 vocab carries suppress tokens (`gguf-py` dump of the tokenizer metadata); if it does,
  the logit-bias sampler is always in the chain and the prefilter never runs, and patch 13's
  condition should be relaxed to "no user biases" (the suppress list is tiny and static).
- I2 Measured 2026-09-07 with I1: server 30.928 vs llama-bench 30.874 ms per token at d128, i.e.
  0.054 ms (0.17%) of sampling plus server bookkeeping with the prefilter live, about 0.5 ms with it
  off; far below the threshold, no work. Original item: `[SRV]` Sampling share: server `predicted_per_second` vs `llama-bench -n` tg at the same
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
- J4b DONE 2026-09-06 (`p100x: 37-server-ckpt-save`, numbers in the Tier 1 table and P100-PATCHES.md 37).
  Correction: the "200 ms DtoH" of the J4 timeline was the log-timestamp gap around `create_checkpoint`, not
  the copy. Measured (`~/p100-opt/log/J4b-notes.md`): the 149.6 MiB device-to-host copy is 768 shard copies in
  30.5 ms (5.14 GB/s), the restore is 36.9 ms HtoD, and the rest of the window was 72.8 ms of
  `std::vector::resize` on a fresh mapping plus 8.5 ms of `free`. `llama_state_seq_get_data_ext` calls
  `ctx->synchronize()`, so host work placed before it is hidden behind whatever decode is still in flight
  (about 66 ms here); anyone timing host work near a state call must account for that. `--ctx-checkpoints`
  defaults to 32 on this branch, not 8. After patches 35 and 37 a follow-up prompt phase is ~417 ms =
  restore 37 + 28-token decode ~200 (+ its ~66 ms tail) + copy 30 + 4-token decode ~82: the two small
  decodes are now the largest pieces (J5 / Area A / Area G).
- J5 DONE 2026-09-08 (measurement only, no code change; scripts in `~/p100-opt/j5/`, numbers in
  `~/p100-opt/j5/J5-notes.md`). `--spec-type draft-mtp` at the default `--spec-draft-n-max 3` is the
  right setting and `--draft-p-min 0` is already the default (`p_min = 0.0f` in `common/common.h`),
  so the recorded production line needs no change. Fresh prose, 512 tokens, 6 reps at `-c 16384`:
  no spec 32.26 t/s, ngram-mod 35.37, MTP 1/2/3/4/5/6/7/15/31/63 = 41.93 / 39.02 / 40.15 / 37.55 /
  33.03 / 30.71 / 26.79 / 13.90 / 7.23 / 3.92. On the realistic 25k chat with follow-ups and a tool
  call the ordering flips, because acceptance rises from 47% to 68%: no spec 30.65, ngram-mod 29.03
  (a loss), MTP 1/2/3/4/5 = 40.48 / 42.12 / 45.75 / 43.45 / 41.75, and the tool call alone runs at
  54.57 t/s (1.81x). Break-even between width 1 and width 3 is ~55-60% acceptance; the chat arms each
  generate their own text, so the controlled check is a fixed verbatim-echo prompt with byte-identical
  output in every arm (main session, 2026-09-08, 300 tokens at `-c 60000`): no spec 32.4 t/s, width
  1/2/3/4 = 44.4 / 47.6 / 50.8 / 51.2 at 86.9 / 76.3 / 68.4 / 63.1% acceptance, i.e. wider is better
  exactly where acceptance is high, and 3 vs 4 is a tie that 4 pays for in memory. Per-position
  acceptance is (0.857, 0.667, 0.476) and one draft position costs ~11 ms in total, which is why
  nothing above 4 pays. Depth is not what flips the ranking, the content is: the same fresh-prose
  request at 140 and 23600 tokens gives the same 46-47% acceptance and the same width-1 win. MTP also
  costs 4.5% of prompt processing (the draft context runs over the prompt too).
- J5 leftovers: `ngram-mod` is a net loss on this workload (-5.3% tg, 8.3% acceptance) and is not
  reproducible (gotcha 34); memory bounds the width through `n_rs_seq`, not through the LM head
  (gotcha 32), so patch 44 does not help MTP at any width that loads; patch 22's premise is dead
  (graph reuse is already at 99% of verify steps with MTP on, see `P100-PATCHES.md` 22). If the
  workload were mostly fresh prose rather than tool calls and quoted code, `--spec-draft-n-max 1`
  would be 4-5% faster there and save 150 MiB per card; for a coding harness it is the wrong way round.
- J6 Request hygiene that keeps the fast paths on: no per-request `logit_bias` (kills the
  prefilter, even with a bias of 0.0; `n_probs` turned out to be a non-issue, see J5), `-fa on` explicitly, `--temp` etc. fixed in the preset rather than
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
| A4 | DONE 2026-09-08 as `p100x: 44-cublas-src0-chunk`: convert and multiply src0 in 256 MiB row chunks (the M dimension of the GEMM) when its compute-type copy would be larger, one pool buffer reused for every chunk. The VMM pool high-water at a wide LM-head call drops 1280 -> 324 MiB per card on Qwen3.8-27B (-956) and 2774 -> 342 MiB on Gemma 4 31B (-2432; its tied `token_embd` LM head is not row-split by the meta backend, so each card converts all 2688 MiB); the `cuMemCreate: out of memory` abort inside `ggml_cuda_mul_mat_cublas_impl` (Qwen at `-c 200000`, Gemma at `-c 65536`, both on a 65-column `ngram-mod` verify batch, and it kills the server rather than the request) is gone, and the largest `-c` that survives such a call goes 190000 -> 215000 (+25000 tokens at 351 MiB per card per 10000). pp512/pp2048/tg64 unchanged at d0 and d16384 (<= 0.13%, inside the spread) because nothing else on these models is above the threshold; where it fires the split is faster, not slower (LM head 65 columns 38.8 -> 36.4 ms per card, 2048 columns 253.9 -> 227.4 ms), so no ping-pong buffer is needed. Not bit exact (cuBLAS picks another kernel for the smaller M): mean KLD -1.2e-5, top-1 identical for 100% of positions, per-chunk perplexity identical to 4 decimals, greedy output of both models byte-identical | left: it does not fire for the recorded production line, whose MTP draft is `n_max` 3 (verify width 4, under the patch 35 Q6_K ceiling of 32): 0 chunked calls at `-c 160000` and identical memory in both arms, so the item pays for `ngram-mod`, for a raised MTP draft width, for all-logits passes and for Gemma; the threshold must stay above the largest ffn weight (see What not to do); `mul_mat_id`, the batched branches and the `convert_nc` path keep the whole-matrix copy; the wide LM head would be cheaper still through the MMVQ column loop (A8 with a higher per-type ceiling, no F16 copy at all: 36.4 ms per card at 65 columns today); splitting Gemma's tied LM head across cards is an Area H meta-backend question; A4b: the split is 0.59x per call at 128 MiB but the byte rule cannot go there, so a rule keyed on rows (keep M per GEMM in the tens of thousands) instead of bytes may be worth 1.6x on every wide LM-head call - measure across widths first, the 128 MiB point is non-monotonic | A1 (optional) |
| F5 | internal allreduce on sm_60 (`__nanosleep` replacement) | A/B vs NCCL at tg; may win 1-3 ms per token | F1 F2 |
| B2 | CLOSED 2026-09-09 (design note, no code). The fused MMVQ path exists only for 1-column matvecs (host gate `dst->ne[1] != 1`, and `mul_mat_vec_q_switch_fusion` asserts `ncols_dst == 1`), so it never fires on the production 4-column MTP verify step. At width 1 it can remove only the FFN GLU launch: 64 x 2.7 us = 0.17 ms = 0.5% of the step; the intermediates are 70 KB per card, and gate/up already share one activation conversion (patch 10 / a16k cache), so no quantize launch goes away - the Tier 0 remark tying `quantize_q8_1` to B2 was wrong. `ggml_cuda_should_fuse_mul_mat` also requires the same src0 type for gate and up, true in 37 of 65 layers of the UD-Q4_K_M (22 of them on the int8 path at width 1, 15 on the HFMA2 path, which refuses fusion). A GLU epilogue in `mmvq-k-f16-sm60.cu` (sequential gate then up, `up * silu(gate_in)` in the store) would cover all widths for 150-250 lines and reach 0.3% per production step. E8 takes more out of the same spot without touching the matvec kernel. The correction of 2026-09-07 stands: lifting the `cc <= PASCAL` gate alone routes the HFMA2 layers back to the int8 kernel | ceiling 0.5% at width 1, 0.3% per production step: below the bench spread | - |
| B4d | DONE 2026-09-07 as `p100x: 41-mmvq-k-f16-range`: the Q6_K (and latently the IQ4_XS) epilogue of the HFMA2 K-quant matvec summed an int8 sub-block scale of -128 times a lane accumulator of 8 products of |q*a'| <= 32 over 4 sub-blocks, which reaches 2^18 against half's 65504; measured peak 71552 on `blk.44.attn_v.weight` of Gemma 4 31B, below the limit on every other tensor of the prompt and on every Qwen shape. Fix: the scale times 1/8 and d times 8, both exact, product unchanged bit for bit. Gemma nan gone, chat byte-identical to stock, tg 28.66 t/s (1.634x stock, -0.46% against the broken build), Qwen byte-identical at -0.08% tg and -0.2/-0.3% pp8/pp32, test-backend-ops 14675/14675 on both cards | left: the F16 accumulation itself costs 0.020 mean KLD and 1.3 points of top-1 on Gemma against the branch's own 0.172 / 90.8% floor (nothing measurable on Qwen), so a model with wider activation swings needs its own KLD check before it is served; new debug knob `GGML_A16K_CHECK=1` | C6 |
| E5b | DONE 2026-09-07 as `p100x: 42-norm-cache-5376`: `max_cache` 5 -> 6 (template default and the `<1024>` launcher), so a row up to 6144 columns is cached; Gemma 4 tg64 d0 28.630 -> 29.075 t/s (+1.56%, four norms per layer over 60 layers at ncols 5376), Qwen unchanged, RMS_NORM 51/51, Qwen greedy byte-identical, registers 30-32 with no spills | left: 7168-wide rows would need `max_cache` 7; the register budget still has room | E5 |
| C4 | NO (2026-09-05, C2 measurement): the `parallel_blocks` search already fills exactly one wave at tg (`ntiles_dst` 6, `max_blocks_per_sm` 4, `pb` 37, 222 of 224 block slots, 99% efficiency at every depth from 4k up); forcing 2x costs 6% at 32k and 21% at 4k, 4x costs 83% at 32k. What the search does miss is the ragged last KV tile (`ntiles_KV % pb`), which makes the `ncols2 = 6` packing of C2 non-monotonic between 2k and 8k: see C2b. The C2b sweep confirms the direction: at the model's tg shape every `pb` above the search's value is slower (n_kv 8192: 52 us at 64, 61.5 at 112, 65.7 at 128); what was left on the table was the smaller value with the same round count (patch 38) | - | C1 |
| E5 | DONE 2026-09-07 as `p100x: 40-norm-cache-5120`: `max_cache` becomes a template parameter with default 5 so a 5120-wide row is cached (patch 17 stopped at 4096 and never fired on this model); 6.33 -> 3.89 us per run at 5120x1 in test-backend-ops perf, 10.2 -> 8.8 us per launch in the model at 129 launches per token per card; tg 31.91 -> 32.11 t/s at d0 (+0.63%) and 30.91 -> 31.10 at d16384 (+0.61%), pp2048 unchanged, output byte-identical; registers unchanged, no spills | left: nothing; E5b (patch 42) raised the limit to 6 for Gemma 4's 5376 | E1 |
| D4 E6 | DONE 2026-09-09 as a measurement (census in `~/p100-opt/d4e6/D4E6-census.md`, log entry 2026-09-09): on 5b8bc7dd7 the tg step is 31.30 ms with 1698 launches and only 18 distinct kernels per card per token, and every adjacent-node pattern in both layer types already fuses (26 launches per layer: 18 + 8 for the FFN). Glue is 993 launches / 3.59 ms / 11.5% of the step, of which the single-block 5120-wide rms_norm is 129 / 1.14 ms / 3.6% and the quantize helpers (not graph nodes) 335 / 0.83 ms / 2.6%. D3 fires, the L2_NORM pair fires, the gated norm is 2 launches not 3 (its halves are separated by the z MUL_MAT, so a defer is needed, not a chain rule), E2 does not fire because `ggml_cuda_should_fuse_rms_norm_mul_rope` rejects MROPE mode (ggml-cuda.cu ~2811). Ranked candidates (estimate = bytes / 450 GB/s + 2.5 us per launch removed): E8 GLU/gated-MUL writes the matvec's quantized activation 0.33 ms (1.0%), the same for the rms_norm producers 0.33 on paper (~0.18 until E7), the conv chain 0.26, conv-state CPY into concat_rows_gather 0.13, D4 gated norm 0.12, E2 MRoPE fused rope 0.12 | no fusion rule written: the largest item is not a fusion (E7), the second is E8 | - |
| E7 | NO (2026-09-09, measured; diff kept in `~/p100-opt/e7/e7.patch`, not committed, tree reverted): multi-block rms_norm for the single-row tg case with a redundant (bit-exact) reduction. Two independent blockers. (1) It cannot be applied to 128 of the 129 target launches: the residual ADD that patch 19 folds into the norm writes into one of its own inputs (`ggml_gallocr` in-place: over 512 pre-add launches `pre_dst == pre_a` 320 times, `== pre_b` 192 times, and `dst` is the other of the same two buffers), so several blocks per row race; forcing it changes the greedy output. (2) Where it applies it is slower: every block still reads the whole row, so the per-block latency chain is unchanged and only the fire-and-forget stores are split: 5120x1 RMS_NORM 4.3 -> 5.2 us in isolation (1.22x slower), in the model 9.32 -> 10.04 us per launch when forced, tg -0.3-0.5%; at pp unchanged by construction. Only a disjoint split with a cross-block reduction (two launches, not bit-exact) could still help, uncertain 0.4-1.3%, not planned | 0 | - |
| E8 | producer writes the matvec's quantized activation: 256 of the 335 quantize launches per token (`quantize_a16k` 197, `quantize_q8_1` 138, 2.3-2.9 us each) directly follow the kernel that produced their input (`unary_gated_op_kernel` 128, `rms_norm_f32` 128). Let the GLU / gated-MUL kernel (later also rms_norm) emit the a16k or q8_1 block format its consumer needs, handed over through the existing single-slot activation cache (the producer fills the cache and sets its key, the consumer's lookup hits); the producer learns the consumer's src0 type from the graph in `ggml_cuda_try_fuse`. Bit-exact (the same F32 value quantizes to the same block). Kill switch `GGML_CUDA_DISABLE_FUSE_QUANT_GLU`. Approved 2026-09-09 as the item after E7. First build measured 2026-09-09 (uncommitted, `~/p100-opt/e8/e8.patch`, report `E8-report.md`, log entry): mechanism correct, self-check 0 mismatching bytes over 8000+ launches in both formats and all three a16k layouts, greedy byte-identical, perplexity identical, 14675/14675 twice; quantize launches 335 -> 207 per token per card exactly as predicted and all 512 folds fire at the width-4 verify shape too (IQ4_XS follows into layout 1). But the a16k epilogue as written stages 256 values in shared memory, syncs, and converts in warp 0 while 7 warps wait: +1.85 us on a 12-block GLU against a 2.59 us quantize removed, so the net is 0.135 ms per token per card (0.43%) in kernel time and tg +0.31/+0.34% at d0, +0.35/+0.23% at d16384 on Qwen, pp unchanged, but -0.41% on Gemma 4 (all of whose folds are a16k layout 2). Not committed. NO (2026-09-09, iteration 2 measured, not committed, tree reverted; diff `~/p100-opt/e8/e8.patch`, report `E8-report.md`): the register-only a16k epilogue (one warp per 256-element block on the quantizer's grid, no shared memory, no sync) brings the fused launch to 3.84-3.86 us against 5.24-5.81 for GLU + quantize on Qwen (saving 0.196 ms per token per card, 0.62% of kernel time) and tg to +0.53/+0.40% at d0 and +0.55/+0.55% at d16384, but Gemma 4 stays at -0.34 to -0.41% over three pairs: all of its folds are a16k layout 2 (Q6_K `ffn_down`), whose lane map (`p0 = 128*(g>>1) + 16*(g&1) + 32*(i>>2) + 4*(i&3)`) makes a warp's run 8 chunks of 64 B instead of layout 0's 4 chunks of 128 B, so the fused kernel is 6.79 us against 3.07 + 2.60 separate, and the elementwise work runs on 11 blocks of 128 threads where the plain kernel had 42 of 256. Structural: the fused kernel must run on the quantizer's narrow grid while the elementwise part wants the wide one. A layout gate would leave Gemma at 0 and Qwen at ~+0.4% for a 680-line diff in upstream-churned files; rejected | first build Qwen +0.3% / Gemma -0.4%; iteration 2 Qwen +0.4-0.55% / Gemma -0.4% | - |
| I1 | DONE 2026-09-07 as `p100x: 43-sampler-prefilter-grammar`: the vocab clause was moot (none of the three GGUFs has suppress tokens); what switched the prefilter off was the request-level lazy grammar of a `tools` request with `tool_choice` auto (the coding-harness shape): 0 of 512 samples prefiltered with thinking off, 21 of 48 on a real tool call with thinking on, 512 of 512 for plain chat. Gating the prefilter on `grammar_first && grammar_should_apply()` instead of `grammar_should_apply()` is exact (the server never sets `grammar_first`; a grammar rejection already rebuilds the whole vocabulary) and recovers it: tools request 31.44 -> 30.98 ms per token (+1.47% tg), plain chat unchanged; the prefilter itself is worth 1.45% tg on plain chat (30.92 vs 31.38 ms) and 3.4% under `--spec-type draft-mtp` (one sample per draft position); replies and tool calls byte-identical, resample count 0 | left: a request `logit_bias` still turns it off (`has_logit_bias`), by design; I2 answered at the same time: sampling plus server bookkeeping is 0.054 ms per token with the prefilter live (server 30.928 vs llama-bench 30.874 ms per token at d128), no work needed | I1 measurement |
| A2 A3 | DONE 2026-09-07 as `p100x: 39-convert-vec`: A3' vectorizes the contiguous F16/BF16 <-> F32 conversion around every cuBLAS GEMM (4 elements per thread, 8-byte transfers, grid covering the data) and A2 issues the contiguous store group of `dequantize_q4_K` and `dequantize_iq4_xs` as one 8-byte transfer with 8 super blocks per 256-thread block; pp2048 420.8 -> 449.4 t/s at `-ub 2048` (+6.8%; A3' +5.55%, A2 +1.22%) and 235.1 -> 248.3 at `-ub 512` (+5.6%; A3' +3.18%, A2 +2.92%), tg unchanged, output byte-identical (perplexity 5.4367 in every arm incl. stock, greedy identical); `convert_unary` 165 -> 520 GB/s (29% -> 93% of the 560 GB/s ceiling), q4_K dequant 168 -> 461 GB/s, iq4_xs 212 -> 330 | left: the plan's A2 premise was wrong (block packing alone is a no-op, the kernels are store bound); q5_K needs a different index assignment inside `dequantize_q5_K` for a 4-element store group, about 0.5% of pp2048; `getrows.cu` could ask for the same vector store; 16-byte transfers and a bounded grid both lose (What not to do); A3 as written (F32 compute) measured dead: -35% pp | A1 |
| J4 | DONE 2026-09-03 (`p100x: 31-server-ckpt-adopt`, worktree wt-j4): adopt the just-restored checkpoint, no forced break at the last user message; `LLAMA_SERVER_CKPT_LEGACY=1` restores upstream | follow-up 1.98 -> 1.41 s wall, prompt phase 1302 -> 732 ms | - |
| A8 | DONE 2026-09-05 as `p100x: 35-mmvq-cols-sm60`: loop the matvec over balanced column chunks (target 7, floor 6) for 9..N columns on GP100, per-type ceiling Q4_K 64, Q5_K 48, Q6_K 32, IQ4_XS 64, int8 32; pp9 3.87x, pp16 2.84x, pp28 2.30x, pp32 1.93x, pp48 1.43x, pp64 1.12x, unchanged at 96 and above; the 25k-chat follow-up prompt phase 710 -> 434 ms and the request 1.34 -> 1.07 s; tg and pp2048 unchanged | the LM head F16 copy is gone at these widths (small-batch half of A4); J4b is now the larger half of a follow-up; verify widths 9-64 are 1.1-3.9x cheaper (J5) | J4 |
| J4b | DONE 2026-09-06 as `p100x: 37-server-ckpt-save`: recycle the storage of an erased context checkpoint instead of allocating a fresh 149.6 MiB vector per save; checkpoint save 112.4 -> 96.4 ms, follow-up prompt phase 433.3 -> 416.9 ms (-3.8%), wall 1.060 -> 1.057 s (inside the noise), RSS unchanged; checkpoints byte-identical (17/17 against a second save of the same state) | the item's premise was wrong: the 149.6 MiB DtoH costs 30.5 ms (5.14 GB/s), faster than the 36.9 ms HtoD restore; the old 112 ms was 8.5 ms free + 72.8 ms allocate/first-touch + the copy, and the allocation was hiding ~66 ms of decode in flight that `ctx->synchronize()` now waits for. Left: that wait plus 30 ms of copy; only a pinned buffer touches the copy (save 30 -> 16 ms, restore 35 -> 18 ms; needs a pinned allocator and up to `--ctx-checkpoints` (default 32) x 149.6 MiB locked per slot); edit/regenerate requests are not covered (their checkpoint is dropped outside `create_checkpoint`) | J4 |
| C2b | DONE 2026-09-06 as `p100x: 38-fattn-pb-tiebreak`: after the `parallel_blocks` search take the smallest value with the same number of KV rounds (one Q column, Pascal; the round count had to be the primary key, no wave-efficiency band reaches the winner at 57% fill); per launch 1.18x at n_kv 8192 on the packed path, 1.06x at 4096 unpacked, more than one Q column unchanged; GQA 6 gate 16384 -> 2560 (7680 at 2 Q columns, knob `GGML_CUDA_FATTN_GQA6_MIN_KV`), the shipped launch 1.10-1.21x from 2560 to 12288 against 36; 4 kv heads 0.88x -> 1.14x at 4096; model tg unchanged within 0.3% (about 16 FA launches per token, 5.7 us each) | left: a tie-break for 2 Q columns (that launch loses up to 14% at n_kv 4608-6656, the naive rule costs 4-7%); the one-kv-head clause above 16384 (1.09-1.28x, one line, own protocol); the rule is measured at D=256 only (C6 measured D=512 GQA 8: 1.094x at n_kv 4096, 1.006-1.009x above; D=128 still unmeasured); `nbatch_fa 32` never measured together with `pb` | C2 |
| B4c | DONE 2026-09-04 as `p100x: 34-mmvq-k-shortk`: prefetch mode 2 (header loaded in its own step) per type and width, at width 1 only for Q5_K below 24 blocks; tg +0.5% (31.7 t/s at d0), speculative verify widths 2/4 +4% / +2%; measured with a new any-shape timing tool (`~/p100-opt/b4c/shape.cpp`) that Area B work should use from now on | the gap itself stands: k=5120 runs at 0.74-0.81 of the k=14336 rate, the streaming part is within 3% of the DRAM ceiling and only a fixed 0.38 step per warp is addressable; a shorter warp step is measured out (see What not to do); the LM head was already at its ceiling | B4 B4b |

### Tier 2: medium changes (design note to the user first, 2-5 days each)

| Item | Change | Expected | Depends on |
|---|---|---|---|
| B4b | DONE 2026-09-04 as `p100x: 33-mmvq-k-hfma2`: Q5_K 1.36x and Q6_K 1.40x at width 1 per call (1.14x / 1.30x in-model, short k again), IQ4_XS only from width 4 (its int8 path already runs at 400 GB/s, see What not to do); tg 30.0 -> 31.6 t/s at d0 (+5.2%; +7.8% over the int8 path), 28.8 -> 30.3 at d16384 | left: two-slot `a16k` cache for mixed-type gate/up pairs at widths 4-8 (~1.7% of kernel time), width 6-7 defaults interpolated. CORRECTNESS 2026-09-07 (C6): nan logits on Gemma 4 31B, a half range overflow in the Q6_K epilogue (the |q*a'| <= 15 bound is a 4-bit one), fixed by B4d (patch 41) | B4 |
| G3 | per-subgraph CUDA graph cache so `-sm tensor` can use graphs | the bulk of the launch-bound share found in G1 | G2 success |
| C2 | DONE 2026-09-05 as `p100x: 36-fattn-tile-p100`: FP16 tile table for D=256 (the model is D=256 GQA 6, not D=128): the tg entry had no register budget (255 registers, 4 blocks and 8 warps per SM) and GQA 6 fell onto `ncols2 = 2` with a 3x K/V re-read; new `ncols2 = 6` packing above n_kv 16384 at 1-2 Q columns plus `(128, 4, 64, 64)` for `ncols 2`; attention kernel 1.12x at 16k, 1.31x at 32k, 1.46x at 64k, 1.70x at 128k (92% of the read ceiling); tg +0.7% at d0, +1.6% at d16384, +3.3% at d32768, +7.9% at d65536; pp unchanged (its table entry is already optimal, 14 candidates lost) | left: the 16384 gate (C2b); the same register-budget fix for the D=128 and small-head `ncols 2/4` entries (unmeasured, llama-class shape); C5 has no headroom at 131k any more | C1 |
| C3 | NO (2026-09-05, C2 measurement): the vector kernel is 1.4-1.8x slower than the tile kernel at every KV length from 4k to 128k on the model's shape; it has no GQA packing at all (6x K/V re-read against the tile kernel's 3x, 1x with patch 36). No KV-length threshold wins | - | C1 |
| F6 | size-dependent allreduce transport | tg wins from F4/F5 without losing pp | F4 F5 |
| F9 | batch the synchronous split-input copies | < 1 ms per token, only if the timeline shows them | F2 |
| D2 | GDN kernel occupancy at tg | a few % of the delta-net share | D1 + nvprof |
| H3 H4 | `-ub` vs compute buffer, unified KV second slot | memory for context or slots | H1 A1 |

### Tier 3: research (weeks, only with Tier 0-2 numbers behind them)

| Item | Change | Why it might pay | Why it might not |
|---|---|---|---|
| B4 | DONE 2026-09-03 as `p100x: 32-mmvq-q4k-hfma2` (Q4_K only): 1.49x at width 1 (471 GB/s), up to 2.85x at widths 5-8 per call; tg +2.5% | Q4_K is only 17.7% of the tg kernel time on the UD-Q4_K_M (Q5_K 24.7%, IQ4_XS 19.4%, Q6_K 9.0%): B4b carries the rest | the k=5120 shapes reach 373 GB/s against 459 at k=14336 (B4c); the Q6_K/IQ4_XS epilogue overflowed half on Gemma 4 31B (C6), fixed by B4d (patch 41) |
| A7 | HFMA2 tiled GEMM with in-register dequant | removes the F16 round trip and temporaries | cuBLAS is already at ~68-80% of peak |
| F7 | P2P direct allreduce kernel for 2 GPUs | < 10 us per collective vs 20-40 | needs stable P2P; Pascal has no `__nanosleep` |
| G4 | fewer subgraph boundaries in the meta backend | 1 CUDA graph per device per token | deep change in `ggml_backend_meta_graph_compute` |
| C7 | quantized KV shards in the meta backend | context or slots | TILE dequantizes the whole cache per call; only with VEC |
| I3 | backend sampling with vocab-sharded logits | removes 1 MB D2H + CPU sort per token | small share; needs meta-aware gather |
| F8 | fewer collectives per layer (sequence parallel) | halves PCIe traffic | model-graph rewrite |

### Deferred code items (user decision 2026-09-09: no more code changes for now, unclear when we get back)

After the D4/E6 census, E7 and E8 the tg step has no fusion left at the graph-node level and the
remaining pool sits inside the matvec kernels and NCCL. Every open code item below is under 1% of tg or
a Tier 3 project, so the code phase stops here and the plan continues with the root session (Final
table). Nothing in this list is killed; the numbers say what each is worth if it is ever picked up.

| Item | What is left | Worth | Notes |
|---|---|---|---|
| E2 | MRoPE variant of the fused `RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)` kernel; the node order already matches, one mode check blocks it | 0.12 ms per token, 0.4% tg | ggml-cuda.cu ~2811, `ggml_cuda_op_rms_norm_mul_rope_fused` |
| D4 | one kernel for `rms_norm(attn_out) * w * silu(z)`, deferred past the z matvec (patch 23 style) | 0.12 ms, 0.4% | 48 launches per token |
| conv chain | `GET_ROWS + CONCAT -> CPY(cache) -> SSM_CONV + SILU` as one kernel, `conv_input` never in DRAM; the safe subset folds the conv-state CPY into `concat_rows_gather` | 0.26 ms, 0.8% (subset 0.13, 0.4%) | ~250 lines; `conv_input` has two consumers |
| F9 | the synchronous split-input copies (62 memcpy/memset ops per token, 0.5 ms at d4096 in Tier 0.2), never examined since | up to 1.6% if they are on the critical path, likely less | needs the timeline, not the kernel sums |
| C5, C8 | 8 vs 16-byte loads on sm_60 (`ggml_cuda_get_max_cpy_bytes`); the `-fa auto` NONE-verdict check | small; a correctness check | one-hour items |
| D2 | GDN kernel occupancy | pp only (GDN is 6.1% of pp, 1.9% of tg) | nvprof |
| A4b, Q6_K ceiling, A6 | row-keyed cuBLAS chunk rule; a higher patch 35 ceiling for the wide LM head; MMQ-for-dense number | non-production shapes only (`ngram-mod`, all-logits, Gemma) | - |
| A7, C7, F7, F8, G4, I3 | Tier 3 research: HFMA2 tiled GEMM, quantized KV shards, P2P allreduce kernel, sequence parallel, fewer meta subgraphs, backend sampling | the only remaining double-digit levers, weeks each | see the Tier 3 table |
| H3, H5, K2, J7, deliverable 4 | not code: compute buffer vs `-ub`, KV share at 200k, host RAM budget with `--cache-ram 0`, router respawn losing the host cache, the final recommended server command line | memory and operations | can ride along with the root session |

Closed at the same time as effectively dead (no measurement needed): A5 (pays only at small `-ub`, the
user runs 2048), B5 (patch 09, MoE only), B6 and B7 (covered by E8 and J5), D5 (patch 26 already fuses
the writeback copies), E4 (rope is 0.3% of the step), G6 (patch 22, premise dead with MTP) and G7 (host
is not the bottleneck: 2.8% idle), H4 (single-stream answer to question 5), J3 (hybrid models cannot
shift), I5 (folded into PL3).

### Final: configuration sweeps (after all Tier 1-3 items; added 2026-09-09)

| Item | Change | Expected | Depends on |
|---|---|---|---|
| PL1 | DONE 2026-09-09 (measurement, `~/p100-opt/pl/PL-report.md`, log entry): 7 arms 250/225/200/175/150/125/250 W, one llama-server per MTP arm. tg is flat to 175 W because each card draws only ~155 W at tg (peak 168): tg d0 32.04 at every limit from 250 to 175, MTP chat 39.5 t/s everywhere down to 175, joules per tg token unchanged. pp follows the clock: pp2048 447.8 / 444.0 / 432.1 / 415.0 / 395.1 / 349.1 t/s at 250 / 225 / 200 / 175 / 150 / 125 W (-0.9 / -3.5 / -7.3 / -11.8 / -22.0%), pp mean clock 1328 -> 1316 -> 1265 -> 1198 -> 1140 -> 984 MHz. At 150 W tg loses 1.0-1.4% and MTP 0.4% for -12% energy per token; at 125 W tg -6.5 to -7.2%. Tolerance summary: within 1% of tg -> 175 W (pp -7.3%), within 2% or 5% -> 150 W (pp -11.8%); the tg knee is between 150 and 165 W. No thermal slowdown at any limit (max 64 C against 82), peak board power 237 W per card, `pviol` 0 at 250 W: the PSU is not stressed. Confirmed by the user's wall meter: whole-system draw unchanged down to 175 W, -50 W at 150 W, -100 W at 125 W. Verdict: keep 250 W (a limit only clips pp and saves nothing at tg); 175 W is the setting if a lower peak (160 W per card instead of 237) is ever wanted for the chassis | no tg to gain; pp -7% at 175 W | - |
| PL2 | DONE 2026-09-09: persistence mode is a NO-OP on this host. The Quadro K2200 display card and `nvidia_modeset` hold the `nvidia` module loaded permanently (refcount 19), so the driver never tears the GPUs down between processes: CUDA init 327 vs 337 ms and init + load + 1 token 5174 vs 5161 ms with persistence on vs off (medians of 3, inside the noise), and a 200 W power limit survived `pm off` plus a CUDA process exit. The user's daemon fix is harmless and stays on. Router respawns pay the 4.2 s model load only (K1) | 0 | - |
| PL3 | DONE 2026-09-09: zero. ASPM not testable (both P100 links report `ASPM not supported`). Governor `performance` vs `schedutil`: tg d0 32.24/32.23 vs 32.24/32.27 t/s, MTP 40.11 vs 40.10, with the CPUs at 3390 vs 1740 MHz mean - the host clock does not matter. `cpu_dma_latency` held at 0 vs not: 32.03/32.04 vs 32.05/32.02, MTP 39.51 vs 39.53. Both together: 32.04 vs 32.04. All MTP arms byte-identical (draft_n 636, accepted 298) | 0 | - |

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
- Do not use the vector flash-attention kernel for single-column tg on Pascal (C3, 2026-09-05): it is
  1.4-1.8x slower than the tile kernel at every KV length because it has no GQA packing.
- Do not force `parallel_blocks` above the search result on Pascal (C4, 2026-09-05): the search fills
  exactly one wave at tg; 2x costs 6-21%, 4x up to 83%. The one thing it misses is the ragged last KV
  tile (C2b).
- Do not pack the K-quant dequantizers into 256-thread blocks and stop there (A2, 2026-09-07): one
  super block per warp with 8 warps per block is byte-identical and a no-op on its own (q4_K 151 -> 153
  GB/s, q5_K 214 -> 214, q6_K 329 -> 331, iq4_xs 177 -> 192, q3_K 189 -> 176). These kernels are store
  bound, not launch bound. The win is in the store: 4 contiguous elements as one 8-byte transfer takes
  q4_K to 377 GB/s, and only then does block packing add the rest (460 GB/s). q6_K writes singles at a
  stride of 32 elements and is already at 58% of the ceiling.
- Do not use 16-byte transfers for the contiguous convert on GP100 (A3', 2026-09-07): Pascal has
  `LDG.128`/`STG.128` and `ggml_cuda_get_max_cpy_bytes()` returns 8 for sm_60, but 16 only ties for
  f32 -> f16 (544 GB/s) and loses 15% for f16 -> f32 (417 against 494). More elements per thread lose
  in every direction, badly for bf16 (16 elements with 8-byte transfers is 157 GB/s for f32 -> bf16,
  slower than the scalar kernel). Data point for C5.
- Do not give the contiguous convert a bounded grid with a grid-stride loop (A3', 2026-09-07): the
  usual streaming-kernel shape (a few blocks per SM) costs 1% for f32 -> f16 and 11% for f16 -> f32
  against letting the grid cover the data. What the P100 could not retire was 40960 blocks moving
  1.5 KB each, not a large block count as such.
- Do not run the cuBLAS path with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` for speed or memory (A3,
  2026-09-07): it drops both conversion passes and the F16 dequant target and still loses 35% of pp2048,
  because the halved HFMA2 rate costs far more. Its perplexity (5.4094 vs 5.4367) is only the price
  tag of F16 accumulation.
- Do not read a low KL divergence or top-1 agreement on Gemma 4 31B as a matvec defect without
  running the kill-switch arm against the same logits file (B4d, 2026-09-07): at `-ub 1` over 4
  wikitext chunks the metric's floor on that model is 0.172 mean KLD and 90.8% top-1 with the HFMA2
  matvec switched off, 400x the Qwen floor of 0.00038 / 99.0%, because the branch still differs from
  stock in the flash-attention combine order (patches 36/38) and this model's logits on that text
  are flat enough for it to flip the argmax. Only the kill-vs-fixed difference against one file is a
  signal, and the absolute `-ub 1` perplexity of that GGUF is not a gate either (C6).
- Do not justify half accumulation with the 4-bit bound (B4d, 2026-09-07): Q6_K values reach 32
  with an int8 scale of 128 and IQ4_XS values reach 127, so any new type on the HFMA2 path needs its
  own worst-case product against 65504, and a run with `GGML_A16K_CHECK=1` on every model.
- Do not lower `GGML_CUDA_CUBLAS_CHUNK_MB` below the largest non-LM-head weight (A4, 2026-09-08):
  64 MiB costs 3.3% of pp2048 and 32 MiB costs 10.9%. Per call the split costs 3.5-5.4% on the
  ffn_gate/up orientation and 1.6-2.6x on `ffn_down` (long k, few rows: M per chunk falls to
  1560-3120 rows and cuBLAS loses far more on the kernel than the split saves). The LM head is a
  wide-M matrix that likes being split; `ffn_down` is a narrow-M matrix that does not.
- Do not chunk a cuBLAS GEMM by a fixed small row count (A4, 2026-09-08): at M=1 cuBLAS changes
  kernel and `MUL_MAT(mxfp4, m=2880, n=32, k=2880)` misses the suite tolerance (3.8e-3 against
  5e-4). `GGML_CUDA_CUBLAS_CHUNK_ROWS` is a test knob; chunks must keep M in the thousands.
- Do not add a two-buffer ping-pong for the A4 convert/GEMM serialization (2026-09-08): the
  single-buffer split is already faster than the unsplit call at every LM-head width measured, so
  there is nothing to hide behind the convert.
- Do not widen the MTP draft to make use of the patch 35 column loop or patch 44 (J5, 2026-09-08):
  the accepted length saturates at ~2.7 tokens per step by width 7 while each draft position costs
  ~11 ms, so `--spec-draft-n-max` 7 / 15 / 31 / 63 run at 26.8 / 13.9 / 7.2 / 3.9 t/s against 40.2 at
  the default 3 and 32.3 with no speculation at all. Widths above 10 do not survive a decode at
  `-c 160000`. The wide-verify machinery pays for `ngram-mod` and all-logits passes, not for MTP.
- Do not add `--spec-type ngram-mod` to the MTP line (J5, 2026-09-08): on the 25k chat it is -5.3%
  against no speculation at all (8.3% acceptance over 48-64 token drafts), and because
  `need_n_rs_seq()` returns 0 for the ngram types every ngram draft of this hybrid model takes the
  full `update_tgt` path - a 149.6 MiB recurrent-state checkpoint per draft plus a restore on partial
  acceptance, where MTP at width 3 creates none. Combined it loses even on the verbatim-echo shape it
  is built for (42.7 vs 53.5 t/s), because the ngram implementation has priority and pre-empts the
  MTP draft.
- Do not build B2 as the upstream fused gate/up MMVQ (2026-09-09): fusion exists only for 1-column
  matvecs, so it never fires on the 4-column production verify step, and at width 1 its whole gain is
  the FFN GLU launch, 0.5% of the step. The B2 row in the Tier 1 table has the numbers; E8 covers the spot.
- Do not split a single-row rms_norm across blocks with redundant reads (E7, 2026-09-09): 1.22x slower per
  launch in isolation, -0.3-0.5% tg when forced in the model, and the pre-add form races (gotcha 37). A
  disjoint two-phase split would cost a second launch and bit-exactness for an uncertain 0.4-1.3%.
- Do not fold the matvec's activation quantization into the GLU / gated-MUL kernel (E8, 2026-09-09, two
  iterations): the mechanism works and is bit-exact, but the per-fold saving is 1.4-1.9 us and Gemma's
  a16k layout 2 turns it into a loss (gotcha 40); +0.4-0.55% Qwen / -0.4% Gemma for 680 lines in
  `mmvq.cu` and friends is not worth the rebase cost. The 207 remaining quantize launches per token
  (0.5 ms, 1.6% of the step) are only reachable by a change to the matvec kernels' input format itself.
- Do not chase `n_probs` in the request hygiene list (J5, 2026-09-08): the OAI endpoint ignores the
  field, and the flag it stands for (`logprobs: true` with `top_logprobs` 5 or 20) costs nothing
  measurable (24.81-24.86 vs 24.83-24.92 ms per token).

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
14. The sampler prefilter (patch 13) turns itself off with any logit-bias sampler in the chain (a
    request `logit_bias`; vocab suppress tokens would too, but none of the three GGUFs has any), and
    until patch 43 with any grammar in the chain, i.e. every request that carries `tools` (I1,
    2026-09-07). `top_k <= 0`, mirostat, DRY on, or a non-penalties sampler before top-k in a custom
    `samplers` order also switch it off. Check with a `LLAMA_SAMPLER_PREFILTER=0` A/B through the
    server; llama-bench does not sample. With MTP the prefilter runs once per draft position as
    well as on the target samples, so it is worth 3.69% there against 1.45-1.50% without
    speculation; a request `logit_bias` only reaches the target sampler (0.80% under MTP 3). A bias
    value of 0.0 still counts: `tools/server/server-schema.cpp` pushes the entry whatever the value
    and `common_sampler_prefilter_nkeep` only tests `!logit_bias.empty()` (J5, 2026-09-08).
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
24. Speculative verify batches leave MMVQ for cuBLAS at the patch 35 per-type ceiling, not at 8:
    Q4_K 64, Q5_K 48, Q6_K 32, so the Qwen LM head (Q6_K) goes to cuBLAS at 33 columns and the Gemma
    one (Q5_K) at 49. Patch 44 chunks that copy, so the draft-width restriction is gone and
    `--spec-type ngram-mod` runs at its default 48-64 token drafts. What still limits MTP is gotcha 32.
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
29. A copy of `build/bin` is not a frozen base arm: the binaries carry
    `RUNPATH=<tree>/build/bin`, so after a rebuild they silently load the new `libggml-cuda.so`
    (found 2026-09-07, A3: the "base" arm reported the new numbers). Export `LD_LIBRARY_PATH` to
    the copy, or `patchelf --set-rpath '$ORIGIN'` on it, before trusting a number from it.
30. `test-backend-ops` has no `RMS_NORM` case between 4096 and 5120 columns and none with one row,
    so a norm change at this model's `n_embd` is invisible to the suite; use the model (greedy and
    perplexity identity) plus a temporary local case that never enters the patch.
31. A kernel validated on one model is not validated on the branch's other models: patches 32/33
    passed every Qwen3.8 check and produce nan on Gemma 4 31B (C6, 2026-09-07). Run the second
    model's perplexity at `-ub 1` (matvec path) and a chat greedy sample before calling a matvec
    change done. For this GGUF `llama-completion`/`llama-cli` need `--jinja` (chat) or `-no-cnv`
    (raw) or they abort with "this custom template is not supported"; raw wikitext perplexity is
    17000-27000 on stock too, so for Gemma 4 use finite/non-finite, chat output and
    `--kl-divergence` against stock logits as the gates, not the absolute perplexity.
32. `--spec-type draft-mtp` costs two things per card: a second `llama_context` (1417 MiB at
    `-c 160000`, 695 MiB at `-c 16384`) and, through `common_params_speculative::need_n_rs_seq()`
    -> `cparams.n_rs_seq` of the *target* context, one extra copy of the GDN recurrent state per
    unit of `--spec-draft-n-max` (74.75 MiB per card, 149.6 MiB total; `llama_memory_recurrent`
    allocates `mem_size * (1 + n_rs_seq)` rows). At `-c 160000` that gives three walls: widths 1-10
    load and run, 11-13 load and then abort the whole server on the first decode with
    `cuMemCreate: out of memory` from the VMM pool, and 14+ fail at load with a 897 MiB `cudaMalloc`
    for the MTP context's compute reserve. At width 3 the largest `-c` that loads and runs is 168000
    (176128 loads and then aborts on the first decode); at width 1, 176128 runs and 184000 fails at
    load. One unit of draft width is worth about 1800 tokens of `-c`. None of this is moved by patch
    44: with MTP the LM head only reaches cuBLAS at 32 draft tokens, which does not load (J5,
    2026-09-08; supersedes the earlier A4 note that put the wall at 40).
33. `test-backend-ops -o MUL_MAT` cannot reach the cuBLAS dense path on this branch: its quantized
    cases are 16 rows x k=256 with n <= 9, which patch 35 routes through the MMVQ column loop, so 0
    of 1253 cases exercise the src0 conversion. Forcing it needs
    `GGML_CUDA_MMVQ_MAX_COLS_SM60=8` together with `GGML_CUDA_CUBLAS_CHUNK_ROWS=N` (A4,
    2026-09-08). Same blind-spot class as gotcha 30.
34. Speculative decoding is only bit-exact if the target logits at batch n equal the logits at
    batch 1, which they do not on this backend: `--spec-type draft-mtp` at widths 3-31 gives one
    byte-identical greedy reply, widths 1-2 give another, 47/63 give two more (the Q6_K MMVQ ceiling
    of 32, gotcha 24) and no speculation gives a fifth. Each MTP width is exactly reproducible with
    itself. `--spec-type ngram-mod` is not reproducible at all: its draft length varies with the
    match, so the verify batch shape varies, and three identical greedy requests in one server
    process produced three different replies. Any greedy A/B must hold the speculation configuration
    fixed (J5, 2026-09-08). On an easy prompt the flips do not appear at all - a verbatim-echo request
    is byte-identical across every width and no speculation - so absence of drift on one prompt is
    not evidence of bit-exactness.
35. At tg every 5120-wide rms_norm (attn_norm, post_attn_norm, output norm: 129 launches per token per
    card) runs as ONE 1024-thread block on one of 56 SMs, because `rms_norm_f32_cuda` maps one row to one
    block (grid = (nrows, nchannels, nsamples)). It costs 8.84 us in the model against 3.89 us for the same
    op in isolation with a warm L2: the launch is latency-bound on cold reads (weights, residual stream,
    x), not bandwidth-bound, and `test-backend-ops perf` undercounts such single-block launches by 2x.
    `rope.cu` has the same grid shape at tg (E4). The limit is one SM's load/store path (~100 KB per launch
    at ~11 GB/s), so extra blocks that read the same bytes again do not help (E7 negative, gotcha 37); only
    a disjoint split with a cross-block reduction could, at the price of a launch and bit-exactness
    (D4/E6 census and E7, 2026-09-09).
36. The quantize helpers of the matvec (`quantize_q8_1`, `quantize_a16k`) are not graph nodes: no fusion
    rule can see them, `GGML_CUDA_FUSE_LOG` never lists them, and only a launch census counts them. Since
    patches 32-34 an activation that feeds a mixed-type gate/up pair is converted twice (once per format),
    so B4 added 78 quantize launches per token (257 -> 335, 0.83 ms = 2.6% of the step) while removing
    2.9 ms of matvec time. Fix: E8 (D4/E6 census, 2026-09-09).
37. The residual ADD that patch 19 folds into `rms_norm_f32` writes into one of its own inputs: `ggml_gallocr`
    allocates ADD/MUL/RMS_NORM in place when the parent has one child, and over 512 pre-add launches at tg
    `pre_dst == pre_a` 320 times and `pre_dst == pre_b` 192 times, never anything else, with `dst` the other
    of the same two buffers (the whole tg graph ping-pongs between two 5120-float buffers). One block per
    row makes that safe, so the pre-add predicate has no non-aliasing check; any multi-block norm that
    reads the whole row races, and forcing it changes the greedy output (E7, 2026-09-09).
38. `test-backend-ops perf` has no norm case at all in `make_test_cases_perf`: `perf -o RMS_NORM` measures
    nothing on a stock tree. E5 and E7 used temporary local cases (never committed). The 5120x1 norm runs at
    4.3 us warm in that harness and 9.3 us cold in the model, so isolation numbers for single-block launches
    understate the in-model cost by ~2x (E5, E7, 2026-09-09).
39. A GLU's rows are not its consumer's rows: the delta-net gate `{SILU, MUL}` writes 128-wide rows
    (per head) that the `ssm_out` matvec reads as one 3072-wide row, so any producer-side quantization
    must take the row width from the consumer's `src1`, not from the GLU's own shape (E8's first version
    keyed on the GLU shape and silently folded 80 of 128 launches). Also: a saving of ~0.13 ms per token
    is invisible in the summed nsys kernel time (NCCL run-to-run noise is larger) and only shows in the
    paired wall-clock A/B, so judge sub-0.5% items by alternating llama-bench passes, not by kernel sums
    (E8, 2026-09-09).
40. Folding a quantization into the kernel that produces the activation is decided by the target FORMAT's
    lane map, not by the launch count: on the a16k layout 0/1 map (`64g + 4i`) a warp's 256-element block
    is 4 runs of 128 B and the fused GLU runs at 3.85 us against 5.2-5.8 for GLU + quantize, on layout 2
    (Q6_K: `128*(g>>1) + 16*(g&1) + 32*(i>>2) + 4*(i&3)`) it is 8 runs of 64 B and the fused kernel is
    6.8 us against 5.7 separate. And the fused kernel must run on the quantizer's narrow grid (3-11 blocks
    of 128 threads) where the elementwise kernel wanted 12-42 blocks of 256. Both together capped E8 at
    +0.5% on Qwen and -0.4% on Gemma (E8, 2026-09-09). Any future producer-side fold must be gated per
    layout and measured on the model whose weights select the bad layout.
41. Persistence mode does nothing on krk-lab: the Quadro K2200 display card and `nvidia_modeset` keep the
    `nvidia` module loaded (refcount 19), so there is no driver teardown between CUDA processes to prevent,
    and a power limit set with `nvidia-smi -pl` survives `pm off` and process exits. Cold start is 0.33 s
    CUDA init + 4.2 s model load either way (PL2, 2026-09-09). On a headless box without a display card
    the usual advice applies again.
42. `nvidia-smi dmon`'s `pviol` column reads 0.0 at 225 and 200 W although the pp clock has already
    dropped (1328 -> 1316 -> 1265 MHz); the mean graphics clock during the run is the honest indicator of
    an active power cap, and tviol stayed 0 everywhere (PL1, 2026-09-09).
43. `llama-bench -p 4 -n 0` (the verify-width shape) is bimodal on this build, ~75.8 or ~80.9 t/s per
    repetition in every arm including two 250 W baselines, so its median is a coin flip; compare means
    or use the J5 chat harness for that shape (PL1, 2026-09-09). Also: `llama-bench` computes a `-d`
    depth once and restores it from a saved state for the other repetitions, so `-d 16384 -r 5` costs one
    prompt pass, not five; a 7-arm sweep with tg at two depths fits in 18 minutes.
44. The kernel's ASPM policy `default` is the boot value and cannot be written back at runtime
    (`Operation not permitted`), so a restore step that echoes `default` must tolerate the error; the
    first version of `p100-root.sh restore` aborted there under `set -e` and left the governor and the
    C1E hold to be reset by hand (PL3, 2026-09-09; fixed in the draft under `~/p100-opt/root/`).

## 6. Open questions for the user (answered 2026-09-03 where marked)

Answered: 1 (ssh krk-lab, paths in 0.1), 2 (no sudo), 3 (production flags in 0.1), 4 (drift allowed
with the perplexity check), 5 (single stream; `-np 1` in production), 6 (Gemma-4 31B is present,
`-sm tensor` accepts gemma4; C6 2026-09-07), 8 (router mode with `--models-dir` is the production
setup, respawn/load time matters). Open: 7 (P2P/NCCL env in production after tests), 9 (how the
power limit gets set for PL1, added 2026-09-09).

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
   arch on this build? Answered 2026-09-07 (C6): yes and yes; the nan C6 found at matvec widths was fixed
   the same day by B4d (patch 41), and with patch 42 Gemma 4 runs at 1.66x stock tg from the branch.
7. May the agent enable `GGML_CUDA_P2P=1` and change NCCL env vars in production after the tests,
   given the documented risk of instability on some boards (P100-PATCHES.md crash history)?
8. Is the router-mode multi-model setup part of the target (respawn/load times, area K), or is a
   single long-lived instance the only case that matters?
9. Root actions for the final session (PL1-PL3, section 1.5 and the Final table): how will they be
   run, since `nvidia-smi -pl`, `-pm`, the ASPM policy, the governor and `/dev/cpu_dma_latency` all
   need root: the user runs each command on request, or grants a sudoers line limited to those
   commands for that session? Everything is restored at the end except what the user decides to keep. Asked 2026-09-09.
   Answered 2026-09-09: a root-owned helper script `/usr/local/sbin/p100-root.sh` with fixed subcommands
   and one NOPASSWD sudoers line for user krk, installed by the user for the session, logged to
   `/var/log/p100-root.log`, removed afterwards.

## 7. Recommended production configuration (deliverable 4, as of 2026-09-09)

The recorded production line in 0.1 is right and stays. Everything below is what the measurements added
or confirmed; nothing in the line itself changes.

```
NCCL_P2P_LEVEL=SYS LLAMA_CACHE=/mnt/hdd/gguf CUDA_VISIBLE_DEVICES=<the two P100 UUIDs> \
llama serve --host 0.0.0.0 --port 8080 --models-dir /mnt/hdd/gguf --tools all --split-mode tensor \
  --cache-ram 16384 --jinja -ngl 99 -c 160000 --parallel 1 -fa on -b 2048 -ub 2048 --no-mmproj-offload \
  --spec-type draft-mtp --draft-p-min 0
```

- `NCCL_P2P_LEVEL=SYS`: +1.5% pp, +0.9% tg (Tier 0.4); the only environment variable worth setting. Do
  not set `GGML_CUDA_P2P=1`, `GGML_CUDA_ALLREDUCE=none`, `NCCL_PROTO`, `NCCL_MAX_NCHANNELS`,
  `NCCL_SHM_DISABLE`, `CUDA_SCALE_LAUNCH_QUEUES` (all measured, none helps, several hurt).
- `--spec-type draft-mtp` at the default `--spec-draft-n-max 3` (J5): +49% tg on 25k chat follow-ups; do
  not widen the draft, do not add `ngram-mod`. `-c 160000` is the right size for MTP width 3 (the
  largest that loads and runs is 168000, gotcha 32); 15.5 GiB per card in use at load.
- `-b 2048 -ub 2048`: `-ub 512` costs 45% of pp (Tier 0). `-t 4 --poll 0` saves host power only.
- `--cache-ram 16384` has no speed effect with `--parallel 1` (the slot is re-selected by prefix match,
  J1); `--cache-ram 0` would free 16 GiB of host RAM if the box ever needs it.
- No request-level `logit_bias`, no per-request sampler exotica (J6) so the sampler prefilter (patch 43)
  and graph reuse stay on.
- GPU and host state: no power limit (PL1: tg draws ~155 W per card and is flat to 175 W; a limit only
  clips pp; 175 W is the value if a lower chassis peak is ever wanted, at -7% pp), persistence mode on
  (harmless, a no-op here), stock governor and C-states, ASPM not applicable, no application clocks
  (boost already sits at 1328 MHz under load). Do not set compute mode EXCLUSIVE_PROCESS (router
  children).
- Binary: the branch build (`p100-b10758`, patches 01-30 + gist + p100x 31-44); the production
  `/home/krk/llama.cpp` (branch `p100`, b10630 base) is 8-10% slower at tg and lacks patches 31-44.
- `~/llama-serve.sh` already carries this exact line including `NCCL_P2P_LEVEL=SYS`; the only step is
  rebuilding its binary from this branch. `P100-README.md` is the quick start for that.

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
| `LLAMA_SAMPLER_PREFILTER=0` | patch 13 off (and 43 with it) |
| `LLAMA_MTP_DRAFT_VOCAB=<file>` | patch 15 draft vocab subset |
| `LLAMA_SERVER_SLOTS_DEBUG=1`, `LLAMA_TRACE=1` | server slot/trace logging (J1) |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` | larger CUDA command buffer (G5) |
| `CUDA_VISIBLE_DEVICES` | single-card runs |
| `GGML_CUDA_FATTN_TILE_LEGACY=1` | patch 36 kill switch: b10758 flash-attention tile kernels, no GQA 6 packing |
| `GGML_CUDA_FATTN_LOG=1` | patch 36: one line per distinct flash-attention launch (kernel, ncols1/ncols2, config, parallel_blocks, grid) |
| `GGML_CUDA_FATTN_PB_TIEBREAK=0` | patch 38 kill switch: b10758 `parallel_blocks` search (implied by `GGML_CUDA_FATTN_TILE_LEGACY=1`) |
| `GGML_CUDA_FATTN_GQA6_MIN_KV=<n>` | patch 38: smallest n_kv for the GQA 6 packing (default 2560, 7680 at 2 Q columns; 16384 = patch 36; 0 = always) |
| `GGML_CUDA_DISABLE_CONVERT_VEC=1` | patch 39 kill switch: scalar `convert_unary` for the contiguous F16/BF16 <-> F32 conversions |
| `GGML_CUDA_DISABLE_DEQUANT_VEC=1` | patch 39 kill switch: scalar stores and one super block per block for the Q4_K/IQ4_XS dequant |
| `GGML_CUDA_NORM_CACHE_LEGACY=1` | patch 40/42 kill switch: `rms_norm` register cache limited to 4096 columns (patch 17) |
| `GGML_A16K_CHECK=1` | patch 41 debug: sync after every HFMA2 K-quant matvec and print the first launches with a non-finite output |
| `GGML_CUDA_CUBLAS_CHUNK_MB=N` (default 256, 0 = off) | patch 44 kill switch and threshold: convert and multiply a src0 copy above N MiB in row chunks |
| `GGML_CUDA_CUBLAS_CHUNK_ROWS=N`, `GGML_CUDA_CUBLAS_CHUNK_LOG=1` | patch 44 test knobs: force N rows per chunk; one stderr line per split call |

## Appendix B: reading order for a new agent

0. `P100-README.md`: how the user builds and starts production from this branch, the smoke test, the
   kill switches that matter, and what is left.

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
