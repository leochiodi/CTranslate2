# CB Decode Performance Optimization — Status & Findings

## The Problem

The continuous batching (CB) Whisper server is **2-4x slower** than the old server (stock faster-whisper with `BatchedInferencePipeline`) at 80 concurrent users on an H100 SXM.

## Root Cause: CB decode step is ~4.5x slower than standard generate()

For the **same batch size** (40 sequences = 8 slots × beam_size 5):

| Decode path | per_step latency | Notes |
|-------------|-----------------|-------|
| `model.generate()` (old server) | ~10ms | Tight loop: decoder → beam search → repeat |
| `ContinuousDecodingEngine::process()` (CB) | **46ms** | Slot mgmt, defrag, scatter/gather, cache padding every step |

The 36ms overhead per step compounds over ~100 decode steps per segment, making each segment take ~4.6s in CB vs ~1s in generate(). This is the **only thing that matters** — everything else has been investigated and ruled out.

## What We Investigated & Ruled Out

### 1. CUDA stream contention from multiple decoder threads
- **Finding**: With 4+ decoder workers, per_step goes from 6ms (solo) to 90ms (contended). But even with 1 decoder (zero contention), per_step is 46ms for 40 seqs.
- **Conclusion**: Contention adds overhead but is NOT the primary issue. The CB engine itself is slow.

### 2. Single decoder with many slots (our latest architecture)
- Implemented N encoder threads feeding 1 decoder thread (code is in place)
- e4_s32 (32 slots = 160 seqs): 123ms/step, GPU 92%
- e4_s16 (16 slots = 80 seqs): 74ms/step, GPU 94%
- e4_s8 (8 slots = 40 seqs): 46ms/step, GPU 94%
- **Conclusion**: Per-step scales linearly with batch size. Single decoder eliminates contention but doesn't fix the per-step overhead.

### 3. GPU mutex to serialize multiple decoders
- **Discarded before implementation**: If worker A finishes a step and is ready for the next, giving the GPU to worker B just delays A. Total GPU work is identical to a single decoder — zero throughput gain.

### 4. Batcher load imbalance / routing
- Fixed langdetect round-robin and batcher round-robin routing
- **Conclusion**: Helped distribute load evenly but didn't fix the fundamental per-step overhead.

### 5. Language detection mutex bottleneck
- **Fixed**: Moved detect_language from `_replica_mutex` to encoder thread. Now p50=0.26s at 80 users (was 21s+).
- **Not the bottleneck anymore.**

### 6. VAD serialization
- Uses Silero ONNX on CPU, not CUDA. Fast enough.

### 7. Encoder batching
- Already implemented: 2ms batching window in encoder_loop, batch-encodes multiple requests in one GPU call.
- **Not the bottleneck.**

## Why the Old Server Is Fast

The old server uses `BatchedInferencePipeline` with 4 workers, `batch_size=16`, `MAX_AGE=2.0s`:
- Preprocessor collects ~5-8 requests per batch (fill ratio 30-60%)
- Each worker calls `model.generate()` which runs the **standard beam search** — a tight loop with no slot management
- The Python GIL naturally serializes workers — only one decoder runs at a time, zero CUDA contention
- Per-step for 40 sequences: ~10ms (estimated from throughput data)
- Total throughput: 2.6 req/s at 80 users, p50=4.6s

## Profiling Results (2026-02-20)

Ran benchmark with `[DECODE PROFILE]` counters at 80 users, e4_s8 config (8 slots, beam_size 5 = 40 seqs):

```
[DECODE PROFILE] steps=1489  total=74239.0ms  per_step=49.86ms  GPU%=92.5%
  slot_mgmt=0.4  cache_pad=0.0  step_setup=846.7  DECODER=68649.2
  attn_accum=66.5  logits=999.3  selection=2411.3  resize_up=0.1
  defrag=12.3  fill=1148.6  rebuild=93.0
```

**Per-step breakdown** (dividing by 1489 steps):
| Phase | Total ms | Per-step ms | % of total |
|-------|----------|-------------|------------|
| **DECODER** | 68649 | **46.1ms** | **92.5%** |
| selection | 2411 | 1.6ms | 3.2% |
| fill | 1149 | 0.8ms | 1.5% |
| logits | 999 | 0.7ms | 1.3% |
| step_setup | 847 | 0.6ms | 1.1% |
| everything else | ~126 | ~0.1ms | <0.2% |

### Key Finding: The bottleneck is INSIDE the decoder call itself

**The surrounding CB machinery (slot mgmt, cache padding, defrag, resize) is negligible.** The `_decoder()` call alone takes 46ms/step — the same 46ms we're trying to optimize. The standard `generate()` path does the equivalent decoder call in ~10ms.

### What was ruled out by this profiling:

- **Cache padding**: 0.0ms — already efficient with 64-token chunk alignment (only triggers on chunk boundary crossings)
- **Defragmentation**: 12.3ms total across 1489 steps = negligible
- **Slot management**: 0.4ms total = negligible
- **Resize up/down**: 0.1ms total = negligible

### Attempted optimization: Pre-allocate KV caches to _max_length

- **Hypothesis**: Eliminate per-step `pad_cache()` calls (up to 64 GPU allocations per step via `ops::Concat`)
- **Result**: cache_pad dropped to 0.0ms but per_step got slightly **worse** (49.86ms vs 46ms)
- **Why it failed**: Pre-allocating caches to `_max_length` (448 tokens) made the attention kernels process the full sequence length including zero-padding. The scatter path writes correctly but attention reads the full time dimension.
- **Reverted**: Changes were reverted.

## Deep Code Analysis: CB vs Standard Decoder Paths (2026-02-20)

### Critical insight: Cache padding is NOT the primary cause

The pre-allocation experiment (64→448 tokens) only added 3.86ms (46→49.86ms). Extrapolating:
- Going from 64→0 padding would save at most ~0.6ms
- **Cache padding accounts for < 1ms of the 36ms overhead**
- The FLOPS overhead of padded self-attention is negligible at these sizes (tiny MatMuls on H100)
- Memory bandwidth overhead of reading padded caches: ~95μs across 32 layers

### Exact differences between the two decoder code paths

Both paths call the same `TransformerDecoderLayer` code. The differences are in how `TransformerDecoder::decode()` sets up the call:

| Aspect | Standard path (`decode(dim_t step, ...)`) | CB path (`decode(StorageView& step_offsets, ...)`) |
|--------|------------------------------------------|---------------------------------------------------|
| **Self-attention mask** | **None** — `input_lengths_mask` is null for iterative decoding (step > 0, ids rank 1) | **Always builds mask** from `cache_lengths` via `prepare_length_mask()` |
| **Cache writes** | `ops::Concat` in attention.cc — single op per layer, creates new allocation with exact size | `scatter_cache_step` — `cudaMemcpy2DAsync` if uniform, `batch_copy_async` with B×H copy descriptors if not |
| **Position encoding** | `_position_encoder(layer_in, scalar_step)` — broadcast | `_position_encoder(layer_in, offsets_cpu)` — per-element gather |
| **Concat path fires** | Always (cache grows by 1, exact size) | Essentially never (requires ALL slots active AND uniform cache AND `current_cache_time == common_cl`) |
| **Cross-attention** | Identical — cached K/V used after step 0 | Identical — cached K/V populated during fill phase |
| **FFN** | Identical | Identical |

### Why the concat path never fires in CB

The condition at `continuous_decoding.cc:976`:
```cpp
use_concat_path = (active_count == _max_slots)       // All slots must be active
    && uniform_cache && common_cl >= 0
    && current_cache_time == static_cast<dim_t>(common_cl);  // Cache must be exact-sized
```

This NEVER fires because:
1. `active_count == _max_slots` — slots finish at different times, rarely all active
2. `current_cache_time == common_cl` — once scatter pads to 64-token chunks, `current_cache_time` is always > `common_cl`
3. Even with all slots active and uniform, chunk padding breaks condition 2

Result: CB **always** uses scatter path with attention mask, never the fast concat path.

### Scatter write overhead analysis

With non-uniform cache_lengths (8 slots at different steps, beam_size=5):
- Positions: `[15,15,15,15,15, 12,12,12,12,12, 3,3,3,3,3, ...]` — NOT globally uniform
- Falls to slow path: `batch_copy_async` with `batch × heads` = 40 × 20 = 800 copy descriptors per call
- 32 layers × 2 (K+V) = 64 calls per step → 51,200 total copy descriptors per step
- Each copy is 64 × 2 = 128 bytes (one head's d_head in fp16)

### What needs investigation

The 36ms overhead is NOT explained by:
- Cache padding compute (~0.6ms)
- Cache padding bandwidth (~0.1ms)
- Mask construction (tiny tensor, done once per step)

**Possible causes** (need per-layer CUDA profiling to confirm):
1. **Scatter kernel launch overhead** — 51,200 small copies via batch_copy_async. If not truly batched into a single kernel, this could be catastrophic
2. **SoftMax masked vs unmasked code path** — CB always passes `values_lengths` to SoftMax; standard path passes null. Different CUDA kernel execution
3. **Hidden GPU↔CPU synchronization** — StorageView operations that trigger implicit sync
4. **CUDA allocator pressure** — scatter path keeps large padded caches; concat path creates/frees each step (potentially better cache locality)

### Recommended next steps

1. **Add CUDA event timing per layer** inside `TransformerDecoder::decode(step_offsets)`:
   - Self-attention total (projection + cache update + dot_product_attention)
   - Cross-attention total
   - FFN total
   - Gate behind `CT2_LAYER_PROFILE` env var

2. **Run the standard decode path with same batch size** and compare per-layer times directly

3. **Quick experiment: reduce `chunk_size` from 64 to 4** in `continuous_decoding.cc:990`
   - Won't fix the 36ms but will reduce cache padding from ~64 to ~4-8 tokens
   - Expected improvement: ~0.5-1ms (minor but free)

4. **Investigate batch_copy_async implementation** — is it a single kernel or N individual cudaMemcpyAsync calls?
   - If individual: this alone could explain the entire 36ms overhead
   - File: `src/cuda/batch_copy.h` or similar

5. **Try forcing concat path** by trimming caches to exact size when cache_lengths are uniform (beams within a slot are always uniform). Process each slot's 5 beams as a sub-batch with concat path

## Current Architecture (in code)

- `WhisperContinuousBatcher` supports `num_encoders` parameter: N encoder threads, 1 decoder thread
- All encoder threads share `_raw_queue` and push to single `_encoded_queue`
- Server.py uses a single batcher instance (no multi-batcher routing)
- Env vars: `MAX_SLOTS` (decode slots), `NUM_ENCODERS` (encoder threads)
- Build: `cd CTranslate2/build && cmake --build . -j$(nproc)` then `CTRANSLATE2_ROOT=$(pwd) LIBRARY_PATH=$(pwd)/build pip install -e python/ --no-build-isolation`

## Key Files

| File | What |
|------|------|
| `CTranslate2/src/continuous_decoding.cc` | **THE FILE TO OPTIMIZE** — ContinuousDecodingEngine::process(), concat/scatter decision at line 958 |
| `CTranslate2/src/layers/transformer.cc` | TransformerDecoder::decode — standard path (line 606), CB path (line 850) |
| `CTranslate2/src/layers/attention.cc` | MultiHeadAttention — scatter_cache_step (line 26), concat vs scatter (line 620), dot_product_attention (line 293) |
| `CTranslate2/src/models/whisper.cc` | WhisperContinuousBatcher (encoder_loop, worker_loop) |
| `CTranslate2/include/ctranslate2/models/whisper.h` | Batcher header |
| `CTranslate2/include/ctranslate2/continuous_decoding.h` | Engine header |
| `ctranslate2-server/server.py` | FastAPI server (multi-encoder, single-decoder) |
| `ctranslate2-server/server_old.py` | Old server baseline (stock faster-whisper, queue-based) |
| `ctranslate2-server/run_benchmark.sh` | A/B benchmark script |
| `ctranslate2-server/bench_compare.py` | Benchmark client |
| `ctranslate2-server/setup_runpod.sh` | RunPod setup (3 venvs, CT2 build) |

## Benchmark Commands

```bash
# Build
cd /workspace/CTranslate2/build && cmake --build . --config Release -j$(nproc)
cd /workspace/CTranslate2 && CTRANSLATE2_ROOT=$(pwd) LIBRARY_PATH=$(pwd)/build /root/venv_new/bin/pip install -e python/ --no-build-isolation

# Run server (single batcher, 4 encoders, 8 slots)
WHISPER_MODEL=/workspace/models/whisper-large-v3 NUM_ENCODERS=4 MAX_SLOTS=8 \
    DEVICE=cuda COMPUTE_TYPE=float16 BEAM_SIZE=5 REQUEST_TIMEOUT=120 DEFAULT_LANGUAGE=fr \
    LD_LIBRARY_PATH=/workspace/CTranslate2/build:$LD_LIBRARY_PATH \
    /root/venv_new/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Benchmark (80 users, 3 loops)
/root/venv_bench/bin/python bench_compare.py --users 80 --loops 3 \
    --url http://localhost:8000/transcribe --audio f1_audio.mp3 --abort-sla 15

# Old server baseline
WHISPER_MODEL=/workspace/models/whisper-large-v3 NUM_WORKERS=4 BATCH_SIZE=16 MAX_AGE=2.0 \
    DEVICE=cuda COMPUTE_TYPE=float16 BEAM_SIZE=5 REQUEST_TIMEOUT=120 DEFAULT_LANGUAGE=fr \
    /root/venv_old/bin/python -m uvicorn server_old:app --host 0.0.0.0 --port 8010
```

## Environment

- RunPod H100 SXM (original benchmarks)
- RunPod L4 (current iteration — much less compute, ~80 users is unrealistic)
- Three venvs: `/root/venv_old` (stock faster-whisper), `/root/venv_new` (custom CT2), `/root/venv_bench` (benchmark client)
- Model: `/workspace/models/whisper-large-v3`
- Audio: `ctranslate2-server/f1_audio.mp3` (~9min F1 commentary, split into 30s chunks)

## Optimization Attempt #1 (2026-02-20, L4)

### Changes made

1. **`src/cuda/batch_copy.cu`**: Increased `MAX_BATCH_COPIES` from 128 → 1024.
   - Rationale: with 8 slots × beam_size 5 × 20 heads = 800 copy descriptors per scatter call, the old limit caused `ceil(800/128) = 7` recursive kernel launches per call. With 1024, all copies fit in a single kernel launch.
   - Expected: reduce scatter kernel launches from ~448/step to ~64/step.

2. **`src/layers/attention.cc`**: Added comment documenting the single-launch benefit.

3. **`src/layers/transformer.cc`**: Added per-layer CUDA event profiling gated behind `CT2_LAYER_PROFILE` env var. When set, synchronizes GPU after each layer and prints per-layer timing every 500 decode calls. Overhead is significant (do NOT use in production).

### What was tried and reverted

- **Grouped scatter with per-slot `cudaMemcpy2D`**: Replaced `batch_copy_async` with per-group `cudaMemcpy2D` calls (one per unique write position = one per slot). This produced 512 API calls per step vs 64 kernel launches with increased `MAX_BATCH_COPIES`. Result: **81.79ms/step — WORSE than baseline (was ~36ms at same users)**. Each `cudaMemcpy2D` for D2D is internally a kernel launch, so 512 calls >> 64 calls. **Reverted.**

### L4 baseline results (e4_s8, 20 users, 2 loops)

| Metric | Value |
|--------|-------|
| p50 latency | 5.09s |
| p95 latency | 6.33s |
| Throughput | 0.63 req/s |
| per_step (high load) | 36.45-36.71ms |
| per_step (low load) | 16.37-18.94ms |
| GPU% (high load) | 89-91% |

### Per-layer profiling (e4_s8, 20 users, CT2_LAYER_PROFILE=1)

With GPU sync after each layer (adds overhead but gives accurate per-layer GPU time):
- Total across 32 layers: ~17.5ms per decode step
- Per layer: 0.43–0.91ms (varies by layer, early layers slightly slower)
- This 17.5ms is **actual GPU compute time** — the remaining gap to the 36ms per_step is kernel launch overhead, stream scheduling, and CB machinery

### L4 config comparison: e4_s8 vs e2_s4 (baseline, 20 users)

| Config | p50 | p95 | Throughput | per_step (high) | GPU% (high) |
|--------|-----|-----|------------|------------------|-------------|
| e4_s8 | 5.09s | 6.33s | 0.63 req/s | 36.45ms | 89-91% |
| **e2_s4** | **2.72s** | **5.60s** | **0.66 req/s** | 19-25ms | 87-89% |

**e2_s4 is clearly better on L4** — 47% lower p50, slightly higher throughput. The L4 doesn't have enough compute to keep 8 slots busy efficiently.

### A/B: `MAX_BATCH_COPIES=1024` vs baseline (e2_s4, 20 users)

| Version | p50 | p95 | Throughput | per_step (high) |
|---------|-----|-----|------------|------------------|
| Baseline (128) | 2.72s | 5.60s | 0.66 req/s | 19-25ms |
| Optimized (1024) | 3.80s | 6.08s | 0.66 req/s | 17-29ms |

**Result: neutral to slightly worse.** On L4 with e2_s4 (4 slots × 5 beams × 20 heads = 400 copies), baseline only does `ceil(400/128) = 4` launches. Reducing 4→1 doesn't save enough to matter. The **actual decoder computation** (~17ms GPU time per step) dominates, not launch overhead.

### Key insight from profiling

The per-layer profile shows the decoder's real GPU time is ~17.5ms per step (all 32 layers). On the L4, the wall-clock per_step is 19-36ms depending on load. The ~2-18ms gap is:
- CB machinery (slot mgmt, cache padding, beam selection): ~2-4ms
- Kernel launch scheduling / stream overhead: ~0-14ms (scales with load and GPU contention)

The scatter launch overhead (reduced by MAX_BATCH_COPIES) is a small part of this gap. The dominant issue remains: **each decoder layer in CB is doing ~0.55ms of GPU compute**, while the standard `generate()` path does the equivalent in ~0.3ms/layer (estimated from 10ms / 32 layers at same batch size on H100).

## Sub-layer Profiling Results (2026-02-20, L4, e2_s4, 20 users)

Added `CT2_SUBLAYER_PROFILE` env var to time self-attention, cross-attention, and FFN separately within each `TransformerDecoderLayer`. GPU sync before/after each sub-operation for accurate timing.

### Per decode step (averaged over 2500 steps, 32 layers each):

| Sub-operation | Per-step (32 layers) | Per-layer | % of in-layer GPU time |
|---|---|---|---|
| Self-attention | 7.04ms | 0.220ms | 35.3% |
| Cross-attention | 7.34ms | 0.229ms | 36.8% |
| FFN | 5.58ms | 0.174ms | 27.9% |
| **Total in-layer** | **19.96ms** | **0.624ms** | 100% |

Wall-clock per_step: 29.57ms → **~10ms gap** between sublayer total and wall clock.

### Critical insight: CB in-layer overhead is negligible

**Cross-attention is identical between CB and standard paths** — same code, same cached K/V from the encoder. It costs 7.34ms per step.

**Self-attention (CB path, with mask + scatter + padding) costs 7.04ms** — actually 0.30ms LESS than cross-attention.

This means: **all the CB-specific changes (mask construction, scatter writes, padded cache) add < 0.5ms total across 32 layers per step.** The self-attention cost is dominated by Q/K/V projections and memory bandwidth, not by CB-specific overhead.

The earlier "0.55 vs 0.3 ms/layer = 1.8× overhead" estimate was misleading — the 0.3ms/layer H100 estimate was from a much faster GPU (11× bandwidth, 4× compute). On L4, the standard path would also take ~0.2ms/layer for cross-attn and FFN.

### Why FFN is the cheapest despite being the most compute

FFN reads 2 weight matrices of d_model × d_ff = 1280 × 5120 = 6.55M params × 2 bytes = ~13.1 MB per layer, ~419 MB for 32 layers. At L4's ~300 GB/s → ~1.4ms theoretical. Measured 5.58ms reflects both up+down projections at near-roofline for fp16 GEMM at batch=20.

### The 10ms gap is the real optimization target

| Component | Estimated cost |
|---|---|
| CB machinery (beam selection, slot mgmt, fill, defrag) | ~3-4ms |
| LM head projection (1280→51864 vocab) | ~0.5-1ms |
| Output norm + embedding + position encoding | ~0.5-1ms |
| Kernel launch overhead (hundreds of launches per step) | ~2-3ms |
| Other (memory allocation, CPU↔GPU transfers) | ~1-2ms |

### Updated conclusions

1. **Optimizing inside layers (mask, scatter, chunk_size) is NOT worthwhile** — total CB overhead there is < 0.5ms
2. **The "fast CB path" (uniform cache_lengths → skip mask) would save < 0.5ms** and almost never triggers anyway
3. **MAX_BATCH_COPIES change was neutral** because scatter overhead was already small
4. The ~10ms out-of-layer gap is split between inherent CB machinery costs and kernel launch overhead

## End-to-End A/B: CB vs Old Server on L4 (2026-02-20)

### Hardware & setup

- **GPU**: NVIDIA L4 (Ada Lovelace, 58 SMs, ~23GB GDDR6, ~300 GB/s bandwidth)
- **Model**: Whisper large-v3 (CT2 fp16), `/workspace/models/whisper-large-v3`
- **Audio**: `f1_audio.mp3` (~9min F1 commentary, split into 30s chunks = 18 chunks)
- **Benchmark**: 20 concurrent users, 3 loops each, jitter 0-30s, SLA threshold 5s
- **Pyannote VAD model** required for old server: `pyannote/segmentation-3.0` downloaded to `/workspace/models/ModelHub-model-huggingface-pyannote/segmentation-3.0/main/`

### CB server config

```bash
WHISPER_MODEL=/workspace/models/whisper-large-v3 NUM_ENCODERS=2 MAX_SLOTS=4 \
    DEVICE=cuda COMPUTE_TYPE=float16 BEAM_SIZE=5 REQUEST_TIMEOUT=120 DEFAULT_LANGUAGE=fr \
    LD_LIBRARY_PATH=/workspace/CTranslate2/build:$LD_LIBRARY_PATH \
    /root/venv_new/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Old server configs tested

```bash
# 2 workers (best on L4):
WHISPER_MODEL=/workspace/models/whisper-large-v3 NUM_WORKERS=2 BATCH_SIZE=16 MAX_AGE=2.0 \
    DEVICE=cuda COMPUTE_TYPE=float16 BEAM_SIZE=5 REQUEST_TIMEOUT=120 DEFAULT_LANGUAGE=fr \
    MODELHUB_DIR=/workspace/models \
    /root/venv_old/bin/python -m uvicorn server_old:app --host 0.0.0.0 --port 8010

# 3 workers (worse on L4 — GPU contention):
# Same as above but NUM_WORKERS=3
```

### Results (20 users, 3 loops, L4)

| Server | Config | p50 | p95 | Throughput | SLA violations (>5s) |
|---|---|---|---|---|---|
| **CB** | **e2_s4 (2 enc, 4 slots)** | **3.74s** | **5.06s** | **0.65 req/s** | **5%** |
| Old | 2 workers, pyannote VAD | 6.18s | 7.79s | 0.65 req/s | 81.7% |
| Old | 3 workers, pyannote VAD | 10.91s | 13.26s | 0.64 req/s | 91.7% |

**CB is 1.65× faster at p50 on L4** with identical throughput. Old server with 3 workers suffers from GPU contention — throughput stays flat but latency nearly doubles.

### Why CB wins despite slower per-step decode

The CB decode step is slower per-step than standard `generate()` (sub-layer profiling shows < 0.5ms of CB-specific overhead, but ~10ms of out-of-layer overhead from CB machinery). However, the end-to-end result is faster because:

1. **No batch-wait delay** — old server waits `MAX_AGE=2.0s` to accumulate a batch; CB starts decoding immediately when a slot is free
2. **Overlapped encoding and decoding** — CB encoder threads run concurrently with the decoder thread; old server's GIL serializes everything
3. **No idle GPU gaps** — CB continuously feeds the GPU; old server has gaps between batch completions and next batch formation

### Replication on H100

On H100 SXM (132 SMs, 80GB HBM3, 3.35 TB/s), the per-step decode overhead is proportionally larger relative to compute time (compute is ~4× faster but overhead stays similar). Use higher concurrency:

```bash
# CB server on H100 — try e4_s8 or e8_s16:
NUM_ENCODERS=4 MAX_SLOTS=8   # or NUM_ENCODERS=8 MAX_SLOTS=16

# Old server on H100 — try 4-8 workers:
NUM_WORKERS=4   # or NUM_WORKERS=8

# Benchmark with more users:
bench_compare.py --users 80 --loops 3 --abort-sla 15
```

The H100 has enough compute and bandwidth to support more concurrent slots/workers. The relative CB vs old server comparison may differ there — the per-step decode overhead (measured at ~46ms for 40 seqs on H100 vs ~10ms for standard) is a larger fraction of total time.

### Remaining investigation

- Profile the **out-of-layer overhead** in detail: what specifically costs ~10ms between sublayer total (20ms) and wall-clock per_step (30ms)?
- Consider **CUDA graphs** for the repeated decoder layer calls (but CB's dynamic batch sizes make this hard)
- Consider **fusing beam selection with GPU** to reduce CPU→GPU roundtrips
- Re-test on **H100** where the per-step overhead is more pronounced relative to compute speed
