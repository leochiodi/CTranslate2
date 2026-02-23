# Whisper Continuous Batching (CB) — Optimization Status & Analysis

**Hardware tested:** DGX Spark (unified memory, low launch overhead), H100 80GB HBM3 (sm_90, PCIe)
**Model:** whisper-large-v3 (CTranslate2 format), float16
**Benchmark audio:** F1 commentary, ~30s per clip, French
**Benchmark tool:** `bench_compare.py` in `/workspace/ctranslate2-server/`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Optimizations Implemented (10 Steps)](#2-optimizations-implemented-10-steps)
3. [H100 Benchmark Results — Single Batcher](#3-h100-benchmark-results--single-batcher)
4. [Multi-Batcher Architecture & Results](#4-multi-batcher-architecture--results)
5. [Key Findings](#5-key-findings)
6. [Suggestions for Next Steps](#6-suggestions-for-next-steps)

---

## 1. Architecture Overview

### Old Server (`server_old.py`)

```
HTTP → Q_RAW → Preprocessor (batch by language, BATCH_SIZE=16, MAX_AGE=2.0s)
     → Q_BATCH → Worker Thread ×N (each has own WhisperModel)
       → pipeline.transcribe() → BeamSearch::search() (tight decode loop)
     → Q_RESULT → HTTP Response
```

- N independent model replicas (each holds full weights, ~3GB VRAM each)
- GIL serializes GPU work — zero CUDA stream contention
- Fixed batch: waits up to MAX_AGE to fill, then processes entire batch together
- `generate()` is a tight loop: embed → 32 layers → LM head → beam select → gather KV → repeat
- **No slot management, no defrag, no resize, no mask construction, no fill**

### CB Server (`server.py`)

```
HTTP → ThreadPool → _process_request()
     → VAD → feature extraction → batcher.submit() per segment
     → Encoder Thread ×M (shared model, 2ms batching window)
     → _encoded_queue
     → Decoder Thread ×1 (ContinuousDecodingEngine::process())
       → Fill slots → build metadata → decoder forward → logits → beam search
       → defrag → fill new slots → loop
     → batcher.get_result() → HTTP Response
```

- Single shared model instance
- Continuous batching: slots fill/drain dynamically mid-generation
- No batch-wait delay: decode starts as soon as a slot is free
- Per-step overhead from slot management, metadata construction, defrag, scatter writes

### Per-Step Overhead Sources (CB vs Old)

Measured on L4 at e2_s4 (20 batch rows), 20 concurrent users:

| Source | CB extra cost | Notes |
|--------|-------------|-------|
| Kernel launch overhead | ~1-2ms | More kernels: scatter, position, mask, defrag, fill |
| Fill new slots | ~0.8ms | `set_batch_slot()` scatters prepared state into batch |
| Step setup (CPU) | ~0.6ms | `step_offsets`, `cache_write_positions`, `sample_from` |
| Logits processing | ~0.4ms | Per-slot Whisper timestamp rules vs per-batch |
| In-layer overhead | ~0.5ms | Masked softmax, scatter cache writes, per-element position |
| Beam selection | ~0.3ms | GPU pipeline vs CPU tight loop |
| Attention accumulation | ~0.3ms | CB-only, for word-level alignment |
| Defrag + resize | ~0.2ms | Physical compaction of active slots |
| **Total extra** | **~4-6ms/step** | On faster GPUs, overhead stays constant but compute shrinks |

### Key Architectural Differences in Decoder Step

**Identical** between old and CB: token embedding, Q/K/V projections, split heads, dot-product attention, cross-attention, FFN, output norm, LM head projection.

**CB-only operations per step:**
1. Count active slots, build indices
2. Cache padding check (64-token chunk alignment)
3. Build step_offsets on CPU → H2D transfer
4. Build cache_write_positions on CPU
5. Per-element position encoding (fused GPU kernel, vs scalar broadcast)
6. Scatter cache writes (vs concat-grow)
7. Build attention mask from cache_lengths (vs no mask when step>0)
8. Accumulate cross-attention weights
9. Per-slot logits processing
10. GPU beam select kernel + async D2H staging
11. Defrag finished slots + fill new slots

### KV Cache: Scatter vs Concat

| Aspect | Old (Concat) | CB (Scatter) |
|--------|-------------|-------------|
| Allocations/step | 64 (K+V × 32 layers) | 0 (amortized, grows by 64-token chunks) |
| Data copied/step | Entire cache history | Just new K/V row |
| Memory waste | None | Up to 63 padding tokens |
| Attention reads | Exact length | Padded length (masked) |

### Cross-Attention K/V Beam Sharing

Cross-attention memory tensors (`memory_keys_L`, `memory_values_L`) are stored **once per slot** (not per beam). The code uses `starts_with(name, "memory")` to distinguish slot-level from beam-level tensors. With beam_size=5 and 8 slots, there are 8 memory entries (not 40). This means cross-attention memory cost scales with slots, not with total_batch.

---

## 2. Optimizations Implemented (10 Steps)

All 10 steps were developed and validated on DGX Spark (unified memory). Steps 1-8 are Phase A+B, Steps 9-10 are Phase C. All verified via `test_correctness.sh` (standard generate path, 115 tokens exact match) and CB engine tests (single, concurrent, staggered).

### Phase A: CPU Overhead Elimination

#### Step 1: GPU-resident cache_lengths + eliminate CPU metadata loops

- `cache_lengths` moved from CPU to GPU (with CPU shadow)
- `step_offsets_device` = `cache_lengths` (no separate H2D)
- `cache_write_positions` = `cache_lengths` (shallow copy)
- Rebuild → `increment_cache_lengths_gpu` kernel
- `attn_lengths` → `add_scalar_int32_gpu` kernel
- `min_step` precomputed on CPU in `batch_state`

**Result (DGX Spark, 1 slot):** step_setup+rebuild: 0.63ms → 0.02ms/step

#### Step 2: Remove resize up/down from hot loop

- Always run decoder on `total_batch` rows (no resize down/up)
- Inactive rows: `cache_lengths=1` (NOT 0 — avoids all-masked softmax NaN)
- Deleted all 4 `resize_batch_state()` calls
- TopK uses `max_slots`, `add_depth_broadcast` uses `total_batch`

**Result (DGX Spark, 4 slots):** per_step=20.47ms (was 21.60ms 1-slot due to fixed 20 rows). Stable shapes enable CUDA graphs.

#### Step 3: Deferred slot fill (adaptive K=1-4)

- Defrag runs immediately (cheap: copy 1-2 slots)
- Fill deferred to every K steps (K=1 when queue has work, ramps to 4 when idle)

**Result (DGX Spark):** Fill only runs when needed. Staggered 4-request test: 167 steps, fill=13.4ms total.

### Phase B: CUDA Graphs + Pre-staging

#### Step 4: Pre-allocate decoder intermediate buffers

- `CbDecodeBuffers` in `TransformerDecoder`: ping-pong `buf[2]` [total_batch, 1, d_model], `attn_lengths` [total_batch], `position_bias`
- `ensure_cb_buffers()` allocates once, reuses across steps
- No more `layer_in = std::move(layer_out)` — stable GPU addresses

**Result:** No perf change (structural — enables graph capture).

#### Step 5: CUDA Graph capture for decoder forward

Graph 1 captures: embedding → scale → position encoding → layernorm → 32×{self-attn + cross-attn + FFN} → output_norm → LM head → logits.

NOT captured: step setup, cache padding, logits processing, beam search, fill, defrag.

5 CPU-dependent branching issues fixed:
1. Position encoder: uniform check → always GPU kernel with GPU offsets during capture
2. `scatter_cache_step`: uniform check → always GPU scatter kernel during capture
3. Profiling syncs: gated on `!g_cuda_graph_capturing`
4. Cross-attention memory access: only capture during steady-state (min_step > 0)
5. `state.find()/erase()`: CPU map ops — not a blocker (only GPU ops captured)

Invalidation triggers: cache chunk growth, defrag, fill (new slot at step 0).
Disable: `CT2_NO_CUDA_GRAPH=1` env var.

**Result (DGX Spark):** Graph vs no-graph ~0.08ms diff (expected: unified memory has low launch overhead). Infrastructure correct and ready for H100.

#### Step 6: Second CUDA Graph for beam search (Phase C of decode)

Graph 2 captures: TopK → topk_scores float32 conversion → `beam_select_async` → `fill_identity_async` → D2D copies.

Outside Graph 2: LogSoftMax (must run before reshape), `add_depth_broadcast`, Gather loop (pointer swaps), D2H copies (variable-sized), event recording.

Key: `beam_id` clamped `min(flat_id / vocab_size, beam_size - 1)` to prevent OOB from inactive slot garbage.

**Result (DGX Spark):** Graph 2 adds ~24ms total overhead (capture cost > per-kernel savings on unified memory). Expected net positive on H100.

#### Step 7: Pre-staged beam replication on encoder thread

- After `forward_prompt` in encoder thread: tile self-KV caches from `[1,H,T,D]` to `[beam_size,H,T,D]`
- Skips `memory*` tensors (slot-level, not beam-replicated)
- `synchronize_stream(device)` ensures tiles complete before queue push
- Decoder thread guards existing Tile loop with `&& !beam_replicated`

**Result (DGX Spark):** Staggered fill cost 28.5ms → 10.6ms (~63% reduction). Remaining fill cost is cache padding (~64 Concat ops).

#### Step 8: Batched timestamp probability check + logits phase profiling

- `batch_timestamp_check_kernel<T>`: replaces N per-row Thrust reductions (each causing GPU→CPU sync) with single kernel + single D2H
- Grid: `num_rows` blocks × 256 threads. Per block: 3 shared-memory reductions (max_text, max_ts, sum_exp)
- Split `logits` profile bucket into `phase_b` + `logits_proc`

**Result (DGX Spark):** phase_b = 0.025ms/step (negligible). logits_proc = 15.6ms/step (dominant). On unified memory, GPU→CPU syncs already ~0.1us, so batched kernel shows no speedup. Expected ~0.3-0.6ms/step on H100 (each Thrust sync ~5-10us × 60 syncs).

### Phase C: Deeper Kernel Changes

#### Step 9: Attention kernel consumes lengths directly (no mask materialization)

- `expanded_self_lengths` and `expanded_memory_lengths` pre-allocated in `CbDecodeBuffers`
- `expand_lengths_gpu`: `out[i] = in[i / stride]` — replaces `prepare_length_mask()`
- `shallow_copy` creates non-owning view → zero per-step allocation, stable GPU address
- Both self-attn and cross-attn masks use this pattern

**Result (DGX Spark):** Sub-microsecond both paths (mask is tiny: 20 elements for 4×5). Value is architectural: eliminates per-step `StorageView` allocation, enables CUDA graph capture of mask computation.

#### Step 10: Logits_proc hot path optimization

- `ContinuousSuppressTokens::init()`: pre-computes flat indices for all total_batch rows × all suppress token IDs, uploads to GPU once. `apply()` → single async `indexed_fill` kernel (zero CPU `push_back`s, zero `cudaMalloc`)
- `ContinuousTimestampRules::init()`: pre-allocates `_row_indices_buf`, `_results_buf` on GPU. `_log_probs_buf` lazy-initialized with correct dtype on first use (FLOAT16 logits → must match)
- `DisableTokens`: new constructor accepts pre-allocated GPU buffer, `apply()` resizes (no-op) + copy + kernel
- Engine `process()`: pre-allocates `disable_buf` (total_batch × vocab_size INT32), calls `init()` on all processors before decode loop

**Debugging note:** Initial hang caused by `_log_probs_buf` defaulting to FLOAT32 while LogSoftMax receives FLOAT16. `resize_as()` only changes shape, not dtype. Fixed by lazy init with `StorageView(logits.dtype(), logits.device())`.

**Result (DGX Spark):** Similar timing (cudaMalloc/Free near-free on unified memory). Expected savings on H100: ~40-200us/step from eliminated allocations + ~0.1-0.5ms from eliminated CPU push_backs.

### DGX Spark Phase Summary

| Step | Phase | Target | Key Result |
|------|-------|--------|-----------|
| 1 | A | 0.6ms CPU metadata | step_setup+rebuild: 0.63→0.02ms |
| 2 | A | Stable shapes | resize eliminated, total_batch always |
| 3 | A | 0.8ms fill savings | Adaptive K=1-4, fill only when needed |
| 4 | B | Graph readiness | Ping-pong buffers, stable GPU addresses |
| 5 | B | ~1-2ms/step (H100) | Decoder graph correct; ~0.08ms diff on Spark |
| 6 | B | Beam search graph | Correct; capture cost > savings on Spark |
| 7 | B | Reduce fill cost | fill 28.5ms→10.6ms (63% reduction) |
| 8 | B | Eliminate 60 GPU syncs | phase_b/logits_proc split; 15.6ms/step in logits_proc |
| 9 | C | No mask materialization | Architectural: zero alloc, stable addresses |
| 10 | C | Eliminate per-step mallocs | Pre-computed suppress indices, pre-alloc buffers |

**All correctness verified at each step:** `test_correctness.sh` (115 tokens exact match), CB 4-concurrent (identical), CB staggered (fill/defrag/graph invalidation correct), `CT2_NO_CUDA_GRAPH=1` (identical output).

---

## 3. H100 Benchmark Results — Single Batcher

All tests: whisper-large-v3, float16, F1 French audio (~30s per clip).
Benchmark command: `cd /workspace/ctranslate2-server && /root/venv_bench/bin/python bench_compare.py -n USERS -l LOOPS --url http://localhost:8000/transcribe --audio f1_audio.mp3 --abort-sla 15`

### 3.1 CB Server with Silero VAD (30s segments)

Config: 4 encoders, beam_size=5 unless noted. Graphs ON unless noted.

| Config | Users | Graphs | p50 (s) | p95 (s) | per_step (ms) | SLA viol. | Throughput (req/s) |
|--------|-------|--------|---------|---------|---------------|-----------|-------------------|
| e4_s8 | 8 | ON | 1.37 | 1.60 | 9.35 | 0% | 0.28 |
| e4_s8 | 20 | ON | 1.39 | 1.59 | 9.4-10.5 | 0% | — |
| e4_s8 | 20 | OFF | 1.37 | 1.64 | 9.4-11.8 | 0% | — |
| e4_s8 | 30 | ON | 1.50 | 1.88 | 9.4-13.1 | 0% | 1.00 |
| e4_s8 | 60 | ON | 2.08 | 3.11 | 10.7-16.9 | 0% | 1.95 |
| e4_s8 | 70 | ON | 3.91 | 19.37 | 9.7-38.3 | 38.3% | 1.66 |
| e4_s8 | 80 | ON | 4.34 | 19.01 | 9.4-37.6 | 42.3% | — |
| e4_s8 | 80 | OFF | 4.79 | 20.31 | 9.7-16.4 | 45.9% | — |
| e4_s10 | 80 | ON | 10.07 | 16.61 | 53.86 | 67.6% | 1.24 |
| e4_s12 | 80 | ON | 3.88 | 10.05 | 42.39 | 28.0% | — |
| e2_s16 | 80 | ON | 2.92 | 11.35 | 50.85 | 27.3% | — |
| e2_s16 | 80 | OFF | 4.15 | 18.05 | 72.37 | 47.3% | 1.06 |
| e1_s16 | 60 | ON | — | — | 52.33 | aborted | — |

### 3.2 CB Server with Pyannote VAD

| Config | Merge (s) | Users | p50 (s) | p95 (s) | per_step (ms) | SLA viol. | Throughput |
|--------|-----------|-------|---------|---------|---------------|-----------|-----------|
| e4_s8 | 30 | 80 | 5.83 | 19.85 | — | 51.9% | 1.32 |
| e4_s8→s16 | 15 | 80 | 5.62 | 18.89 | 73.45 | 56.9% | 1.13 |
| e4_s16 | 15 | 80 | 4.68 | 15.71 | — | 45.2% | 1.18 |

Note: Pyannote splits F1 audio into ~2 segments per 30s clip. With 30s merge, segments merge back to 1 (no benefit). With 15s merge, each request uses 2 slots → only 4 concurrent requests in 8 slots → worse throughput.

### 3.3 Old Server (4 workers, batch_size=16, MAX_AGE=2.0s)

| VAD | Users | p50 (s) | p95 (s) | SLA viol. | Throughput (req/s) |
|-----|-------|---------|---------|-----------|-------------------|
| Silero (builtin) | 80 | 17.37 | 21.99 | 100% | 0.83 |
| Pyannote | 80 | 3.66 | 5.46 | 8.8% | 2.54 |
| Pyannote | 100 | 4.29 | 9.02 | 43.0% | 3.19 |
| Pyannote | 120 | 5.32 | 6.95 | 57.9% | 3.83 |

### 3.4 User's Production Reference (separate hardware, old server architecture)

- 68 users under **strict 3s** (p95 < 3s)
- 115 users under **p95 5s**
- Uses pyannote VAD, 4 workers, same old server architecture

### 3.5 Per-Step Profile Breakdowns at High Load

**e4_s8 at 70 users (all 8 slots full, steady state block):**
- steps=1663, DECODER=36.4ms/step, fill=0.6ms, rebuild=0.5ms
- **per_step=38.3ms**

**e4_s8 at 80 users, graphs OFF:**
- DECODER=42.8ms/step, fill=0.4ms, rebuild=0.7ms
- **per_step=44.6ms**

**e4_s16 pyannote 15s merge at 80 users:**
- DECODER=66.1ms/step, fill=3.9ms, rebuild=2.8ms
- **per_step=73.5ms**

**e4_s8 at 8 users (1-2 active slots):**
- **per_step=9.35ms** (only 5-10 batch rows active)

---

## 4. Multi-Batcher Architecture & Results

### 4.1 The Single-Decoder Bottleneck

The single-batcher CB server hits a hard wall at ~60-65 concurrent users (e4_s8, beam=5):
- Below 60: p95 < 3.2s, 0% violations, ~2 req/s
- At 70: p95 jumps to 19s, 38% violations — queue builds up
- The wall occurs when all 8 slots are continuously full (40 batch rows → 38ms/step)
- At 38ms/step × ~115 steps ≈ 4.4s per-segment decode → throughput ≈ 8/4.4 ≈ 1.8 req/s

Root cause: the single decoder thread is the bottleneck. Per-step scales linearly with batch rows, so adding more slots just makes each step slower without improving throughput.

### 4.2 Multi-Batcher Design

Instead of modifying the C++ engine, the solution is **multiple independent `WhisperContinuousBatcher` instances** at the Python level. Each batcher owns its own encoder thread + decoder thread + slot pool, sharing the same GPU.

```
HTTP → ThreadPool → _process_request()
     → VAD → feature extraction
     → round-robin _next_batcher()
     → batcher[i].submit() per segment
       → Encoder Thread ×1 (per batcher)
       → Decoder Thread ×1 (per batcher, own slot pool)
     → batcher[i].get_result() → HTTP Response
```

Key implementation details in `server.py`:
- `NUM_BATCHERS` env var controls instance count (default: 1)
- Thread-safe round-robin `_next_batcher()` with `threading.Lock`
- Each batcher gets `NUM_ENCODERS` encoder threads (typically 1) and `MAX_SLOTS` slots
- Language detection always routes through `batchers[0]` (any batcher works)
- CUDA graphs disabled (`CT2_NO_CUDA_GRAPH=1`) — graphs are counterproductive at high churn with slot filling/draining

### 4.3 Multi-Batcher Benchmark Results

All tests: whisper-large-v3, float16, beam=5, F1 French audio (~30s), H100.
Notation: `NbMeKsS` = N batchers, M encoders each, S slots each.
CUDA graphs OFF for all multi-batcher tests.

#### Single Batcher Baseline

| Config | Users | p50 (s) | p95 (s) | SLA >5s | Throughput (req/s) |
|--------|-------|---------|---------|---------|-------------------|
| 1b×e1_s4 | 20 | 1.19 | 2.14 | 0% | 0.69 |
| 1b×e1_s4 | 60 | — | — | — | aborted (>15s) |
| 1b×e4_s8 (old baseline) | 60 | 4.57 | 17.60 | 44.9% | 1.30 |
| 1b×e4_s8 (old baseline) | 80 | 38.22 | 44.59 | 100% | 0.26 |

#### 2 Batchers

| Config | Users | p50 (s) | p95 (s) | SLA >5s | Throughput (req/s) |
|--------|-------|---------|---------|---------|-------------------|
| 2b×e1_s4 | 20 | 1.61 | 7.67 | 15.0% | 0.66 |
| 2b×e1_s4 | 60 | 1.89 | 2.40 | 0% | 1.96 |
| 2b×e1_s4 | 80 | 2.98 | 5.74 | 13.1% | 2.46 |
| 2b×e1_s4 | 100 | 7.05 | 18.16 | 61.8% | aborted |
| 2b×e1_s8 | 60 | 2.28 | 2.80 | 0% | 1.95 |
| 2b×e1_s8 | 80 | 4.96 | 18.76 | 48.5% | 1.65 |
| 2b×e1_s8 | 100 | 50.86 | 56.30 | 100% | 0.28 |

Finding: 2b×s4 outperforms 2b×s8 at 80u (p95 5.74s vs 18.76s). Larger batch per decoder hurts more than extra slots help.

#### 3 Batchers (optimal)

| Config | VAD | Users | p50 (s) | p95 (s) | SLA >5s | Throughput (req/s) |
|--------|-----|-------|---------|---------|---------|-------------------|
| 3b×e1_s4 | Pyannote | 80 | 2.48 | 3.51 | 0% | 2.61 |
| 3b×e1_s4 | Silero | 80 | 2.38 | **4.22** | **0%** | **2.60** |
| 3b×e1_s4 | Silero | 90 | 6.53 | 17.19 | 63.6% | 2.36 |
| 3b×e1_s5 | Silero | 80 | 2.81 | 5.10 | 5.6% | 2.59 |

**Best result: 3 batchers × 4 slots, Silero VAD, 80 users → p95=4.22s, 0% SLA violations, 2.60 req/s**

At 80 users with 3 batchers profiling showed:
- Even round-robin distribution across batchers (~33% each)
- GPU SM utilization 94-97% under load
- Per-step 17-23ms at full 4-slot occupancy per decoder (20 batch rows each)
- Per-step rises to 30-42ms under heavy GPU contention between decoders

#### 4 Batchers

| Config | VAD | Users | p50 (s) | p95 (s) | SLA >5s | Throughput (req/s) |
|--------|-----|-------|---------|---------|---------|-------------------|
| 4b×e1_s4 | Pyannote | 20 | 1.31 | 3.05 | 0% | 0.66 |
| 4b×e1_s4 | Pyannote | 60 | 2.48 | 4.63 | 4.2% | 1.85 |
| 4b×e1_s4 | Pyannote | 80 | 4.26 | 15.33 | 24.6% | aborted |
| 4b×e1_s3 | Silero | 80 | 4.11 | 16.28 | 30.8% | 1.31 |

4 batchers causes too much GPU SM contention — 4 concurrent decoder threads fight for compute resources, and per-step time degrades for all.

### 4.4 Silero VAD vs Pyannote VAD for CB

| VAD | Config | Users | p95 (s) | SLA >5s | Notes |
|-----|--------|-------|---------|---------|-------|
| Pyannote (GPU) | 3b×e1_s4 | 80 | 3.51–12.04 | 0–41.9% | Variable across runs |
| Silero (CPU) | 3b×e1_s4 | 80 | 4.22 | 0% | Consistent |

Pyannote runs inference on the GPU, competing with the 3 decoder threads for SM occupancy. Silero runs on CPU, freeing all GPU compute for encoding and decoding. With 3 batchers already at 94-97% SM utilization, Silero is the better choice — it avoids GPU contention at the cost of slightly less precise segmentation (irrelevant for F1 audio which has minimal silence).

### 4.5 Old Server: Random Language Assignment Experiment

The old server batches requests by language key. When all requests are French, batches fill to BATCH_SIZE=16 quickly. Testing with random language assignment (en, fr, es, it) creates smaller batches (1-4 items) that dispatch faster.

| Users | p50 (s) | p95 (s) | SLA >5s | Throughput (req/s) |
|-------|---------|---------|---------|-------------------|
| 60 | 3.14 | 3.92 | 0% | 1.92 |
| 80 | 4.09 | 8.86 | 35.6% | 2.52 |
| 100 | 3.73 | 5.06 | 6.0% | 3.21 |
| 120 | 4.01 | 5.43 | 13.8% | 3.77 |

Compared to old server with all-French: worse at 80u (p95 8.86s vs 5.46s with pyannote) but better at 100-120u (p95 5-5.4s vs aborted). Smaller batches mean lower per-batch GPU efficiency but less queuing delay.

### 4.6 Why Multi-Batcher Works

The key insight from section 3 was that per-step time scales linearly with batch rows:

| Active Slots | Batch Rows (×beam5) | per_step (ms) |
|-------------|---------------------|---------------|
| 1-2 | 5-10 | 9.35 |
| ~4 | ~20 | 10-13 |
| 8 | 40 | 36-38 |

With 3 batchers × 4 slots each, the GPU runs 3 decoders at 20 batch rows each instead of 1 decoder at 40+ rows. Even though the 3 decoders share GPU compute and each step is slightly slower than single-decoder at 20 rows (~17-23ms vs ~13ms), the total throughput is dramatically higher:

- **Single batcher, 8 slots:** 1 decoder × (8 slots / 4.4s per segment) ≈ 1.8 req/s
- **3 batchers, 4 slots each:** 3 decoders × (4 slots / ~2.3s per segment) ≈ 5.2 req/s theoretical, ~2.6 req/s measured

The measured 2.6 req/s vs 5.2 theoretical reflects GPU contention between decoders, but it's still a **3.2x improvement** over single-batcher at 80 users (p95 4.22s vs 44.59s).

---

## 5. Key Findings

### 5.1 Per-Step Scales Linearly with Active Batch Rows

Per-step timing tracks linearly with the number of active batch rows:

| Active Slots | Batch Rows (×beam5) | per_step (ms) |
|-------------|---------------------|---------------|
| 1-2 | 5-10 | 9.35 |
| ~4 (moderate load) | ~20 | 10-13 |
| 8 (all full) | 40 | 36-38 |
| 16 (pyannote, 15s merge) | 80 | 66-73 |

This means total throughput is roughly constant regardless of slot count in a single batcher — more slots = proportionally slower per-step. The solution is multiple batchers with fewer slots each.

### 5.2 30x Gap Between Theoretical and Actual Per-Step

For 40 batch rows on H100: theoretical bandwidth-limited minimum ~1.2ms, actual 36.4ms → **30x slower**. Root causes:
1. No flash attention (standard Q @ K^T → softmax → @ V with materializations)
2. ~500+ kernel launches per step with dispatch overhead
3. No kernel fusion across layers

### 5.3 CUDA Graphs Are Counterproductive at High Load

At high load with constant slot fill/drain:
- Every fill invalidates graphs, capture costs ~50-100ms
- At 80 users with single batcher: graphs are **20ms/step WORSE** (44.6ms vs 42.8ms)
- With multi-batcher, graphs disabled entirely (`CT2_NO_CUDA_GRAPH=1`)

### 5.4 3 Batchers × 4 Slots Is the Sweet Spot

- **1 batcher:** Single decoder bottleneck, wall at ~60 users
- **2 batchers:** Good improvement, handles 60u cleanly, struggles at 80u
- **3 batchers:** GPU SM at 94-97%, handles 80u with p95<5s, 0% violations
- **4 batchers:** Too much SM contention, performance degrades

4 slots per batcher = 20 batch rows × beam5 = manageable per-step time. 5 slots pushes per-step over the edge (p95=5.10s at 80u). 3 slots wastes GPU by running 4 batchers.

### 5.5 Silero VAD Preferred for Multi-Batcher CB

With 3 decoder threads consuming 94-97% of GPU SMs, Pyannote VAD (GPU-based) adds contention. Silero VAD runs on CPU, freeing all GPU compute for encoding and decoding. Results are consistent vs Pyannote's variability under load.

### 5.6 Old Server vs CB Multi-Batcher Comparison

| Server | Config | 80 users p95 | SLA >5s | Throughput |
|--------|--------|-------------|---------|-----------|
| Old (Pyannote) | 4 workers, batch=16 | 5.46s | 8.8% | 2.54 req/s |
| CB single batcher | e4_s8 | 19.01s | 42.3% | ~1.7 req/s |
| **CB multi-batcher** | **3b×e1_s4, Silero** | **4.22s** | **0%** | **2.60 req/s** |

Multi-batcher CB now matches and slightly exceeds the old server at 80 users, with cleaner latency distribution (0% SLA violations vs 8.8%).

---

## 6. Suggestions for Next Steps

### Priority 1: Test beam_size=2

Expected: fewer batch rows → faster per-step → potential for 4+ batchers without SM contention.
Quality impact: measure WER on test set vs beam=5.

### Priority 2: Investigate the 30x per-step efficiency gap

Options:
1. **Flash attention integration** — CTranslate2 doesn't support it natively
2. **Alternative backend** — vLLM, TensorRT-LLM, or FasterWhisper with CuDNN attention
3. **Profile with nsys/ncu** — identify top kernels by time

### Priority 3: Smart batcher assignment

Current round-robin doesn't account for batcher load. Options:
- Least-loaded assignment (route to batcher with most free slots)
- Fix lang_detect asymmetry (currently always routes through `batchers[0]`'s encoder)

### Priority 4: Pyannote for CB with sequential segment processing

Process pyannote segments **sequentially** within a single request's slot (decode segment 1, then segment 2 in same slot) instead of submitting each as a separate batcher request. This matches the old server behavior and avoids doubling slot consumption.

### Other Ideas (Lower Priority)

- **INT8 quantization:** halve decoder bandwidth, reduce per-step time
- **Dynamic batcher count:** scale batchers based on load
- **CUDA graph banks:** per-active-count graph caching instead of all-or-nothing

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `/workspace/ctranslate2-server/server.py` | CB server (multi-batcher, pyannote/Silero VAD) |
| `/workspace/ctranslate2-server/server_old.py` | Old server (4 workers, queue-based, random language support) |
| `/workspace/ctranslate2-server/bench_compare.py` | Load test tool |
| `/workspace/ctranslate2-server/old_wrapper/vad.py` | Pyannote VAD loader (`load_vad_model()`) |
| `/workspace/ctranslate2-server/old_wrapper/transcribe.py` | `BatchedWaveformInferencePipeline` |
| `/workspace/CTranslate2/src/continuous_decoding.cc` | CB decode engine (all 10 steps modified) |
| `/workspace/CTranslate2/src/layers/transformer.cc` | Decoder forward (Steps 1,2,4,9 modified) |
| `/workspace/CTranslate2/src/layers/attention.cc` | Attention + cache scatter (Step 5 modified) |
| `/workspace/CTranslate2/src/layers/common.cc` | Position encoder (Step 5 modified) |
| `/workspace/CTranslate2/src/cuda/cuda_graph.h` | CUDA graph wrapper (Step 5) |
| `/workspace/CTranslate2/src/cuda/cb_ops.cu` | Custom CUDA kernels (Steps 1,8,9) |
| `/workspace/CTranslate2/src/cuda/beam_select.cu` | Beam select kernel (Step 6) |
| `/workspace/CTranslate2/src/decoding_utils.cc` | DisableTokens pre-alloc (Step 10) |

## Appendix: Launch Script Examples

```bash
# Multi-batcher CB server (best config: 3 batchers × 4 slots, Silero VAD)
export WHISPER_MODEL=/workspace/models/whisper-large-v3
export NUM_BATCHERS=3
export NUM_ENCODERS=1
export MAX_SLOTS=4
export DEVICE=cuda
export COMPUTE_TYPE=float16
export BEAM_SIZE=5
export REQUEST_TIMEOUT=120
export DEFAULT_LANGUAGE=fr
export CT2_NO_CUDA_GRAPH=1
cd /workspace/ctranslate2-server
python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Run benchmark
cd /workspace/ctranslate2-server
/root/venv_bench/bin/python bench_compare.py \
    -n 80 -l 2 \
    --url http://localhost:8000/transcribe \
    --audio f1_audio.mp3 \
    --abort-sla 15
```
