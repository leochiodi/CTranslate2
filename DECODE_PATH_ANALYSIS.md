# Whisper Decode Path Analysis: Old Server vs Continuous Batching

## Purpose

This document traces every step of the decode path for both implementations so that
engineers can identify architectural differences and optimization opportunities. Each
section documents exact file locations, data structures, memory patterns, and timing
characteristics.

---

## Table of Contents

1. [High-Level Architecture Comparison](#1-high-level-architecture-comparison)
2. [Old Server: Full Request Lifecycle](#2-old-server-full-request-lifecycle)
3. [CB Server: Full Request Lifecycle](#3-cb-server-full-request-lifecycle)
4. [Side-by-Side: Decoder Step Internals](#4-side-by-side-decoder-step-internals)
5. [Side-by-Side: KV Cache Management](#5-side-by-side-kv-cache-management)
6. [Side-by-Side: Beam Search](#6-side-by-side-beam-search)
7. [Side-by-Side: Position Encoding](#7-side-by-side-position-encoding)
8. [Side-by-Side: Attention Mask](#8-side-by-side-attention-mask)
9. [Overhead Breakdown: Where the 10ms/step Gap Lives](#9-overhead-breakdown-where-the-10msstep-gap-lives)
10. [Key Data Structures Reference](#10-key-data-structures-reference)

---

## 1. High-Level Architecture Comparison

### Old Server (`server_old.py`)

```
HTTP Request
    │
    ▼
Q_RAW (maxsize=1024)
    │
    ▼
Preprocessor Process (1)
    ├── ProcessPoolExecutor(8) for decode_resample()
    ├── Accumulates by language until BATCH_SIZE(16) or MAX_AGE(1.0s)
    ▼
Q_BATCH (maxsize=256)
    │
    ▼
Worker Thread ×N (each has own WhisperModel)
    ├── pipeline.transcribe(audios=[...], batch_size=16)
    │       └── WhisperReplica::generate()
    │               ├── encode(features)
    │               ├── forward_prompt(prompt_tokens)
    │               └── BeamSearch::search() ← tight decode loop
    ▼
Q_RESULT (maxsize=1024)
    │
    ▼
Result Listener → resolve asyncio.Future → HTTP Response
```

**Key characteristics:**
- N independent WhisperModel instances (each holds full model weights)
- GIL naturally serializes GPU work → zero CUDA stream contention
- Fixed batch size (up to 16), waits up to 1.0s to fill batch
- `generate()` is a tight loop: decode → beam select → repeat (no slot management)

### CB Server (`server.py`)

```
HTTP Request
    │
    ▼
Thread pool executor → _process_request()
    ├── decode_audio() (CPU)
    ├── VAD (CPU)
    ├── detect_language() via batcher (GPU, async)
    ├── feature_extractor() per segment (CPU)
    ▼
batcher.submit(features, prompt) × N_segments
    │
    ▼
Encoder Thread ×M (shared model)
    ├── 2ms batching window
    ├── Batch encode (single GPU call)
    ├── forward_prompt() per request
    ▼
_encoded_queue
    │
    ▼
Decoder Thread ×1
    └── ContinuousDecodingEngine::process()
            ├── Fill slots from queue
            ├── Build step_offsets, cache_write_positions
            ├── decoder(step_offsets, tokens, state, &logits)  ← CB decode
            ├── Logits processing (per-slot Whisper rules)
            ├── Beam search (GPU pipelined or CPU)
            ├── Defragment finished slots
            ├── Fill new slots mid-generation
            └── Loop until all slots done
    │
    ▼
batcher.get_result(req_id) → HTTP Response
```

**Key characteristics:**
- Single shared model instance
- N encoder threads + 1 decoder thread (no CUDA contention)
- Continuous batching: slots fill/drain dynamically mid-generation
- No batch-wait delay: decode starts as soon as a slot is free
- Per-step overhead from slot management, defrag, scatter writes, mask construction

---

## 2. Old Server: Full Request Lifecycle

### 2.1 HTTP → Preprocessing

| Step | Location | What happens | Timing |
|------|----------|-------------|--------|
| HTTP receive | `server_old.py:449-456` | Read uploaded file bytes | ~10-50ms (I/O) |
| Enqueue | `server_old.py:472` | `Q_RAW.put({audio_bytes, params})` | <1ms |
| Preprocessor pickup | `server_old.py:124` | `Q_RAW.get()` in preprocessor process | 0-100ms (if idle) |
| Audio decode | `server_old.py:133` | `executor.submit(decode_resample, ...)` in ProcessPoolExecutor(8) | 100-500ms (parallel) |
| Batch accumulation | `server_old.py:153-163` | Wait for `BATCH_SIZE(16)` requests OR `MAX_AGE(1.0s)` timeout | **0-1000ms** |
| Language grouping | `server_old.py:179-196` | Group by language, form batch dict | <1ms |
| Enqueue to workers | `server_old.py:196` | `Q_BATCH.put({lang, waveforms, params})` | <1ms |

**Critical delay**: MAX_AGE=1.0s. If only 1 request arrives, it waits a full second before dispatching.

### 2.2 Worker Thread: Inference

| Step | Location | What happens | Timing |
|------|----------|-------------|--------|
| Dequeue | `server_old.py:271` | `Q_IN.get()` (worker-specific queue) | 0-∞ms |
| Timeout check | `server_old.py:283-295` | Drop requests older than REQUEST_TIMEOUT | <1ms |
| **Transcribe** | `server_old.py:299-318` | `pipeline.transcribe(audios=waveforms, batch_size=16)` | **1-5s** |
| Result enqueue | `server_old.py:357` | `Q_RESULT.put({req_id, segments, ...})` | <1ms |

### 2.3 Inside `generate()`: The Tight Decode Loop

**Entry**: `WhisperReplica::generate()` at `whisper.cc:237`

```
generate(features, prompts, options)
    │
    ├── 1. encode(features) → memory              whisper.cc:256
    │       └── WhisperEncoder::operator()         single batched GPU call
    │
    ├── 2. decoder.initial_state()                 whisper.cc:255
    │       └── Creates empty DecoderState map
    │
    ├── 3. forward_prompt(prompt_tokens, state)    whisper.cc:267-282
    │       └── Full forward pass on prompt tokens
    │       └── Fills KV caches: self_keys_L, self_values_L
    │       └── Projects cross-attn: memory_keys_L, memory_values_L
    │
    └── 4. BeamSearch::search()                    decoding.cc:424
            │
            └── FOR step = 0 .. max_length:
                    │
                    ├── A. decoder(step, ids, state, &logits)
                    │       └── TransformerDecoder::decode()      transformer.cc:669
                    │           ├── Embed tokens                  line 703
                    │           ├── Add position encoding (scalar) line 720
                    │           ├── Layer norm                    line 726
                    │           └── FOR layer = 0..31:
                    │                   ├── Self-attention          attention.cc:529
                    │                   │   ├── QKV projection     line 555
                    │                   │   ├── Split heads         line 590
                    │                   │   ├── Cache concat        line 622-627
                    │                   │   ├── Dot-product attn    line 647
                    │                   │   └── Output projection   line 685
                    │                   ├── Cross-attention         (same structure)
                    │                   └── FFN
                    │
                    ├── B. LogSoftMax(logits)                     decoding.cc:545
                    │
                    ├── C. Add cumulative beam scores              decoding.cc:549
                    │
                    ├── D. TopK selection                          decoding.cc:562
                    │       └── sampler(log_probs, topk_ids, topk_scores)
                    │
                    ├── E. Unflatten beam/word indices             decoding.cc:564
                    │
                    ├── F. Check EOS / finalize hypotheses         decoding.cc:598-670
                    │
                    └── G. Gather state (reorder KV caches)        decoding.cc:683
                            └── decoder.gather_state(state, keep)
```

**Per-step operations (standard path):**
- Embedding lookup: 1 op
- Position encoding: broadcast add (scalar offset)
- 32× { self-attn + cross-attn + FFN }
- LogSoftMax: 1 op
- TopK: 1 op
- Gather KV: reorder batch dim of all caches

**What's NOT here (compared to CB):**
- No slot management
- No defragmentation
- No cache padding / scatter writes
- No step_offsets construction
- No per-element position encoding
- No attention mask construction (skipped when step > 0)
- No resize up/down
- No fill new slots mid-generation
- No per-slot logits processors

### 2.4 Result Collection

| Step | Location | What happens |
|------|----------|-------------|
| Worker puts result | `server_old.py:357` | `Q_RESULT.put({...})` |
| Listener receives | `server_old.py:395` | `await loop.run_in_executor(None, Q_RESULT.get)` |
| Future resolved | `server_old.py:402` | `fut.set_result(data)` |
| HTTP returns | `server_old.py:483` | `await asyncio.wait_for(fut, timeout=...)` unblocks |

---

## 3. CB Server: Full Request Lifecycle

### 3.1 HTTP → Preprocessing

| Step | Location | What happens | Timing |
|------|----------|-------------|--------|
| HTTP receive | `server.py:276-288` | Read uploaded file, parse params | ~10-50ms |
| Thread dispatch | `server.py:316` | `loop.run_in_executor(None, _process_request, ...)` | <1ms |
| Audio decode | `server.py:427` | `decode_audio(file_bytes)` — soundfile + torchaudio resample | 100-500ms |
| VAD | `server.py:432` | `get_speech_timestamps()` — Silero ONNX on CPU | 50-200ms |
| Segment merge | `server.py:435-438` | `merge_vad_segments()` — combine adjacent segments ≤30s | <1ms |
| Language detect | `server.py:451` | `detect_language_via_batcher()` → encoder thread → wait | 50-100ms |
| Mel extraction | `server.py:488` | `feature_extractor(chunk)` per merged segment | 10-50ms each |
| **Submit** | `server.py:495` | `batcher.submit(features, prompt)` per segment | <5ms |

**No batch-wait delay**: each segment is submitted immediately to the batcher.

### 3.2 Encoder Thread: Encode + Forward Prompt

**File**: `whisper.cc:933-1132`

| Step | Location | What happens | Timing |
|------|----------|-------------|--------|
| Wait for work | `whisper.cc:958-960` | `_raw_queue_cv.wait()` | 0-∞ms |
| Batching window | `whisper.cc:965-971` | Wait up to 2ms for more requests | 0-2ms |
| Drain queue | `whisper.cc:974-981` | Pull all pending requests from `_raw_queue` | <1ms |
| Batch encode | `whisper.cc:1007-1028` | Concat features → single encoder call → scatter back | **50-200ms** |
| Forward prompt | `whisper.cc:1111-1115` | Per-request: `prep_decoder.forward_prompt(tokens, state)` | 5-20ms each |
| GPU sync | `whisper.cc:1122` | `synchronize_stream(device)` | 0-5ms |
| Push to decoder | `whisper.cc:1126-1130` | Lock + push to `_encoded_queue` + notify | <1ms |

**Prepared state after encoder thread** (per request):
- `memory`: encoder output `[1, enc_time, d_model]`
- `self_keys_L`, `self_values_L`: `[1, heads, prompt_len, d_head]` (32 layers)
- `memory_keys_L`, `memory_values_L`: projected encoder K/V (32 layers)
- `prompt_length`: number of tokens in KV cache

### 3.3 Decoder Thread: ContinuousDecodingEngine::process()

**File**: `continuous_decoding.cc:563-2201`

#### Phase 1: Initial Batch Setup (lines 576-918)

| Step | Location | What happens |
|------|----------|-------------|
| Drain queue | `line 592-603` | Pull up to `_max_slots` requests from `_encoded_queue` |
| Beam replicate | `line 621-636` | If beam_size>1: tile each request's state ×beam_size |
| Pad to uniform | `line 653-670` | Pad self-attn caches to max prompt_length across requests |
| Concatenate batch | `line 672-685` | `ops::Concat(0)` all requests into single batch state |
| Pad inactive slots | `line 687-706` | Zero-pad remaining rows to max_slots×beam_size |
| Init cache_lengths | `line 710-717` | `[total_batch]` INT32 CPU tensor, each = prompt_length |
| Init slot states | `line 719-743` | Per-slot: active=true, step=prompt_len, beam state |
| Alloc GPU beam state | `line 804-896` | Pinned staging + GPU tensors for pipelined beam search |
| Alloc scratch buffers | `line 748-918` | `sample_from`, `logits`, `step_offsets`, `gather_indices` |

#### Phase 2: Main Decode Loop (lines 935-2000)

Each iteration of this loop is one decode step for ALL active slots simultaneously.

```
WHILE active_count > 0:
    │
    ├── 2a. Count active slots                       line 939-945
    │       └── Scan slots[0..max_slots-1], break on first inactive
    │       └── Contiguity invariant: active slots are always packed at front
    │
    ├── 2b. Cache padding decision                   line 955-1005
    │       ├── Check concat path conditions           line 959-979
    │       │   └── ALL slots active AND uniform cache_lengths AND exact cache size
    │       │   └── Almost never true → always scatter path
    │       └── Scatter path: pad caches to 64-token aligned size
    │           └── ops::Concat to extend time dimension with zeros
    │
    ├── 2c. Resize down                              line 1009-1010
    │       └── If active < max_slots: shrink batch tensors to active_count
    │       └── resize_batch_state(): adjusts dim(0) of all tensors
    │
    ├── 2d. Build step_offsets                       line 1015-1027
    │       └── CPU loop: for each slot, for each beam:
    │           step_offsets[s*beam+b] = slot.step (+ speculative +1 if GPU pipeline)
    │       └── Store as "step_offsets_cpu" in batch_state (avoids GPU sync)
    │       └── Copy to GPU if needed
    │
    ├── 2e. Build cache_write_positions              line 1041-1045
    │       └── CPU loop: positions[i] = cache_lengths[i] for each active row
    │       └── Only for scatter path (always)
    │
    ├── 2f. Build active_sample_from                 line 1049-1070
    │       └── Tokens to feed this step (from previous beam selection or start_id)
    │       └── GPU pipeline: D2D copy from _bs_sample_from
    │       └── CPU path: build on CPU, convert to device
    │
    ├── 2g. DECODER FORWARD PASS                     line 1086-1092
    │       └── _decoder(step_offsets, tokens, batch_state, &logits)
    │       └── TransformerDecoder::decode()           transformer.cc:1002
    │           ├── Read step_offsets_cpu               line 1039
    │           ├── Embed tokens                        line 1059
    │           ├── Position encoding (per-element)     line 1079-1080
    │           │   └── common.cc:186 → cb_ops.cu:add_position_encoding_gpu
    │           ├── Layer norm                          line 1085
    │           └── FOR layer = 0..31:
    │                   ├── Set scatter positions        transformer.cc:1205
    │                   ├── Self-attention
    │                   │   ├── QKV projection
    │                   │   ├── Split heads
    │                   │   ├── Scatter cache write     attention.cc:617-620
    │                   │   │   └── cb_ops.cu:scatter_cache_step_gpu
    │                   │   ├── Build attention mask     from cache_lengths
    │                   │   ├── Dot-product attention
    │                   │   └── Output projection
    │                   ├── Cross-attention             (same as old path)
    │                   └── FFN                         (same as old path)
    │
    ├── 2h. Attention accumulation                   line 1104-1193
    │       └── Grow accumulated_attention buffer geometrically
    │       └── Write step's cross-attention weights
    │
    ├── 2i. GPU beam Phase B (sync previous step)    line 1200-1349
    │       └── cudaEventSynchronize(d2h_event)
    │       └── Process staged results from previous step
    │       └── Collect EOS hypotheses, reorder beam_tokens
    │       └── Mark finished slots
    │
    ├── 2j. Logits processing                        line 1353-1356
    │       └── ContinuousSuppressTokens: disable per-slot
    │       └── ContinuousSuppressBlank: disable at gen_step==0
    │       └── ContinuousTimestampRules: Whisper timestamp enforcement
    │
    ├── 2k. Beam search / greedy selection           line 1361-1877
    │       ├── GREEDY (beam==1): sampler → append token → check EOS
    │       ├── CPU BEAM: LogSoftMax → add scores → TopK → per-slot selection → gather state
    │       └── GPU BEAM: LogSoftMax → add scores → GPU TopK → beam_select_async
    │                     → gather KV → D2D next_tokens → async D2H staging
    │
    ├── 2l. Resize back up                           line 1884-1885
    │       └── If greedy and active < max: resize to max_slots
    │
    ├── 2m. Defragment finished slots                line 1890-1914
    │       └── Compact active slots to front (contiguity invariant)
    │       └── batch_copy_async for all tensor rows
    │       └── Update staging_map if GPU pipeline pending
    │
    ├── 2n. Fill new slots                           line 1919-1926
    │       └── For empty slot positions: pull from queue_provider
    │       └── set_batch_slot(): scatter prepared state into batch
    │       └── Init slot state, update cache_lengths
    │
    └── 2o. Rebuild sample_from                      line 1934-1957
            └── CPU path only: rebuild token buffer for next step
```

### 3.4 Result Collection

| Step | Location | What happens |
|------|----------|-------------|
| Slot finishes | `continuous_decoding.cc` | Result callback invoked with tokens + attention |
| Callback stores | `whisper.cc:1272-1285` | Convert to WhisperGenerationResult, store in `_results[req_id]` |
| Notify client | `whisper.cc:1285` | Condition variable signal |
| Python collects | `server.py:525` | `batcher.get_result(req_id)` blocks until ready |
| Alignment | `server.py:552-559` | `align_from_attention()` if word timestamps requested |
| HTTP response | `server.py:637` | Return JSON with segments |

**Key latency**: `get_result()` blocks per-segment sequentially (line 524). If segment 2 finishes before segment 1, the handler still waits on segment 1 first.

---

## 4. Side-by-Side: Decoder Step Internals

### What's identical

| Operation | Code path | Notes |
|-----------|-----------|-------|
| Token embedding | `_embeddings(ids, layer_in)` | Same lookup, same output shape |
| Self-attn Q/K/V projection | `_linear[0](*q, fused_proj)` | Same weight matrix, same GEMM |
| Self-attn split heads | `split_heads(fused_proj, ...)` | Same reshape |
| Self-attn dot-product | `dot_product_attention(Q, K, V, ...)` | Same GEMM (but CB passes mask, standard passes null) |
| Self-attn output proj | `_linear.back()(context, output)` | Same weight matrix |
| Cross-attention (entire block) | `MultiHeadAttention::operator()` | Identical — cached K/V, same compute |
| FFN | `_ff(context, output)` | Identical — same weight matrices, same compute |
| Output norm | `_output_norm(layer_in, layer_in)` | Same |
| LM head | `_proj(layer_in, *outputs)` | Same GEMM → `[batch, vocab_size]` |

### What's different

| Operation | Old (standard) | CB | Overhead |
|-----------|---------------|-----|----------|
| **Position encoding** | `pos_encoder(input, step)` — scalar broadcast add | `pos_encoder(input, offsets_cpu)` — fused GPU kernel per-element | ~0.1ms (after fusion) |
| **Self-attn mask** | **None** (null pointer, step>0) | Built from `cache_lengths` via `prepare_length_mask()` | ~0.1ms |
| **KV cache write** | `ops::Concat` — allocate new `[b,h,t+1,d]`, copy old+new | `scatter_cache_step_gpu` — write at position, no alloc | ~0.1ms (after fusion) |
| **Decoder call signature** | `decode(dim_t step, ids, state)` | `decode(StorageView& step_offsets, ids, state)` | N/A |

**Total in-layer CB overhead: < 0.5ms per step** (measured via sublayer profiling).

---

## 5. Side-by-Side: KV Cache Management

### Old Path: Concat Growth

**File**: `attention.cc:622-627`

```
Step 0: cache = K_proj              → [batch×beam, heads, 1, d_head]     NEW ALLOC
Step 1: tmp = cache                 → move semantics (no copy)
         cache = Concat(tmp, K_proj) → [batch×beam, heads, 2, d_head]    NEW ALLOC
         tmp freed
Step 2: cache = Concat(old, new)    → [batch×beam, heads, 3, d_head]    NEW ALLOC
...
Step T: cache = Concat(old, new)    → [batch×beam, heads, T+1, d_head]  NEW ALLOC
```

**Memory pattern**:
- New allocation every step (old freed immediately)
- Size grows linearly: O(T × batch × heads × d_head)
- CUDA allocator overhead per step: depends on allocator strategy (caching helps)
- Concat copies entire old cache + new row

**Per-step allocation count**: 2 per layer (K+V) × 32 layers = **64 allocations per step**

### CB Path: Pre-allocated Scatter

**File**: `attention.cc:615-620` → `cb_ops.cu`

```
Init:   cache = zeros([batch×beam, heads, chunk_size, d_head])     ONE-TIME ALLOC
Step 0: scatter(cache, K_proj, position=0)                         IN-PLACE WRITE
Step 1: scatter(cache, K_proj, position=1)                         IN-PLACE WRITE
...
Step 63: scatter(cache, K_proj, position=63)                       IN-PLACE WRITE
Step 64: cache = Concat(cache, zeros(64))                          GROW BY CHUNK
         scatter(cache, K_proj, position=64)                       IN-PLACE WRITE
```

**Memory pattern**:
- Pre-allocated in 64-token chunks
- Zero allocations per step (except at chunk boundaries)
- Scatter writes at designated position — no copy of existing data
- Cache always over-sized by up to 63 tokens (padded with zeros)

**Per-step allocation count**: **0** (except every ~64 steps: 2 × 32 = 64 allocs for chunk growth)

### Trade-offs

| Aspect | Concat (old) | Scatter (CB) |
|--------|-------------|-------------|
| Allocations/step | 64 | 0 (amortized) |
| Data copied/step | Entire cache history | Just new K/V row |
| Memory waste | None (exact size) | Up to 63 padding tokens |
| Attention reads | Exact sequence length | Padded length (but masked) |
| Implementation | Simple `ops::Concat` | Fused CUDA kernel |
| Dynamic batch | Fixed batch from start | Scatter at per-row positions |

---

## 6. Side-by-Side: Beam Search

### Old Path: CPU Tight Loop

**File**: `decoding.cc:424-707`

```
PER STEP:
    1. logits = decoder(step, tokens, state)        → GPU
    2. log_probs = LogSoftMax(logits)                → GPU
    3. log_probs += beam_scores (broadcast)          → GPU
    4. log_probs.reshape([batch, beam×vocab])        → metadata only
    5. sampler(log_probs, topk_ids, topk_scores)     → GPU TopK
    6. unflatten beam/word indices                   → CPU (small)
    7. FOR each batch element:                       → CPU loop
         check EOS, collect hypotheses
         swap finished beams with candidates
    8. gather_state(state, gather_indices)            → GPU: Gather on all KV caches
```

**Characteristics**:
- Steps 1-5: GPU compute
- Steps 6-7: CPU (minimal, N < 20 elements)
- Step 8: GPU gather (reorder batch dimension)
- **No D2H transfer for beam selection** — TopK result stays on GPU except for small gather indices
- **Implicit sync**: TopK result needed on CPU for hypothesis tracking → sync point

### CB Path: GPU-Pipelined Beam Search

**File**: `continuous_decoding.cc:1663-1877`

```
PER STEP:
    Phase C (current step):
    1. logits = decoder(step_offsets, tokens, state)     → GPU
    2. log_probs = LogSoftMax(logits)                     → GPU
    3. log_probs += gpu_beam_scores (from device)         → GPU
    4. log_probs.reshape([active_slots, beam×vocab])      → metadata
    5. TopK(log_probs, gpu_topk_ids, gpu_topk_scores)    → GPU
    6. beam_select_async(topk, state, ...)                → GPU kernel
    7. Gather KV caches with gpu_gather_indices           → GPU
    8. D2D: next_tokens → sample_from                     → GPU
    9. Async D2H: next_tokens, scores, finished, etc.     → GPU→pinned
    10. Record event, set pending_beam_sync                → GPU event

    Phase B (NEXT step, after decoder forward):
    11. cudaEventSynchronize(d2h_event)                   → CPU waits for D2H
    12. Process staged results on CPU                     → CPU
        - Collect EOS hypotheses
        - Reorder beam_tokens
        - Update beam scores, mark finished slots
```

**Characteristics**:
- Steps 1-10: all GPU, no CPU sync needed
- Step 11: CPU waits only AFTER next step's decoder forward is already running
- Steps 12: CPU processes previous step's beam results (overlapped with GPU compute)
- **beam_select_async**: custom CUDA kernel (`cb_ops.cu`) that does beam selection entirely on GPU
- **D2H transfer**: 7 async cudaMemcpy calls to pinned staging buffers per step

**Pipeline overlap**:
```
Step N:   [decoder forward][beam select + D2H launch]
Step N+1: [decoder forward          ][sync D2H][CPU beam process]
                                      ↑ overlapped with decoder
```

### CB Path: Greedy (beam_size == 1)

**File**: `continuous_decoding.cc:1361-1409`

```
PER STEP:
    1. logits = decoder(step_offsets, tokens, state)  → GPU
    2. sampler(logits, best_ids, best_probs)           → GPU
    3. FOR each active slot:                           → CPU
         token = best_ids[s]  (requires GPU→CPU sync!)
         append to generated_tokens
         check EOS, check max_length
         if done: collect result, mark inactive
```

**Note**: Greedy path still goes through full CB machinery every step (resize, defrag, fill) even though there's no beam search overhead.

### Overhead comparison

| Aspect | Old (CPU beam) | CB (GPU pipeline) | CB (Greedy) |
|--------|---------------|-------------------|-------------|
| Beam selection | CPU loop (~0.1ms) | GPU kernel (~0.2ms) | N/A |
| KV gather | GPU Gather | GPU Gather | No gather |
| D2H transfer | Implicit sync (~1ms) | Async pipeline (~0.5ms) | Implicit sync |
| CPU processing | Per-batch (~0.1ms) | Per-slot (~0.2ms) | Per-slot (~0.1ms) |
| **Total per step** | **~1.3ms** | **~1.6ms** (decoding.cc profiling: 1.6ms) | **~0.7ms** |

---

## 7. Side-by-Side: Position Encoding

### Old Path: Scalar Broadcast

**File**: `common.cc:156-179`

```cpp
void PositionEncoder::operator()(StorageView& input, dim_t index) {
    // index = step (e.g., 0, 1, 2, ...)
    const dim_t time = input.dim(1);    // = 1 for generation
    const dim_t depth = input.dim(-1);  // = d_model
    const StorageView& encodings = get_position_encoding(max_time);

    // Single broadcast add: encoding[index, :] added to all batch rows
    primitives<D>::add_batch_broadcast(
        encodings.data<T>() + index * depth,   // pointer to row `index`
        input.data<T>(),
        time * depth,       // elements per broadcast group
        input.size());      // total elements
}
```

**Cost**: Single GPU kernel, O(batch × d_model) — negligible.

### CB Path: Per-Element Fused Kernel

**File**: `common.cc:186-261` → `cb_ops.cu`

```cpp
void PositionEncoder::operator()(StorageView& input, const StorageView& offsets) {
    // offsets = [batch_size] INT32, different per slot
    // e.g., [45, 45, 45, 45, 45,  12, 12, 12, 12, 12, ...]

    // Fast path: all offsets identical → same as scalar
    if (all_same(offsets)) {
        return (*this)(input, offsets.at<int32_t>(0));  // delegate to scalar
    }

    // GPU path: fused kernel
    StorageView offsets_gpu(offsets.to(Device::CUDA));
    cuda::add_position_encoding_gpu(
        input, encodings, offsets_gpu.data<int32_t>(),
        batch_size, time, depth, elem_bytes);
}
```

**Kernel** (`cb_ops.cu`):
```cpp
// Grid: (batch_size * time) blocks, 256 threads each
// Each block: adds encodings[offsets[b] + t, :] to input[b, t, :]
// Uses vectorized __half2 adds for fp16
```

**Cost**: Single GPU kernel + 160-byte H2D for offsets — ~0.1ms.

**When fast path fires**: Only when all active slots are at the same step (rare after first few steps).

---

## 8. Side-by-Side: Attention Mask

### Old Path: No Mask for Step > 0

**File**: `transformer.cc:765-776`

```cpp
// Standard iterative decoding: step > 0, ids.rank() == 1
// → input_lengths_mask is NOT set (remains null)
// → MultiHeadAttention receives null mask
// → dot_product_attention skips mask application in SoftMax
```

The self-attention computes `Q @ K^T` over the full cache length. Since the cache
grows exactly by 1 each step (via concat), and all positions are valid, no mask is
needed — the SoftMax naturally attends to all cached positions.

### CB Path: Always Build Mask from cache_lengths

**File**: `transformer.cc:1080-1085` (estimated from CB decode path)

```cpp
// CB path: always builds mask because different slots have different cache lengths
// The padded cache has zeros beyond each slot's actual length → must mask those

if (cache_lengths_ptr && !no_self_attn_mask_marker) {
    input_lengths_mask = MultiHeadAttention::prepare_length_mask(
        cache_lengths_device,    // [active_batch] INT32 — actual lengths per row
        num_heads,
        max_time,                // padded cache time dimension
        mask_future,             // true
        multi_query);
}
```

**Why it's needed**: CB pads caches to 64-token chunks. A slot at step 15 has cache
shape `[beam, heads, 64, d_head]` — positions 15-63 are zero-padded. Without the mask,
SoftMax would attend to those zeros, corrupting output.

**Cost**: `prepare_length_mask()` is a small CPU+GPU operation, ~0.05ms. But it means
the SoftMax kernel takes a different code path (masked vs unmasked) — potential minor
performance difference.

---

## 9. Overhead Breakdown: Where the 10ms/step Gap Lives

Measured on L4 at e2_s4 (4 slots × 5 beams = 20 batch rows), 20 concurrent users.

| Component | CB per-step | Old per-step | Delta | Source |
|-----------|-------------|-------------|-------|--------|
| **In-layer compute (32 layers)** | 19.96ms | ~19.5ms (est.) | **~0.5ms** | Sublayer profiling |
| **LM head projection** | ~0.5-1.0ms | ~0.5-1.0ms | ~0ms | Same GEMM both paths |
| **Embedding + position enc** | ~0.3ms | ~0.2ms | ~0.1ms | Fused kernel vs broadcast |
| **Step setup** (step_offsets, cache_write_pos, sample_from) | ~0.6ms | 0ms | **~0.6ms** | CB-only, profiled |
| **Beam selection** | ~1.6ms | ~1.3ms | **~0.3ms** | GPU pipeline vs CPU |
| **Logits processing** (per-slot Whisper rules) | ~0.7ms | ~0.3ms | **~0.4ms** | Per-slot vs per-batch |
| **Fill new slots** | ~0.8ms | 0ms | **~0.8ms** | CB-only, profiled |
| **Attention accumulation** | ~0.3ms | 0ms | **~0.3ms** | CB-only (for alignment) |
| **Defrag (amortized)** | ~0.1ms | 0ms | **~0.1ms** | CB-only, profiled |
| **Resize up/down** | ~0.1ms | 0ms | **~0.1ms** | CB-only, profiled |
| **Kernel launch overhead** | ~2-3ms | ~1-2ms | **~1-2ms** | More kernels in CB |
| **Cache padding (amortized)** | ~0.1ms | 0ms | **~0.1ms** | CB-only, every ~64 steps |
| **TOTAL** | ~29.6ms | ~23-25ms (est.) | **~5-7ms** | |

**Note**: The "old per-step" is estimated for the same batch size on L4. On H100, the gap
is larger because compute is faster but overhead stays constant.

### The overhead that matters most

1. **Kernel launch overhead (~1-2ms extra)**: CB path launches more CUDA kernels per step
   due to scatter writes, per-element position encoding, mask construction, defrag copies.
   Each kernel launch costs ~5-10μs, and there are hundreds more per step in CB.

2. **Fill new slots (~0.8ms)**: When a slot finishes mid-generation, the engine fills it
   with a new request. This involves `set_batch_slot()` which does `batch_copy_async` to
   scatter prepared state into the batch tensor — a significant per-step cost when slots
   turn over frequently.

3. **Step setup (~0.6ms)**: Building step_offsets, cache_write_positions, and sample_from
   on CPU every single step. These are simple loops but involve CPU→GPU transfers.

4. **Logits processing (~0.4ms extra)**: Whisper-specific per-slot token rules (timestamp
   enforcement, blank suppression) require iterating over each slot's history every step.

5. **In-layer CB overhead (~0.5ms)**: The masked SoftMax, scatter cache writes, and per-element
   position encoding — all minimized by the fused CUDA kernels, but still nonzero.

---

## 10. Key Data Structures Reference

### DecoderState (both paths)

A `std::unordered_map<std::string, StorageView>` containing:

| Key | Shape | Notes |
|-----|-------|-------|
| `memory` | `[batch, enc_time, d_model]` | Encoder output (cross-attn K/V source) |
| `self_keys_L` | `[batch×beam, heads, time, d_head]` | Self-attn K cache, layer L |
| `self_values_L` | `[batch×beam, heads, time, d_head]` | Self-attn V cache, layer L |
| `memory_keys_L` | `[batch, heads, enc_time, d_head]` | Cross-attn K (projected once) |
| `memory_values_L` | `[batch, heads, enc_time, d_head]` | Cross-attn V (projected once) |

**CB-only additions:**
| Key | Shape | Notes |
|-----|-------|-------|
| `cache_lengths` | `[total_batch]` INT32 CPU | Actual filled length per row |
| `step_offsets_cpu` | `[active_batch]` INT32 CPU | Position offsets per row (transient) |
| `cache_write_positions` | `[active_batch]` INT32 | Where to scatter new K/V |
| `accumulated_attention` | `[total_batch, heads, steps, enc_time]` | Cross-attn weights (for alignment) |
| `_retain_memory` | flag | Prevent encoder output erasure |

### SlotState (CB only)

```cpp
struct SlotState {                              // continuous_decoding.h:35-67
    size_t request_id;
    dim_t step;                                 // absolute position (prompt + generated)
    dim_t prompt_length;
    bool active;
    std::vector<size_t> generated_tokens;       // greedy path
    std::vector<std::vector<size_t>> beam_tokens;  // [beam_size][tokens]
    std::vector<float> beam_scores;             // [beam_size] cumulative log-probs
    std::vector<bool> beam_finished;            // [beam_size]
    std::vector<Hypothesis> finished_hypotheses;
    dim_t attention_step_offset;                // for slicing accumulated_attention
};
```

### GPU Beam Pipeline State (CB only)

Pinned staging buffers (CPU, for async D2H):
- `staging_next_tokens[total_batch]`
- `staging_gather_indices[total_batch]`
- `staging_beam_scores[total_batch]`
- `staging_beam_finished[total_batch]`
- `staging_slot_finished[max_slots]`
- `staging_num_finished[max_slots]`
- `staging_eos_beam_ids[total_batch]`
- `staging_eos_scores[total_batch]`
- `staging_num_eos[max_slots]`
- `staging_needs_gather[max_slots]`

GPU-resident beam state:
- `_bs_sample_from[total_batch]` — tokens to feed next step
- `_bs_beam_scores[total_batch]` — cumulative scores
- `_bs_beam_finished[total_batch]` — finished flags
- `_bs_num_finished[max_slots]` — count per slot
- `_bs_steps[max_slots]` — current step per slot
- `_bs_prompt_lengths[max_slots]` — prompt length per slot

---

## Summary: Why CB Has Per-Step Overhead

The standard `generate()` path is a **tight loop** that does exactly:
1. Embed + position encode (scalar)
2. 32 layers of { self-attn (no mask, concat cache) + cross-attn + FFN }
3. LM head
4. Beam selection
5. Gather KV caches

The CB path does all of the above **plus**:
1. Count active slots, build indices
2. Check cache padding, potentially grow caches
3. Resize batch state down to active count
4. Build step_offsets on CPU, transfer to GPU
5. Build cache_write_positions on CPU
6. Build sample_from (D2D or CPU→GPU)
7. Position encode with per-element kernel
8. Scatter cache writes (per-layer fused kernel)
9. Build attention mask from cache_lengths
10. Accumulate cross-attention weights
11. Per-slot logits processing (Whisper timestamp rules)
12. GPU beam selection kernel + async D2H
13. Sync previous D2H + CPU beam processing
14. Resize batch state back up
15. Defragment finished slots (batch_copy_async)
16. Fill new slots from queue (set_batch_slot)
17. Rebuild sample_from

Operations 1-6 and 10-17 are the ~5-10ms per-step overhead. They exist because CB
must manage a **dynamic batch** where slots arrive and depart at different times, have
different cache lengths, and need independent beam search state. The standard path
avoids all of this because the batch is fixed from start to finish.
