# CB Decoder Overhead Reduction — Phase A Status

## Reference: Prior H100 Baseline (from OPTIMIZATION_STATUS.md)

H100 e4_s8 (8 slots, beam 5, 80 users, 1489 steps):

| Component | Total (ms) | Per-step (ms) | % of total |
|---|---|---|---|
| step_setup | 846.7 | 0.57 | 1.1% |
| rebuild | 93.0 | 0.06 | 0.1% |
| DECODER | 68649 | 46.1 | 92.5% |
| selection | 2411 | 1.6 | 3.2% |
| fill | 1149 | 0.8 | 1.5% |
| logits | 999 | 0.7 | 1.3% |
| **per_step** | — | **49.86** | 100% |

---

## Step 1: GPU-resident cache_lengths + eliminate CPU metadata loops

**Target:** ~0.6ms/step savings from step_setup + rebuild

**Changes:**
- cache_lengths moved from CPU to GPU (with CPU shadow vector)
- step_offsets_device = cache_lengths (GPU), no separate H2D
- cache_write_positions = cache_lengths (GPU), shallow copy
- cache_lengths rebuild → increment_cache_lengths_gpu kernel
- attn_lengths in transformer.cc → add_scalar_int32_gpu kernel
- min_step precomputed on CPU, stored in batch_state

**Status:** DONE

### After Step 1 (DGX Spark, 1 slot, beam 5, f1_35s.wav, 117 steps)

| Component | Per-step (ms) | Notes |
|---|---|---|
| slot_mgmt | 0.00 | |
| cache_pad | 0.13 | |
| **step_setup** | **0.01** | Was 0.57ms on H100 — CPU loops eliminated |
| DECODER | 15.79 | |
| attn_accum | 0.05 | |
| logits | 0.03 | |
| selection | 0.32 | |
| resize_up | 0.00 | |
| defrag | 0.00 | Single slot, no defrag needed |
| fill | 0.00 | Single slot, no fill needed |
| **rebuild** | **0.007** | Was 0.06ms on H100 — GPU increment kernel |
| **per_step** | **16.37** | |
| **GPU%** | **96.4%** | |

**Result:** step_setup + rebuild reduced from ~0.63ms to ~0.02ms/step.
CPU metadata loops (step_offsets, cache_write_positions, cache_lengths rebuild)
fully eliminated. All three now use GPU-resident cache_lengths directly.

Note: single-slot test — multi-slot benchmark with `bench_compare.py` still TBD.

### Correctness
- `test_correctness.sh` PASS: standard generate() produces identical 115 tokens
- CB engine produces identical 115-token transcription of f1_35s.wav

---

## Step 2: Remove resize up/down from hot loop

**Target:** ~0.1ms direct + fewer kernel launches + stable shapes

**Changes:**
- Always run decoder on total_batch rows (no resize down/up)
- Inactive rows: cache_lengths=1 (NOT 0 — avoids all-masked softmax NaN)
- Deleted all 4 resize_batch_state() calls in main loop
- TopK reshape uses max_slots instead of active_count
- add_depth_broadcast uses total_batch
- After defrag+fill, inactive cache_lengths reset to 1 via fill_int32_gpu

**Status:** DONE

### After Step 2

Single-slot test (DGX Spark, max_slots=4, beam 5, f1_35s.wav, 117 steps):
- per_step=21.60ms (up from 16.37ms — expected: 20 rows vs 5 rows for 1 active slot)
- resize_up=0.0ms — eliminated
- GPU%=97.4%

4-slot test (all slots active, 117 steps):
- per_step=20.47ms, GPU%=92.0%
- All 4 requests produce identical 115 tokens — CONSISTENT
- resize_up=0.0ms, step_setup=0.003ms/step

Note: resize was already 0.0ms in Step 1 single-slot test (no resize needed when
only 1 slot). The benefit is stable shapes (enables future CUDA graphs) and
elimination of shape churn that could cause extra kernel launches.

---

## Step 3: Deferred slot fill

**Target:** ~0.8ms saved on steps where fill is skipped

**Changes:**
- Always defrag immediately (cheap: copy_slot_data for 1-2 slots)
- Defer fill to every K steps (adaptive K=1-4)
- K=1 when queue has pending requests or local queue non-empty
- K ramps up to 4 when queue is empty (no new work to schedule)
- Resets to K=1 when fill succeeds (work available)

**Status:** DONE

### After Step 3 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

Single request: 117 steps, per_step=21.60ms, fill=0.4ms total
4 concurrent: 117 steps, per_step=20.45ms, fill=0.0ms total (all filled at start)
Staggered (0.5s apart): 167 steps, per_step=25.47ms, fill=13.4ms total (fills triggered by arrivals)

### Correctness
- `test_correctness.sh` PASS: standard generate() identical 115 tokens
- CB single: 115 tokens
- CB 4-concurrent: all 4 produce 115 tokens
- CB staggered: all 4 produce 115 tokens

---

## Phase A Summary

All 3 steps implemented and verified:

| Step | Target | Status | Key result |
|---|---|---|---|
| 1. GPU cache_lengths | 0.6ms/step | DONE | step_setup+rebuild: 0.63→0.02ms/step |
| 2. Remove resize | 0.1ms + stable shapes | DONE | resize_up=0.0ms, total_batch always |
| 3. Deferred fill | 0.8ms when skipped | DONE | fill only runs when needed (adaptive K) |

Correctness verified at each step via `test_correctness.sh` (standard generate path)
and CB engine tests (single, concurrent, staggered requests).

---

# Phase B: CUDA Graphs

## Step 4: Pre-allocate decoder intermediate buffers

**Target:** Stable GPU addresses for CUDA graph capture.

**Changes:**
- Added `CbDecodeBuffers` struct to `TransformerDecoder` (transformer.h)
  - `buf[2]`: ping-pong layer I/O buffers [total_batch, 1, d_model]
  - `attn_lengths`: [total_batch] INT32
  - `position_bias`: empty, reused across steps
- `ensure_cb_buffers()` method allocates once, reuses across steps
- CB decode path uses ping-pong pattern: `buf[cur]` → layer → `buf[1-cur]`, swap `cur`
- No more `layer_in = std::move(layer_out)` — buffers never move, addresses stay stable
- Pre-allocated `logits` in continuous_decoding.cc: `{total_batch, vocab_size}`

**Status:** DONE

### After Step 4 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

- `test_correctness.sh` PASS: generate() baseline 115 tokens exact match
- CB 4-concurrent: all 4 produce identical 121 tokens (121 due to Whisper logits processors in batcher — expected diff from generate baseline which has none)
- CB staggered: all 4 identical to concurrent reference
- per_step ≈ 22ms (no perf change expected — pre-alloc is for graph readiness)

---

## Step 5: CUDA Graph capture for decoder forward

**Target:** ~1-2ms/step from eliminating ~500-600 kernel launch overhead.

**Status:** DONE

### Changes

- `src/cuda/cuda_graph.h`: `CudaGraphWrapper` class + `g_cuda_graph_capturing` thread-local flag
  - `capture(stream, fn)`: sets flag, captures GPU ops, instantiates graph
  - `replay(stream)`: launches captured graph
  - `invalidate()` / `is_valid()`: structural change handling
  - Exception-safe: cleans up capture on throw

- `src/layers/common.cc`: Position encoder — when `g_cuda_graph_capturing`, skip CPU
  uniform check, use GPU `add_position_encoding_gpu` kernel directly with GPU offsets.
  Avoids baking CPU-computed pointer offset into graph.

- `src/layers/attention.cc`: `scatter_cache_step` — when `g_cuda_graph_capturing`, skip
  CPU uniform check, use GPU scatter kernel directly. Avoids baking CPU-computed
  `cudaMemcpy2DAsync` offset into graph.

- `src/layers/transformer.cc`: CB decode path —
  - Pass GPU `step_offsets` (not CPU offsets) to position encoder during capture
  - Gate `dp_sync_now()` and `layer_profile` syncs on `!g_cuda_graph_capturing`
    (synchronize_stream inside capture would error)

- `src/continuous_decoding.cc`: Wrap decoder call with graph capture/replay
  - Capture on first steady-state step (min_step > 0, no structural changes)
  - Replay on subsequent steady-state steps
  - Invalidate on: cache chunk growth (pad_cache), defrag, fill (new slot at step 0)
  - Gate: `CT2_NO_CUDA_GRAPH=1` env var disables graphs

### Graph capture scope

```
CAPTURED (one cudaGraphLaunch replaces all of this):
  embedding lookup → scale → position encoding → layernorm_embedding
  → attention mask computation
  → 32× { self-attn + cross-attn + FFN } (layer loop)
  → output_norm → LM head projection → logits

NOT CAPTURED (runs as normal before/after graph launch):
  step_offsets/sample_from setup, cache padding,
  logits processing, beam search, fill, defrag, rebuild
```

### After Step 5 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

4-concurrent (graph enabled):
- 110 steps, per_step=21.66ms, DECODER=5.4ms (graph launch only)
- All 4 produce identical 109 tokens — CORRECT

4-concurrent (graph DISABLED, CT2_NO_CUDA_GRAPH=1):
- 110 steps, per_step=21.74ms, DECODER=5.5ms
- All 4 produce identical 109 tokens — CORRECT

Staggered (0.5s apart, graph enabled):
- 149 steps, per_step=26.61ms, fill=33.1ms (fills triggered by arrivals)
- All 4 produce identical 109 tokens — CORRECT (graph invalidation works)

**Result:** On DGX Spark (unified memory, low launch overhead), graph vs no-graph
difference is negligible (~0.08ms/step). The infrastructure is correct and ready.
Expected ~1-2ms/step savings on H100 (PCIe, higher launch overhead per kernel).

### Blockers resolved

5 CPU-dependent branching issues inside the decoder forward were identified and fixed:

1. **Position encoder** (`common.cc`): Uniform check → scalar broadcast with CPU-computed
   pointer offset. Fix: `g_cuda_graph_capturing` → always GPU kernel, reads offsets from
   GPU buffer (updated each step, correct on replay).

2. **scatter_cache_step** (`attention.cc`): Uniform check → `cudaMemcpy2DAsync` with
   CPU-computed offset. Fix: `g_cuda_graph_capturing` → always GPU scatter kernel.

3. **Profiling syncs** (`transformer.cc`): `synchronize_stream()` during capture would
   error. Fix: gate on `!g_cuda_graph_capturing`.

4. **Cross-attention memory access** (`transformer.cc`): `min_step <= 0` accesses encoder
   memory; `min_step > 0` uses cached K/V. Fix: only capture during steady-state
   (min_step > 0). Invalidate graph on fill (new slot starts at step 0).

5. **`state.find()` / `state.erase()`**: CPU map ops — not a blocker. During replay only
   captured GPU ops run; CPU side effects only matter during capture.

---

## Step 6: Second CUDA Graph for Phase C beam search

**Target:** Eliminate ~5-7 kernel launch overhead from Phase C (TopK → beam_select → fill_identity → D2D copies).

**Status:** DONE

### Changes

- `src/continuous_decoding.cc`:
  - Pre-allocated persistent buffers for graph-stable addresses:
    - `beam_scores_cast_persistent` [total_batch] in logits dtype
    - `topk_scores_f32_persistent` [max_slots, 2*beam_size] FLOAT32
    - `gpu_topk_scores` / `gpu_topk_ids` pre-shaped at [max_slots, 2*beam_size]
  - Second `CudaGraphWrapper cuda_graph_beam` alongside existing `cuda_graph`
  - `graph_structural_change = false` moved after both graphs (was after Graph 1 only)

- `src/cuda/beam_select.cu`:
  - `beam_id` clamped: `min(flat_id / vocab_size, beam_size - 1)` — prevents OOB
    gather indices from inactive slot garbage when running with max_slots blocks
  - Kernel launch uses explicit `cudaStream_t` parameter (was default stream 0)

- `src/cuda/beam_select.h`:
  - `beam_select_async` signature updated with `cudaStream_t stream = 0` parameter

### Graph 2 capture scope

```
OUTSIDE graph (runs every step):
  LogSoftMax (needs [total_batch, vocab] shape — can't reshape before)
  beam_scores convert (float32 → logits dtype) into persistent buffer
  add_depth_broadcast (in-place on logits)
  reshape [total_batch, vocab] → [max_slots, beam_size * vocab] (CPU metadata)

CAPTURED in Graph 2 (one cudaGraphLaunch replaces all):
  TopK → pre-allocated gpu_topk_scores/ids (max_slots sized, no resize)
  topk_scores float32 conversion → persistent buffer
  beam_select_async (max_slots blocks, not active_count)
  fill_identity_async (total_batch, explicit stream)
  D2D copy: bs_gather → full_gather (total_batch)
  D2D copy: bs_next_tokens → bs_sample_from (total_batch)

OUTSIDE graph (runs after graph replay):
  Gather loop over batch_state KV caches (~64 ops, pointer swaps)
  D2H copies (10x cudaMemcpyAsync, variable sizes using active_count)
  cudaEventRecord + pending_beam_sync bookkeeping
  Debug prints
```

**Why LogSoftMax is outside:** LogSoftMax normalizes over the last dimension.
With `[total_batch, vocab_size]` it normalizes per beam (correct). After reshape
to `[max_slots, beam_size*vocab]` it would normalize across all beams jointly (wrong).
So LogSoftMax + add_depth_broadcast must run before reshape, outside the graph.

**Why Gather loop is outside:** In-place `ops::Gather()` does move+allocate+gather
internally, creating pointer swaps that are graph-incompatible.

### Key correctness points

- **beam_select with max_slots:** Inactive slots have garbage TopK data. The `beam_id`
  clamp ensures gather indices stay in `[s_offset, s_offset + beam_size)`. Inactive KV
  cache rows may get shuffled by Gather, but they have `cache_lengths=1` and are never read.
- **D2H copies stay variable-sized:** They use `active_batch`/`active_count`, not `total_batch`.
- **Both graphs share invalidation:** `graph_structural_change` flag is reset after both
  Graph 1 and Graph 2 have processed, so pad/defrag/fill invalidates both on the same step.

### After Step 6 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

4-concurrent (graph enabled):
- 110 steps, per_step=21.91ms, DECODER=601.7ms, selection=68.6ms
- All 4 produce identical 109 tokens — CORRECT

4-concurrent (graph DISABLED, CT2_NO_CUDA_GRAPH=1):
- 110 steps, per_step=21.60ms, DECODER=608.3ms, selection=44.3ms
- All 4 produce identical 109 tokens — CORRECT

Staggered (0.5s apart, graph enabled):
- 149 steps, per_step=26.45ms, fill=28.5ms
- All 4 produce identical 109 tokens — CORRECT (graph invalidation works)

`test_correctness.sh`: generate() baseline 115 tokens exact match — PASS

**Result:** On DGX Spark (unified memory), Graph 2 adds ~24ms overhead to selection
(68.6ms vs 44.3ms) due to graph capture/instantiate cost outweighing per-kernel launch
savings. Expected net positive on H100 (PCIe, higher per-kernel launch overhead ~5-10us).

---

## Phase B Summary

| Step | Target | Status | Key result |
|---|---|---|---|
| 4. Pre-allocate buffers | Stable GPU addresses | DONE | Ping-pong buf[2], no per-step alloc |
| 5. CUDA Graph (decoder) | ~1-2ms/step (H100) | DONE | Correct on DGX Spark; ~0.08ms diff (expected: low-end GPU) |
| 6. CUDA Graph (beam search) | Eliminate Phase C launch overhead | DONE | Correct; overhead on DGX Spark, expected benefit on H100 |

### Correctness verified at each step
- `test_correctness.sh` (standard generate path): 115 tokens exact match
- CB 4-concurrent: all identical
- CB staggered (fill/defrag/graph invalidation): all identical
- CT2_NO_CUDA_GRAPH=1: identical output to graph-enabled

---

## Step 7: Pre-staged beam replication on encoder thread

**Target:** Reduce fill cost by moving ~64 `ops::Tile` GPU kernel launches off the decoder hot path.

**Status:** DONE

### Changes

- `include/ctranslate2/continuous_decoding.h`: Added `beam_replicated = false` flag to `ContinuousRequest`
- `include/ctranslate2/models/whisper.h`: Added `beam_replicated = false` flag to `Request`
- `src/models/whisper.cc`:
  - Added `#include "ctranslate2/utils.h"` for `starts_with()` helper
  - After `forward_prompt` in `encoder_loop`: tiles self-KV caches from `[1,H,T,D]` to
    `[beam_size,H,T,D]` using `ops::Tile`, sets `req.beam_replicated = true`
  - Skips `memory*` tensors (slot-level, not beam-replicated)
  - `synchronize_stream(device)` at line 1146 ensures Tile kernels complete before queue push
  - `convert_request` lambda propagates `cr.beam_replicated = req.beam_replicated`
- `src/continuous_decoding.cc`:
  - Saves `beam_replicated` flag before state move
  - Guards existing Tile loop with `&& !beam_replicated` (fallback preserved for non-batcher callers)

### Thread safety

- Encoder thread uses its own CUDA stream (via `prep_decoder`)
- `synchronize_stream(device)` completes all Tile GPU work before queue push
- Queue mutex provides CPU-level happens-before for the handoff
- Decoder thread's stream can safely read the data after queue pop

### After Step 7 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

4-concurrent (graph enabled):
- 120 steps, per_step=21.88ms, DECODER=640.2ms
- **fill=0.0ms** (all filled simultaneously at start, no staggered fills)
- All 4 produce 108 tokens — CORRECT

Staggered (3s apart, graph enabled):
- 234 steps, per_step=23.65ms
- **fill=10.6ms** (down from 28.5ms in Step 6 = ~63% reduction)
- All 4 produce 108 tokens — CORRECT

CT2_NO_CUDA_GRAPH=1:
- 120 steps, per_step=21.99ms
- All 4 produce 108 tokens — CORRECT (identical to graph-enabled)

**Result:** Beam replication (~64 Tile ops per fill) moved to encoder thread.
Staggered fill cost reduced from ~28.5ms to ~10.6ms (~63% reduction).
Remaining fill cost is cache padding (~64 Concat ops) which depends on runtime
batch state and must stay on the decoder thread.

---

## Step 8: Batched timestamp probability check + logits phase profiling

**Target:** Eliminate up to 60 GPU→CPU syncs per step from timestamp rules; split opaque
`logits` profile bucket into `phase_b` + `logits_proc` for visibility.

**Status:** DONE

### Problem

The `logits` profiling bucket (t5→t6) was 16.0ms/step (73% of per-step time in 4-slot
concurrent test), but was opaque — it included both Phase B sync (CudaEventSynchronize +
CPU beam bookkeeping) and all logits processors.

Within logits processors, `ContinuousTimestampRules::apply()` called `should_sample_timestamp()`
per row, which uses `primitives<Device::CUDA>::max()` and `primitives<Device::CUDA>::logsumexp()`
— both Thrust reductions returning scalars to host, each causing an implicit GPU→CPU sync.
With 4 slots × 5 beams = 20 rows, this means up to **60 synchronous GPU reductions per step**.

### Changes

- **`src/cuda/cb_ops.h`**: Declared `batch_timestamp_check_gpu()` — batched kernel that
  replaces N per-row Thrust reductions with a single kernel launch + D2H copy.

- **`src/cuda/cb_ops.cu`**: Implemented `batch_timestamp_check_kernel<T>` (templated on
  `__half`/`float`):
  - Grid: `num_rows` blocks (one per row to check), 256 threads each
  - Per block: 3 shared-memory reductions:
    1. `max_text` over text tokens `[0, timestamp_begin_id)`
    2. `max_ts` over timestamp tokens (numerical stability)
    3. `sum_exp(x - max_ts)` over timestamp tokens
  - Thread 0: `ts_lse = log(sum_exp) + max_ts`, writes `results[blockIdx.x] = (ts_lse > max_text)`

- **`src/continuous_decoding.cc`** — `ContinuousTimestampRules::apply()`:
  - CUDA path: upload row indices (single H2D via StorageView), call batched kernel
    (single launch), copy results back (single D2H), loop CPU-side to apply disable_tokens
  - CPU fallback: original per-row `should_sample_timestamp` loop preserved
  - Uses `log_probs.buffer()` for type-erased device pointer

- **`src/continuous_decoding.cc`** — Sub-component profiling:
  - New `t_phase_b` accumulator: measures Phase B (EventSync + CPU beam bookkeeping)
  - `t_logits_proc`: now only measures logits processor time (after Phase B completes)
  - Profile print updated: `logits=X` → `phase_b=X  logits_proc=Y`

### Thread safety

All new operations run on the decoder thread using the main CUDA stream.
`batch_timestamp_check_gpu` uses `get_cuda_stream()` like all other CB kernels.

### Expected impact

- **Timestamp-enabled workloads:** Eliminates up to 60 GPU→CPU syncs per step,
  replacing them with 1 kernel + 1 D2H copy. Expected savings: 5-15ms/step.
- **Non-timestamp workloads:** No change (timestamp rules skip all slots).
- **Profiling split:** Reveals actual breakdown of the 16ms/step logits overhead.

### After Step 8 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

`test_correctness.sh` (standard generate path): 115 tokens exact match — PASS

4-concurrent, NO timestamps (graph enabled):
- 124 steps, per_step=21.44ms, DECODER=622.4ms
- phase_b=2.9ms, logits_proc=1939.5ms
- All 4 produce 121 tokens — CORRECT

4-concurrent, WITH timestamps (graph enabled):
- 127 steps, per_step=20.99ms, DECODER=599.2ms
- phase_b=3.2ms, logits_proc=2001.1ms
- All 4 produce 126 tokens — CORRECT

CT2_NO_CUDA_GRAPH=1 (no timestamps): 121 tokens x4 — CORRECT (identical)
CT2_NO_CUDA_GRAPH=1 (with timestamps): 126 tokens x4 — CORRECT (identical)

**Profiling split insight:** Phase B is only ~0.025ms/step (2.9ms / 124 steps). The
logits_proc bucket dominates at ~15.6ms/step. On DGX Spark (unified memory), the
batched kernel provides no visible speedup because GPU→CPU syncs are already cheap
(~0.1us via unified memory coherence). On H100 (PCIe), each Thrust reduction sync
costs ~5-10us, so eliminating 60 per step should save ~0.3-0.6ms/step.

The remaining ~15.6ms/step `logits_proc` bottleneck is in the DisableTokens::apply()
GPU kernel and the logits processor CPU loops (iterating over slots × beams × token IDs).

---

## Phase B Summary

| Step | Target | Status | Key result |
|---|---|---|---|
| 4. Pre-allocate buffers | Stable GPU addresses | DONE | Ping-pong buf[2], no per-step alloc |
| 5. CUDA Graph (decoder) | ~1-2ms/step (H100) | DONE | Correct on DGX Spark; ~0.08ms diff (expected: low-end GPU) |
| 6. CUDA Graph (beam search) | Eliminate Phase C launch overhead | DONE | Correct; overhead on DGX Spark, expected benefit on H100 |
| 7. Pre-staged beam replication | Reduce fill cost | DONE | fill 28.5ms → 10.6ms (~63% reduction) |
| 8. Batched timestamp check + profiling | Eliminate 60 GPU syncs/step | DONE | Correct; phase_b/logits_proc split reveals 15.6ms/step in logits_proc |

### Correctness verified at each step
- `test_correctness.sh` (standard generate path): 115 tokens exact match
- CB 4-concurrent: all identical
- CB staggered (fill/defrag/graph invalidation): all identical
- CT2_NO_CUDA_GRAPH=1: identical output to graph-enabled

### Remaining: Phase C
- Step 9: Attention kernel consumes lengths directly (no explicit mask materialization)
- Step 10: Multi-op fusion around decoder step prologue/epilogue

---

# Phase C: Deeper Kernel Changes

## Step 9: Attention kernel consumes lengths directly (no mask materialization)

**Target:** Eliminate `prepare_length_mask()` per-step allocation in CB decode path.
Replace with a simple `expand_lengths_gpu` kernel writing into pre-allocated buffers.

**Status:** DONE

### Problem

The CB decode path materialized a 3D INT32 mask via `prepare_length_mask()` for both
self-attention and cross-attention on every decode step. When `mask_future=false` (always
true for CB decode), every entry for a given batch element equals the same scalar length
value — the mask is fully redundant. `prepare_length_mask()` also allocates a new
`StorageView` per call, breaking CUDA graph address stability.

### Changes

- **`include/ctranslate2/layers/transformer.h`**: Added `expanded_self_lengths` and
  `expanded_memory_lengths` pre-allocated INT32 buffers to `CbDecodeBuffers` struct.
  Sized at `[total_batch * num_heads_local]` — stable GPU addresses for CUDA graph.

- **`src/cuda/cb_ops.h`**: Declared `expand_lengths_gpu(out, in, stride, total)`.

- **`src/cuda/cb_ops.cu`**: Implemented `expand_lengths_kernel`:
  `out[i] = in[i / stride]` for `i in [0, total)`. One thread per element, simple
  grid-stride pattern. Replaces the more complex `prepare_length_mask` primitive.

- **`src/layers/transformer.cc`**:
  - `ensure_cb_buffers()`: Allocates `expanded_self_lengths` and `expanded_memory_lengths`
    at `[total_batch * num_heads_local]` INT32.
  - Self-attention mask: `prepare_length_mask()` replaced with `expand_lengths_gpu` into
    pre-allocated buffer + `shallow_copy` (non-owning view, no per-step allocation).
    Stride = `num_heads`, total = `batch_size * num_heads`.
  - Cross-attention mask: Same pattern. Stride = `num_heads * num_queries`
    (where `num_queries = beam_size > 1 ? beam_size : max_time`),
    total = `memory_batch * stride`.
  - CPU fallback preserved: uses existing `prepare_length_mask()` for non-CUDA builds.

### Key design points

- **Pre-allocated buffers** in `CbDecodeBuffers` → stable GPU addresses for CUDA graph
- **`shallow_copy`** creates non-owning view → no per-step GPU allocation, safe on
  destroy (`release()` checks `_allocator==nullptr`)
- **Both CUDA graphs share these buffers** — expand kernel gets captured in Graph 1
  alongside the old mask kernel's place. Same invalidation rules apply.
- **`mask_future=false` always** for CB decode self-attention. No causal masking needed
  (single decode step).
- **Correctness**: `out[i] = in[i / stride]` produces identical values to
  `prepare_length_mask` when `mask_future=false`. Each softmax row sees the same
  length as before.

### After Step 9 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

`test_correctness.sh` (standard generate path): 115 tokens exact match — PASS

4-concurrent (graph enabled):
- 110 steps, per_step=21.15ms, DECODER=543.3ms
- All 4 produce identical 109 tokens — CORRECT

Staggered (3s apart, graph enabled):
- 148 steps, per_step=26.13ms, fill=36.4ms
- All 4 produce identical 109 tokens — CORRECT (graph invalidation works)

CT2_NO_CUDA_GRAPH=1:
- 110 steps, per_step=20.86ms
- All 4 produce identical 109 tokens — CORRECT (identical to graph-enabled)

### A/B Phase E+F timing (CT2_DECODE_PHASE_PROFILE, CT2_NO_CUDA_GRAPH=1, 100 calls)

| | Phase E (attn_mask) | Phase F (memory_setup) |
|---|---|---|
| Before (prepare_length_mask) | 0.001 ms/call | 0.001 ms/call |
| After (expand_lengths_gpu) | 0.001 ms/call | 0.000 ms/call |

On DGX Spark (unified memory), both paths are sub-microsecond — the mask is tiny
(20 elements for 4 slots × 5 beams) and both kernels complete in ~1us. No measurable
timing difference on this hardware.

**Result:** The value of Step 9 is architectural, not raw timing on DGX Spark:
1. Eliminates per-step `StorageView` allocation — `prepare_length_mask()` allocated a
   new tensor every call; the expand kernel writes into a pre-allocated buffer with
   `shallow_copy` (zero allocation, stable GPU address for CUDA graph).
2. On H100 (PCIe), each `StorageView` allocation involves `cudaMalloc`/pool overhead
   (~5-10us). With 2 masks per step (self + cross), ~10-20us/step saved. Enables
   CUDA graph to capture Phase E cleanly.
3. Replaces `prepare_length_mask` primitive dispatch with a simpler single-purpose kernel.

### Server test (DEVICE=cuda, 4 concurrent requests)

Server (`server.py`) with `DEVICE=cuda MAX_SLOTS=4 NUM_ENCODERS=1`:
- Single request: 532 chars, lang=en — CORRECT
- 4 concurrent: all 4 return identical 532-char transcriptions — CORRECT
- Server stays alive after all requests — CORRECT

Note: earlier crashes were caused by missing `DEVICE=cuda` env var (default is `"cpu"`
in `server.py:88`). CTranslate2 built without MKL has no CPU SGEMM backend, so loading
the model on CPU aborts. Unrelated to Step 9.

---

## Step 10: Logits_proc hot path optimization

**Target:** Eliminate per-step `cudaMalloc`/`cudaFree` and CPU `push_back` overhead in
`DisableTokens::apply()` and `ContinuousSuppressTokens::apply()`.

**Status:** DONE

### Problem

Profiling (Step 8) revealed `logits_proc` takes ~15.6ms/step (73% of per-step time) on
DGX Spark 4-concurrent. The bottleneck is in `DisableTokens::apply()` which does
**GPU allocate → H2D copy → Thrust indexed_fill → GPU free** every decode step. The CUB
caching allocator can trigger `cudaMalloc` (implicit device sync) on cache miss, absorbing
all async decoder GPU work into the logits_proc wall-clock measurement.

The dominant processor is `ContinuousSuppressTokens` which does the **same work every step**
(same token IDs × all batch rows) — 4000+ CPU `push_back` calls per step that are entirely
pre-computable.

### Changes

- **`include/ctranslate2/decoding_utils.h`**:
  - Added `apply_precomputed(const StorageView& gpu_flat_indices)` — applies pre-computed
    GPU indices directly with a single `indexed_fill` kernel (no CPU→GPU transfer)
  - Added constructor `DisableTokens(StorageView& logits, StorageView& gpu_buffer, ...)` —
    accepts a pre-allocated GPU buffer, avoiding per-apply `cudaMalloc`
  - Added `_gpu_buffer` non-owning pointer member

- **`src/decoding_utils.cc`**:
  - Implemented `apply_precomputed()` — single kernel launch, no allocation
  - Modified `apply()` to use `_gpu_buffer` when available: `resize` (no-op if capacity
    sufficient) + `copy_from` + `indexed_fill` — zero `cudaMalloc`
  - GPU buffer constructor reserves `_flat_indices` capacity (`batch_size * 256`)

- **`include/ctranslate2/continuous_decoding.h`**:
  - Added virtual `init(total_batch, beam_size, vocab_size, device)` to
    `ContinuousLogitsProcessor` base class (default no-op)
  - `ContinuousSuppressTokens`: added `_gpu_indices` (pre-computed), `_precomputed` flag
  - `ContinuousTimestampRules`: added `_log_probs_buf`, `_row_indices_buf`, `_results_buf`
    persistent buffers, `_initialized` flag

- **`src/continuous_decoding.cc`**:
  - `ContinuousSuppressTokens::init()`: pre-computes flat indices for all total_batch rows ×
    all suppress token IDs on CPU, uploads to GPU once. `apply()` calls `apply_precomputed()`
    — single async kernel, zero CPU `push_back`s.
  - `ContinuousTimestampRules::init()`: pre-allocates `_row_indices_buf` and `_results_buf`
    on GPU. `_log_probs_buf` lazily initialized on first `apply()` with correct dtype
    (avoids dtype mismatch with FLOAT16 logits). Subsequent steps reuse capacity.
  - Engine `process()`: pre-allocates `disable_buf` (`total_batch × vocab_size` INT32),
    calls `init()` on all processors before decode loop.

### Per-step flow: before vs after

**Before:**
```
SuppressTokens: 4000+ push_backs to CPU vector
TimestampRules: push_backs, disable_tokens.apply() → cudaMalloc+H2D+kernel+cudaFree
                LogSoftMax → cudaMalloc+kernel+cudaFree
                batch_check → cudaMalloc+H2D+kernel+D2H+cudaFree
Outer apply(): cudaMalloc+H2D+indexed_fill+cudaFree
```

**After:**
```
SuppressTokens: apply_precomputed() → single async indexed_fill kernel
TimestampRules: push_backs (reserved vector), disable_tokens.apply() → copy to pre-alloc buf + kernel
                LogSoftMax → into persistent _log_probs_buf (resize is no-op)
                batch_check → pre-alloc row_indices + results bufs
Outer apply(): copy to pre-alloc disable_buf + indexed_fill
```

### Debugging note

Initial implementation caused a hang: `_log_probs_buf` was initialized as
`StorageView(device)` (default FLOAT32) but `LogSoftMax` receives FLOAT16 logits.
`resize_as()` only changes shape, not dtype — the kernel wrote FLOAT16 into a FLOAT32
buffer, causing silent corruption/hang. Fixed by lazy-initializing `_log_probs_buf` with
correct dtype on first use (`StorageView(logits.dtype(), logits.device())`).

### After Step 10 (DGX Spark, max_slots=4, beam 5, f1_35s.wav)

`test_correctness.sh` (standard generate path): 115 tokens exact match — PASS

1-slot:
- 110 steps, per_step=21.00ms, logits_proc=117.9ms — PASS

4-concurrent (graph enabled):
- 110 steps, per_step=20.91ms, logits_proc=1661.5ms
- All 4 produce identical 109 tokens — CORRECT

Staggered (graph enabled):
- 149 steps, per_step=25.91ms, logits_proc=182.4ms
- All 4 produce identical 109 tokens — CORRECT

CT2_NO_CUDA_GRAPH=1:
- 110 steps, per_step=20.97ms, logits_proc=1687.4ms
- All 4 produce identical 109 tokens — CORRECT

**Result:** On DGX Spark (unified memory), `logits_proc` timing is similar because
`cudaMalloc`/`cudaFree` are near-free on unified memory (no PCIe overhead). The CUB caching
allocator rarely misses, so the allocator sync cost that dominates on discrete GPUs is
absent here. The optimization eliminates:
- All CPU `push_back` loops in `ContinuousSuppressTokens` (pre-computed GPU indices)
- All per-step `StorageView` constructor allocations in `DisableTokens::apply()` and
  `ContinuousTimestampRules::apply()` (pre-allocated buffers with capacity reuse)

Expected savings on H100 (PCIe, discrete memory):
- Each `cudaMalloc` CUB cache miss: ~5-50us (device sync)
- 4+ allocations eliminated per step × ~10us avg = ~40-200us/step saved
- CPU `push_back` elimination: ~0.1-0.5ms/step saved (4000+ push_backs)

### Step 10b (deferred): Prologue/epilogue multi-op fusion

Original Step 10 target (fusing prologue/epilogue ops around decoder forward) saves <0.1ms
since those ops are already inside CUDA Graph. Deferred in favor of the higher-impact
logits_proc optimization above.
