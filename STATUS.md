# Continuous Batching — Current Status

## Background: The Selection Phase Bottleneck

The beam search **selection phase** dominated decode loop wall time. On every step,
after the GPU computes logits, the selection phase had to:

1. `LogSoftMax` on logits (GPU)
2. `TopK` to find candidates (GPU)
3. **GPU→CPU transfer** of TopK results (`.to(Device::CPU)` — implicit sync, ~13ms wait)
4. CPU-side beam hypothesis management (expand, prune, finalize)
5. `ops::Gather` to reorder 64 state tensors for surviving beams (GPU, async)

Steps 3–4 were the bottleneck. Profile from DGX Spark (4 slots, beam=5):

| per_step | DECODER enqueue | selection (incl. sync) | other |
|----------|-----------------|------------------------|-------|
| 19.1ms   | 3.4ms           | 15.2ms                 | 0.5ms |

Selection was **80% of wall time** and completely GPU-idle.

---

## What Has Been Done

### Phase 1: Scatter path (merged, staged)
- GPU scatter kernel for K/V cache writes (`scatter_step_async`)
- Upstream in staged changes — see git status

### Phase 2: GPU beam selection (complete)

#### 2a. GPU beam selection kernel (`src/cuda/beam_select.cu`, `src/cuda/beam_select.h`)
- One CUDA block per active slot; sequential candidate iteration on GPU
- Replaces the entire CPU beam search loop
- Produces: `gather_indices`, `next_tokens`, `beam_scores`, `beam_finished`, EOS data
- GPU state buffers (scores, finished, etc.) are persistent across steps
- CPU staging buffers via `cudaMallocHost` for small async D2H (~200 bytes/step)

#### 2b. GPU gather kernel (`src/cuda/batch_copy.cu`, `src/cuda/batch_copy.h`)
- `gather_rows_async`: reads gather indices from device memory (no CPU sync needed)
- Two-phase: Phase A (kernel scatter to scratch), Phase B (contiguous D2D memcpy back)
- KV cache reordering now fully GPU-side, enqueued before the sync

### Phase 3: Pipeline restructuring (complete — on `continuous-batching` branch)

The decode loop for `beam_size > 1` on CUDA has been restructured into a 4-phase
pipeline. The CPU fallback path is preserved and selectable via env var.

**Loop structure (implemented):**
```
while (true) {
  Phase A: slot_mgmt → cache_pad → step_setup → decoder(N) enqueue
           Uses GPU-resident _bs_sample_from; speculative step_offsets when pending.

  Phase B: if pending_beam_sync:
             cudaEventSynchronize(d2h_event)
             CPU bookkeep step N-1: collect EOS, reorder beam_tokens,
             update scores/finished, slot.step++, detect finished slots.

  Phase C: LogSoftMax → add_depth_broadcast(beam_scores→fp16, logits fp16)
           → TopK → topk_scores.to_float32() → beam_select_async (GPU kernel)
           → resize up → Gather(batch_state, GPU gather_indices)
           → D2D next_tokens→sample_from → async D2H → cudaEventRecord

  Phase D: defrag finished slots, fill new slots (re-init GPU beam state),
           speculative step++, update steps_gpu, cache_lengths++
}
Final sync: process last pending step after loop exits.
```

**Key implementation details:**
- GPU-persistent beam state in `batch_state` with `"_bs_"` prefix (beam-level,
  dim0=total_batch) and `"memory_bs_"` prefix (slot-level, dim0=max_slots).
  Prefixes ensure `defragment_slots` / `copy_slot_data` / `resize_batch_state`
  handle them automatically.
- Pinned CPU staging buffers (~2KB) via `cudaMallocHost` for 9 arrays.
- CUDA event (`cudaEventCreateWithFlags(DisableTiming)`) for precise D2H sync.
- Logits stay in native fp16 throughout — only the tiny `topk_scores`
  (active_count × 2×beam_size values) is converted to float32 for the kernel.
  `add_depth_broadcast` converts beam_scores (float32, ~20 values) → fp16
  before adding to fp16 logits. No full-tensor dtype conversion.
- `CT2_CPU_BEAM_FALLBACK=1` env var forces CPU fallback for A/B testing.

---

## Plan: GPU Pipeline Fixes + Multi-Batcher for 130 Users

**Full plan**: `/root/.claude/plans/cuddly-dreaming-church.md`

**Goal**: Handle 130 simultaneous users under 5s SLA on H100.

The plan had 7 changes. Here is the status of each:

### C1 — Fix GPU pipeline sync points — DONE, WORKING

- `fill_identity_async` kernel added to `beam_select.cu/h` (replaces sync `cudaMemcpy` for identity gather init)
- `full_gather_persistent` pre-allocated before decode loop (avoids per-step GPU alloc)
- Env var flipped: GPU pipeline is now the default, `CT2_CPU_BEAM_FALLBACK=1` to disable
- **Verified working**: 35s audio transcribed correctly, GPU% up to 85%

### C2 — Multi-batcher support — DONE, UNTESTED WITH NUM_WORKERS>1

- `server.py` rewritten: `batchers` list, `NUM_WORKERS` env var, round-robin via `itertools.cycle`
- Single-worker verified working
- Needs load testing with NUM_WORKERS=2+ on H100

### C3 — CUDA stream verification — DONE

- Verified: each batcher thread gets its own CUDA stream via `thread_local` in `src/cuda/utils.cc`
- `CudaStream` class creates a new `cudaStreamCreate` for each non-main thread
- With NUM_WORKERS=N: 2N worker streams (encoder + decoder per batcher) + 1 default (main)
- All streams share the same GPU device; no concurrency conflicts

### C4 — Pre-allocate TopK buffers — DONE

- `gpu_topk_scores` and `gpu_topk_ids` initialized on GPU device at loop start
- Dtype check at TopK call: resets `gpu_topk_scores` if logits dtype changes (fp16 vs fp32)
- TopK `resize()` reuses allocation when shape matches (no per-step GPU malloc)
- fp32 conversion for beam_select done into separate `topk_scores_f32` (not in-place on pre-alloc buffer)

### C5 — Skip Gather for identity reordering — DONE

- `slot_needs_gather` output param added back to `beam_select_kernel` and `beam_select_async`
- Kernel checks if gather_indices is identity per slot, writes 1 (needs gather) or 0
- D2H copy of `memory_bs_needs_gather` → `staging_needs_gather` added to async section
- Phase B reads `staging_needs_gather` to set `prev_step_all_identity`
- Condition flipped: `if (!prev_step_all_identity)` — skips expensive KV-cache Gather when all beams are identity

### C6 — CPU fallback optimization — DONE

- `gather_device_persistent` pre-allocated on GPU, reused via `cudaMemcpyAsync` instead of per-step `StorageView(...).to(device)`
- Working (verified with `CT2_CPU_BEAM_FALLBACK=1`)

### C7 — Batch H2D for new slots — DONE

- New slot GPU state init uses batched H2D (build contiguous staging, one memcpy, scatter)
- In the "fill new slots" section of the decode loop

---

## Bugs Found and Fixed

1. **`_bs_end_ids` in batch_state** — A constant-size tensor (dim0 = num_end_ids, typically 1) was stored in `batch_state`. `resize_batch_state()` iterates ALL entries and resizes to `target_batch` (e.g. 20), corrupting this buffer. **Fix**: moved to standalone `end_ids_device` variable.

2. **Out-of-bounds on inactive slots** (the actual segfault) — The beam state init loop accessed `slots[s].beam_scores[b]` and `slots[s].beam_finished[b]` for ALL slots 0..max_slots, but inactive slots have empty vectors. **Fix**: bounds checking with `slots[s].active && b < slots[s].beam_scores.size()`.

3. **In-place `beam_finished` read-after-write hazard** — `beam_finished_in` and `beam_finished_out` aliased the same GPU buffer (`_bs_beam_finished`). The kernel could write `beam_finished_out[s_offset + filled] = 0` then later read `beam_finished_in[s_offset + beam_id]` from the same location if `filled == beam_id`, getting the output value instead of the input. **Fix**: cache input `beam_finished` into `local_finished[32]` registers before the selection loop.

---

## BUG FIXED: GPU pipeline produces different text than CPU fallback

**Status**: FIXED.

**Root cause**: The GPU pipeline's Phase B (CPU bookkeeping that updates `slot.beam_tokens`)
was executing AFTER the Whisper logits processor. The logits processor reads `slot.beam_tokens[b]`
to determine timestamp/token suppression rules. Because Phase B ran after logits processing,
the processor saw stale token history (off by one step), causing it to suppress the wrong tokens
(e.g., suppressing word tokens while allowing timestamps when it should have been the reverse).

**Fix**: Moved Phase B from inside the GPU pipeline branch to before the logits processor call.
This ensures `slot.beam_tokens` is fully up-to-date before suppression decisions are made.
The decoder forward pass (Phase A) doesn't depend on Phase B's output, so this reordering
is safe and doesn't affect the pipelining of GPU work.

**Verification**: GPU and CPU paths now produce nearly identical text (minor fp16 rounding
differences in beam scores can cause slightly different beam selections, but output is coherent).

---

## Benchmark (single request, 35s French audio, beam=5, 4 slots)

### Previous (DGX Spark)

| Mode | Steps | Total | per_step | GPU% | Selection | RTFx |
|------|-------|-------|----------|------|-----------|------|
| CPU fallback | 56 | 713ms | 12.7ms | 57.8% | 291.7ms | 26.1 |
| GPU pipeline | 57 | 749ms | 13.1ms | 84.9% | 89.2ms | 25.2 |

### Current (L4 GPU, fp16)

| Mode | Steps | Total | per_step | GPU% | RTFx |
|------|-------|-------|----------|------|------|
| CPU fallback | 168 | 2412ms | 14.4ms | 39.7% | 13.5 |
| GPU pipeline (C1-C7) | 137 | 1932ms | 14.1ms | 41.5% | 16.7 |

**NOTE**: GPU pipeline text divergence bug has been fixed. Benchmarks need re-running.

---

## What To Do Next

1. ~~Re-enable C4 (TopK pre-alloc)~~ — DONE
2. ~~Re-enable C5 (gather skip)~~ — DONE
3. ~~Verify C3 (CUDA streams per batcher)~~ — DONE
4. ~~FIX GPU pipeline text divergence~~ — FIXED (Phase B moved before logits processing)
5. Test NUM_WORKERS=2+ with concurrent load
6. Load test on H100 with 130 concurrent users
7. Remove debug fprintf + `[DECODE PROFILE]` once tuned on target hardware

---

## Files Modified

| File | Change |
|------|--------|
| `src/cuda/beam_select.h` | Header for GPU beam selection kernel + `fill_identity_async` |
| `src/cuda/beam_select.cu` | CUDA kernel: one block/slot + identity fill kernel |
| `CMakeLists.txt` | Added `beam_select.cu` to `CUDA_SOURCES` |
| `src/continuous_decoding.cc` | Pipeline restructure, GPU beam state, C1/C4-C7 changes |
| `server.py` | Multi-batcher support (C2) |
| `DEPLOYMENT.md` | Updated with NUM_WORKERS and CT2_CPU_BEAM_FALLBACK docs |

## Environment Flags

| Flag | Effect |
|------|--------|
| `CT2_CPU_BEAM_FALLBACK=1` | Force CPU fallback beam path (for A/B testing) |
| `CT2_NO_ATTN_CAPTURE=1` | Disable attention capture in whisper decoder |
| `NUM_WORKERS` | Number of batcher instances (default: 1) |
| `MAX_SLOTS` | Slots per worker (default: 4) |
