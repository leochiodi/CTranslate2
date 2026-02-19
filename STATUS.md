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

## Bug 4: GPU pipeline wrong text (timestamp suppression) — FIXED

**Root cause**: Phase B (CPU bookkeeping updating `slot.beam_tokens`) ran AFTER the Whisper
logits processor. The logits processor reads `slot.beam_tokens[b]` for timestamp/token
suppression rules, so it saw stale token history (off by one step), suppressing the wrong tokens.

**Fix**: Moved Phase B from inside the GPU pipeline branch to before the logits processor call
(line ~1190 in continuous_decoding.cc). Single-slot transcription now matches CPU fallback.

---

## Bug 5: GPU pipeline empty/truncated results with multiple concurrent slots — FIXED

**Root cause**: Slot index mismatch between Phase B staging buffers and current slot positions.
`defragment_slots()` in Phase D shifted slot indices after D2H was launched in Phase C.
Phase B (next iteration) read staging buffers using post-defrag indices but buffers had
pre-defrag data. Additionally, `pending_beam_active_count` didn't account for defrag
removing slots then fill adding new ones back to the same count.

**Fix (Option A — staging index mapping)**:
1. Added `pending_staging_map` vector mapping post-defrag slot position → pre-defrag
   staging buffer index. Built before `defragment_slots()` by recording active slot
   positions in order.
2. Phase B and final sync use `pending_staging_map[s]` instead of `s` for all staging
   buffer reads (`staging_*[stg]` / `staging_*[stg_offset + b]`).
3. `pending_beam_active_count` updated to post-defrag count (before fill) so Phase B
   won't iterate into newly-filled slots that have no staging data.
4. Map reset to identity at each D2H launch (Phase C), since staging indices match
   current positions at that point.

**Verified**: 19 x 30s chunks, max_slots=4, beam_size=5. GPU pipeline and CPU fallback
produce equivalent results (17-18 OK, 1-2 empty from VAD/server issues, not beam search).

---

## ACTIVE BUG 6: GPU beam kernel early termination — n_finished accumulates too fast

**Status**: Root-caused, NOT YET FIXED. Affects single-slot too (no defrag involved).

**Symptom**: GPU pipeline returns only 1-2 words for a 30s audio chunk. CPU fallback
(`CT2_CPU_BEAM_FALLBACK=1`) returns full correct transcription. Tested on H100 with
whisper-large-v3, beam_size=5, single 30s chunk via server.

**H100 debug logs** (single slot, `CT2_DEBUG_DEFRAG=1`):
```
[D2H] active=1 slots=[0:r0:s3]
[PHASE_B] sync_active=1 cur_active=1 map=[0->0]
[PHASE_B] s=0 stg=0 req=0 step=3 slot_finished=0 n_eos=0 n_finished=0 bf=[0,0,0,0,0]
[D2H] active=1 slots=[0:r0:s4]
[PHASE_B] sync_active=1 cur_active=1 map=[0->0]
[PHASE_B] s=0 stg=0 req=0 step=4 slot_finished=0 n_eos=1 n_finished=1 eos_beams=[0] bf=[0,0,0,0,0]
[D2H] active=1 slots=[0:r0:s5]
[PHASE_B] sync_active=1 cur_active=1 map=[0->0]
[PHASE_B] s=0 stg=0 req=0 step=5 slot_finished=0 n_eos=0 n_finished=1 bf=[0,0,0,0,0]
[D2H] active=1 slots=[0:r0:s6]
[PHASE_B] sync_active=1 cur_active=1 map=[0->0]
[PHASE_B] s=0 stg=0 req=0 step=6 slot_finished=0 n_eos=1 n_finished=2 eos_beams=[1] bf=[0,0,0,0,0]
[D2H] active=1 slots=[0:r0:s7]
[PHASE_B] sync_active=1 cur_active=1 map=[0->0]
[PHASE_B] s=0 stg=0 req=0 step=7 slot_finished=1 n_eos=3 n_finished=5 eos_beams=[2,1,0] bf=[0,0,0,0,1]
[DECODE PROFILE] steps=5 total=95.4ms per_step=19.08ms GPU%=36.4%
```

**Key observations**:
1. `n_finished` accumulates across steps: 0 → 1 → 1 → 2 → 5 (hits max_candidates=5)
2. `bf=[0,0,0,0,0]` — beams that produce EOS are **never marked finished**
3. At step 7, three beams (2,1,0) hit EOS simultaneously, pushing n_finished from 2 to 5
4. Total only 5 decode steps, ~2 generated tokens before termination
5. CPU fallback produces full transcription for the same audio

**Root cause analysis**:

The GPU beam_select kernel (`src/cuda/beam_select.cu`) accumulates `n_finished` via
`num_finished_in[slot]` which persists across steps. Each step, if any beam's best
candidate is EOS, `n_finished++`. But the beam is NOT marked as finished in
`beam_finished_out` because:

1. When a beam hits EOS, the kernel does `continue` (doesn't fill an output position)
2. There are enough non-EOS candidates (num_candidates = 2*beam_size = 10) to fill
   all beam_size=5 output positions even after skipping the EOS
3. So all output positions get `beam_finished_out = 0`
4. Next step, the same beam can hit EOS again → increments n_finished again

The CPU fallback path has the same accumulation logic (`slot.num_finished_beams++`),
but it produces much longer output. The difference is likely in the **logits** being
fed to the kernel — the Whisper logits processor uses `slot.beam_tokens` for
timestamp/token suppression, and if the token history differs between GPU and CPU
paths, different tokens get suppressed, changing EOS frequency.

**Possible explanations for the logits difference**:
- The GPU pipeline's Phase B updates `slot.beam_tokens` one step behind (pipelined),
  which was supposed to be fixed by Bug 4, but there may still be an off-by-one
- The `add_depth_broadcast` that adds beam_scores to logits in fp16 may introduce
  precision differences vs the CPU path
- The speculative step increment in Phase D may cause `cache_lengths` to be wrong,
  feeding the decoder wrong positional info

**Investigation needed**:
1. Add debug logging of `beam_tokens[0]` (best beam's token history) at each step
   to compare GPU vs CPU path token-by-token
2. Check if the logits processor sees the same `beam_tokens` in both paths
3. Compare the actual logits values for the EOS token between GPU and CPU paths

**Debug env var**: `CT2_DEBUG_DEFRAG=1` enables Phase B / D2H / defrag logging to stderr.
Needs more fields added (beam_tokens) — uncommitted edit exists locally.

**Files involved**:
- `src/cuda/beam_select.cu` — kernel accumulates n_finished, may need logic change
- `src/continuous_decoding.cc` — Phase B beam_tokens update, logits processor interaction

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
5. ~~FIX GPU pipeline empty/truncated results with concurrent slots~~ — FIXED (staging index mapping)
6. **FIX GPU beam kernel early termination** — Bug 6, active, see above
7. Test NUM_WORKERS=2+ with concurrent load
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
| `CT2_DEBUG_DEFRAG=1` | Enable Phase B / D2H / defrag debug logging to stderr |
| `CT2_NO_ATTN_CAPTURE=1` | Disable attention capture in whisper decoder |
| `NUM_WORKERS` | Number of batcher instances (default: 1) |
| `MAX_SLOTS` | Slots per worker (default: 4) |
