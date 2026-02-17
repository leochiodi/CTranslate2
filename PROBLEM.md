
Leonardo Chiodi <leochiodi@gmail.com>
08:24 (il y a 1 heure)
À leonardo.chiodi

# Contention Fix: Batch-Encode in Worker Thread

## Problem
Under sustained load (100-130 users), `queue_wait` climbs to 20-48s despite the callback fix delivering results incrementally.

## Root Cause
**The encoder runs inside the decode loop's hot path.** Every time a slot finishes and gets recycled via `fill_slot()`, the `slot_initializer` runs a full encoder pass (~150-400ms). During this time, ALL other active slots are blocked — no decode steps happen for any slot.

With 4 slots continuously recycling, the worker thread spends a large fraction of its time encoding instead of decoding, cutting effective throughput roughly in half.

## First Attempt: Pre-encode in `submit()` (REVERTED)
Moved encoding into the caller's thread via `_replica_mutex`. This freed the decode loop but introduced a NEW problem: the mutex serialized all submit() calls, causing requests to trickle into the queue over ~600ms. The worker's 50ms batching window only captured 1-2 requests instead of all 4, so the batch started with partial slots and requests processed serially.

## Second Attempt: Async encoder thread (MERGED, TESTED)

Introduced a dedicated `encoder_loop()` thread that:
- Drains `_raw_queue` as submissions arrive (with a 10ms batching window)
- Batch-encodes all pending features in a single GPU call
- Pushes pre-encoded requests to `_encoded_queue`

The worker thread reads exclusively from `_encoded_queue` — `fill_slot()` never blocks on the encoder.

### Result (8 concurrent × 35s audio, L40S, MAX_SLOTS=16, BEAM_SIZE=5, float16)
```
req 6: 13.3s
req 0: 26.1s
req 1: 38.9s
req 4: 52.0s
req 2: 65.2s
req 7: 78.7s
req 5: 92.3s
req 3: 106.2s
Total wall time: 106s for 8 requests
```

**Requests processed sequentially (~13s each) — no concurrency gain.**

### Why
With MAX_SLOTS=16 and BEAM_SIZE=5, the decode batch is always 80 rows (16 × 5), even when only 1 slot is active. A single 35s request already occupies one slot and generates ~115 tokens. The per-step compute cost is the same whether 1 or 8 slots are active, so requests effectively serialize through the decode loop rather than genuinely parallelizing.

The encoder thread works correctly (encoding no longer blocks decoding), but the fundamental bottleneck has shifted: the batch size configuration itself is causing each decode step to be expensive, and slots are not being kept full.

## Third Attempt: Proactive slot filling (MERGED, TESTED)

### Root Cause of Serialization

The decode loop in `ContinuousDecodingEngine::process()` only called `fill_slot()` for **finished** slots (slots that were active and hit EOT). Empty/inactive slots were never proactively filled during the decode loop.

When `process()` started with only 1 request (because the rest hadn't been encoded yet when the worker drained the queue), only 1 slot was active. The other 15 slots sat empty for the entire decode run. When that 1 slot finished after ~115 tokens, `fill_slot` recycled it with the next request — which then decoded alone for another 115 steps. Pure FIFO.

### Fix

Added a scan of **all** inactive slots at every decode step, right after the existing finished-slot recycling and before the active-slot recount. If there are queued requests (from `request_queue` or `queue_provider` reading `_encoded_queue`), they immediately get loaded into empty slots via `fill_slot()`.

```cpp
// src/continuous_decoding.cc — after finished_slots loop, before recount
for (size_t s = 0; s < _max_slots; ++s) {
  if (!slots[s].active) {
    const auto& accum = batch_state["accumulated_attention"];
    const dim_t attn_time = (accum && accum.rank() >= 3) ? accum.dim(2) : 0;
    if (!fill_slot(s, request_queue, queue_provider, slots, batch_state))
      break;  // No more requests available.
    slots[s].attention_step_offset = attn_time;
  }
}
```

The existing sample_from rebuild and cache_lengths update already iterate all slots, so newly-filled slots are handled correctly without further changes.

### Result (8 concurrent × 35s audio, L40S, MAX_SLOTS=16, BEAM_SIZE=5, float16)
```
req 2: 14.17s
req 7: 14.17s
req 0: 14.18s
req 4: 14.18s
req 1: 14.19s
req 6: 14.19s
req 5: 14.20s
req 3: 14.20s
Total wall time: 14.2s for 8 requests
```

**All 8 requests complete together — 7.5x faster than before (14s vs 106s).**

### Result (32 concurrent × 35s audio, same config)
```
First 16 (filling all slots):  ~17s
Next 16 (recycled slots):      ~30s
Total wall time: 29.9s for 32 requests
```

Two-wave pattern as expected with MAX_SLOTS=16. Would have been ~416s with the old FIFO behavior — **14x speedup**.
