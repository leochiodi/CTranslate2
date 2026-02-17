# Contention Fix: Batch-Encode in Worker Thread

## Problem
Under sustained load (100-130 users), `queue_wait` climbs to 20-48s despite the callback fix delivering results incrementally.

## Root Cause
**The encoder runs inside the decode loop's hot path.** Every time a slot finishes and gets recycled via `fill_slot()`, the `slot_initializer` runs a full encoder pass (~150-400ms). During this time, ALL other active slots are blocked — no decode steps happen for any slot.

With 4 slots continuously recycling, the worker thread spends a large fraction of its time encoding instead of decoding, cutting effective throughput roughly in half.

## First Attempt: Pre-encode in `submit()` (REVERTED)
Moved encoding into the caller's thread via `_replica_mutex`. This freed the decode loop but introduced a NEW problem: the mutex serialized all submit() calls, causing requests to trickle into the queue over ~600ms. The worker's 50ms batching window only captured 1-2 requests instead of all 4, so the batch started with partial slots and requests processed serially.

## Current Fix: Batch-encode in `worker_loop()` (TESTING)
- `submit()` stays instant (no encoding, no mutex)
- All requests arrive in the queue together within milliseconds
- Worker drains the queue, then **batch-encodes** up to `_max_slots` features in a single encoder call (GPU processes them in parallel)
- The `slot_initializer` sees `is_encoded() == true` and skips encoding for the initial batch
- Mid-decode slot fills still encode individually (~150ms per fill, acceptable overhead)

### Expected improvement
- **Before**: encode + decode serial per batch cycle → ~50% of GPU time wasted on sequential encoding
- **After**: single batched encode call (~200ms for all slots), then uninterrupted decode loop
