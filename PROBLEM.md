# Continuous Batching — History

## Original Problem
Under sustained load, `queue_wait` climbed to 20-48s because the encoder ran inside the decode loop — every `fill_slot()` blocked all active slots for 150-400ms.

## Solved: Async Encoder Thread
Dedicated `encoder_loop()` thread batch-encodes features in a separate GPU call. Worker thread reads from `_encoded_queue` — `fill_slot()` never blocks on encoding.

## Solved: Proactive Slot Filling
Inactive slots were never filled mid-decode. Added scan of all inactive slots every decode step to load queued requests immediately. Result: 8 requests finish together in 14s instead of serializing at 106s (7.5x speedup).

## Solved: Decode Loop Performance (Phases 1-6)
Closed ~2x perf gap between continuous batcher and `model.generate()` for beam=1. See optimization plan in `.claude/plans/hashed-inventing-thimble.md`. Ratio went from ~2x to 1.09x.
