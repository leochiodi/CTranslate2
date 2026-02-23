You already did the hard part: the overhead is now well localized.

If I had to pick **one high-impact direction** to reduce CB overhead without losing continuous batching, it would be:

## Make the decoder loop mostly GPU-resident and event-driven (remove per-step CPU orchestration)

The biggest avoidable cost in your breakdown is not math—it’s **control-plane work repeated every step**:

* step setup on CPU (`step_offsets`, `cache_write_positions`, `sample_from`)
* per-slot logits processing on CPU-ish flow
* fill/defrag decisions every step
* extra kernel launches from fragmented operations

That’s the part to attack.

---

## The practical plan (highest ROI first)

### 1) **Stop rebuilding per-step metadata on CPU every step**

**Target:** `~0.6ms/step` (and some launch overhead)

Right now CB does:

* CPU loop for `step_offsets`
* CPU loop for `cache_write_positions`
* CPU path for `sample_from` (in some modes)
* small H2D copies every step

### Replace with GPU-maintained slot state

Keep these tensors permanently on GPU:

* `slot_steps[max_slots]`
* `slot_prompt_lengths[max_slots]`
* `slot_active[max_slots]`
* `cache_lengths[total_batch]` (already present, but keep GPU-authoritative)
* `sample_from[total_batch]` (GPU-authoritative for *all* modes, not just pipelined beam)

Then launch **one tiny kernel** per step to build:

* `step_offsets[active_batch]`
* `cache_write_positions[active_batch]`
* maybe even active row map / gather indices base

That removes:

* CPU loops
* H2D copies for these tiny tensors
* synchronization pressure around “CPU owns step state”

**Why this matters:** your decode math is already fast; shaving 0.5–1.0ms from orchestration is a big percentage.

---

### 2) **Eliminate resize-down / resize-up in the hot loop**

**Target:** `~0.1ms/step` directly, plus fewer launches and less shape churn

You currently resize batch tensors to `active_count` then resize back.

That is expensive not because resizing is huge, but because it creates:

* shape bookkeeping
* branchy paths
* extra kernel launches / dispatch decisions
* potential allocator/metadata churn

### Replace with fixed-shape execution + active mask

Always run decoder on:

* `total_batch = max_slots * beam_size`

Use:

* `slot_active` / row-active mask
* `cache_lengths == 0` or masked rows
* logits masking for inactive rows

Yes, this may compute a bit more on inactive rows, but:

* your contiguity invariant already keeps actives front-packed
* at steady load, most slots are active anyway
* fixed shapes improve kernel selection / graph capture opportunities (huge)

This also sets up the next optimization (CUDA Graphs).

---

### 3) **Capture the per-step decode path with CUDA Graphs (or graph-like static launch bundles)**

**Target:** `~1–2ms/step` kernel launch overhead

Your own analysis points to extra launch overhead as one of the largest deltas.

CB has more kernels because of:

* scatter writes
* position encoding
* mask prep
* logits rules
* beam kernels
* defrag/fill helpers

If the hot path uses **stable shapes**, you can graph-capture a step for common regimes, e.g.:

* beam size fixed (5)
* slots fixed (`e2_s4` => 20 rows)
* cache chunk size fixed (64-aligned)
* same dtypes/layouts

Then replay each step with updated buffers.

### Why this is probably your biggest win after GPU slot-state

Launch overhead is mostly constant and hurts more on faster GPUs (as you observed). CUDA Graphs directly attack that.

Even if you can’t graph the entire loop (due to dynamic fill/defrag), graph:

* decoder forward path
* logits + beam topk + select path
* common no-defrag/no-fill step

And fall back to normal launches only on structural changes (slot turnover/chunk growth).

---

### 4) **Decouple slot refill/defrag from every decode step (“micro-epochs”)**

**Target:** `fill new slots ~0.8ms/step`, `defrag ~0.1ms`, plus launch overhead

This is likely the single biggest CB-specific overhead in your table.

Right now, CB pays slot management tax **inside every step**:

* defragment
* fill new slots
* scatter prepared state into batch
* rebuild sample_from

### Better strategy: defer structural updates

Run decode in **micro-epochs** of K steps (e.g. 2–4 steps), where:

* finished slots are marked inactive immediately,
* but **actual defrag + refill** only happens every K steps (or when active count drops below threshold).

This keeps responsiveness while reducing:

* `set_batch_slot()` frequency
* row-copy churn
* state mutation complexity in the hottest path

**Tradeoff:** a finished slot may sit idle for 1–3 steps before refill.
At 25–30ms/step, K=2 adds ~25–30ms average refill delay worst-case, which may be acceptable depending on SLA. You can make it adaptive:

* high queue pressure → refill every step
* low queue pressure → refill every 2–4 steps

This is a very good throughput/latency knob.

---

### 5) **Pre-stage encoded requests into slot-ready GPU layouts**

**Target:** reduce `fill new slots` cost

`set_batch_slot()` is expensive because it scatters/copies many tensors into the live batch state.

Instead, in encoder thread, produce each request not just as “encoded state”, but as **slot-ready contiguous slabs** matching batch layout:

* self KV padded to chunk size and beam-tiled upfront (if beam>1)
* memory K/V already in target layout
* prompt state already on GPU in slot-compatible row-major arrangement

Then slot fill becomes closer to:

* a small number of contiguous row copies (or pointer swaps if architecture allows segmented state)

Even better: use a **pool of preallocated slot buffers**, and decoder reads from an indirection table rather than physically copying into a monolithic batch tensor (see #7 below).

---

### 6) **Move Whisper logits processors fully onto GPU**

**Target:** `~0.4ms` extra logits processing

Your per-slot timestamp/blank suppression rules are a classic “small but frequent” overhead.

They are also highly parallel and deterministic.

Implement a fused GPU logits postprocessor kernel that takes:

* current step
* prompt length / generation step
* per-slot token histories (or compact state flags)
* timestamp config
* suppress token lists / bitsets

and modifies logits in-place before sampling/topk.

This reduces:

* CPU iteration over slots
* sync points around logits inspection
* extra launches if you currently chain multiple suppressors

If histories are expensive to read, maintain compact per-slot state on GPU (e.g., last token, timestamp phase flags, no-speech flags, etc.) updated during beam select.

---

### 7) **Consider logical compaction instead of physical defragmentation**

**Target:** `defrag + fill` overhead and copy churn

Right now you maintain a “contiguous active slots at front” invariant and physically compact rows.

That simplifies loops but costs row copies.

Alternative:

* keep a **logical-to-physical slot map**
* active slot list is compact, physical storage is not
* decoder kernels gather row indices through an indirection array

This adds index indirection inside kernels, but can eliminate a lot of `batch_copy_async`.

Whether it wins depends on your kernel implementation:

* If your ops are heavily row-major contiguous and index indirection hurts memory access, maybe not.
* If defrag churn is frequent (high turnover segments), it can win.

A hybrid works well:

* logical compaction every step
* physical defrag only when fragmentation exceeds threshold (e.g., >25%)

---

### 8) **Reuse masks / avoid rebuilding self-attn mask when cache lengths are unchanged pattern-wise**

**Target:** small but cumulative (`~0.05–0.1ms` + launches)

In CB, mask is needed due to padding, but for many consecutive steps:

* all active beams within a slot advance together
* lengths change predictably (+1 per active row)
* padded max length stays same until chunk boundary

You may be able to:

* precompute/cached masks for common `(active_batch, padded_len, length_pattern)` cases, or
* use an attention kernel that directly consumes `cache_lengths` without materializing a separate mask tensor (best option)

If `prepare_length_mask()` materializes full mask tensors, that’s a lot of overhead for small logical info.

---

## The highest-impact redesign in one sentence

### **Turn CB from “GPU compute + CPU step scheduler” into “GPU state machine with occasional CPU decisions.”**

That is the architectural difference between “CB works” and “CB is as fast as fixed-batch decode.”

---

## A concrete prioritized roadmap

### Phase A (fastest wins, lowest risk)

1. **GPU-authoritative step metadata** (`step_offsets`, `cache_write_positions`, `sample_from`)
2. **Remove resize up/down** (fixed-shape batches)
3. **Fuse logits processors on GPU**
4. **Refill/defrag every K steps (adaptive K=1/2/4)**

Expected combined improvement: **~1.5–3.0ms/step** (often more under churn)

---

### Phase B (larger engineering, bigger win)

5. **CUDA Graph capture for common decode steps**
6. **Pre-staged slot-ready encoded states**
7. **Fragmentation-threshold defrag (not every step)**

Expected additional improvement: **~2–4ms/step** depending on GPU and concurrency

---

### Phase C (deeper kernel changes)

8. **Logical compaction / indirection-based slot mapping**
9. **Attention kernel consumes lengths directly (no explicit mask materialization)**
10. **Multi-op fusion around decoder step prologue/epilogue**

---

## If you want one very specific change to implement first

If you only pick **one** thing next week:

## ✅ Implement a GPU kernel that builds `step_offsets`, `cache_write_positions`, and next-step token buffer from GPU slot state (and delete CPU per-step setup)

Why first:

* clear target in your profile (`~0.6ms/step`)
* localized change
* low algorithmic risk
* unlocks CUDA Graphs later
* reduces synchronization complexity

---

## Sanity check: don’t regress quality / semantics

When optimizing, protect these invariants:

* exact Whisper timestamp rules behavior
* identical beam search hypothesis ordering / tie-breaking
* cache length correctness per row after fill/defrag
* no stale row state after slot reuse
* attention accumulation alignment offsets remain correct if refill is deferred

Golden tests:

* token-level exact match vs old CB on fixed seeds
* beam trace diff per step
* timestamp output diff on tricky audio (speech/music/silence)
* slot churn stress tests (many short segments)

---

## Bonus: one latency-specific improvement (not per-step, but user-visible)

You noted this already:

> `get_result()` blocks per-segment sequentially.

That can hurt request latency even if decoding is fast. Change to:

* wait for **all segment futures concurrently** (or completion queue by segment id),
* then reorder at the end.

This won’t reduce per-step overhead, but it **improves p95 request latency** and makes CB look faster immediately.
