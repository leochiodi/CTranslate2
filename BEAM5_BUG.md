# Beam=5 Correctness Bug

## Symptom
With `beam_size=5` and 4 concurrent requests (same audio), some requests intermittently generate **446 tokens** (hitting `max_length`) instead of the correct **28 tokens**. The beam search fails to find/detect EOS.

## Reproduction
```bash
LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH python bench_compare.py
```
Typical output — some runs correct (2.7s, 28 tokens), others broken (25-36s, 446 tokens):
```
Run 1: 25.137s — req 1: 446 tokens, req 2: 446 tokens, req 3: 28, req 4: 28
Run 2: 2.714s  — req 5: 5 tokens, req 6: 28, req 7: 28, req 8: 28
Run 3: 36.685s — req 9: 446 tokens, req 10: 28, req 11: 28, req 12: 28
```
**beam=1 works perfectly** (1.09x ratio, always 28 tokens).

## Analysis So Far

### What's NOT the cause
- **Phase 1-6 optimizations**: beam=1 uses the same code paths (attention accumulation, position encoder, scatter cache) and works flawlessly.
- **Mid-decode slot filling**: With 4 requests and 4 slots, all slots fill initially — no mid-decode filling occurs.
- **Cross-run state**: `process()` creates fresh local state each invocation.

### Known bug found (but probably not the cause here)
`sample_from` is NOT updated for newly filled slots in the beam search path. After `fill_slot()` inserts a new request mid-decode, beam rows still contain the previous request's last tokens. **Fixed** in current code (rebuild `sample_from` for beam path same as greedy), but this only matters for mid-decode slot filling which doesn't happen in the benchmark.

### Where to investigate next
1. **Beam state corruption between slots**: When 4 slots with beam=5 run concurrently (total_batch=20), the beam selection logic at `continuous_decoding.cc:900-980` processes each slot independently. Check if one slot's beam reordering (gather_indices) corrupts another slot's KV cache or attention state.

2. **Identity gather skip (Phase 4)**: The optimization at line 1074-1092 skips beam reordering when gather_indices is identity. Verify this correctly handles the case where SOME slots need reordering but OTHERS don't (the identity check is global across all 20 rows, not per-slot).

3. **LogitsProcessor interaction**: `suppress_blank=True` and `suppress_tokens=[-1]` apply to all batch rows. Check if `ApplyTimestampRules` or other processors incorrectly affect beam rows across slot boundaries.

4. **Initial beam score setup**: Beams 1-4 start with score `-1e9f`. After LogSoftMax + TopK, verify these beams' candidates are properly deprioritized and don't contaminate slot 0's beam selection.

## Key Files
- `src/continuous_decoding.cc:870-1072` — beam search path
- `src/continuous_decoding.cc:1074-1092` — identity gather skip (Phase 4)
- `src/continuous_decoding.cc:900-980` — per-slot beam candidate selection
- `bench_compare.py` — reproduction script
- `bench_clean.py` — timing benchmark (8 runs)

## Config
- Model: whisper-large-v3 (float16, CUDA)
- 4 slots, beam_size=5, patience=1.0, max_length=448
- Audio: jfk.npy (11s), 4 identical requests
