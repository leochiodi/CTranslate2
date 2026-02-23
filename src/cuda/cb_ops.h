#pragma once

#include <cstddef>
#include <cstdint>

namespace ctranslate2 {
  namespace cuda {

    // Scatter-write K/V cache step using a single GPU kernel.
    // cache:     device ptr, shape [batch*heads, cache_time, d] (flattened over batch & heads).
    // step:      device ptr, shape [batch*heads, 1, d] (flattened, contiguous).
    // positions: device ptr, [batch] INT32 — write position per batch row.
    // batch, heads, cache_time, d: dimensions.
    // elem_bytes: sizeof one element (2 for float16, 4 for float32).
    void scatter_cache_step_gpu(void* cache,
                                const void* step,
                                const int32_t* positions,  // device ptr [batch]
                                int batch,
                                int heads,
                                int cache_time,
                                int d,
                                int elem_bytes);

    // Add per-element position encodings on GPU.
    // input:     device ptr [batch, time, depth]  (float16 or float32)
    // encodings: device ptr [max_positions, depth] (same type)
    // offsets:   device ptr [batch] INT32
    // This computes: input[b, t, :] += encodings[offsets[b] + t, :]
    void add_position_encoding_gpu(void* input,
                                   const void* encodings,
                                   const int32_t* offsets,  // device ptr [batch]
                                   int batch,
                                   int time,
                                   int depth,
                                   int elem_bytes);

    // Increment cache_lengths for active batch rows: cl[0..n-1] += 1.
    // cache_lengths: device ptr [total_batch] INT32 (only first n elements modified).
    void increment_cache_lengths_gpu(int32_t* cache_lengths, int n);

    // Element-wise add scalar: out[i] = in[i] + scalar, for i in [0, n).
    // out and in may alias (in-place safe).
    void add_scalar_int32_gpu(int32_t* out, const int32_t* in, int32_t scalar, int n);

    // Fill n elements with a constant value: out[0..n-1] = value.
    void fill_int32_gpu(int32_t* out, int32_t value, int n);

    // Batched timestamp probability check for Whisper continuous batching.
    // Replaces per-row should_sample_timestamp() calls that each do 2-3 synchronous
    // Thrust reductions, with a single kernel launch + D2H copy.
    //
    // log_probs:       device ptr [batch_size, vocab_size], after LogSoftMax
    // row_indices:     device ptr [num_rows] INT32, which rows to check
    // num_rows:        number of rows to check
    // vocab_size:      second dimension of log_probs
    // num_text_tokens: timestamp_begin_id (text tokens are [0, num_text_tokens))
    // num_ts_tokens:   timestamp_end_id - timestamp_begin_id + 1
    // results:         device ptr [num_rows] INT32, output: 1 if should sample timestamp
    // elem_bytes:      2 for float16, 4 for float32
    void batch_timestamp_check_gpu(const void* log_probs,
                                   const int32_t* row_indices,
                                   int num_rows,
                                   int vocab_size,
                                   int num_text_tokens,
                                   int num_ts_tokens,
                                   int32_t* results,
                                   int elem_bytes);

    // Expand per-element lengths into per-head lengths: out[i] = in[i / stride].
    // total = batch * stride (e.g. batch * num_heads).
    void expand_lengths_gpu(int32_t* out, const int32_t* in, int stride, int total);

  }
}
