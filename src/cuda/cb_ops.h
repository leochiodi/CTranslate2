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

  }
}
