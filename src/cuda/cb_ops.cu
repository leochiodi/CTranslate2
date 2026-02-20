#include "cuda/cb_ops.h"
#include "cuda/utils.h"

#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace cuda {

    // =========================================================================
    // scatter_cache_step_gpu: scatter-write decode step into K/V cache
    // =========================================================================
    //
    // Each thread-block handles one (batch_row, head) pair.
    // Threads cooperatively copy `d` elements from step to cache at the
    // designated position, using 4-byte (int) copies for coalescing.
    //
    // Grid:  (batch * heads)
    // Block: min(256, ceil(d * elem_bytes / 4))

    __global__ void scatter_cache_step_kernel(
        char* __restrict__ cache,
        const char* __restrict__ step,
        const int32_t* __restrict__ positions,
        int batch,
        int heads,
        int cache_time,
        int d,
        int elem_bytes)
    {
      const int bh = blockIdx.x;   // flattened (batch, head) index
      if (bh >= batch * heads)
        return;

      const int b = bh / heads;

      const int pos = positions[b];
      const int row_bytes = d * elem_bytes;

      // Source: step is [batch*heads, 1, d] contiguous
      const char* src = step + static_cast<size_t>(bh) * row_bytes;

      // Dest: cache is [batch*heads, cache_time, d]
      char* dst = cache + (static_cast<size_t>(bh) * cache_time + pos)
                  * static_cast<size_t>(row_bytes);

      // Copy using 4-byte words for coalescing.
      const int n4 = row_bytes / 4;
      const int* src4 = reinterpret_cast<const int*>(src);
      int* dst4 = reinterpret_cast<int*>(dst);

      for (int i = threadIdx.x; i < n4; i += blockDim.x)
        dst4[i] = src4[i];

      // Trailing bytes (0-3), single thread.
      if (threadIdx.x == 0) {
        for (int i = n4 * 4; i < row_bytes; ++i)
          dst[i] = src[i];
      }
    }

    void scatter_cache_step_gpu(void* cache,
                                const void* step,
                                const int32_t* positions,
                                int batch,
                                int heads,
                                int cache_time,
                                int d,
                                int elem_bytes) {
      const int num_blocks = batch * heads;
      if (num_blocks == 0)
        return;

      const int row_bytes = d * elem_bytes;
      const int threads = std::min(256, (row_bytes + 3) / 4);

      scatter_cache_step_kernel<<<num_blocks, threads, 0, get_cuda_stream()>>>(
          reinterpret_cast<char*>(cache),
          reinterpret_cast<const char*>(step),
          positions,
          batch, heads, cache_time, d, elem_bytes);
    }


    // =========================================================================
    // add_position_encoding_gpu: fused per-element position encoding add
    // =========================================================================
    //
    // Computes: input[b, t, :] += encodings[offsets[b] + t, :]
    //
    // For float16: uses __half2 vectorized adds (2 elements per op).
    // For float32: uses float adds.
    //
    // Grid:  (batch * time)
    // Block: 256
    // Each block handles one (b, t) row of `depth` elements.

    __global__ void add_position_encoding_f16_kernel(
        __half* __restrict__ input,
        const __half* __restrict__ encodings,
        const int32_t* __restrict__ offsets,
        int batch,
        int time,
        int depth)
    {
      const int bt = blockIdx.x;
      if (bt >= batch * time)
        return;

      const int b = bt / time;
      const int t = bt % time;
      const int enc_row = offsets[b] + t;

      __half* inp_row = input + static_cast<size_t>(bt) * depth;
      const __half* enc_row_ptr = encodings + static_cast<size_t>(enc_row) * depth;

      // Vectorized half2 path.
      const int depth2 = depth / 2;
      __half2* inp2 = reinterpret_cast<__half2*>(inp_row);
      const __half2* enc2 = reinterpret_cast<const __half2*>(enc_row_ptr);

      for (int i = threadIdx.x; i < depth2; i += blockDim.x)
        inp2[i] = __hadd2(inp2[i], enc2[i]);

      // Handle odd trailing element.
      if (depth % 2 != 0 && threadIdx.x == 0)
        inp_row[depth - 1] = __hadd(inp_row[depth - 1], enc_row_ptr[depth - 1]);
    }

    __global__ void add_position_encoding_f32_kernel(
        float* __restrict__ input,
        const float* __restrict__ encodings,
        const int32_t* __restrict__ offsets,
        int batch,
        int time,
        int depth)
    {
      const int bt = blockIdx.x;
      if (bt >= batch * time)
        return;

      const int b = bt / time;
      const int t = bt % time;
      const int enc_row = offsets[b] + t;

      float* inp_row = input + static_cast<size_t>(bt) * depth;
      const float* enc_row_ptr = encodings + static_cast<size_t>(enc_row) * depth;

      for (int i = threadIdx.x; i < depth; i += blockDim.x)
        inp_row[i] += enc_row_ptr[i];
    }

    void add_position_encoding_gpu(void* input,
                                   const void* encodings,
                                   const int32_t* offsets,
                                   int batch,
                                   int time,
                                   int depth,
                                   int elem_bytes) {
      const int num_blocks = batch * time;
      if (num_blocks == 0)
        return;

      const int threads = 256;

      if (elem_bytes == 2) {
        add_position_encoding_f16_kernel<<<num_blocks, threads, 0, get_cuda_stream()>>>(
            reinterpret_cast<__half*>(input),
            reinterpret_cast<const __half*>(encodings),
            offsets,
            batch, time, depth);
      } else {
        add_position_encoding_f32_kernel<<<num_blocks, threads, 0, get_cuda_stream()>>>(
            reinterpret_cast<float*>(input),
            reinterpret_cast<const float*>(encodings),
            offsets,
            batch, time, depth);
      }
    }

  }
}
