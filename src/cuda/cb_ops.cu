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


    // =========================================================================
    // increment_cache_lengths_gpu: cl[0..n-1] += 1
    // =========================================================================
    //
    // Tiny kernel: one thread per element.  n is typically active_batch
    // (max_slots * beam_size, e.g. 20).  A single warp handles it.

    __global__ void increment_cache_lengths_kernel(int32_t* __restrict__ cl, int n) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
        cl[i] += 1;
    }

    void increment_cache_lengths_gpu(int32_t* cache_lengths, int n) {
      if (n <= 0)
        return;
      const int threads = std::min(256, n);
      const int blocks = (n + threads - 1) / threads;
      increment_cache_lengths_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
          cache_lengths, n);
    }


    // =========================================================================
    // add_scalar_int32_gpu: out[i] = in[i] + scalar, for i in [0, n)
    // =========================================================================

    __global__ void add_scalar_int32_kernel(
        int32_t* __restrict__ out,
        const int32_t* __restrict__ in,
        int32_t scalar,
        int n) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
        out[i] = in[i] + scalar;
    }

    void add_scalar_int32_gpu(int32_t* out, const int32_t* in, int32_t scalar, int n) {
      if (n <= 0)
        return;
      const int threads = std::min(256, n);
      const int blocks = (n + threads - 1) / threads;
      add_scalar_int32_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
          out, in, scalar, n);
    }


    // =========================================================================
    // fill_int32_gpu: out[0..n-1] = value
    // =========================================================================

    __global__ void fill_int32_kernel(int32_t* __restrict__ out, int32_t value, int n) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
        out[i] = value;
    }

    void fill_int32_gpu(int32_t* out, int32_t value, int n) {
      if (n <= 0)
        return;
      const int threads = std::min(256, n);
      const int blocks = (n + threads - 1) / threads;
      fill_int32_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
          out, value, n);
    }


    // =========================================================================
    // batch_timestamp_check_gpu: batched timestamp probability check
    // =========================================================================
    //
    // For each row in row_indices, computes:
    //   max_text = max(log_probs[row, 0 : num_text_tokens])
    //   ts_lse   = logsumexp(log_probs[row, num_text_tokens : num_text_tokens + num_ts_tokens])
    //   results[i] = (ts_lse > max_text) ? 1 : 0
    //
    // One thread-block per row. Shared memory reduction for max + logsumexp.
    // Replaces N × (Thrust::max + Thrust::logsumexp) = N × 3 GPU→CPU syncs
    // with 1 kernel launch + 1 small D2H copy.
    //
    // Grid:  num_rows blocks
    // Block: 256 threads

    template <typename T>
    __global__ void batch_timestamp_check_kernel(
        const T* __restrict__ log_probs,   // [batch_size, vocab_size]
        const int32_t* __restrict__ row_indices,  // [num_rows]
        int vocab_size,
        int num_text_tokens,
        int num_ts_tokens,
        int32_t* __restrict__ results)     // [num_rows]
    {
      // Shared memory for reductions: [0..255] for first reduction, [256..511] for second
      extern __shared__ char smem_raw[];
      float* smem = reinterpret_cast<float*>(smem_raw);

      const int row_idx = row_indices[blockIdx.x];
      const T* row_ptr = log_probs + static_cast<size_t>(row_idx) * vocab_size;

      // --- Phase 1: max over text tokens [0, num_text_tokens) ---
      float local_max_text = -1e30f;
      for (int i = threadIdx.x; i < num_text_tokens; i += blockDim.x) {
        float val = static_cast<float>(row_ptr[i]);
        if (val > local_max_text)
          local_max_text = val;
      }

      // Shared-memory reduction for max_text
      smem[threadIdx.x] = local_max_text;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          if (smem[threadIdx.x + stride] > smem[threadIdx.x])
            smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
      }
      float max_text = smem[0];

      // --- Phase 2: max over timestamp tokens for numerical stability ---
      const T* ts_ptr = row_ptr + num_text_tokens;
      float local_max_ts = -1e30f;
      for (int i = threadIdx.x; i < num_ts_tokens; i += blockDim.x) {
        float val = static_cast<float>(ts_ptr[i]);
        if (val > local_max_ts)
          local_max_ts = val;
      }

      smem[threadIdx.x] = local_max_ts;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          if (smem[threadIdx.x + stride] > smem[threadIdx.x])
            smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
      }
      float max_ts = smem[0];

      // --- Phase 3: sum of exp(x - max_ts) over timestamp tokens ---
      float local_sum_exp = 0.0f;
      for (int i = threadIdx.x; i < num_ts_tokens; i += blockDim.x) {
        float val = static_cast<float>(ts_ptr[i]);
        local_sum_exp += expf(val - max_ts);
      }

      smem[threadIdx.x] = local_sum_exp;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
          smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
      }

      // Thread 0: compute logsumexp and compare
      if (threadIdx.x == 0) {
        float ts_lse = logf(smem[0]) + max_ts;
        results[blockIdx.x] = (ts_lse > max_text) ? 1 : 0;
      }
    }

    void batch_timestamp_check_gpu(const void* log_probs,
                                   const int32_t* row_indices,
                                   int num_rows,
                                   int vocab_size,
                                   int num_text_tokens,
                                   int num_ts_tokens,
                                   int32_t* results,
                                   int elem_bytes) {
      if (num_rows <= 0)
        return;

      const int threads = 256;
      const size_t smem_bytes = threads * sizeof(float);

      if (elem_bytes == 2) {
        batch_timestamp_check_kernel<__half>
            <<<num_rows, threads, smem_bytes, get_cuda_stream()>>>(
            reinterpret_cast<const __half*>(log_probs),
            row_indices, vocab_size, num_text_tokens, num_ts_tokens, results);
      } else {
        batch_timestamp_check_kernel<float>
            <<<num_rows, threads, smem_bytes, get_cuda_stream()>>>(
            reinterpret_cast<const float*>(log_probs),
            row_indices, vocab_size, num_text_tokens, num_ts_tokens, results);
      }
    }

  }
}
