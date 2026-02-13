#include "cuda/batch_copy.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // One block per copy descriptor, threads cooperatively copy 4 bytes at a time.
    __global__ void batch_copy_kernel(const CopyDescriptor* descs, int num_copies) {
      const int copy_idx = blockIdx.x;
      if (copy_idx >= num_copies)
        return;

      const CopyDescriptor& d = descs[copy_idx];

      // Copy 4 bytes at a time for coalesced access.
      const int* src4 = reinterpret_cast<const int*>(d.src);
      int* dst4 = reinterpret_cast<int*>(d.dst);
      const size_t n4 = d.num_bytes / 4;

      for (size_t i = threadIdx.x; i < n4; i += blockDim.x) {
        dst4[i] = src4[i];
      }

      // Handle trailing 0-3 bytes (single thread).
      if (threadIdx.x == 0) {
        const size_t tail_start = n4 * 4;
        for (size_t i = tail_start; i < d.num_bytes; ++i) {
          d.dst[i] = d.src[i];
        }
      }
    }

    static constexpr int MAX_BATCH_COPIES = 128;

    void batch_copy_async(const std::vector<CopyDescriptor>& copies) {
      if (copies.empty())
        return;

      const int num = static_cast<int>(copies.size());

      // Persistent thread-local device buffer for descriptors.
      // Leaked on thread exit (~3KB) — matches CTranslate2's thread_local CUDA resource pattern.
      static thread_local CopyDescriptor* d_descs = nullptr;
      if (!d_descs) {
        CUDA_CHECK(cudaMalloc(&d_descs, MAX_BATCH_COPIES * sizeof(CopyDescriptor)));
      }

      // Upload descriptors (sub-KB, effectively instant even from pageable memory).
      const int batch = (num < MAX_BATCH_COPIES) ? num : MAX_BATCH_COPIES;
      CUDA_CHECK(cudaMemcpyAsync(d_descs, copies.data(),
                                  batch * sizeof(CopyDescriptor),
                                  cudaMemcpyHostToDevice, get_cuda_stream()));

      dim3 block(256);
      dim3 grid(batch);
      batch_copy_kernel<<<grid, block, 0, get_cuda_stream()>>>(d_descs, batch);

      // If more than MAX_BATCH_COPIES, process remaining in another pass.
      if (num > MAX_BATCH_COPIES) {
        std::vector<CopyDescriptor> remaining(copies.begin() + MAX_BATCH_COPIES, copies.end());
        batch_copy_async(remaining);
      }
    }

  }
}
