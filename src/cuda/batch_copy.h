#pragma once

#include <vector>
#include <cstddef>

namespace ctranslate2 {
  namespace cuda {

    // Descriptor for a single device-to-device memory copy.
    struct CopyDescriptor {
      const char* src;
      char* dst;
      size_t num_bytes;
    };

    // Execute all copies in a single CUDA kernel launch.
    // All pointers must be device pointers on the current CUDA device.
    // Much lower overhead than N separate cudaMemcpyAsync calls.
    void batch_copy_async(const std::vector<CopyDescriptor>& copies);

  }
}
