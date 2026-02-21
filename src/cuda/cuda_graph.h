#pragma once

#ifdef CT2_WITH_CUDA

#include <functional>
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // Thread-local flag: true when inside CudaGraphWrapper::capture().
    // Checked by position encoder and scatter_cache_step to force GPU kernel paths,
    // avoiding CPU-dependent branching that would bake stale values into the graph.
    inline thread_local bool g_cuda_graph_capturing = false;

    // Wraps CUDA graph capture/replay for a repeatable GPU workload.
    // Usage:
    //   if (!graph.is_valid())
    //     graph.capture(stream, [&]{ decoder_forward(); });
    //   graph.replay(stream);
    class CudaGraphWrapper {
    public:
      CudaGraphWrapper() = default;

      ~CudaGraphWrapper() {
        release();
      }

      // Non-copyable.
      CudaGraphWrapper(const CudaGraphWrapper&) = delete;
      CudaGraphWrapper& operator=(const CudaGraphWrapper&) = delete;

      // Movable.
      CudaGraphWrapper(CudaGraphWrapper&& other) noexcept
        : _graph(other._graph)
        , _exec(other._exec)
        , _valid(other._valid)
      {
        other._graph = nullptr;
        other._exec = nullptr;
        other._valid = false;
      }

      CudaGraphWrapper& operator=(CudaGraphWrapper&& other) noexcept {
        if (this != &other) {
          release();
          _graph = other._graph;
          _exec = other._exec;
          _valid = other._valid;
          other._graph = nullptr;
          other._exec = nullptr;
          other._valid = false;
        }
        return *this;
      }

      // Capture: begin stream capture, run fn, end capture, instantiate.
      // Sets g_cuda_graph_capturing=true during fn() so that GPU kernel paths
      // are forced (avoiding CPU-dependent branching in position encoder, scatter, etc.).
      void capture(cudaStream_t stream, std::function<void()> fn) {
        release();

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        g_cuda_graph_capturing = true;
        try {
          fn();
        } catch (...) {
          g_cuda_graph_capturing = false;
          cudaGraph_t abandoned = nullptr;
          cudaStreamEndCapture(stream, &abandoned);
          if (abandoned) cudaGraphDestroy(abandoned);
          throw;
        }
        g_cuda_graph_capturing = false;
        CUDA_CHECK(cudaStreamEndCapture(stream, &_graph));
        CUDA_CHECK(cudaGraphInstantiate(&_exec, _graph, 0));
        _valid = true;
      }

      // Replay the captured graph on the given stream.
      void replay(cudaStream_t stream) {
        CUDA_CHECK(cudaGraphLaunch(_exec, stream));
      }

      bool is_valid() const { return _valid; }

      void invalidate() {
        // Mark invalid but keep resources — they'll be freed on next capture().
        _valid = false;
      }

    private:
      void release() {
        if (_exec) {
          cudaGraphExecDestroy(_exec);
          _exec = nullptr;
        }
        if (_graph) {
          cudaGraphDestroy(_graph);
          _graph = nullptr;
        }
        _valid = false;
      }

      cudaGraph_t _graph = nullptr;
      cudaGraphExec_t _exec = nullptr;
      bool _valid = false;
    };

  }  // namespace cuda
}  // namespace ctranslate2

#endif  // CT2_WITH_CUDA
