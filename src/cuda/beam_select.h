#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace ctranslate2 {
namespace cuda {

// GPU beam selection kernel for continuous batching.
// One CUDA block per active slot; sequential candidate iteration.
// Replaces the CPU beam search loop by producing gather_indices,
// next_tokens, and updated beam state entirely on GPU.
//
// EOS hypothesis info (beam_id + normalized score) is written to
// output buffers for CPU to collect token histories later via async D2H.
void beam_select_async(
    const int32_t* topk_ids,          // [active_count, num_candidates]
    const float* topk_scores,         // [active_count, num_candidates]
    const float* beam_scores_in,      // [active_count * beam_size]
    const int32_t* beam_finished_in,  // [active_count * beam_size]
    const int32_t* num_finished_in,   // [active_count]
    const int32_t* steps,             // [active_count]
    const int32_t* prompt_lengths,    // [active_count]
    const int32_t* end_ids,           // [num_end_ids]
    int32_t* gather_indices,          // [active_count * beam_size] output
    int32_t* next_tokens,             // [active_count * beam_size] output
    float* beam_scores_out,           // [active_count * beam_size] output
    int32_t* beam_finished_out,       // [active_count * beam_size] output
    int32_t* slot_finished,           // [active_count] output
    int32_t* num_finished_out,        // [active_count] output
    int32_t* eos_beam_ids,            // [active_count * beam_size] output
    float* eos_scores,                // [active_count * beam_size] output
    int32_t* num_eos_per_slot,        // [active_count] output
    int32_t* slot_needs_gather,       // [active_count] output: 1 if gather is non-identity
    int32_t active_count,
    int32_t beam_size,
    int32_t num_candidates,
    int32_t vocab_size,
    int32_t max_length,
    int32_t max_candidates,
    float length_penalty,
    int32_t num_end_ids);

// Fill out[0..n-1] = 0, 1, ..., n-1 asynchronously on the given stream.
void fill_identity_async(int32_t* out, int32_t n, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace ctranslate2
