#include "beam_select.h"

#include <cuda_runtime.h>
#include <math.h>

namespace ctranslate2 {
namespace cuda {

__global__ void beam_select_kernel(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_scores,
    const float* __restrict__ beam_scores_in,
    const int32_t* __restrict__ beam_finished_in,
    const int32_t* __restrict__ num_finished_in,
    const int32_t* __restrict__ steps,
    const int32_t* __restrict__ prompt_lengths,
    const int32_t* __restrict__ end_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ next_tokens,
    float* __restrict__ beam_scores_out,
    int32_t* __restrict__ beam_finished_out,
    int32_t* __restrict__ slot_finished,
    int32_t* __restrict__ num_finished_out,
    int32_t* __restrict__ eos_beam_ids,
    float* __restrict__ eos_scores,
    int32_t* __restrict__ num_eos_per_slot,
    int32_t* __restrict__ slot_needs_gather,
    int32_t beam_size,
    int32_t num_candidates,
    int32_t vocab_size,
    int32_t max_length,
    int32_t max_candidates,
    float length_penalty,
    int32_t num_end_ids)
{
  const int slot = blockIdx.x;
  if (threadIdx.x != 0)
    return;

  const int s_offset = slot * beam_size;
  const int step = steps[slot];
  const int prompt_len = prompt_lengths[slot];
  const float hyp_len = static_cast<float>(step - prompt_len + 1);

  // Cache input beam_finished in registers to avoid read-after-write hazard
  // when beam_finished_in and beam_finished_out alias the same buffer.
  int local_finished[32];  // max beam_size we'll ever use
  for (int b = 0; b < beam_size && b < 32; ++b)
    local_finished[b] = beam_finished_in[s_offset + b];

  int filled = 0;
  int n_finished = num_finished_in[slot];
  int n_eos = 0;

  for (int k = 0; k < num_candidates && filled < beam_size; ++k) {
    const int flat_id = topk_ids[slot * num_candidates + k];
    const float score = topk_scores[slot * num_candidates + k];
    const int beam_id = flat_id / vocab_size;
    const int word_id = flat_id % vocab_size;

    if (local_finished[beam_id])
      continue;

    bool is_eos = false;
    for (int e = 0; e < num_end_ids; ++e) {
      if (word_id == end_ids[e]) { is_eos = true; break; }
    }

    if (is_eos) {
      if (n_eos < beam_size) {
        eos_beam_ids[slot * beam_size + n_eos] = beam_id;
        eos_scores[slot * beam_size + n_eos] =
            score / powf(hyp_len, length_penalty);
        n_eos++;
      }
      // Mark this source beam as finished so it can't produce more EOS
      // candidates (within this step or, via beam_finished_out, in future steps).
      local_finished[beam_id] = 1;
      n_finished++;

      // CRITICAL: Fill an output position for this finished beam so it stays
      // marked finished in subsequent steps. Without this, the beam slot gets
      // filled by a non-EOS candidate with beam_finished_out=0, allowing the
      // same beam to hit EOS again on future steps and double-count n_finished.
      gather_indices[s_offset + filled] = s_offset + beam_id;
      next_tokens[s_offset + filled] = word_id;
      beam_scores_out[s_offset + filled] = score;
      beam_finished_out[s_offset + filled] = 1;
      filled++;

      if (n_finished >= max_candidates) {
        while (filled < beam_size) {
          gather_indices[s_offset + filled] = s_offset;
          next_tokens[s_offset + filled] = 0;
          beam_scores_out[s_offset + filled] = -1e9f;
          beam_finished_out[s_offset + filled] = 1;
          filled++;
        }
        break;
      }
      continue;
    }

    gather_indices[s_offset + filled] = s_offset + beam_id;
    next_tokens[s_offset + filled] = word_id;
    beam_scores_out[s_offset + filled] = score;
    beam_finished_out[s_offset + filled] = 0;
    filled++;
  }

  while (filled < beam_size) {
    gather_indices[s_offset + filled] = s_offset;
    next_tokens[s_offset + filled] = 0;
    beam_scores_out[s_offset + filled] = -1e9f;
    beam_finished_out[s_offset + filled] = 1;
    filled++;
  }

  num_finished_out[slot] = n_finished;
  num_eos_per_slot[slot] = n_eos;

  // Check if gather_indices is identity for this slot.
  int needs = 0;
  for (int b = 0; b < beam_size; ++b) {
    if (gather_indices[s_offset + b] != s_offset + b) { needs = 1; break; }
  }
  slot_needs_gather[slot] = needs;

  slot_finished[slot] =
      (n_finished >= max_candidates || step + 1 >= max_length) ? 1 : 0;
}

void beam_select_async(
    const int32_t* topk_ids,
    const float* topk_scores,
    const float* beam_scores_in,
    const int32_t* beam_finished_in,
    const int32_t* num_finished_in,
    const int32_t* steps,
    const int32_t* prompt_lengths,
    const int32_t* end_ids,
    int32_t* gather_indices,
    int32_t* next_tokens,
    float* beam_scores_out,
    int32_t* beam_finished_out,
    int32_t* slot_finished,
    int32_t* num_finished_out,
    int32_t* eos_beam_ids,
    float* eos_scores,
    int32_t* num_eos_per_slot,
    int32_t* slot_needs_gather,
    int32_t active_count,
    int32_t beam_size,
    int32_t num_candidates,
    int32_t vocab_size,
    int32_t max_length,
    int32_t max_candidates,
    float length_penalty,
    int32_t num_end_ids)
{
  if (active_count <= 0)
    return;

  beam_select_kernel<<<active_count, 1>>>(
      topk_ids, topk_scores,
      beam_scores_in, beam_finished_in,
      num_finished_in, steps, prompt_lengths, end_ids,
      gather_indices, next_tokens,
      beam_scores_out, beam_finished_out,
      slot_finished, num_finished_out,
      eos_beam_ids, eos_scores, num_eos_per_slot,
      slot_needs_gather,
      beam_size, num_candidates, vocab_size,
      max_length, max_candidates,
      length_penalty, num_end_ids);
}

__global__ void fill_identity_kernel(int32_t* out, int32_t n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) out[idx] = idx;
}

void fill_identity_async(int32_t* out, int32_t n, cudaStream_t stream) {
  if (n <= 0) return;
  fill_identity_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, n);
}

}  // namespace cuda
}  // namespace ctranslate2
