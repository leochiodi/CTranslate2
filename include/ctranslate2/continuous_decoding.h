#pragma once

#include <functional>
#include <optional>
#include <queue>
#include <vector>

#include "ctranslate2/decoding.h"
#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  // A request to be processed by the continuous batching engine.
  // The prepared_state contains the decoder state after forward_prompt
  // (KV caches initialized with prompt tokens, cross-attention projected).
  // NOT beam-replicated — fill_slot handles that.
  struct ContinuousRequest {
    size_t id;
    layers::DecoderState prepared_state;  // after forward_prompt, single-element
    dim_t prompt_length = 0;
    std::vector<size_t> start_tokens;     // tokens to start autoregressive decode
    bool use_timestamps = true;           // whether timestamp rules apply to this request
    bool beam_replicated = false;         // true if self-KV caches already tiled to beam_size
  };

  // Result from a completed slot.
  struct ContinuousResult {
    size_t request_id;
    DecodingResult result;
    StorageView attention_weights;  // [num_alignment_heads, gen_steps, enc_time] (optional)
  };

  // State of a single slot in the continuous batch.
  struct SlotState {
    size_t request_id = 0;
    dim_t step = 0;                       // current decode step (absolute: prompt_length + generated)
    dim_t prompt_length = 0;              // number of prompt tokens that filled the KV cache
    bool active = false;
    bool use_timestamps = true;           // whether timestamp rules apply to this slot
    size_t start_id = 0;                  // first token for decode

    // Greedy state (beam_size == 1).
    std::vector<size_t> generated_tokens;
    float cum_score = 0.0f;

    // Beam search state (beam_size > 1).
    dim_t beam_size = 1;
    std::vector<std::vector<size_t>> beam_tokens;  // [B][time] per beam
    std::vector<float> beam_scores;                 // [B] cumulative log-prob
    std::vector<bool> beam_finished;                // [B] which beams hit EOT
    size_t num_finished_beams = 0;

    struct Hypothesis {
      std::vector<size_t> tokens;
      float score;
    };
    std::vector<Hypothesis> finished_hypotheses;

    // Helper: generation step (0-based, relative to start of generation).
    dim_t gen_step() const { return step - prompt_length; }

    // Index into accumulated_attention's time dimension where this slot's
    // attention starts (used when slots are recycled mid-decode).
    dim_t attention_step_offset = 0;
  };

  // Base class for logits processors in continuous batching mode.
  // Unlike the standard LogitsProcessor which takes a scalar step,
  // this interface receives the full slot state vector so it can
  // apply per-slot logic with per-slot steps and token histories.
  class ContinuousLogitsProcessor {
  public:
    virtual ~ContinuousLogitsProcessor() = default;

    // One-time initialization after total_batch/vocab_size are known.
    // Called before the decode loop begins. Default: no-op.
    virtual void init(dim_t /*total_batch*/, dim_t /*beam_size*/,
                      dim_t /*vocab_size*/, Device /*device*/) {}

    // Apply logits processing for the current batch.
    // slots: all slot states (max_slots elements).
    // active_slot_indices: indices of currently active slots.
    // beam_size: number of beams per slot (batch row = slot_idx * beam_size + beam).
    // logits: [max_slots * beam_size, vocab_size] — modify in place.
    // disable_tokens: helper to disable specific tokens per batch row.
    virtual void apply(const std::vector<SlotState>& slots,
                       const std::vector<size_t>& active_slot_indices,
                       dim_t beam_size,
                       StorageView& logits,
                       DisableTokens& disable_tokens) = 0;
  };

  // Suppress specific token IDs for all active batch rows.
  class ContinuousSuppressTokens : public ContinuousLogitsProcessor {
  public:
    explicit ContinuousSuppressTokens(std::vector<size_t> ids);
    void init(dim_t total_batch, dim_t beam_size,
              dim_t vocab_size, Device device) override;
    void apply(const std::vector<SlotState>& slots,
               const std::vector<size_t>& active_slot_indices,
               dim_t beam_size,
               StorageView& logits,
               DisableTokens& disable_tokens) override;
  private:
    const std::vector<size_t> _ids;
    StorageView _gpu_indices;   // pre-computed [total_batch * _ids.size()] INT32 on GPU
    bool _precomputed = false;
  };

  // Suppress blank tokens at generation step 0 only.
  class ContinuousSuppressBlank : public ContinuousLogitsProcessor {
  public:
    explicit ContinuousSuppressBlank(std::vector<size_t> ids);
    void apply(const std::vector<SlotState>& slots,
               const std::vector<size_t>& active_slot_indices,
               dim_t beam_size,
               StorageView& logits,
               DisableTokens& disable_tokens) override;
  private:
    const std::vector<size_t> _ids;
  };

  // Whisper timestamp rules adapted for continuous batching.
  // Port of ApplyTimestampRules with per-slot step and token history.
  class ContinuousTimestampRules : public ContinuousLogitsProcessor {
  public:
    ContinuousTimestampRules(size_t eot_id,
                             size_t no_timestamps_id,
                             size_t timestamp_begin_id,
                             size_t timestamp_end_id,
                             size_t max_initial_timestamp_id);
    void init(dim_t total_batch, dim_t beam_size,
              dim_t vocab_size, Device device) override;
    void apply(const std::vector<SlotState>& slots,
               const std::vector<size_t>& active_slot_indices,
               dim_t beam_size,
               StorageView& logits,
               DisableTokens& disable_tokens) override;
  private:
    const size_t _eot_id;
    const size_t _no_timestamps_id;
    const size_t _timestamp_begin_id;
    const size_t _timestamp_end_id;
    const size_t _max_initial_timestamp_id;

    // Pre-allocated buffers for apply() (avoids per-step GPU allocation).
    StorageView _log_probs_buf;       // [total_batch, vocab_size] — reused across steps
    StorageView _row_indices_buf;     // [total_batch] INT32 on GPU
    StorageView _results_buf;         // [total_batch] INT32 on GPU
    bool _initialized = false;
  };

  // Callback that returns the next request if available.
  // Must be thread-safe (caller provides locking).
  using QueueProvider = std::function<std::optional<ContinuousRequest>()>;

  // Callback invoked when a slot completes, delivering the result immediately.
  using ResultCallback = std::function<void(ContinuousResult)>;

  // Continuous batching decode engine.
  //
  // Instead of decoding a fixed batch until all elements finish,
  // this engine processes requests from a queue. When a slot finishes
  // (EOT or max_length), it is replaced with a new request from the queue.
  //
  // Always decodes with max_slots elements to maintain consistent batch
  // dimensions with the KV cache. Inactive slots receive dummy tokens
  // and their outputs are ignored.
  class ContinuousDecodingEngine {
  public:
    ContinuousDecodingEngine(
        layers::Decoder& decoder,
        size_t max_slots,
        dim_t beam_size,
        float length_penalty,
        float patience,
        size_t num_hypotheses,
        const std::vector<size_t>& end_ids,
        dim_t max_length,
        const Sampler& sampler,
        std::vector<std::shared_ptr<ContinuousLogitsProcessor>> logits_processors = {},
        bool capture_attention = false);

    // Process requests from the queue.
    // queue_provider: optional callback to pull additional requests mid-decode
    //   (e.g. from a thread-safe shared queue).
    // result_callback: invoked immediately when each slot completes, delivering
    //   results incrementally instead of waiting for all slots to drain.
    void process(
        std::queue<ContinuousRequest>& request_queue,
        QueueProvider queue_provider = nullptr,
        ResultCallback result_callback = nullptr);

  private:
    layers::Decoder& _decoder;
    const size_t _max_slots;
    const dim_t _beam_size;
    const float _length_penalty;
    const size_t _max_candidates;  // ceil(beam_size * patience)
    const size_t _num_hypotheses;
    const std::vector<size_t> _end_ids;
    const dim_t _max_length;
    const Sampler& _sampler;
    std::vector<std::shared_ptr<ContinuousLogitsProcessor>> _logits_processors;
    const bool _capture_attention;

    // Fill a free slot with a new request. Returns true if a request was available.
    bool fill_slot(size_t slot_idx,
                   std::queue<ContinuousRequest>& request_queue,
                   QueueProvider& queue_provider,
                   std::vector<SlotState>& slots,
                   layers::DecoderState& batch_state);

    // Scatter a single-element state into the batch state at a specific slot.
    void set_batch_slot(layers::DecoderState& batch_state,
                        dim_t slot_idx,
                        const layers::DecoderState& single_state,
                        dim_t target_cache_len);

    // Zero-pad a cache tensor along the time dimension to reach target_len.
    static void pad_cache(StorageView& cache, dim_t target_len, dim_t time_dim);

    // Get the current max cache length across all active slots.
    dim_t max_cache_length(const layers::DecoderState& batch_state) const;

    // Resize all batch_state tensors to target_count active slots.
    // Beam-level tensors get dim(0) = target_count * beam_size,
    // slot-level (memory*) tensors get dim(0) = target_count.
    // Skips cache_lengths, accumulated_attention, and _retain_memory markers.
    void resize_batch_state(layers::DecoderState& batch_state,
                            dim_t target_count) const;

    // Copy all tensor rows from src_slot position to dst_slot position
    // within batch_state (for defragmentation).
    void copy_slot_data(layers::DecoderState& batch_state,
                        size_t src_slot, size_t dst_slot) const;

    // Compact active slots to positions 0..N-1 by moving data forward.
    // Returns the new active_count.
    size_t defragment_slots(std::vector<SlotState>& slots,
                            layers::DecoderState& batch_state) const;
  };

}
