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
  struct ContinuousRequest {
    size_t id;
    StorageView encoder_output;           // [1, T_enc, D] — pre-encoded features
    std::vector<size_t> prompt_tokens;    // tokens for forward_prompt (sot, lang, task, ...)
    std::vector<size_t> start_tokens;     // tokens to start autoregressive decode
    bool use_timestamps = true;           // whether timestamp rules apply to this request
  };

  // Result from a completed slot.
  struct ContinuousResult {
    size_t request_id;
    DecodingResult result;
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
  };

  // Base class for logits processors in continuous batching mode.
  // Unlike the standard LogitsProcessor which takes a scalar step,
  // this interface receives the full slot state vector so it can
  // apply per-slot logic with per-slot steps and token histories.
  class ContinuousLogitsProcessor {
  public:
    virtual ~ContinuousLogitsProcessor() = default;

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
    void apply(const std::vector<SlotState>& slots,
               const std::vector<size_t>& active_slot_indices,
               dim_t beam_size,
               StorageView& logits,
               DisableTokens& disable_tokens) override;
  private:
    const std::vector<size_t> _ids;
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
  };

  // Callback that returns the next request if available.
  // Must be thread-safe (caller provides locking).
  using QueueProvider = std::function<std::optional<ContinuousRequest>()>;

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
    // Callback to initialize a slot's decoder state (encode + forward_prompt).
    // Receives: the decoder, a single-element state, the request.
    // Must fill the state's KV caches via forward_prompt.
    using SlotInitializer = std::function<void(
        layers::Decoder& decoder,
        layers::DecoderState& state,
        const ContinuousRequest& request)>;

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
        SlotInitializer slot_initializer,
        std::vector<std::shared_ptr<ContinuousLogitsProcessor>> logits_processors = {});

    // Process requests from the queue. Returns results for completed requests.
    // queue_provider: optional callback to pull additional requests mid-decode
    //   (e.g. from a thread-safe shared queue).
    std::vector<ContinuousResult> process(
        std::queue<ContinuousRequest>& request_queue,
        QueueProvider queue_provider = nullptr);

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
    SlotInitializer _slot_initializer;
    std::vector<std::shared_ptr<ContinuousLogitsProcessor>> _logits_processors;

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
  };

}
