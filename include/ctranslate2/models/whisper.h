#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "ctranslate2/continuous_decoding.h"
#include "ctranslate2/generation.h"
#include "ctranslate2/layers/whisper.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"
#include "ctranslate2/sampling.h"

namespace ctranslate2 {
  namespace models {

    struct WhisperOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 5;

      // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
      // The decoding will continue until beam_size*patience hypotheses are finished.
      float patience = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Include log probs of each token in the result
      bool return_logits_vocab = false;

      // Include the probability of the no speech token in the result.
      bool return_no_speech_prob = false;

      // Maximum index of the first predicted timestamp.
      size_t max_initial_timestamp_index = 50;

      // Suppress blank outputs at the beginning of the sampling.
      bool suppress_blank = true;

      // List of token IDs to suppress.
      // -1 will suppress a default set of symbols as defined in the model config.json file.
      std::vector<int> suppress_tokens = {-1};
    };

    struct WhisperGenerationResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;
      std::vector<std::vector<StorageView>> logits;
      float no_speech_prob = 0;
      StorageView attention_weights;  // [num_alignment_heads, gen_steps, enc_time] (optional)

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    struct WhisperAlignmentResult {
      std::vector<std::pair<dim_t, dim_t>> alignments;
      std::vector<float> text_token_probs;
    };

    class WhisperModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class WhisperReplica : public ModelReplica {
    public:
      static std::unique_ptr<WhisperReplica> create_from_model(const Model& model);

      WhisperReplica(const std::shared_ptr<const WhisperModel>& model);

      bool is_multilingual() const {
        return _is_multilingual;
      }

      size_t n_mels() const {
        return _n_mels;
      }

      size_t num_languages() const {
        return _num_languages;
      }

      StorageView encode(StorageView features, const bool to_cpu);

      std::vector<WhisperGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<std::string>>& prompts,
               const WhisperOptions& options);

      std::vector<WhisperGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<size_t>>& prompts,
               const WhisperOptions& options);

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(StorageView features);

      std::vector<WhisperAlignmentResult>
      align(StorageView features,
            const std::vector<size_t>& start_sequence,
            const std::vector<std::vector<size_t>>& text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width);

    private:
      const std::shared_ptr<const WhisperModel> _model;
      const std::unique_ptr<layers::WhisperEncoder> _encoder;
      const std::unique_ptr<layers::WhisperDecoder> _decoder;

      size_t _sot_id;
      size_t _eot_id;
      size_t _no_timestamps_id;
      size_t _no_speech_id;
      size_t _n_mels;
      size_t _num_languages;
      bool _is_multilingual;

      StorageView maybe_encode(StorageView features);
    };

    class Whisper : public ReplicaPool<WhisperReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      bool is_multilingual() const;
      size_t n_mels() const;
      size_t num_languages() const;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

      std::vector<std::future<WhisperGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<std::string>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<WhisperGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<size_t>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<std::vector<std::pair<std::string, float>>>>
      detect_language(const StorageView& features);

      std::vector<std::future<WhisperAlignmentResult>>
      align(const StorageView& features,
            std::vector<size_t> start_sequence,
            std::vector<std::vector<size_t>> text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width);

    };

    // Continuous batching engine for Whisper.
    //
    // Runs a background worker thread that consumes a queue of transcription
    // requests. Each request gets a "slot" in a persistent decode batch.
    // When a slot finishes (EOT or max_length), its result is stored and
    // the slot is recycled for the next queued request.
    //
    // Usage:
    //   WhisperContinuousBatcher batcher(model_path, /*max_slots=*/4);
    //   batcher.start();
    //   size_t id = batcher.submit(features, prompt);
    //   auto result = batcher.get_result(id);  // blocks until ready
    //   batcher.stop();
    class WhisperContinuousBatcher {
    public:
      WhisperContinuousBatcher(
        const std::string& model_path,
        size_t max_slots,
        const WhisperOptions& default_options = {},
        Device device = Device::CUDA,
        int device_index = 0,
        ComputeType compute_type = ComputeType::DEFAULT,
        size_t num_encoders = 1);

      ~WhisperContinuousBatcher();

      // Non-copyable, non-movable.
      WhisperContinuousBatcher(const WhisperContinuousBatcher&) = delete;
      WhisperContinuousBatcher& operator=(const WhisperContinuousBatcher&) = delete;

      // Submit a request. Thread-safe, can be called from any thread.
      // features: [1, n_mels, T] audio features (raw or pre-encoded).
      // prompt: full prompt token IDs (e.g. [sot, lang, task, notimestamps]).
      // Returns a unique request ID.
      size_t submit(StorageView features, std::vector<size_t> prompt);

      // Block until the result for request_id is ready, then return it.
      WhisperGenerationResult get_result(size_t request_id);

      // Non-blocking check if a result is ready.
      bool is_ready(size_t request_id) const;

      // Start the background worker thread.
      void start();

      // Signal stop and join the worker thread.
      // Completes any in-flight decode batch before returning.
      void stop();

      // Synchronous operations using a dedicated replica (thread-safe with worker_loop).
      StorageView encode(StorageView features, bool to_cpu = false);

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(StorageView features);

      // Submit a language detection request through the encoder thread (no mutex).
      // Returns a unique request ID.
      size_t submit_langdetect(StorageView features);

      // Block until the language detection result for request_id is ready.
      std::vector<std::pair<std::string, float>> get_langdetect_result(size_t request_id);

      std::vector<WhisperAlignmentResult>
      align(StorageView features,
            const std::vector<size_t>& start_sequence,
            const std::vector<std::vector<size_t>>& text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width = 7);

      // Compute alignments from pre-captured attention weights (no decoder pass needed).
      // attention_weights: [num_alignment_heads, gen_steps, enc_time] from a single request.
      // text_tokens: the text tokens for this request (excluding start sequence and EOT).
      // num_frames: number of audio frames for this request.
      // sot_sequence_length: length of the start-of-transcript sequence (sot + lang + task + notimestamps).
      WhisperAlignmentResult
      align_from_attention(StorageView attention_weights,
                           const std::vector<size_t>& text_tokens,
                           size_t num_frames,
                           dim_t sot_sequence_length,
                           dim_t median_filter_width = 7);

      bool is_multilingual() const;

    private:
      std::shared_ptr<const WhisperModel> _model;
      std::vector<std::unique_ptr<layers::WhisperEncoder>> _encoders;
      std::unique_ptr<layers::WhisperDecoder> _decoder;
      std::vector<std::unique_ptr<layers::WhisperDecoder>> _prep_decoders;

      size_t _max_slots;
      size_t _num_encoders;
      WhisperOptions _options;

      size_t _sot_id;
      size_t _eot_id;
      size_t _no_timestamps_id;
      size_t _no_speech_id;

      // Internal request type.
      // Raw requests (from submit) carry features + prompt.
      // After encoder_loop: features consumed, prepared_state populated.
      struct Request {
        size_t id;
        StorageView features;
        std::vector<size_t> prompt;
        // Set by encoder_loop after encode + forward_prompt:
        layers::DecoderState prepared_state;
        dim_t prompt_length = 0;
        std::vector<size_t> start_tokens;
        bool use_timestamps = true;
      };

      // Language detection request type.
      struct LangDetectRequest {
        size_t id;
        StorageView features;
      };

      // Raw request queue (un-encoded features from submit()).
      std::queue<Request> _raw_queue;
      // Language detection queue (shares mutex/cv with _raw_queue).
      std::queue<LangDetectRequest> _langdetect_queue;
      mutable std::mutex _raw_queue_mutex;
      std::condition_variable _raw_queue_cv;

      // Language detection results.
      std::unordered_map<size_t, std::vector<std::pair<std::string, float>>> _langdetect_results;
      mutable std::mutex _langdetect_results_mutex;
      std::condition_variable _langdetect_results_cv;

      // Prepared request queue (encoded + forward_prompt KV caches ready).
      std::queue<Request> _encoded_queue;
      mutable std::mutex _encoded_queue_mutex;
      std::condition_variable _encoded_queue_cv;

      // Results storage.
      std::unordered_map<size_t, WhisperGenerationResult> _results;
      std::string _worker_error;  // set on worker exception; checked in get_result()
      mutable std::mutex _results_mutex;
      std::condition_variable _results_cv;

      // Encoder threads: encode features from _raw_queue → _encoded_queue.
      std::vector<std::thread> _encoder_threads;
      // Worker thread: single decoder from _encoded_queue.
      std::thread _worker;
      std::atomic<bool> _running{false};
      std::atomic<size_t> _next_id{0};

      // Dedicated replica for synchronous methods (encode/detect_language/align).
      // Shares model weights with the batcher but has its own encoder/decoder,
      // so concurrent use with worker_loop is safe.
      std::unique_ptr<WhisperReplica> _replica;
      mutable std::mutex _replica_mutex;

      void encoder_loop(size_t encoder_idx);
      void worker_loop();
    };

  }
}
