#include "ctranslate2/continuous_decoding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "ctranslate2/devices.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/utils.h"
#include "dispatch.h"

#ifdef CT2_WITH_CUDA
#include "cuda/batch_copy.h"
#endif

namespace ctranslate2 {

  // --- ContinuousLogitsProcessor implementations ---

  ContinuousSuppressTokens::ContinuousSuppressTokens(std::vector<size_t> ids)
    : _ids(std::move(ids))
  {
  }

  void ContinuousSuppressTokens::apply(
      const std::vector<SlotState>& slots,
      const std::vector<size_t>& active_slot_indices,
      dim_t beam_size,
      StorageView& /*logits*/,
      DisableTokens& disable_tokens) {
    for (const size_t s : active_slot_indices) {
      for (dim_t b = 0; b < beam_size; ++b) {
        const dim_t row = static_cast<dim_t>(s) * beam_size + b;
        for (const size_t id : _ids)
          disable_tokens.add(row, id);
      }
    }
  }

  ContinuousSuppressBlank::ContinuousSuppressBlank(std::vector<size_t> ids)
    : _ids(std::move(ids))
  {
  }

  void ContinuousSuppressBlank::apply(
      const std::vector<SlotState>& slots,
      const std::vector<size_t>& active_slot_indices,
      dim_t beam_size,
      StorageView& /*logits*/,
      DisableTokens& disable_tokens) {
    for (const size_t s : active_slot_indices) {
      if (slots[s].gen_step() != 0)
        continue;
      for (dim_t b = 0; b < beam_size; ++b) {
        const dim_t row = static_cast<dim_t>(s) * beam_size + b;
        for (const size_t id : _ids)
          disable_tokens.add(row, id);
      }
    }
  }

  ContinuousTimestampRules::ContinuousTimestampRules(
      size_t eot_id,
      size_t no_timestamps_id,
      size_t timestamp_begin_id,
      size_t timestamp_end_id,
      size_t max_initial_timestamp_id)
    : _eot_id(eot_id)
    , _no_timestamps_id(no_timestamps_id)
    , _timestamp_begin_id(timestamp_begin_id)
    , _timestamp_end_id(timestamp_end_id)
    , _max_initial_timestamp_id(max_initial_timestamp_id)
  {
  }

  // Helper: check if timestamp log-prob exceeds max text token log-prob.
  template <Device D, typename T>
  static bool should_sample_timestamp(const StorageView& log_probs,
                                      dim_t batch_id,
                                      size_t timestamp_begin_id,
                                      size_t timestamp_end_id) {
    const dim_t num_text_tokens = timestamp_begin_id;
    const dim_t num_timestamp_tokens = timestamp_end_id - timestamp_begin_id + 1;

    const T* text_log_probs = log_probs.index<T>({batch_id, 0});
    const T* timestamp_log_probs = text_log_probs + num_text_tokens;

    const float max_text_token_log_prob = primitives<D>::max(text_log_probs, num_text_tokens);
    const float timestamp_log_prob = primitives<D>::logsumexp(timestamp_log_probs,
                                                              num_timestamp_tokens);
    return timestamp_log_prob > max_text_token_log_prob;
  }

  void ContinuousTimestampRules::apply(
      const std::vector<SlotState>& slots,
      const std::vector<size_t>& active_slot_indices,
      dim_t beam_size,
      StorageView& logits,
      DisableTokens& disable_tokens) {

    std::vector<dim_t> check_timestamps_prob_rows;

    for (const size_t s : active_slot_indices) {
      const auto& slot = slots[s];

      // Skip slots that don't use timestamps (e.g. <|notimestamps|> in prompt).
      if (!slot.use_timestamps)
        continue;

      const dim_t gs = slot.gen_step();

      for (dim_t b = 0; b < beam_size; ++b) {
        const dim_t row = static_cast<dim_t>(s) * beam_size + b;

        // Get the token history for this beam.
        const std::vector<size_t>& tokens =
          (beam_size > 1 && b < static_cast<dim_t>(slot.beam_tokens.size()))
            ? slot.beam_tokens[b]
            : slot.generated_tokens;

        // Always suppress <|notimestamps|>.
        disable_tokens.add(row, _no_timestamps_id);

        if (gs == 0) {
          // At the very beginning: suppress non-timestamps, apply max_initial_timestamp.
          for (size_t i = 0; i < _timestamp_begin_id; ++i)
            disable_tokens.add(row, i);
          for (size_t i = _max_initial_timestamp_id + 1; i <= _timestamp_end_id; ++i)
            disable_tokens.add(row, i);

        } else if (gs > 0 && !tokens.empty()) {
          const size_t last_token = tokens.back();

          if (last_token >= _timestamp_begin_id) {
            // Last token was a timestamp.
            const size_t penultimate_token =
              (tokens.size() >= 2) ? tokens[tokens.size() - 2] : last_token;

            if (penultimate_token >= _timestamp_begin_id) {
              // Two timestamps in a row — must be non-timestamp next.
              for (size_t i = _timestamp_begin_id; i <= _timestamp_end_id; ++i)
                disable_tokens.add(row, i);
            } else {
              // Text then timestamp — must be another timestamp or EOT.
              for (size_t i = 0; i < _eot_id; ++i)
                disable_tokens.add(row, i);
              for (size_t i = _timestamp_begin_id; i < last_token; ++i)
                disable_tokens.add(row, i);
              check_timestamps_prob_rows.push_back(row);
            }
          } else {
            // Last token was text.
            check_timestamps_prob_rows.push_back(row);

            // Timestamps shouldn't decrease.
            for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
              if (*it >= _timestamp_begin_id) {
                for (size_t i = _timestamp_begin_id; i <= *it; ++i)
                  disable_tokens.add(row, i);
                break;
              }
            }
          }
        }
      }
    }

    // Two-phase check: compute log-softmax and check timestamp probability.
    if (!check_timestamps_prob_rows.empty()) {
      disable_tokens.apply();

      StorageView log_probs(logits.dtype(), logits.device());
      ops::LogSoftMax()(logits, log_probs);

      for (const dim_t row : check_timestamps_prob_rows) {
        bool sample_ts = false;

        DEVICE_AND_FLOAT_DISPATCH(
          "ContinuousTimestampRules", log_probs.device(), log_probs.dtype(),
          (sample_ts = should_sample_timestamp<D, T>(
            log_probs, row, _timestamp_begin_id, _timestamp_end_id)));

        if (sample_ts) {
          for (size_t i = 0; i < _timestamp_begin_id; ++i)
            disable_tokens.add(row, i);
        }
      }
    }
  }


  // --- ContinuousDecodingEngine ---

  ContinuousDecodingEngine::ContinuousDecodingEngine(
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
      std::vector<std::shared_ptr<ContinuousLogitsProcessor>> logits_processors,
      bool capture_attention)
    : _decoder(decoder)
    , _max_slots(max_slots)
    , _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _max_candidates(static_cast<size_t>(std::ceil(beam_size * patience)))
    , _num_hypotheses(num_hypotheses)
    , _end_ids(end_ids)
    , _max_length(max_length)
    , _sampler(sampler)
    , _slot_initializer(std::move(slot_initializer))
    , _logits_processors(std::move(logits_processors))
    , _capture_attention(capture_attention)
  {
  }

  dim_t ContinuousDecodingEngine::max_cache_length(
      const layers::DecoderState& batch_state) const {
    const auto it = batch_state.find("self_keys_0");
    if (it == batch_state.end() || !it->second)
      return 0;
    const auto& shape = it->second.shape();
    return shape.size() == 4 ? shape[2] : shape[1];
  }

  void ContinuousDecodingEngine::pad_cache(StorageView& cache,
                                           dim_t target_len,
                                           dim_t time_dim) {
    if (!cache || cache.shape()[time_dim] >= target_len)
      return;

    const dim_t current_len = cache.shape()[time_dim];
    const dim_t pad_len = target_len - current_len;

    Shape pad_shape = cache.shape();
    pad_shape[time_dim] = pad_len;
    StorageView padding(std::move(pad_shape), cache.dtype(), cache.device());
    DEVICE_AND_TYPE_DISPATCH(cache.device(), cache.dtype(),
                             primitives<D>::fill(padding.data<T>(), T(0), padding.size()));

    StorageView result(cache.dtype(), cache.device());
    const ops::Concat concat_op(time_dim);
    concat_op({&cache, &padding}, result);  // Append zeros after real data
    cache = std::move(result);
  }

  void ContinuousDecodingEngine::set_batch_slot(
      layers::DecoderState& batch_state,
      dim_t slot_idx,
      const layers::DecoderState& single_state,
      dim_t target_cache_len) {
    const Device device = batch_state.begin()->second.device();

    // On CUDA: collect all copy descriptors and execute in a single kernel launch.
    // On CPU: copy each tensor immediately.
#ifdef CT2_WITH_CUDA
    std::vector<cuda::CopyDescriptor> copies;
    copies.reserve(32);
    // Keep source tensors alive until after batch_copy_async launches the kernel.
    // Without this, single_value destructors would cudaFreeAsync the source memory
    // before the copy kernel is enqueued on the stream.
    std::vector<StorageView> source_keepalive;
    source_keepalive.reserve(32);
#endif

    for (auto& [name, batch_value] : batch_state) {
      if (name.find("_retain_memory") != std::string::npos)
        continue;
      if (name == "cache_lengths")
        continue;

      const auto it = single_state.find(name);
      if (it == single_state.end())
        continue;

      StorageView single_value = it->second;
      if (!single_value || !batch_value)
        continue;

      // Memory tensors are at slot-level (1 row per slot), others at beam-level.
      const bool is_mem = starts_with(name, "memory");
      const dim_t rows_per_slot = is_mem ? 1 : _beam_size;

      const bool is_cache = (name.find("keys_") != std::string::npos
                             || name.find("values_") != std::string::npos);
      if (is_cache && !is_mem && single_value.rank() >= 3) {
        const dim_t time_dim = single_value.rank() == 4 ? 2 : 1;
        pad_cache(single_value, target_cache_len, time_dim);
      }

      if (batch_value.rank() < 1)
        continue;

      // Copy rows_per_slot rows into batch at [slot_idx * rows_per_slot, ...).
      const dim_t slot_size_per_row = batch_value.size() / batch_value.dim(0);
      const dim_t expected_single_size = slot_size_per_row * rows_per_slot;
      if (single_value.size() != expected_single_size)
        continue;

      const dim_t batch_offset = slot_idx * rows_per_slot * slot_size_per_row;

#ifdef CT2_WITH_CUDA
      if (device == Device::CUDA) {
        const dim_t elem_bytes = batch_value.item_size();
        const char* src = reinterpret_cast<const char*>(single_value.buffer());
        char* dst = reinterpret_cast<char*>(batch_value.buffer())
                    + batch_offset * elem_bytes;
        copies.push_back({src, dst, static_cast<size_t>(expected_single_size * elem_bytes)});
        source_keepalive.push_back(std::move(single_value));
        continue;
      }
#endif

      DEVICE_AND_TYPE_DISPATCH(
        device, batch_value.dtype(),
        primitives<D>::copy(single_value.data<T>(),
                            batch_value.data<T>() + batch_offset,
                            expected_single_size));
    }

#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA && !copies.empty())
      cuda::batch_copy_async(copies);
    // source_keepalive is destroyed here, after the kernel is enqueued.
    // cudaFreeAsync ordering ensures memory stays valid until the kernel completes.
#endif
  }

  bool ContinuousDecodingEngine::fill_slot(
      size_t slot_idx,
      std::queue<ContinuousRequest>& request_queue,
      QueueProvider& queue_provider,
      std::vector<SlotState>& slots,
      layers::DecoderState& batch_state) {

    // Try to get a request from the local queue or the provider.
    ContinuousRequest request;
    if (!request_queue.empty()) {
      request = std::move(request_queue.front());
      request_queue.pop();
    } else if (queue_provider) {
      auto opt = queue_provider();
      if (!opt.has_value())
        return false;
      request = std::move(*opt);
    } else {
      return false;
    }

    // Create a single-element state and initialize it via forward_prompt.
    layers::DecoderState single_state = _decoder.initial_state(/*iterative_decoding=*/true);
    single_state["memory"] = std::move(request.encoder_output);

    _slot_initializer(_decoder, single_state, request);

    const dim_t prompt_length = request.prompt_tokens.size();

    // For beam search: replicate the single-element state beam_size times.
    // Skip memory* tensors — they stay at slot-level (one per request).
    if (_beam_size > 1) {
      for (auto& [name, value] : single_state) {
        if (!value || name.find("_retain_memory") != std::string::npos)
          continue;
        if (starts_with(name, "memory"))
          continue;
        if (value.dim(0) == 1) {
          value.expand_dims(1);
          ops::Tile(1, _beam_size)(value);
          // Merge: [1, B, ...] -> [B, ...]
          Shape shape = value.shape();
          shape[0] = _beam_size;
          shape.erase(shape.begin() + 1);
          value.reshape(std::move(shape));
        }
      }
    }

    // Pad and scatter into the batch state.
    const dim_t target_len = max_cache_length(batch_state);
    if (target_len > 0) {
      set_batch_slot(batch_state, slot_idx, single_state, target_len);
    } else {
      // First element — pad single_state to full batch dimensions.
      const Device device = _decoder.device();
      const dim_t total_batch = static_cast<dim_t>(_max_slots) * _beam_size;

      for (auto& [name, value] : single_state) {
        if (!value || value.rank() < 1 || name.find("_retain_memory") != std::string::npos)
          continue;

        // Memory tensors: dim(0)=1 (slot-level), pad to max_slots.
        // Other tensors: dim(0)=beam_size, pad to total_batch.
        const bool is_mem = starts_with(name, "memory");
        const dim_t expected = is_mem ? 1 : _beam_size;
        const dim_t target = is_mem ? static_cast<dim_t>(_max_slots) : total_batch;
        const dim_t current_batch = value.dim(0);

        if (current_batch == expected && target > expected) {
          Shape pad_shape = value.shape();
          pad_shape[0] = target - current_batch;
          StorageView padding(std::move(pad_shape), value.dtype(), device);
          DEVICE_AND_TYPE_DISPATCH(device, value.dtype(),
            primitives<D>::fill(padding.data<T>(), T(0), padding.size()));

          StorageView expanded(value.dtype(), device);
          const ops::Concat concat_op(0);
          concat_op({&value, &padding}, expanded);
          value = std::move(expanded);
        }
      }

      batch_state = std::move(single_state);
      batch_state["cache_lengths"] = StorageView({total_batch}, int32_t(0));  // CPU: avoid GPU sync roundtrips
    }

    // Update cache_lengths for this slot's rows (always on CPU).
    auto& cache_lengths = batch_state["cache_lengths"];
    if (cache_lengths) {
      for (dim_t b = 0; b < _beam_size; ++b)
        cache_lengths.at<int32_t>(slot_idx * _beam_size + b) = prompt_length;
    }

    // Set slot state.
    SlotState& slot = slots[slot_idx];
    slot.request_id = request.id;
    slot.step = prompt_length;
    slot.prompt_length = prompt_length;
    slot.active = true;
    slot.use_timestamps = request.use_timestamps;
    slot.start_id = request.start_tokens.empty() ? 0 : request.start_tokens.front();
    slot.generated_tokens.clear();
    slot.cum_score = 0.0f;

    // Initialize beam state.
    slot.beam_size = _beam_size;
    slot.beam_tokens.assign(_beam_size, {});
    slot.beam_scores.assign(_beam_size, 0.0f);
    if (_beam_size > 1) {
      // Only the first beam is active initially; others get -inf score.
      for (dim_t b = 1; b < _beam_size; ++b)
        slot.beam_scores[b] = -1e9f;
    }
    slot.beam_finished.assign(_beam_size, false);
    slot.num_finished_beams = 0;
    slot.finished_hypotheses.clear();

    return true;
  }

  void ContinuousDecodingEngine::resize_batch_state(
      layers::DecoderState& batch_state,
      dim_t target_count) const {
    const dim_t target_batch = target_count * _beam_size;

    for (auto& [name, value] : batch_state) {
      if (!value || value.rank() < 1)
        continue;
      if (name.find("_retain_memory") != std::string::npos)
        continue;
      if (name == "cache_lengths" || name == "accumulated_attention")
        continue;

      const bool is_mem = starts_with(name, "memory");
      const dim_t target_dim0 = is_mem ? target_count : target_batch;

      if (value.dim(0) == target_dim0)
        continue;

      Shape new_shape = value.shape();
      new_shape[0] = target_dim0;
      value.resize(std::move(new_shape));
    }
  }

  void ContinuousDecodingEngine::copy_slot_data(
      layers::DecoderState& batch_state,
      size_t src_slot, size_t dst_slot) const {
    if (src_slot == dst_slot)
      return;

    const Device device = batch_state.begin()->second.device();

#ifdef CT2_WITH_CUDA
    std::vector<cuda::CopyDescriptor> copies;
    copies.reserve(72);  // ~32 layers × 2 (K+V) + memory tensors + extras
#endif

    for (auto& [name, value] : batch_state) {
      if (!value || value.rank() < 1)
        continue;
      if (name.find("_retain_memory") != std::string::npos)
        continue;
      // cache_lengths is on CPU, handled separately below.
      if (name == "cache_lengths")
        continue;

      const bool is_mem = starts_with(name, "memory");
      const dim_t rows_per_slot = is_mem ? 1 : _beam_size;

      const dim_t row_stride = value.size() / value.dim(0);
      const dim_t elem_bytes = value.item_size();
      const dim_t copy_elems = rows_per_slot * row_stride;
      const dim_t src_offset = static_cast<dim_t>(src_slot) * rows_per_slot * row_stride;
      const dim_t dst_offset = static_cast<dim_t>(dst_slot) * rows_per_slot * row_stride;

#ifdef CT2_WITH_CUDA
      if (device == Device::CUDA) {
        const char* src = reinterpret_cast<const char*>(value.buffer())
                          + src_offset * elem_bytes;
        char* dst = reinterpret_cast<char*>(value.buffer())
                    + dst_offset * elem_bytes;
        copies.push_back({src, dst, static_cast<size_t>(copy_elems * elem_bytes)});
        continue;
      }
#endif

      DEVICE_AND_TYPE_DISPATCH(
        device, value.dtype(),
        primitives<D>::copy(value.data<T>() + src_offset,
                            value.data<T>() + dst_offset,
                            copy_elems));
    }

    // Copy cache_lengths entries (CPU tensor).
    {
      auto& cl = batch_state["cache_lengths"];
      if (cl) {
        for (dim_t b = 0; b < _beam_size; ++b) {
          cl.at<int32_t>(static_cast<dim_t>(dst_slot) * _beam_size + b) =
            cl.at<int32_t>(static_cast<dim_t>(src_slot) * _beam_size + b);
        }
      }
    }

#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA && !copies.empty())
      cuda::batch_copy_async(copies);
#endif
  }

  size_t ContinuousDecodingEngine::defragment_slots(
      std::vector<SlotState>& slots,
      layers::DecoderState& batch_state) const {
    size_t write = 0;
    for (size_t read = 0; read < _max_slots; ++read) {
      if (slots[read].active) {
        if (write != read) {
          copy_slot_data(batch_state, read, write);
          slots[write] = std::move(slots[read]);
          slots[write].active = true;
          slots[read].active = false;
        }
        ++write;
      }
    }
    return write;
  }

  void ContinuousDecodingEngine::process(
      std::queue<ContinuousRequest>& request_queue,
      QueueProvider queue_provider,
      ResultCallback result_callback) {
    if (request_queue.empty() && (!queue_provider || !queue_provider().has_value()))
      return;

    const Device device = _decoder.device();
    const DataType dtype = _decoder.output_type();
    const dim_t total_batch = static_cast<dim_t>(_max_slots) * _beam_size;

    std::vector<SlotState> slots(_max_slots);

    // ---- Batched initial fill ----
    // Collect initial requests, process each independently (encode + forward_prompt),
    // then stack into batch_state by concatenating along the batch dimension.
    // This avoids set_batch_slot for initial fill and mirrors the standard
    // model.generate() batching approach.

    struct InitialSlot {
      ContinuousRequest request;
      layers::DecoderState single_state;
      dim_t prompt_length = 0;
    };

    std::vector<InitialSlot> initial_slots;
    initial_slots.reserve(_max_slots);

    // Drain requests from queue.
    while (initial_slots.size() < _max_slots && !request_queue.empty()) {
      initial_slots.emplace_back();
      initial_slots.back().request = std::move(request_queue.front());
      request_queue.pop();
    }
    // Also try queue_provider.
    while (initial_slots.size() < _max_slots && queue_provider) {
      auto opt = queue_provider();
      if (!opt.has_value()) break;
      initial_slots.emplace_back();
      initial_slots.back().request = std::move(*opt);
    }

    if (initial_slots.empty())
      return;

    // Process each request: encode + forward_prompt + beam replication.
    dim_t max_prompt_len = 0;
    for (auto& is : initial_slots) {
      is.single_state = _decoder.initial_state(/*iterative_decoding=*/true);
      is.single_state["memory"] = std::move(is.request.encoder_output);
      _slot_initializer(_decoder, is.single_state, is.request);
      is.prompt_length = static_cast<dim_t>(is.request.prompt_tokens.size());
      max_prompt_len = std::max(max_prompt_len, is.prompt_length);

      // Beam replication — skip memory* tensors (cross-attention K/V cache,
      // encoder output, memory_lengths).  These stay at slot-level (one row
      // per request, not per beam), matching the standard generate() path.
      // The attention layer's beam_size computation handles the mismatch:
      //   beam_size = queries.dim(0) / cached_keys.dim(0)
      if (_beam_size > 1) {
        for (auto& [name, value] : is.single_state) {
          if (!value || name.find("_retain_memory") != std::string::npos)
            continue;
          if (starts_with(name, "memory"))
            continue;
          if (value.dim(0) == 1) {
            value.expand_dims(1);
            ops::Tile(1, _beam_size)(value);
            Shape shape = value.shape();
            shape[0] = _beam_size;
            shape.erase(shape.begin() + 1);
            value.reshape(std::move(shape));
          }
        }
      }
    }

    // Build batch state by concatenating all initial states + zero padding.
    const dim_t num_initial = static_cast<dim_t>(initial_slots.size());
    const dim_t initial_batch = num_initial * _beam_size;

    layers::DecoderState batch_state;

    for (auto& [name, first_value] : initial_slots[0].single_state) {
      if (!first_value || first_value.rank() < 1
          || name.find("_retain_memory") != std::string::npos)
        continue;

      // Pad self-attention caches to uniform time dimension if prompt lengths differ.
      const bool is_self_cache = (name.find("self_keys_") != std::string::npos
                                  || name.find("self_values_") != std::string::npos);
      if (is_self_cache && first_value.rank() >= 3 && num_initial > 1) {
        const dim_t time_dim = first_value.rank() == 4 ? 2 : 1;
        for (auto& is : initial_slots) {
          auto& val = is.single_state[name];
          if (val && val.shape()[time_dim] < max_prompt_len) {
            const dim_t pad_len = max_prompt_len - val.shape()[time_dim];
            Shape pad_shape = val.shape();
            pad_shape[time_dim] = pad_len;
            StorageView padding(std::move(pad_shape), val.dtype(), device);
            DEVICE_AND_TYPE_DISPATCH(device, val.dtype(),
              primitives<D>::fill(padding.data<T>(), T(0), padding.size()));
            StorageView result(val.dtype(), device);
            const ops::Concat concat_time(time_dim);
            concat_time({&val, &padding}, result);  // Append zeros
            val = std::move(result);
          }
        }
      }

      // Collect pointers for batch-dimension concatenation.
      std::vector<const StorageView*> parts;
      parts.reserve(num_initial);
      for (auto& is : initial_slots)
        parts.push_back(&is.single_state.at(name));

      // Concatenate along batch dimension (dim 0).
      StorageView concatenated(first_value.dtype(), device);
      if (parts.size() == 1) {
        concatenated = std::move(initial_slots[0].single_state[name]);
      } else {
        const ops::Concat concat_batch(0);
        concat_batch(parts, concatenated);
      }

      // Pad remaining rows with zeros for inactive slots.
      // Memory tensors are at slot-level (dim(0) = num_slots, not total_batch).
      const bool is_mem = starts_with(name, "memory");
      const dim_t expected_dim0 = is_mem ? num_initial : initial_batch;
      const dim_t target_dim0 = is_mem ? static_cast<dim_t>(_max_slots) : total_batch;
      const dim_t pad_amount = target_dim0 - expected_dim0;

      if (pad_amount > 0 && concatenated.dim(0) == expected_dim0) {
        Shape pad_shape = concatenated.shape();
        pad_shape[0] = pad_amount;
        StorageView padding(std::move(pad_shape), concatenated.dtype(), device);
        DEVICE_AND_TYPE_DISPATCH(device, concatenated.dtype(),
          primitives<D>::fill(padding.data<T>(), T(0), padding.size()));
        StorageView expanded(concatenated.dtype(), device);
        const ops::Concat expand_concat(0);
        expand_concat({&concatenated, &padding}, expanded);
        batch_state[name] = std::move(expanded);
      } else {
        batch_state[name] = std::move(concatenated);
      }
    }

    // Set cache_lengths (build on CPU, then move to device).
    {
      StorageView cl_cpu({total_batch}, int32_t(0));
      for (size_t s = 0; s < initial_slots.size(); ++s) {
        for (dim_t b = 0; b < _beam_size; ++b)
          cl_cpu.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = initial_slots[s].prompt_length;
      }
      batch_state["cache_lengths"] = std::move(cl_cpu);  // Keep on CPU to avoid GPU sync roundtrips
    }

    // Set slot states.
    size_t active_count = 0;
    for (size_t s = 0; s < initial_slots.size(); ++s) {
      SlotState& slot = slots[s];
      const auto& req = initial_slots[s].request;
      slot.request_id = req.id;
      slot.step = initial_slots[s].prompt_length;
      slot.prompt_length = initial_slots[s].prompt_length;
      slot.active = true;
      slot.use_timestamps = req.use_timestamps;
      slot.start_id = req.start_tokens.empty() ? 0 : req.start_tokens.front();
      slot.generated_tokens.clear();
      slot.cum_score = 0.0f;
      slot.beam_size = _beam_size;
      slot.beam_tokens.assign(_beam_size, {});
      slot.beam_scores.assign(_beam_size, 0.0f);
      if (_beam_size > 1) {
        for (dim_t b = 1; b < _beam_size; ++b)
          slot.beam_scores[b] = -1e9f;
      }
      slot.beam_finished.assign(_beam_size, false);
      slot.num_finished_beams = 0;
      slot.finished_hypotheses.clear();
      ++active_count;
    }

    if (active_count == 0)
      return;

    // Build sample_from for ALL batch rows (always total_batch elements).
    StorageView sample_from({total_batch}, DataType::INT32);
    for (size_t s = 0; s < _max_slots; ++s) {
      for (dim_t b = 0; b < _beam_size; ++b) {
        const dim_t row = static_cast<dim_t>(s) * _beam_size + b;
        sample_from.at<int32_t>(row) = slots[s].active ? slots[s].start_id : 0;
      }
    }

    StorageView logits(dtype, device);
    StorageView best_ids(DataType::INT32);
    StorageView best_probs(dtype);

    // Pre-allocated attention accumulation buffer tracking.
    dim_t attention_capacity = 0;
    dim_t attention_time = 0;

    // Reusable scratch buffers (avoid per-step allocation).
    StorageView step_offsets({total_batch}, DataType::INT32);
    StorageView gather_indices_scratch({total_batch}, DataType::INT32);

    // Main decode loop.
    while (true) {
      // Count active slots. After defragmentation, active slots are always
      // contiguous at positions 0..active_count-1.
      active_count = 0;
      for (size_t s = 0; s < _max_slots; ++s) {
        if (slots[s].active)
          ++active_count;
        else
          break;  // Contiguity invariant: first inactive means no more active.
      }
      if (active_count == 0)
        break;
      const dim_t active_batch = static_cast<dim_t>(active_count) * _beam_size;
      std::vector<size_t> active_slot_indices(active_count);
      std::iota(active_slot_indices.begin(), active_slot_indices.end(), 0);

      // --- Determine concat vs scatter path (at MAX size, before resize) ---
      // Concat path is only allowed when ALL slots are active (active_count == max_slots)
      // because concat creates new allocations that would break the over-allocate invariant.
      bool use_concat_path = false;
      {
        const auto& cache_lengths = batch_state["cache_lengths"];
        const dim_t current_cache_time = max_cache_length(batch_state);

        bool uniform_cache = true;
        int32_t common_cl = -1;
        for (size_t s = 0; s < active_count; ++s) {
          int32_t cl = cache_lengths.at<int32_t>(
            static_cast<dim_t>(s) * _beam_size);
          if (common_cl < 0)
            common_cl = cl;
          else if (cl != common_cl) {
            uniform_cache = false;
            break;
          }
        }

        use_concat_path =
          (active_count == _max_slots)
          && uniform_cache && common_cl >= 0
          && current_cache_time == static_cast<dim_t>(common_cl);

        // Scatter path: pre-allocate cache to chunk-aligned size at MAX size.
        // pad_cache uses ops::Concat which creates new allocations, so it must
        // happen before we resize the tensors down.
        if (!use_concat_path) {
          dim_t max_cl = 0;
          for (dim_t i = 0; i < active_batch; ++i)
            max_cl = std::max(max_cl, dim_t(cache_lengths.at<int32_t>(i)));

          if (max_cl >= current_cache_time) {
            constexpr dim_t chunk_size = 64;
            const dim_t target = std::min(
              ((max_cl / chunk_size) + 1) * chunk_size,
              _max_length);
            for (auto& [name, value] : batch_state) {
              if (name.find("self_keys_") != std::string::npos
                  || name.find("self_values_") != std::string::npos) {
                pad_cache(value, target, /*time_dim=*/2);
              }
            }
          }
        }
      }

      // --- Resize batch_state DOWN to active dimensions ---
      // StorageView::resize() to a smaller dim(0) is free (no reallocation).
      if (active_count < _max_slots)
        resize_batch_state(batch_state, static_cast<dim_t>(active_count));

      // Build step_offsets for active batch rows only (on CPU).
      step_offsets.resize({active_batch});
      for (size_t s = 0; s < active_count; ++s) {
        const int32_t step_val = slots[s].step;
        for (dim_t b = 0; b < _beam_size; ++b)
          step_offsets.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = step_val;
      }

      // Store CPU step_offsets in batch_state so transformer.cc can skip GPU→CPU copy.
      batch_state["step_offsets_cpu"] = step_offsets;

      // Create GPU copy for the decode step.
      StorageView step_offsets_device(step_offsets);
      if (device != Device::CPU)
        step_offsets_device = step_offsets_device.to(device);

      // Set concat/scatter flags in batch_state.
      if (use_concat_path) {
        batch_state["no_self_attn_mask"] = StorageView();
      } else {
        batch_state.erase("no_self_attn_mask");

        // Scatter path: set write positions for active rows only.
        const auto& cache_lengths = batch_state["cache_lengths"];
        StorageView active_write_pos({active_batch}, DataType::INT32);
        for (dim_t i = 0; i < active_batch; ++i)
          active_write_pos.at<int32_t>(i) = cache_lengths.at<int32_t>(i);
        batch_state["cache_write_positions"] = std::move(active_write_pos);
      }

      // Build active sample_from for the decoder.
      StorageView active_sample_from({active_batch}, DataType::INT32);
      for (dim_t i = 0; i < active_batch; ++i)
        active_sample_from.at<int32_t>(i) = sample_from.at<int32_t>(i);

      // Run one decode step with only active_batch rows.
      StorageView step_attention(device);
      _decoder(step_offsets_device, active_sample_from.to(device), batch_state, &logits,
               _capture_attention ? &step_attention : nullptr);

      // Remove temporary state entries after the decode step.
      batch_state.erase("cache_write_positions");
      batch_state.erase("step_offsets_cpu");
      batch_state.erase("no_self_attn_mask");

      // Accumulate cross-attention weights per step using pre-allocated buffer.
      // step_attention: [active_batch, num_alignment_heads, enc_time] (3D).
      // accum buffer is always allocated at total_batch (never resized by compaction).
      // We write each step directly into accum[:, :, attention_time, :] to avoid
      // O(N^2) concat copies. Buffer grows geometrically when needed.
      if (_capture_attention && step_attention) {
        step_attention.expand_dims(2);  // [active_batch, heads, 1, enc_time]
        auto& accum = batch_state["accumulated_attention"];

        const dim_t n_batch = step_attention.dim(0);  // active_batch
        const dim_t n_heads = step_attention.dim(1);
        const dim_t enc_time = step_attention.dim(3);
        const dim_t elem_bytes = step_attention.item_size();
        const dim_t row_bytes = enc_time * elem_bytes;

        // Grow buffer if needed (geometric doubling).
        // Always allocate at total_batch rows (not n_batch) to support defragmentation.
        if (attention_time >= attention_capacity) {
          const dim_t old_cap = attention_capacity;
          const dim_t new_cap = (old_cap == 0) ? 64 : old_cap * 2;

          StorageView new_accum({total_batch, n_heads, new_cap, enc_time},
                                step_attention.dtype(), device);

          // Copy old data with strided copy (time dim is not last).
          // Copy all total_batch rows to preserve data from prior defragmentations.
          if (attention_time > 0 && accum) {
            const char* old_data = static_cast<const char*>(accum.buffer());
            char* new_data = static_cast<char*>(new_accum.buffer());
            const dim_t copy_bytes = attention_time * row_bytes;
            const dim_t accum_batch = accum.dim(0);

            if (device == Device::CPU) {
              for (dim_t b = 0; b < accum_batch; ++b) {
                for (dim_t h = 0; h < n_heads; ++h) {
                  const dim_t bh = b * n_heads + h;
                  std::memcpy(new_data + bh * new_cap * row_bytes,
                              old_data + bh * old_cap * row_bytes,
                              copy_bytes);
                }
              }
            } else {
#ifdef CT2_WITH_CUDA
              std::vector<cuda::CopyDescriptor> copies;
              copies.reserve(accum_batch * n_heads);
              for (dim_t b = 0; b < accum_batch; ++b) {
                for (dim_t h = 0; h < n_heads; ++h) {
                  const dim_t bh = b * n_heads + h;
                  copies.push_back({old_data + bh * old_cap * row_bytes,
                                    new_data + bh * new_cap * row_bytes,
                                    static_cast<size_t>(copy_bytes)});
                }
              }
              cuda::batch_copy_async(copies);
#endif
            }
          }

          accum = std::move(new_accum);
          attention_capacity = new_cap;
        }

        // Write step_attention[:, :, 0, :] into accum[:, :, attention_time, :].
        {
          const char* src = static_cast<const char*>(step_attention.buffer());
          char* dst = static_cast<char*>(accum.buffer());

          if (device == Device::CPU) {
            for (dim_t b = 0; b < n_batch; ++b) {
              for (dim_t h = 0; h < n_heads; ++h) {
                const dim_t bh = b * n_heads + h;
                std::memcpy(dst + (bh * attention_capacity + attention_time) * row_bytes,
                            src + bh * row_bytes,
                            row_bytes);
              }
            }
          } else {
#ifdef CT2_WITH_CUDA
            std::vector<cuda::CopyDescriptor> copies;
            copies.reserve(n_batch * n_heads);
            for (dim_t b = 0; b < n_batch; ++b) {
              for (dim_t h = 0; h < n_heads; ++h) {
                const dim_t bh = b * n_heads + h;
                copies.push_back({src + bh * row_bytes,
                                  dst + (bh * attention_capacity + attention_time) * row_bytes,
                                  static_cast<size_t>(row_bytes)});
              }
            }
            cuda::batch_copy_async(copies);
#endif
          }
        }

        attention_time++;
      }

      // Apply logits processors.
      DisableTokens disable_tokens(logits);
      for (const auto& proc : _logits_processors)
        proc->apply(slots, active_slot_indices, _beam_size, logits, disable_tokens);
      disable_tokens.apply();

      // --- Greedy path (beam_size == 1) ---
      std::vector<size_t> finished_slots;

      if (_beam_size == 1) {
        _sampler(logits, best_ids, best_probs);

        for (const size_t s : active_slot_indices) {
          const size_t token_id = best_ids.at<int32_t>(s);
          const float score = best_probs.scalar_at<float>({static_cast<dim_t>(s), 0});

          slots[s].generated_tokens.push_back(token_id);
          slots[s].beam_tokens[0].push_back(token_id);
          slots[s].cum_score += score;
          slots[s].step++;

          if (is_eos(token_id, _end_ids)
              || slots[s].step >= _max_length) {
            ContinuousResult cr;
            cr.request_id = slots[s].request_id;
            cr.result.hypotheses.push_back(std::move(slots[s].generated_tokens));
            cr.result.scores.push_back(slots[s].cum_score);

            // Extract this slot's accumulated attention weights.
            // accumulated_attention is always total_batch-sized (not resized).
            if (_capture_attention) {
              const auto& accum = batch_state["accumulated_attention"];
              if (accum) {
                cr.attention_weights = StorageView(accum.dtype(), accum.device());
                StorageView slot_idx({1}, int32_t(s), device);
                ops::Gather()(accum, slot_idx, cr.attention_weights);
                cr.attention_weights.squeeze(0);  // [heads, capacity, enc_time]

                const dim_t offset = slots[s].attention_step_offset;
                const dim_t slot_time = attention_time - offset;
                if (slot_time > 0 && slot_time < cr.attention_weights.dim(1)) {
                  StorageView sliced(cr.attention_weights.dtype(),
                                     cr.attention_weights.device());
                  const ops::Slide slice_op(1, offset, slot_time);
                  slice_op(cr.attention_weights, sliced);
                  cr.attention_weights = std::move(sliced);
                }
              }
            }

            if (result_callback) result_callback(std::move(cr));

            slots[s].active = false;
            finished_slots.push_back(s);
          }
        }

      } else {
        // --- Beam search path (beam_size > 1) ---

        // 1. In-place LogSoftMax.
        ops::LogSoftMax()(logits);

        // 2. Add cumulative beam scores on GPU.
        //    Build active-sized beam_scores from slot metadata.
        {
          StorageView active_beam_scores({active_batch}, 0.0f);
          for (size_t s = 0; s < active_count; ++s) {
            for (dim_t b = 0; b < _beam_size; ++b)
              active_beam_scores.at<float>(
                static_cast<dim_t>(s) * _beam_size + b) = slots[s].beam_scores[b];
          }

          StorageView beam_scores_device(active_beam_scores);
          if (device != Device::CPU) {
            if (beam_scores_device.dtype() != logits.dtype())
              beam_scores_device = beam_scores_device.to(logits.dtype());
            beam_scores_device = beam_scores_device.to(device);
          }
          DEVICE_AND_TYPE_DISPATCH(logits.device(), logits.dtype(),
            primitives<D>::add_depth_broadcast(beam_scores_device.data<T>(),
                                               logits.data<T>(),
                                               beam_scores_device.size(),
                                               logits.size()));
        }

        // 3. Reshape [active_batch, vocab] → [active_count, beam_size * vocab].
        const dim_t vocab_size = logits.dim(-1);
        logits.reshape({static_cast<dim_t>(active_count),
                        _beam_size * vocab_size});

        // 4. GPU TopK: [active_count, 2*beam_size].
        const dim_t num_candidates = 2 * _beam_size;
        StorageView topk_scores_step(logits.dtype(), logits.device());
        StorageView topk_ids(DataType::INT32, logits.device());
        const ops::TopK topk_op(num_candidates);
        topk_op(logits, topk_scores_step, topk_ids);

        // 5. Transfer to CPU.
        if (topk_scores_step.dtype() != DataType::FLOAT32)
          topk_scores_step = topk_scores_step.to_float32();
        topk_scores_step = topk_scores_step.to(Device::CPU);
        topk_ids = topk_ids.to(Device::CPU);

        // Initialize gather_indices to identity for ALL rows (total_batch).
        // Active rows get updated below; inactive rows stay identity (no-op for Gather).
        StorageView& gather_indices = gather_indices_scratch;
        for (dim_t i = 0; i < total_batch; ++i)
          gather_indices.at<int32_t>(i) = i;

        for (const size_t s : active_slot_indices) {
          auto& slot = slots[s];
          const dim_t s_offset = static_cast<dim_t>(s) * _beam_size;
          const dim_t s_dim = static_cast<dim_t>(s);

          std::vector<dim_t> new_beam_from(_beam_size, -1);
          std::vector<size_t> new_beam_token(_beam_size, 0);
          std::vector<float> new_beam_score(_beam_size, -1e9f);
          dim_t filled = 0;

          for (dim_t k = 0; k < num_candidates && filled < _beam_size; ++k) {
            const int32_t flat_id = topk_ids.at<int32_t>({s_dim, k});
            const float score = topk_scores_step.at<float>({s_dim, k});
            const dim_t beam_id = flat_id / vocab_size;
            const size_t word_id = flat_id % vocab_size;

            if (slot.beam_finished[beam_id])
              continue;

            if (is_eos(word_id, _end_ids)) {
              std::vector<size_t> hyp_tokens = slot.beam_tokens[beam_id];
              const float hyp_len = static_cast<float>(hyp_tokens.size() + 1);
              const float normalized_score = score / std::pow(hyp_len, _length_penalty);

              slot.finished_hypotheses.push_back(
                SlotState::Hypothesis{std::move(hyp_tokens), normalized_score});
              slot.num_finished_beams++;

              if (slot.num_finished_beams >= _max_candidates) {
                while (filled < _beam_size) {
                  new_beam_from[filled] = 0;
                  new_beam_token[filled] = 0;
                  new_beam_score[filled] = -1e9f;
                  filled++;
                }
                break;
              }
              continue;
            }

            new_beam_from[filled] = beam_id;
            new_beam_token[filled] = word_id;
            new_beam_score[filled] = score;
            filled++;
          }

          while (filled < _beam_size) {
            new_beam_from[filled] = 0;
            new_beam_token[filled] = 0;
            new_beam_score[filled] = -1e9f;
            slot.beam_finished[filled] = true;
            filled++;
          }

          // Check if slot is fully finished.
          if (slot.num_finished_beams >= _max_candidates
              || slot.step + 1 >= _max_length) {
            if (slot.step + 1 >= _max_length) {
              for (dim_t b = 0; b < _beam_size; ++b) {
                if (!slot.beam_finished[b] && new_beam_from[b] >= 0) {
                  std::vector<size_t> hyp_tokens = slot.beam_tokens[new_beam_from[b]];
                  hyp_tokens.push_back(new_beam_token[b]);
                  const float hyp_len = static_cast<float>(hyp_tokens.size());
                  const float normalized_score = new_beam_score[b] / std::pow(hyp_len, _length_penalty);
                  slot.finished_hypotheses.push_back(
                    SlotState::Hypothesis{std::move(hyp_tokens), normalized_score});
                }
              }
            }

            std::sort(slot.finished_hypotheses.begin(), slot.finished_hypotheses.end(),
                      [](const SlotState::Hypothesis& a, const SlotState::Hypothesis& b) {
                        return a.score > b.score;
                      });

            ContinuousResult cr;
            cr.request_id = slot.request_id;
            const size_t n = std::min(_num_hypotheses, slot.finished_hypotheses.size());
            for (size_t h = 0; h < n; ++h) {
              cr.result.hypotheses.push_back(std::move(slot.finished_hypotheses[h].tokens));
              cr.result.scores.push_back(slot.finished_hypotheses[h].score);
            }

            // Extract beam 0's accumulated attention for this slot.
            if (_capture_attention) {
              const auto& accum = batch_state["accumulated_attention"];
              if (accum) {
                cr.attention_weights = StorageView(accum.dtype(), accum.device());
                StorageView slot_idx({1}, int32_t(s_offset), device);
                ops::Gather()(accum, slot_idx, cr.attention_weights);
                cr.attention_weights.squeeze(0);

                const dim_t offset = slot.attention_step_offset;
                const dim_t slot_time = attention_time - offset;
                if (slot_time > 0 && slot_time < cr.attention_weights.dim(1)) {
                  StorageView sliced(cr.attention_weights.dtype(),
                                     cr.attention_weights.device());
                  const ops::Slide slice_op(1, offset, slot_time);
                  slice_op(cr.attention_weights, sliced);
                  cr.attention_weights = std::move(sliced);
                }
              }
            }

            if (result_callback) result_callback(std::move(cr));

            slot.active = false;
            finished_slots.push_back(s);
            continue;
          }

          // Update beam state: reorder beams according to new_beam_from.
          std::vector<std::vector<size_t>> updated_beam_tokens(_beam_size);
          for (dim_t b = 0; b < _beam_size; ++b) {
            if (new_beam_from[b] >= 0) {
              updated_beam_tokens[b] = slot.beam_tokens[new_beam_from[b]];
              updated_beam_tokens[b].push_back(new_beam_token[b]);
            }
          }
          slot.beam_tokens = std::move(updated_beam_tokens);
          slot.beam_scores = std::move(new_beam_score);
          slot.step++;

          if (!slot.beam_tokens.empty() && !slot.beam_tokens[0].empty())
            slot.generated_tokens = slot.beam_tokens[0];

          // Build gather indices for this slot's beams.
          for (dim_t b = 0; b < _beam_size; ++b) {
            gather_indices.at<int32_t>(s_offset + b) =
              s_offset + (new_beam_from[b] >= 0 ? new_beam_from[b] : b);
          }

          // Update sample_from for this slot's beams.
          for (dim_t b = 0; b < _beam_size; ++b)
            sample_from.at<int32_t>(s_offset + b) = new_beam_token[b];
        }

        // --- Resize back to max dimensions before Gather ---
        // Gather on GPU reallocates via clone, so tensors must be at max size
        // to preserve the over-allocate invariant.
        if (active_count < _max_slots)
          resize_batch_state(batch_state, static_cast<dim_t>(_max_slots));

        // Apply beam reordering at max size (skip if identity permutation).
        bool is_identity = true;
        for (dim_t i = 0; i < total_batch && is_identity; ++i)
          is_identity = (gather_indices.at<int32_t>(i) == i);

        if (!is_identity) {
          StorageView gather_device(gather_indices);
          if (device != Device::CPU)
            gather_device = gather_device.to(device);

          for (auto& [name, value] : batch_state) {
            if (name.find("_retain_memory") != std::string::npos)
              continue;
            if (name == "cache_lengths")
              continue;
            if (starts_with(name, "memory"))
              continue;
            if (name == "accumulated_attention")
              continue;
            if (value && value.dim(0) == total_batch)
              ops::Gather()(value, gather_device);
          }
        }
      }

      // --- Resize back (greedy path — beam path already resized above) ---
      if (_beam_size == 1 && active_count < _max_slots)
        resize_batch_state(batch_state, static_cast<dim_t>(_max_slots));

      // --- Defragment: compact active slots to positions 0..N-1 ---
      // This ensures the contiguity invariant for the next iteration.
      if (!finished_slots.empty()) {
        active_count = defragment_slots(slots, batch_state);
      }

      if (active_count == 0)
        break;

      // --- Fill new slots at the end of the compacted region ---
      for (size_t s = active_count; s < _max_slots; ++s) {
        if (!fill_slot(s, request_queue, queue_provider, slots, batch_state))
          break;
        slots[s].attention_step_offset = attention_time;
        if (_beam_size > 1) {
          // beam_scores are rebuilt from slot metadata each iteration,
          // so no need to update the persistent buffer here.
        }
        ++active_count;
      }

      if (active_count == 0)
        break;

      // Rebuild sample_from for all slots (positions changed by defragmentation).
      sample_from.resize({total_batch});
      if (_beam_size == 1) {
        for (size_t s = 0; s < _max_slots; ++s) {
          if (slots[s].active) {
            if (slots[s].generated_tokens.empty())
              sample_from.at<int32_t>(s) = slots[s].start_id;
            else
              sample_from.at<int32_t>(s) = slots[s].generated_tokens.back();
          } else {
            sample_from.at<int32_t>(s) = 0;
          }
        }
      } else {
        for (size_t s = 0; s < _max_slots; ++s) {
          const dim_t s_offset = static_cast<dim_t>(s) * _beam_size;
          if (slots[s].active) {
            if (slots[s].generated_tokens.empty()) {
              for (dim_t b = 0; b < _beam_size; ++b)
                sample_from.at<int32_t>(s_offset + b) = (b == 0) ? slots[s].start_id : 0;
            }
            // For continuing slots, beam selection already set sample_from above.
            // But defrag may have shifted positions, so rebuild from slot state.
            else {
              // For beam path, the last selected tokens are in beam_tokens.
              // Use beam_tokens[b].back() for each beam.
              for (dim_t b = 0; b < _beam_size; ++b) {
                if (b < static_cast<dim_t>(slots[s].beam_tokens.size())
                    && !slots[s].beam_tokens[b].empty())
                  sample_from.at<int32_t>(s_offset + b) = slots[s].beam_tokens[b].back();
                else
                  sample_from.at<int32_t>(s_offset + b) = 0;
              }
            }
          } else {
            for (dim_t b = 0; b < _beam_size; ++b)
              sample_from.at<int32_t>(s_offset + b) = 0;
          }
        }
      }

      // Update cache_lengths for active slots (always on CPU).
      auto& cache_lengths = batch_state["cache_lengths"];
      if (cache_lengths) {
        for (size_t s = 0; s < _max_slots; ++s) {
          if (slots[s].active) {
            for (dim_t b = 0; b < _beam_size; ++b)
              cache_lengths.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = slots[s].step;
          }
        }
      }

    }

    // Synchronize the CUDA stream before returning. This ensures all async
    // operations (scatter writes, batch_copy) have completed so that the
    // batch_state tensors (destroyed when this function returns) are not
    // freed while kernels are still in-flight.
    synchronize_stream(device);

  }

}
