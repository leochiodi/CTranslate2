#include "ctranslate2/continuous_decoding.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "ctranslate2/ops/ops.h"
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
      std::vector<std::shared_ptr<ContinuousLogitsProcessor>> logits_processors)
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
    const dim_t rows_per_slot = _beam_size;

    // On CUDA: collect all copy descriptors and execute in a single kernel launch.
    // On CPU: copy each tensor immediately.
#ifdef CT2_WITH_CUDA
    std::vector<cuda::CopyDescriptor> copies;
    copies.reserve(32);
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

      const bool is_cache = (name.find("keys_") != std::string::npos
                             || name.find("values_") != std::string::npos);
      if (is_cache && single_value.rank() >= 3) {
        const dim_t time_dim = single_value.rank() == 4 ? 2 : 1;
        pad_cache(single_value, target_cache_len, time_dim);
      }

      if (batch_value.rank() < 1)
        continue;

      // For beam search: single_state has rows_per_slot rows for this slot.
      // Copy them into batch rows [slot_idx * rows_per_slot, ...).
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
    if (_beam_size > 1) {
      for (auto& [name, value] : single_state) {
        if (!value || name.find("_retain_memory") != std::string::npos)
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
      // First element — pad single_state from beam_size to total_batch rows.
      const Device device = _decoder.device();
      const dim_t total_batch = static_cast<dim_t>(_max_slots) * _beam_size;

      for (auto& [name, value] : single_state) {
        if (!value || value.rank() < 1 || name.find("_retain_memory") != std::string::npos)
          continue;

        const dim_t current_batch = value.dim(0);
        if (current_batch == _beam_size && total_batch > _beam_size) {
          // Expand batch dimension: pad with zeros from beam_size to total_batch.
          Shape pad_shape = value.shape();
          pad_shape[0] = total_batch - current_batch;
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
      batch_state["cache_lengths"] = StorageView({total_batch}, int32_t(0), device);
    }

    // Update cache_lengths for this slot's rows.
    auto& cache_lengths = batch_state["cache_lengths"];
    if (cache_lengths) {
      if (cache_lengths.device() == Device::CPU) {
        // Write directly — no shallow_copy/move dance (which causes use-after-free).
        for (dim_t b = 0; b < _beam_size; ++b)
          cache_lengths.at<int32_t>(slot_idx * _beam_size + b) = prompt_length;
      } else {
        StorageView cl_cpu(DataType::INT32);
        cl_cpu.copy_from(cache_lengths.to(Device::CPU));
        for (dim_t b = 0; b < _beam_size; ++b)
          cl_cpu.at<int32_t>(slot_idx * _beam_size + b) = prompt_length;
        cache_lengths = cl_cpu.to(cache_lengths.device());
      }
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

  std::vector<ContinuousResult> ContinuousDecodingEngine::process(
      std::queue<ContinuousRequest>& request_queue,
      QueueProvider queue_provider) {
    if (request_queue.empty() && (!queue_provider || !queue_provider().has_value()))
      return {};

    const Device device = _decoder.device();
    const DataType dtype = _decoder.output_type();
    const dim_t total_batch = static_cast<dim_t>(_max_slots) * _beam_size;

    std::vector<SlotState> slots(_max_slots);
    std::vector<ContinuousResult> results;

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
      return results;

    // Process each request: encode + forward_prompt + beam replication.
    dim_t max_prompt_len = 0;
    for (auto& is : initial_slots) {
      is.single_state = _decoder.initial_state(/*iterative_decoding=*/true);
      is.single_state["memory"] = std::move(is.request.encoder_output);
      _slot_initializer(_decoder, is.single_state, is.request);
      is.prompt_length = static_cast<dim_t>(is.request.prompt_tokens.size());
      max_prompt_len = std::max(max_prompt_len, is.prompt_length);

      // Beam replication.
      if (_beam_size > 1) {
        for (auto& [name, value] : is.single_state) {
          if (!value || name.find("_retain_memory") != std::string::npos)
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
    const dim_t pad_rows = total_batch - initial_batch;

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

      // Pad remaining batch rows with zeros for inactive slots.
      if (pad_rows > 0 && concatenated.dim(0) == initial_batch) {
        Shape pad_shape = concatenated.shape();
        pad_shape[0] = pad_rows;
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
      batch_state["cache_lengths"] = cl_cpu.to(device);
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
      return results;

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

    // Main decode loop.
    while (true) {
      // Recompute active count from slot states each iteration.
      active_count = 0;
      std::vector<size_t> active_slot_indices;
      active_slot_indices.reserve(_max_slots);
      for (size_t s = 0; s < _max_slots; ++s) {
        if (slots[s].active) {
          active_slot_indices.push_back(s);
          ++active_count;
        }
      }
      if (active_count == 0)
        break;

      // Build step_offsets for ALL batch rows.
      StorageView step_offsets({total_batch}, DataType::INT32);
      for (size_t s = 0; s < _max_slots; ++s) {
        const int32_t step_val = slots[s].active ? slots[s].step : 0;
        for (dim_t b = 0; b < _beam_size; ++b)
          step_offsets.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = step_val;
      }

      if (device != Device::CPU)
        step_offsets = step_offsets.to(device);

      // Run one decode step.
      _decoder(step_offsets, sample_from.to(device), batch_state, &logits, nullptr);



      // Apply logits processors.
      DisableTokens disable_tokens(logits);
      for (const auto& proc : _logits_processors)
        proc->apply(slots, active_slot_indices, _beam_size, logits, disable_tokens);
      disable_tokens.apply();

      // --- Greedy path (beam_size == 1) ---
      if (_beam_size == 1) {
        _sampler(logits, best_ids, best_probs);

        std::vector<size_t> finished_slots;
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
            results.push_back(std::move(cr));

            slots[s].active = false;
            finished_slots.push_back(s);
          }
        }

        // NOTE: mid-decode slot filling is disabled because appending new
        // K/V entries via ops::Concat always goes to the END of the cache,
        // but a freshly-added slot needs entries at position prompt_len
        // (not at the batch's current cache_time).  This mismatch causes
        // hallucination.  Instead, finished slots stay inactive and new
        // requests are processed in the next process() call.

      } else {
        // --- Beam search path (beam_size > 1) ---

        // Compute log_softmax.
        StorageView log_probs(logits.dtype(), logits.device());
        ops::LogSoftMax()(logits, log_probs);

        // Move log_probs to CPU for per-slot beam selection.
        if (log_probs.dtype() != DataType::FLOAT32)
          log_probs = log_probs.to_float32();
        if (log_probs.device() != Device::CPU)
          log_probs = log_probs.to(Device::CPU);

        const dim_t vocab_size = log_probs.dim(1);
        StorageView gather_indices({total_batch}, DataType::INT32);

        // Initialize gather_indices to identity (inactive slots keep their rows).
        for (dim_t i = 0; i < total_batch; ++i)
          gather_indices.at<int32_t>(i) = i;

        std::vector<size_t> finished_slots;

        for (const size_t s : active_slot_indices) {
          auto& slot = slots[s];
          const dim_t s_offset = static_cast<dim_t>(s) * _beam_size;

          // Collect candidates across all beams for this slot.
          struct Candidate {
            dim_t beam;
            size_t word_id;
            float score;
          };
          std::vector<Candidate> candidates;
          candidates.reserve(_beam_size * 2 * _beam_size);

          for (dim_t b = 0; b < _beam_size; ++b) {
            if (slot.beam_finished[b])
              continue;

            const float* row_probs = log_probs.index<float>({s_offset + b, 0});
            const float beam_score = slot.beam_scores[b];

            // Find top 2*beam_size candidates from this beam.
            std::vector<std::pair<float, size_t>> beam_candidates;
            beam_candidates.reserve(vocab_size);
            for (dim_t v = 0; v < vocab_size; ++v)
              beam_candidates.emplace_back(beam_score + row_probs[v], v);

            std::partial_sort(beam_candidates.begin(),
                              beam_candidates.begin() + std::min(static_cast<dim_t>(2 * _beam_size),
                                                                  vocab_size),
                              beam_candidates.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            for (dim_t k = 0; k < std::min(static_cast<dim_t>(2 * _beam_size), vocab_size); ++k) {
              candidates.push_back(Candidate{b, beam_candidates[k].second,
                                             beam_candidates[k].first});
            }
          }

          // Sort all candidates by score descending.
          std::sort(candidates.begin(), candidates.end(),
                    [](const Candidate& a, const Candidate& b) {
                      return a.score > b.score;
                    });

          // Select top beam_size non-finished candidates.
          std::vector<dim_t> new_beam_from(_beam_size, -1);  // Which old beam each new beam comes from.
          std::vector<size_t> new_beam_token(_beam_size, 0);
          std::vector<float> new_beam_score(_beam_size, -1e9f);
          dim_t filled = 0;

          for (const auto& cand : candidates) {
            if (filled >= _beam_size)
              break;

            if (is_eos(cand.word_id, _end_ids)) {
              // This beam produced EOT — record hypothesis.
              std::vector<size_t> hyp_tokens = slot.beam_tokens[cand.beam];
              // Don't include EOT in hypothesis.
              const float hyp_len = static_cast<float>(hyp_tokens.size() + 1);
              const float normalized_score = cand.score / std::pow(hyp_len, _length_penalty);

              slot.finished_hypotheses.push_back(
                SlotState::Hypothesis{std::move(hyp_tokens), normalized_score});
              slot.num_finished_beams++;

              if (slot.num_finished_beams >= _max_candidates) {
                // Slot is done — fill remaining beams with dummy.
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

            new_beam_from[filled] = cand.beam;
            new_beam_token[filled] = cand.word_id;
            new_beam_score[filled] = cand.score;
            filled++;
          }

          // If we couldn't fill all beams (all candidates were EOT), fill with dummy.
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
            // If we hit max_length, add remaining active beams as hypotheses.
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

            // Sort hypotheses by normalized score, collect top num_hypotheses.
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
            results.push_back(std::move(cr));

            slot.active = false;
            finished_slots.push_back(s);

            // Keep identity gather for this slot's rows.
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

          // Also update generated_tokens with best beam's last token for compatibility.
          if (!slot.beam_tokens.empty() && !slot.beam_tokens[0].empty())
            slot.generated_tokens = slot.beam_tokens[0];

          // Build gather indices for this slot's beams.
          for (dim_t b = 0; b < _beam_size; ++b) {
            gather_indices.at<int32_t>(s_offset + b) =
              s_offset + (new_beam_from[b] >= 0 ? new_beam_from[b] : b);
          }

          // Update sample_from for this slot's beams.
          for (dim_t b = 0; b < _beam_size; ++b) {
            sample_from.at<int32_t>(s_offset + b) = new_beam_token[b];
          }
        }

        // Apply beam reordering to all state tensors.
        if (device != Device::CPU)
          gather_indices = gather_indices.to(device);

        for (auto& [name, value] : batch_state) {
          if (name.find("_retain_memory") != std::string::npos)
            continue;
          if (name == "cache_lengths")
            continue;
          if (value && value.dim(0) == total_batch)
            ops::Gather()(value, gather_indices);
        }

        // Mid-decode slot filling disabled (see greedy path comment).
      }

      // Recount active slots.
      active_count = 0;
      for (size_t s = 0; s < _max_slots; ++s) {
        if (slots[s].active)
          ++active_count;
      }

      if (active_count == 0)
        break;

      // Rebuild sample_from for greedy path (beam path already updated above).
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
        // For beam path, sample_from was already updated per-slot above.
        // Just ensure inactive slots have dummy values.
        for (size_t s = 0; s < _max_slots; ++s) {
          if (!slots[s].active) {
            for (dim_t b = 0; b < _beam_size; ++b)
              sample_from.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = 0;
          }
        }
      }

      // Update cache_lengths for active slots.
      auto& cache_lengths = batch_state["cache_lengths"];
      if (cache_lengths) {
        if (cache_lengths.device() == Device::CPU) {
          for (size_t s = 0; s < _max_slots; ++s) {
            if (slots[s].active) {
              for (dim_t b = 0; b < _beam_size; ++b)
                cache_lengths.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = slots[s].step;
            }
          }
        } else {
          StorageView cl_cpu(DataType::INT32);
          cl_cpu.copy_from(cache_lengths.to(Device::CPU));
          for (size_t s = 0; s < _max_slots; ++s) {
            if (slots[s].active) {
              for (dim_t b = 0; b < _beam_size; ++b)
                cl_cpu.at<int32_t>(static_cast<dim_t>(s) * _beam_size + b) = slots[s].step;
            }
          }
          cache_lengths = cl_cpu.to(cache_lengths.device());
        }
      }
    }

    return results;
  }

}
