#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/utils.h"


#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "dispatch.h"
#include "cpu/parallel.h"

#ifdef CT2_WITH_CUDA
#include "cuda/batch_copy.h"
#include "cuda/cb_ops.h"
#include "cuda/cuda_graph.h"
#include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace layers {

    // Scatter-write a single decode step into the K/V cache at per-row positions.
    // cache:   [batch, heads, time, d]  (time_dim=2)  or  [batch, time, d]  (time_dim=1)
    // step:    [batch, heads, 1, d]                   or  [batch, 1, d]
    // positions: [batch] INT32 — write position for each batch row.
    static void scatter_cache_step(StorageView& cache,
                                   const StorageView& step,
                                   const StorageView& positions,
                                   dim_t time_dim) {
      const Device device = cache.device();
      const dim_t batch = cache.dim(0);
      const dim_t heads = (time_dim == 2) ? cache.dim(1) : 1;
      const dim_t cache_time = cache.dim(time_dim);
      const dim_t d = cache.dim(time_dim + 1);
      const dim_t elem_bytes = cache.item_size();

      StorageView pos_cpu(DataType::INT32);
      if (positions.device() != Device::CPU)
        pos_cpu.copy_from(positions.to(Device::CPU));
      else
        pos_cpu.shallow_copy(const_cast<StorageView&>(positions));

      // Fast path: all positions identical → single strided copy.
      const int32_t pos0 = pos_cpu.at<int32_t>(0);
      bool uniform = true;
      for (dim_t b = 1; b < batch; ++b) {
        if (pos_cpu.at<int32_t>(b) != pos0) { uniform = false; break; }
      }

#ifdef CT2_WITH_CUDA
      if (device == Device::CUDA) {
        // CUDA graph capture: skip CPU uniform check, always use GPU scatter kernel.
        // The uniform fast-path uses cudaMemcpy2DAsync with a CPU-computed offset that
        // gets baked into the graph; on replay it would write to the stale position.
        // The scatter kernel reads positions from a GPU buffer updated each step.
        if (cuda::g_cuda_graph_capturing) {
          const int32_t* pos_gpu;
          StorageView pos_gpu_tmp;
          if (positions.device() == Device::CUDA) {
            pos_gpu = positions.data<int32_t>();
          } else {
            pos_gpu_tmp = positions.to(Device::CUDA);
            pos_gpu = pos_gpu_tmp.data<int32_t>();
          }
          cuda::scatter_cache_step_gpu(
              cache.buffer(),
              step.buffer(),
              pos_gpu,
              static_cast<int>(batch),
              static_cast<int>(heads),
              static_cast<int>(cache_time),
              static_cast<int>(d),
              static_cast<int>(elem_bytes));
          return;
        }

        if (uniform) {
          // All rows write to the same time position → use cudaMemcpy2DAsync.
          // Source: step[batch*heads, 1, d] contiguous, pitch = d * elem_bytes.
          // Dest: cache[batch*heads, cache_time, d] at time pos0, pitch = cache_time * d * elem_bytes.
          const size_t row_bytes = static_cast<size_t>(d * elem_bytes);
          CUDA_CHECK(cudaMemcpy2DAsync(
              reinterpret_cast<char*>(cache.buffer()) + pos0 * d * elem_bytes,
              cache_time * row_bytes,                           // dst pitch
              reinterpret_cast<const char*>(step.buffer()),
              row_bytes,                                        // src pitch
              row_bytes,                                        // width
              static_cast<size_t>(batch * heads),               // height
              cudaMemcpyDeviceToDevice,
              cuda::get_cuda_stream()));
          return;
        }

        // Non-uniform positions: use fused GPU scatter kernel.
        // Upload positions to GPU (tiny: batch * 4 bytes, e.g. 160 bytes for 40 slots).
        // This eliminates the CPU loop that built batch*heads copy descriptors.
        StorageView pos_gpu_sv(pos_cpu.to(Device::CUDA));
        cuda::scatter_cache_step_gpu(
            cache.buffer(),
            step.buffer(),
            pos_gpu_sv.data<int32_t>(),
            static_cast<int>(batch),
            static_cast<int>(heads),
            static_cast<int>(cache_time),
            static_cast<int>(d),
            static_cast<int>(elem_bytes));
        return;
      }
#endif

      // CPU path.
      if (uniform) {
        const size_t row_bytes = d * elem_bytes;
        for (dim_t bh = 0; bh < batch * heads; ++bh) {
          std::memcpy(reinterpret_cast<char*>(cache.buffer())
                        + (bh * cache_time + pos0) * d * elem_bytes,
                      reinterpret_cast<const char*>(step.buffer())
                        + bh * row_bytes,
                      row_bytes);
        }
        return;
      }

      for (dim_t b = 0; b < batch; ++b) {
        const dim_t pos = pos_cpu.at<int32_t>(b);
        for (dim_t h = 0; h < heads; ++h) {
          dim_t src_off;
          dim_t dst_off;
          if (time_dim == 2) {
            src_off = (b * heads + h) * 1 * d;
            dst_off = (b * heads + h) * cache_time * d + pos * d;
          } else {
            src_off = b * 1 * d;
            dst_off = b * cache_time * d + pos * d;
          }
          std::memcpy(reinterpret_cast<char*>(cache.buffer()) + dst_off * elem_bytes,
                      reinterpret_cast<const char*>(step.buffer()) + src_off * elem_bytes,
                      d * elem_bytes);
        }
      }
    }

    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position) {
      StorageView positions({queries_length, keys_length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      const dim_t offset = keys_length - queries_length;

      for (dim_t i = 0; i < queries_length; ++i) {
        auto* row = positions_data + i * keys_length;
        for (dim_t j = 0; j < keys_length; ++j) {
          row[j] = std::min(std::max(j - (i + offset), -max_position), max_position) + max_position;
        }
      }

      return positions;
    }

    StorageView make_asymmetric_relative_positions(dim_t queries_length,
                                                  dim_t keys_length,
                                                  dim_t left_max_position,
                                                  dim_t right_max_position) {
      StorageView positions({queries_length, keys_length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      for (dim_t i = 0; i < queries_length; ++i) {
        auto* row = positions_data + i * keys_length;
        for (dim_t j = 0; j < keys_length; ++j) {
          row[j] = std::max(std::min(j - i, right_max_position), -left_max_position) + left_max_position;
        }
      }

      return positions;
    }

    static StorageView get_relative_position_bucket(bool bidirectional,
                                                    dim_t query_length,
                                                    dim_t key_length,
                                                    dim_t num_buckets,
                                                    dim_t max_distance,
                                                    dim_t query_offset = 0) {
      StorageView relative_buckets({query_length, key_length}, DataType::INT32);
      int32_t* relative_buckets_data = relative_buckets.data<int32_t>();

      if (bidirectional)
        num_buckets /= 2;

      const dim_t max_exact = num_buckets / 2;
      const float log_max_distance_over_max_exact = std::log(float(max_distance) / float(max_exact));

      cpu::parallel_for(0, query_length * key_length, 8, [&](const dim_t begin, const dim_t end) {
        for (dim_t flat_index = begin; flat_index < end; ++flat_index) {
          const dim_t i = flat_index / key_length;
          const dim_t j = flat_index % key_length;

          int32_t relative_position = j - (i + query_offset);
          int32_t relative_bucket = 0;

          if (bidirectional) {
            if (relative_position > 0)
              relative_bucket += num_buckets;
            else
              relative_position = std::abs(relative_position);
          } else {
            relative_position = -std::min(relative_position, 0);
          }

          const bool is_small = relative_position < max_exact;

          if (!is_small) {
            relative_position = std::min(
              int32_t(float(max_exact)
                      + std::log(float(relative_position) / float(max_exact))
                      / log_max_distance_over_max_exact
                      * float(num_buckets - max_exact)),
              int32_t(num_buckets - 1));
          }

          relative_bucket += relative_position;
          relative_buckets_data[flat_index] = relative_bucket;
        }
      });

      return relative_buckets;
    }

    static StorageView compute_relative_bias(const StorageView& relative_attention_bias,
                                             dim_t query_length,
                                             dim_t key_length,
                                             dim_t max_distance,
                                             bool is_decoder,
                                             dim_t query_offset = 0) {
      const Device device = relative_attention_bias.device();
      const DataType dtype = relative_attention_bias.dtype();
      const dim_t num_buckets = relative_attention_bias.dim(0);

      StorageView relative_attention_bucket = get_relative_position_bucket(!is_decoder,
                                                                           query_length,
                                                                           key_length,
                                                                           num_buckets,
                                                                           max_distance,
                                                                           query_offset);

      StorageView values(dtype, device);
      ops::Gather()(relative_attention_bias, relative_attention_bucket.to(device), values);

      StorageView values_t(dtype, device);
      ops::Transpose({2, 0, 1})(values, values_t);

      return values_t;
    }

    static void matmul_with_relative_representations(const ops::MatMul& matmul_op,
                                                     const StorageView& a,
                                                     const StorageView& b,
                                                     StorageView& c) {
      const Device device = a.device();
      const DataType dtype = a.dtype();
      const dim_t batch = a.dim(0);
      const dim_t head = a.dim(1);
      const dim_t time = a.dim(2);

      StorageView a_t(dtype, device);
      ops::Transpose({2, 0, 1, 3})(a, a_t);
      a_t.reshape({time, batch * head, -1});

      StorageView c_t(dtype, device);
      matmul_op(a_t, b, c_t);
      c_t.reshape({time, batch, head, -1});
      ops::Transpose({1, 2, 0, 3})(c_t, c);
    }

    static void add_relative_representations(const StorageView& queries,
                                             const StorageView& relative_positions,
                                             const StorageView& relative_values,
                                             const ops::MatMul& matmul_op,
                                             StorageView& dot) {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      StorageView relative_representations(dtype, device);
      ops::Gather()(relative_values, relative_positions, relative_representations);

      StorageView dot_relative(dtype, device);
      matmul_with_relative_representations(matmul_op,
                                           queries,
                                           relative_representations,
                                           dot_relative);
      ops::Add()(dot_relative, dot, dot);
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    static inline void save_attention(StorageView& attention, StorageView weights, dim_t beam_size) {
      if (beam_size == 1)
        attention = std::move(weights);
      else {
        // Ensure output dtype matches input (may differ in mixed precision).
        if (attention.dtype() != weights.dtype())
          attention = StorageView(weights.dtype(), weights.device());
        transpose_op(weights, attention);
        attention.reshape({-1, weights.dim(1), 1, weights.dim(-1)});
      }
    }

    static void dot_product_attention(const StorageView& queries,
                                      const StorageView& keys,
                                      const StorageView& values,
                                      const StorageView* values_lengths,
                                      const StorageView* relative_position_keys,
                                      const StorageView* relative_asymmetric_position_keys,
                                      const StorageView* relative_position_values,
                                      const StorageView* relative_attention_bias,
                                      dim_t relative_left_max_position,
                                      dim_t relative_right_max_position,
                                      dim_t maximum_relative_position,
                                      StorageView& output,
                                      StorageView* attention = nullptr,
                                      bool return_normalized_attention = true,
                                      float queries_scale = 1,
                                      bool is_decoder = false,
                                      bool with_cache = false,
                                      dim_t beam_size = 1,
                                      Alibi* alibi = nullptr,
                                      StorageView* position_bias = nullptr) {
      PROFILE("dot_product_attention");

      std::unique_ptr<const StorageView> relative_positions;
      if (relative_position_keys || relative_position_values || relative_asymmetric_position_keys) {
        const dim_t query_length = queries.dim(2);
        const dim_t key_length = keys.dim(2);
        if (relative_asymmetric_position_keys)
          relative_positions = std::make_unique<StorageView>(
            make_asymmetric_relative_positions(query_length,
                                    key_length,
                                    relative_left_max_position,
                                    relative_right_max_position).to(queries.device()));
        else relative_positions = std::make_unique<StorageView>(
               make_relative_positions(query_length,
                                       key_length,
                                       maximum_relative_position).to(queries.device()));
      }

      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, queries_scale);
      keys_matmul(queries, keys, output);
      if (relative_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_position_keys,
                                     keys_matmul,
                                     output);

      if (relative_asymmetric_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_asymmetric_position_keys,
                                     keys_matmul,
                                     output);
      if (relative_attention_bias) {
        StorageView local_position_bias(output.dtype(), output.device());

        if (!position_bias)
          position_bias = &local_position_bias;

        if (position_bias->empty()) {
          const dim_t query_length = queries.dim(2);
          const dim_t key_length = keys.dim(2);
          *position_bias = compute_relative_bias(*relative_attention_bias,
                                                 query_length,
                                                 key_length,
                                                 maximum_relative_position,
                                                 is_decoder,
                                                 with_cache ? key_length - 1 : 0);
        }
        StorageView* position_bias_per_gpu = position_bias;
        StorageView position_bias_tmp(position_bias->dtype(), position_bias->device());
        if (ScopedMPISetter::getCurRank() != 0) {
          const dim_t num_head_per_gpu = SAFE_DIVIDE(position_bias->dim(0), ScopedMPISetter::getNRanks());
          ops::Slide slide_ops(0, num_head_per_gpu * ScopedMPISetter::getCurRank(),
                               num_head_per_gpu, true);
          slide_ops(*position_bias, position_bias_tmp);
          position_bias_per_gpu = &position_bias_tmp;
        }

        DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(),
                                 primitives<D>::add_batch_broadcast(position_bias_per_gpu->data<T>(),
                                                                    output.data<T>(),
                                                                    position_bias_per_gpu->size(),
                                                                    output.size()));
      }

      if (alibi)
        alibi->apply(output, queries_scale);

      StorageView attn(values.dtype(), values.device());
      ops::SoftMax()(output, values_lengths, attn);

      if (attention && !return_normalized_attention)
        save_attention(*attention, std::move(output), beam_size);

      const ops::MatMul values_matmul;
      values_matmul(attn, values, output);
      if (relative_position_values)
        add_relative_representations(attn,
                                     *relative_positions,
                                     *relative_position_values,
                                     values_matmul,
                                     output);

      if (attention && return_normalized_attention)
        save_attention(*attention, std::move(attn), beam_size);
    }



    static void replicate_heads(StorageView& x, dim_t repeats) {
      x.expand_dims(2);
      ops::Tile(2, repeats)(x);
      x.reshape({x.dim(0), x.dim(1) * x.dim(2), x.dim(3), x.dim(4)});
    }

    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder,
                                           Alibi* alibi)
      : AttentionLayer(model, scope, num_heads, self_attention, pre_norm, is_decoder, alibi, false)
      , _relative_attention_bias(model.get_variable_if_exists(scope + "/relative_attention_bias"))
      , _relative_position_keys(model.get_variable_if_exists(scope + "/relative_position_keys"))
      , _relative_asymmetric_position_keys(model.get_variable_if_exists(scope + "/relative_asymmetric_position_keys"))
      , _relative_position_values(model.get_variable_if_exists(scope + "/relative_position_values"))
      , _merge_time_and_head_dims(_multi_query
                                  && !_relative_attention_bias
                                  && !_relative_position_keys
                                  && !_relative_position_values)
      ,_cache_time_dim(_merge_time_and_head_dims ? 1 : 2)
      , _q_norm(build_optional_layer<LayerNorm>(model, scope + "/q_norm"))
      , _k_norm(build_optional_layer<LayerNorm>(model, scope + "/k_norm"))
    {
      if (_relative_position_keys)
        _maximum_relative_position = (_relative_position_keys->dim(0) - 1) / 2;
      else if (_relative_asymmetric_position_keys) {
        _relative_left_max_position = model.get_attribute<int32_t>(
          scope + "/relative_left_max_position");
        _relative_right_max_position = model.get_attribute<int32_t>(
          scope + "/relative_right_max_position");
      }
      else if (_relative_attention_bias)
        _maximum_relative_position = model.get_attribute<int32_t>(
          scope + "/relative_attention_max_distance");
      else
        _maximum_relative_position = 0;
    }

    DataType MultiHeadAttention::output_type() const {
      return _linear.back().output_type();
    }

    dim_t MultiHeadAttention::output_size() const {
      return _d_model;
    }

    void MultiHeadAttention::apply_k_norm(StorageView& keys_proj) const {
      if (_k_norm) {
        StorageView keys_normed(keys_proj.dtype(), keys_proj.device());
        (*_k_norm)(keys_proj, keys_normed);
        keys_proj = std::move(keys_normed);
      }
    }

    void MultiHeadAttention::apply_qk_norm(StorageView& queries_proj, StorageView& keys_proj) const {
      if (_q_norm) {
        StorageView queries_normed(queries_proj.dtype(), queries_proj.device());
        (*_q_norm)(queries_proj, queries_normed);
        queries_proj = std::move(queries_normed);
      }

      apply_k_norm(keys_proj);
    }

    void MultiHeadAttention::process_cross_attention(
        const StorageView& queries,
        const StorageView& values,
        StorageView& fused_proj,
        StorageView& queries_proj,
        StorageView& keys_proj,
        StorageView& values_proj,
        StorageView* cached_keys,
        StorageView* cached_values,
        const Padder* queries_padder,
        const Padder* values_padder,
        dim_t& beam_size) const {

      queries_proj = std::move(fused_proj);

      if (cached_keys == nullptr || cached_keys->empty()) {
        _linear[1](values, fused_proj);

        if (_num_heads_kv == 1) { // MQA (Multi-Query Attention)
          if (values_padder)
            values_padder->add_padding(fused_proj);
          ops::Split(2, {_d_head, _d_head})(fused_proj, keys_proj, values_proj);

          apply_k_norm(keys_proj);
          keys_proj.expand_dims(1);
          values_proj.expand_dims(1);
          replicate_heads(keys_proj, _num_heads);
          replicate_heads(values_proj, _num_heads);

        } else if (_num_heads_kv < _num_heads) { // GQA (Grouped-Query Attention)
          if (values_padder)
            values_padder->add_padding(fused_proj);

          const ops::Split split_op(2, {_num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(fused_proj, keys_proj, values_proj);

          split_heads(keys_proj, _num_heads_kv);
          split_heads(values_proj, _num_heads_kv);

          apply_k_norm(keys_proj);

          replicate_heads(keys_proj, _num_heads / _num_heads_kv);
          replicate_heads(values_proj, _num_heads / _num_heads_kv);
        } else { //Standard Multi-Head Attention (MHA)
          split_heads(fused_proj, 2 * _num_heads, values_padder);
          ops::Split(1)(fused_proj, keys_proj, values_proj);

          apply_k_norm(keys_proj);
        }

        if (cached_keys != nullptr) {
          *cached_keys = std::move(keys_proj);
          *cached_values = std::move(values_proj);
        }
      }

      if (_q_norm) {
        StorageView queries_normed(queries_proj.dtype(), queries_proj.device());
        (*_q_norm)(queries_proj, queries_normed);
        queries_proj = std::move(queries_normed);
      }

      if (queries_proj.dim(1) == 1 && cached_keys)
        beam_size = queries_proj.dim(0) / cached_keys->dim(0);

      split_heads(queries_proj, _num_heads, queries_padder, beam_size);
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView& values,
                                        const StorageView* values_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        StorageView* attention,
                                        const Padder* queries_padder,
                                        const Padder* values_padder,
                                        bool return_normalized_attention,
                                        StorageView* position_bias,
                                        dim_t offset) const {
      PROFILE("MultiHeadAttention");
      const Device device = queries.device();
      const DataType dtype = queries.dtype();
      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);

      const StorageView* q = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, queries_proj);
        q = &queries_proj;
      }

      _linear[0](*q, fused_proj);

      dim_t beam_size = 1;

      bool prefilling = (_sliding_window > 0 && values_lengths);

      if (!_self_attention) {
      
        process_cross_attention(queries, values, fused_proj, queries_proj, keys_proj,
                                values_proj, cached_keys, cached_values,
                                queries_padder, values_padder, beam_size);
      } else {

        if (_num_heads_kv < _num_heads) {// MQA or GQA: queries stay in merged time/head format
          if (queries_padder)
            queries_padder->add_padding(fused_proj);

          const ops::Split split_op(2, {_num_heads * _d_head, _num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(fused_proj, queries_proj, keys_proj, values_proj);

          if (_merge_time_and_head_dims) {
            queries_proj.reshape({queries_proj.dim(0), -1, _d_head});
            apply_qk_norm(queries_proj, keys_proj);
          } else {
            split_heads(queries_proj, _num_heads);
            split_heads(keys_proj, _num_heads_kv);
            split_heads(values_proj, _num_heads_kv);

            apply_qk_norm(queries_proj, keys_proj);

            replicate_heads(keys_proj, _num_heads / _num_heads_kv);
            replicate_heads(values_proj, _num_heads / _num_heads_kv);
          }

        } else {
          split_heads(fused_proj, 3 * _num_heads, queries_padder);
          ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);

          apply_qk_norm(queries_proj, keys_proj);
        }

        if (_rotary_embeddings) {
          if (_merge_time_and_head_dims) {
            queries_proj.reshape({queries_proj.dim(0), -1, _d_model});
            split_heads(queries_proj, _num_heads);
          }

          _rotary_embeddings->apply(queries_proj, offset);
          _rotary_embeddings->apply(keys_proj, offset);

          if (_merge_time_and_head_dims) {
            combine_heads(queries_proj, _num_heads);
            queries_proj.reshape({queries_proj.dim(0), -1, _d_head});
          }
        }

        if (cached_keys != nullptr) {
          if (cached_keys->empty()) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          } else if (_cache_write_positions) {
            // Scatter path: write each batch row's K/V at its designated position.
            scatter_cache_step(*cached_keys, keys_proj,
                               *_cache_write_positions, _cache_time_dim);
            scatter_cache_step(*cached_values, values_proj,
                               *_cache_write_positions, _cache_time_dim);
          } else {
            const ops::Concat concat_op(_cache_time_dim);
            StorageView& tmp = fused_proj;  // Reuse storage.
            tmp = std::move(*cached_keys);
            concat_op({&tmp, &keys_proj}, *cached_keys);
            tmp = std::move(*cached_values);
            concat_op({&tmp, &values_proj}, *cached_values);

            if (!prefilling && _sliding_window > 0 && cached_keys->shape()[2] > _sliding_window) {
              // only for generation
              const ops::Slide slide_op(2, 1, cached_keys->shape()[2] - 1);
              slide_op(*cached_keys, tmp);
              *cached_keys = std::move(tmp);
              slide_op(*cached_values, tmp);
              *cached_values = std::move(tmp);
            }
          }
        }
      }

      if (cached_keys) {
        keys_proj.shallow_copy(*cached_keys);
        values_proj.shallow_copy(*cached_values);
      }

      StorageView& context = fused_proj;  // Reuse storage.
      dot_product_attention(queries_proj,
                            keys_proj,
                            values_proj,
                            values_lengths,
                            _relative_position_keys,
                            _relative_asymmetric_position_keys,
                            _relative_position_values,
                            _relative_attention_bias,
                            _relative_left_max_position,
                            _relative_right_max_position,
                            _maximum_relative_position,
                            context,
                            attention,
                            return_normalized_attention,
                            _queries_scale,
                            _is_decoder,
                            bool(cached_keys),
                            beam_size,
                            _alibi,
                            position_bias);

      if (prefilling && cached_keys && cached_keys->shape()[2] > _sliding_window) {
        // set only last sliding_window tokens to cached_keys and cached_values after computing attention
        const ops::Slide slide_op(2, cached_keys->shape()[2] - _sliding_window, _sliding_window);
        StorageView tmp(dtype, device);
        slide_op(*cached_keys, tmp);
        *cached_keys = std::move(tmp);
        slide_op(*cached_values, tmp);
        *cached_values = std::move(tmp);
      }

      if (_merge_time_and_head_dims) {
        context.reshape(queries.shape());
        if (queries_padder)
          queries_padder->remove_padding(context);
      } else {
        combine_heads(context, _num_heads, queries_padder, beam_size);
      }
      _linear.back()(context, output, _layer_norm ? &queries : nullptr);

      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll ops_reduce_all(ops::ReduceAll::RED_OP::SUM);
        ops_reduce_all(output, tmp);
        output = std::move(tmp);
      }
      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }

    void MultiHeadAttention::split_heads(StorageView& x,
                     dim_t num_heads,
                     const Padder* padder,
                     dim_t beam_size) {
      if (padder)
        padder->add_padding(x);

      if (beam_size > 1)
        x.reshape({x.dim(0) / beam_size, beam_size, x.dim(2)});

      // x has shape [batch_size, time, depth]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t head_dim = x.dim(2) / num_heads;

      if (time == 1) {
        x.reshape({batch_size, num_heads, 1, head_dim});
      } else {
        x.reshape({batch_size, time, num_heads, head_dim});
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }
    }

    void MultiHeadAttention::combine_heads(StorageView& x,
                                         dim_t num_heads,
                                         const Padder* padder,
                                         dim_t beam_size) {
      // x has shape [batch_size, num_heads, time, head_dim]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(2);
      const dim_t depth = x.dim(3) * num_heads;

      if (time > 1) {
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }

      x.reshape({batch_size, time, depth});

      if (beam_size > 1)
        x.reshape({batch_size * beam_size, 1, depth});

      if (padder)
        padder->remove_padding(x);
    }
  }
}
