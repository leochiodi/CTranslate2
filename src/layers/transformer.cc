#include "ctranslate2/layers/transformer.h"

#include <cmath>
#include <cstdlib>
#include <chrono>

#include "ctranslate2/devices.h"

#ifdef CT2_WITH_CUDA
#include "cuda/utils.h"
#include "cuda/cb_ops.h"
#include "cuda/cuda_graph.h"
#endif

namespace ctranslate2 {
  namespace layers {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _ff1(model, scope + "/linear_0", &_activation_type)
      , _ff1_noact(build_optional_layer<Dense>(model, scope + "/linear_0_noact"))
      , _ff2(model, scope + "/linear_1", nullptr, true)
      , _tensor_parallel(model.tensor_parallel()) {
    }

    void FeedForwardNetwork::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(input, output);
        x = &output;
      }

      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView inner(dtype, device);
      _ff1(*x, inner);
      if (_ff1_noact) {
        StorageView linear(dtype, device);
        (*_ff1_noact)(*x, linear);
        ops::Mul()(linear, inner, inner);
      }

      _ff2(inner, output, _layer_norm ? &input : nullptr);

      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll red_op(ops::ReduceAll::RED_OP::SUM);
        red_op(output, tmp);
        output = std::move(tmp);
      }

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_feedforward_layer_norm"))
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }


    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder,
                                             StorageView* position_bias) const {
      PROFILE("TransformerEncoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      // Check if using pre_post_layer_norm pattern (T5Gemma style)
      const bool pre_post_layer_norm = _input_layer_norm && _post_attention_layer_norm
                                        && _pre_feedforward_layer_norm && _post_feedforward_layer_norm;

      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);

        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                          hidden,
                          lengths,
                          context,
                          nullptr,
                          nullptr,
                          nullptr,
                          padder,
                          padder,
                          true,
                          position_bias);

        // post_self_attn_layernorm
        (*_post_attention_layer_norm)(context, output);

        // residual + hidden_states
        ops::Add()(input, output, output);

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        // mlp
        _ff(hidden, output);

        // post_feedforward_layernorm
        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);

        // residual + hidden_states
        ops::Add()(context, output, output);
        return;
      }

      // Original path for standard pre-norm/post-norm architectures
      StorageView context(dtype, device);
      if (_self_attention)
        (*_self_attention)(input,
                        input,
                        lengths,
                        context,
                        nullptr,
                        nullptr,
                        nullptr,
                        padder,
                        padder,
                        true,
                        position_bias);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention,
                                                     Alibi* alibi)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)))
      , _shared_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/shared_layer_norm"))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layer_norm"))
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
      , _external_pre_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_pre_encoder_attention_layer_norm"))
      , _external_post_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_post_encoder_attention_layer_norm"))
      {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* input_length,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView* cached_self_attn_keys,
                                             StorageView* cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention,
                                             const Padder* input_padder,
                                             const Padder* memory_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      const bool pre_post_layer_norm = _post_feedforward_layer_norm && _pre_feedforward_layer_norm;
      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);
        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                             hidden,
                             input_length,
                             context,
                             cached_self_attn_keys,
                             cached_self_attn_values,
                             nullptr,
                             input_padder,
                             input_padder,
                             true,
                             position_bias,
                             offset);

        (*_post_attention_layer_norm)(context, output);
        ops::Add()(output, input, output);

        if (_encoder_attention) {
            StorageView cross_attn_in = output;  // save for residual

            StorageView query_normalized(dtype, device);
            if (_external_pre_encoder_attention_layer_norm) {
                (*_external_pre_encoder_attention_layer_norm)(output, query_normalized);
            }
            else {
                query_normalized.shallow_copy(output);
            }

            (*_encoder_attention)(query_normalized,
                                  *memory,
                                  memory_lengths,
                                  context,
                                  cached_attn_keys,
                                  cached_attn_values,
                                  attention,
                                  input_padder,
                                  memory_padder,
                                  return_normalized_attention);

            if (_external_post_encoder_attention_layer_norm) {
                (*_external_post_encoder_attention_layer_norm)(context, context);
            }
            ops::Add()(context, cross_attn_in, output);
        }

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        _ff(hidden, output);

        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);
        ops::Add()(output, context, output);
        return;
      }

      const bool use_parallel_residual = _shared_layer_norm || _input_layer_norm;

      if (use_parallel_residual) {
        // The parallel residual implementation assumes there is no cross attention.
        StorageView hidden(dtype, device);

        if (_shared_layer_norm)
          (*_shared_layer_norm)(input, hidden);
        else
          (*_input_layer_norm)(input, hidden);

        StorageView attn(dtype, device);
        if (_self_attention)
          (*_self_attention)(hidden,
                        hidden,
                        input_length,
                        attn,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        nullptr,
                        input_padder,
                        input_padder,
                        true,
                        position_bias,
                        offset);

        if (_post_attention_layer_norm)
          (*_post_attention_layer_norm)(input, hidden);

        _ff(hidden, output);

        ops::Add()(output, input, output);
        ops::Add()(output, attn, output);

        return;
      }
      // --- Sub-layer profiling (CT2_SUBLAYER_PROFILE env var) ---
      static const bool sp_enabled = std::getenv("CT2_SUBLAYER_PROFILE") != nullptr;
      static thread_local size_t sp_calls = 0;
      static thread_local double sp_self_attn_ms = 0;
      static thread_local double sp_cross_attn_ms = 0;
      static thread_local double sp_ffn_ms = 0;

#ifdef CT2_WITH_CUDA
      if (sp_enabled && device == Device::CUDA)
        synchronize_stream(device);
      auto sp_t0 = std::chrono::steady_clock::now();
#endif

      if (_self_attention)
        (*_self_attention)(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder,
                      true,
                      position_bias,
                      offset);

#ifdef CT2_WITH_CUDA
      if (sp_enabled && device == Device::CUDA)
        synchronize_stream(device);
      auto sp_t1 = std::chrono::steady_clock::now();
#endif

      StorageView context(dtype, device);
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder,
                              return_normalized_attention);
      }
      else {
        context = std::move(output);
      }

#ifdef CT2_WITH_CUDA
      if (sp_enabled && device == Device::CUDA)
        synchronize_stream(device);
      auto sp_t2 = std::chrono::steady_clock::now();
#endif

      _ff(context, output);

#ifdef CT2_WITH_CUDA
      if (sp_enabled && device == Device::CUDA) {
        synchronize_stream(device);
        auto sp_t3 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
          return std::chrono::duration<double, std::milli>(b - a).count();
        };
        sp_self_attn_ms += ms(sp_t0, sp_t1);
        sp_cross_attn_ms += ms(sp_t1, sp_t2);
        sp_ffn_ms += ms(sp_t2, sp_t3);
        ++sp_calls;

        // Print every 16000 calls (= 500 decode steps × 32 layers).
        if (sp_calls % 16000 == 0) {
          const double steps = sp_calls / 32.0;
          fprintf(stderr, "[SUBLAYER PROFILE] calls=%zu (%.0f steps)\n"
                          "  self_attn: total=%.1fms  per_layer=%.3fms  per_step=%.2fms (%.1f%%)\n"
                          "  cross_attn: total=%.1fms  per_layer=%.3fms  per_step=%.2fms (%.1f%%)\n"
                          "  ffn:       total=%.1fms  per_layer=%.3fms  per_step=%.2fms (%.1f%%)\n",
                  sp_calls, steps,
                  sp_self_attn_ms, sp_self_attn_ms / sp_calls, sp_self_attn_ms / steps,
                  100.0 * sp_self_attn_ms / (sp_self_attn_ms + sp_cross_attn_ms + sp_ffn_ms),
                  sp_cross_attn_ms, sp_cross_attn_ms / sp_calls, sp_cross_attn_ms / steps,
                  100.0 * sp_cross_attn_ms / (sp_self_attn_ms + sp_cross_attn_ms + sp_ffn_ms),
                  sp_ffn_ms, sp_ffn_ms / sp_calls, sp_ffn_ms / steps,
                  100.0 * sp_ffn_ms / (sp_self_attn_ms + sp_cross_attn_ms + sp_ffn_ms));
        }
      }
#endif
    }


    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    static std::unique_ptr<const StorageView>
    build_embeddings_scale(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");

      // Backward compatibility with older models.
      if (!scale)
        scale = model.get_variable_if_exists(scope + "/embeddings/multiply_by_sqrt_depth");

      StorageView value;

      // The attribute can either be a boolean flag or the actual scale value.
      if (!scale || (scale->dtype() == DataType::INT8 && scale->as_scalar<int8_t>()))
        value = StorageView(std::sqrt(static_cast<float>(embeddings.output_size())));
      else if (scale->dtype() != DataType::INT8 && scale->as_scalar<float>() != 1.f)
        value = *scale;
      else
        return nullptr;

      return std::make_unique<StorageView>(value.to(embeddings.output_type()));
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings",
                    model.get_enum_value<EmbeddingsMerge>(scope + "/embeddings_merge"))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerEncoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _tensor_parallel(model.tensor_parallel())
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView* lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_embeddings_scale)
        ops::Mul()(input, *_embeddings_scale, input);
      if (_position_encoder)
        (*_position_encoder)(input);
      if (_layernorm_embedding)
        (*_layernorm_embedding)(input, input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      std::unique_ptr<StorageView> lengths_mask;

      if (lengths) {
        if (Padder::allow_padding_removal(output.device(), _compute_type)) {
          padder = std::make_unique<Padder>(*lengths, max_time);
          padder->remove_padding(input);
        }

        int num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }
        lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths, num_heads, max_time));
      }

      StorageView position_bias(output.dtype(), output.device());

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), &position_bias);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    static std::unique_ptr<Alibi> make_alibi(const models::Model& model, const std::string& scope) {
      const bool use_alibi = model.get_flag_with_default(scope + "/alibi", false);
      if (!use_alibi)
        return nullptr;

      const bool use_positive_positions = model.get_flag_with_default(
        scope + "/alibi_use_positive_positions", true);
      const bool scale_alibi = model.get_flag_with_default(
        scope + "/scale_alibi", false);

      return std::make_unique<Alibi>(use_positive_positions, scale_alibi);
    }

    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _alibi(make_alibi(model, scope))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation"),
                  _use_flash_attention,
                  _alibi.get()))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection")
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
      , _tensor_parallel(model.tensor_parallel()) {

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);

      const auto* outputs_scale = model.get_variable_if_exists(scope + "/scale_outputs");
      if (outputs_scale) {
        const DataType dtype = get_default_float_type(_compute_type);
        _outputs_scale = std::make_unique<StorageView>(outputs_scale->to(dtype));
      }
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;

      if (iterative_decoding) {
        const size_t state_size = _layers.size() * (_with_encoder_attention ? 4 : 2);
        state.reserve(state_size);

        const DataType dtype = output_type();

        for (size_t i = 0; i < _layers.size(); ++i) {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention) {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
        }
      }

      return state;
    }

    bool TransformerDecoder::replicate_state(const std::string& name) const {
      // No need to replicate projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::set_alignment_heads(const dim_t layer,
                                                 const dim_t num_heads_to_average) {
      std::vector<dim_t> range(num_heads_to_average);
      std::iota(range.begin(), range.end(), dim_t(0));

      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      _alignment_heads[layer] = std::move(range);

      _average_alignment_heads = true;
    }

    void TransformerDecoder::set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads) {
      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      for (const auto& [layer, head] : alignment_heads)
        _alignment_heads[layer].push_back(head);

      _average_alignment_heads = false;
    }

    std::unique_ptr<StorageView>
    TransformerDecoder::get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const {
      if (_alignment_heads.empty())
        return nullptr;

      const auto& heads = _alignment_heads[layer];
      const dim_t num_heads = heads.size();

      if (heads.empty())
        return nullptr;

      std::vector<int32_t> indices;
      indices.reserve(batch_size * num_heads);
      for (dim_t i = 0; i < batch_size; ++i)
        indices.insert(indices.end(), heads.begin(), heads.end());

      return std::make_unique<StorageView>(Shape{batch_size, num_heads}, indices, _device);
    }

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView& ids,
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      return decode(ids, nullptr, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView& step_offsets,
                                        const StorageView& ids,
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      return decode(ids, nullptr, step_offsets, state, logits, attention,
                    /*return_logits=*/true, /*retain_memory=*/true);
    }

    void TransformerDecoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        DecoderState& state,
                                        StorageView& logits,
                                        StorageView* attention) {
      return decode(ids, &lengths, -1, state, &logits, attention);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits) {
      PROFILE("TransformerDecoder");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      // --- Decode phase profiling (CT2_DECODE_PHASE_PROFILE env var) ---
      static const bool dp_std_enabled = std::getenv("CT2_DECODE_PHASE_PROFILE") != nullptr;
      static thread_local size_t dp_std_calls = 0;
      static thread_local double dp_std_B = 0, dp_std_C = 0, dp_std_D = 0;
      static thread_local double dp_std_E = 0, dp_std_F = 0, dp_std_G = 0;
      static thread_local double dp_std_H = 0, dp_std_I = 0, dp_std_total = 0;

#ifdef CT2_WITH_CUDA
      auto dp_std_sync_now = [&]() -> std::chrono::steady_clock::time_point {
        if (dp_std_enabled && device == Device::CUDA)
          synchronize_stream(device);
        return std::chrono::steady_clock::now();
      };
      auto dp_std_ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      auto dp_std_t_start = dp_std_sync_now();
#endif

      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);

      _embeddings(ids, layer_in);
      if (_start_from_zero_embedding)
        zero_first_timestep(layer_in, step);
      if (_embeddings_scale && (!_start_from_zero_embedding || step != 0))
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (_project_in) {
        (*_project_in)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);

#ifdef CT2_WITH_CUDA
      auto dp_std_t2 = dp_std_sync_now();  // End Phase B (embed+scale)
#endif

      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));

#ifdef CT2_WITH_CUDA
      auto dp_std_t3 = dp_std_sync_now();  // End Phase C (pos encoding)
#endif

      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

#ifdef CT2_WITH_CUDA
      auto dp_std_t4 = dp_std_sync_now();  // End Phase D (layernorm_emb)
#endif

      const dim_t batch_size = layer_in.dim(0);
      dim_t max_time;

      if (_sliding_window > 0 && layer_in.dim(1) > _sliding_window) {
        max_time = _sliding_window;
      } else
        max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }

        dim_t num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths,
          num_heads,
          max_time,
          /*mask_future=*/true,
          multi_query);


        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

#ifdef CT2_WITH_CUDA
      auto dp_std_t5 = dp_std_sync_now();  // End Phase E (attn mask)
#endif

      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        if (step <= 0) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          dim_t num_heads = _num_heads;
          if (_tensor_parallel) {
            num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
          }
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

#ifdef CT2_WITH_CUDA
      auto dp_std_t6 = dp_std_sync_now();  // End Phase F (memory setup)
#endif

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);

      std::vector<StorageView> layer_ins;

      while (true) {
        dim_t prompt_size = layer_in.dim(1);
        if (_sliding_window == 0 || prompt_size <= _sliding_window || _use_flash_attention) {
          layer_ins.push_back(std::move(layer_in));
          break;
        }
        if (layer_in.dim(1) > _sliding_window) {
          StorageView tmp(dtype, device);
          const ops::Split split_op(1, {_sliding_window, prompt_size - _sliding_window});
          split_op(layer_in, tmp, layer_in);
          layer_ins.push_back(std::move(tmp));
        }
      }

      for (size_t i = 0; i < layer_ins.size(); ++i) {
        StorageView* layer_in_chunk = &layer_ins[i];
        for (size_t l = 0; l < _layers.size(); ++l) {
          StorageView* cached_self_attn_keys = nullptr;
          StorageView* cached_self_attn_values = nullptr;
          StorageView* cached_attn_keys = nullptr;
          StorageView* cached_attn_values = nullptr;

          if (step >= 0) {
            const std::string l_str = std::to_string(l);
            cached_self_attn_keys = &state.at("self_keys_" + l_str);
            cached_self_attn_values = &state.at("self_values_" + l_str);
            if (_with_encoder_attention) {
              cached_attn_keys = &state.at("memory_keys_" + l_str);
              cached_attn_values = &state.at("memory_values_" + l_str);
            }
          }

          std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
          std::unique_ptr<StorageView> layer_attention;
          if (attention && heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          dim_t offset = _sliding_window * i + step;
          offset = offset < 0 ? 0 : offset;
          if (i > 0) {
            auto max_tokens = _sliding_window + layer_in_chunk->dim(1);
            StorageView tmp_lengths = StorageView(Shape{layer_in_chunk->dim(0)}, int32_t(max_tokens), device);
            int num_heads = _num_heads;
            if (_tensor_parallel) {
              num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
            }
            StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
              tmp_lengths,
              num_heads,
              max_tokens,
              /*mask_future=*/true,
              multi_query);

            const ops::Slide slide_lengths_op(2, _sliding_window, layer_in_chunk->dim(1));
            // reuse tmp_lengths
            slide_lengths_op(lengths_mask, tmp_lengths);
            input_lengths_mask = std::make_unique<StorageView>(std::move(tmp_lengths));
          }

          (*_layers[l])(*layer_in_chunk,
                        input_lengths_mask.get(),
                        memory,
                        memory_lengths_mask.get(),
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        cached_attn_keys,
                        cached_attn_values,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        memory_padder.get(),
                        return_normalized_attention(),
                        &position_bias,
                        offset);
          *layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        layer_in = std::move(*layer_in_chunk);
      }

#ifdef CT2_WITH_CUDA
      auto dp_std_t7 = dp_std_sync_now();  // End Phase G (layer loop)
#endif

      if (step == 0 && state.count("_retain_memory") == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (attention && !alignment_heads.empty()) {
        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);

        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);

          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

#ifdef CT2_WITH_CUDA
      auto dp_std_t8 = dp_std_sync_now();  // End Phase H (attn concat)
#endif

      if (outputs) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (_project_out) {
          (*_project_out)(layer_in, layer_out);
          layer_in = std::move(layer_out);
        }

        if (_outputs_scale)
          ops::Mul()(layer_in, *_outputs_scale, layer_in);

        if (return_logits)
          _proj(layer_in, *outputs);
        else
          *outputs = std::move(layer_in);

        if (!is_sequence)
          outputs->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*outputs);
      }

#ifdef CT2_WITH_CUDA
      auto dp_std_t9 = dp_std_sync_now();  // End Phase I (norm+lm_head)

      if (dp_std_enabled && device == Device::CUDA) {
        dp_std_B += dp_std_ms(dp_std_t_start, dp_std_t2);
        dp_std_C += dp_std_ms(dp_std_t2, dp_std_t3);
        dp_std_D += dp_std_ms(dp_std_t3, dp_std_t4);
        dp_std_E += dp_std_ms(dp_std_t4, dp_std_t5);
        dp_std_F += dp_std_ms(dp_std_t5, dp_std_t6);
        dp_std_G += dp_std_ms(dp_std_t6, dp_std_t7);
        dp_std_H += dp_std_ms(dp_std_t7, dp_std_t8);
        dp_std_I += dp_std_ms(dp_std_t8, dp_std_t9);
        dp_std_total += dp_std_ms(dp_std_t_start, dp_std_t9);
        ++dp_std_calls;

        if (dp_std_calls % 500 == 0) {
          auto pct = [&](double v) { return dp_std_total > 0 ? 100.0 * v / dp_std_total : 0.0; };
          fprintf(stderr,
            "[DECODE PHASE STD] calls=%zu  total=%.1fms (%.2fms/call)\n"
            "  B embed+scale:    %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  C pos_encoding:   %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  D layernorm_emb:  %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  E attn_mask:      %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  F memory_setup:   %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  G layer_loop:     %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  H attn_concat:    %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  I norm+lm_head:   %8.1fms  %5.2fms/call  %5.1f%%\n",
            dp_std_calls, dp_std_total, dp_std_total / dp_std_calls,
            dp_std_B, dp_std_B / dp_std_calls, pct(dp_std_B),
            dp_std_C, dp_std_C / dp_std_calls, pct(dp_std_C),
            dp_std_D, dp_std_D / dp_std_calls, pct(dp_std_D),
            dp_std_E, dp_std_E / dp_std_calls, pct(dp_std_E),
            dp_std_F, dp_std_F / dp_std_calls, pct(dp_std_F),
            dp_std_G, dp_std_G / dp_std_calls, pct(dp_std_G),
            dp_std_H, dp_std_H / dp_std_calls, pct(dp_std_H),
            dp_std_I, dp_std_I / dp_std_calls, pct(dp_std_I));
        }
      }
#endif
    }

    void TransformerDecoder::ensure_cb_buffers(dim_t total_batch,
                                               DataType dtype,
                                               Device device) {
      if (_cb_buffers.allocated_batch == total_batch
          && _cb_buffers.buf[0].device() == device
          && _cb_buffers.buf[0].dtype() == dtype)
        return;

      const dim_t d_model = _embeddings.output_size();
      _cb_buffers.buf[0] = StorageView({total_batch, dim_t(1), d_model}, dtype, device);
      _cb_buffers.buf[1] = StorageView({total_batch, dim_t(1), d_model}, dtype, device);
      _cb_buffers.attn_lengths = StorageView({total_batch}, DataType::INT32, device);
      _cb_buffers.position_bias = StorageView(dtype, device);
      _cb_buffers.allocated_batch = total_batch;
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    const StorageView& step_offsets,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits,
                                    bool retain_memory) {
      PROFILE("TransformerDecoder_continuous");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      // --- Decode phase profiling (CT2_DECODE_PHASE_PROFILE env var) ---
      static const bool dp_enabled = std::getenv("CT2_DECODE_PHASE_PROFILE") != nullptr;
      static thread_local size_t dp_calls = 0;
      static thread_local double dp_A = 0, dp_B = 0, dp_C = 0, dp_D = 0;
      static thread_local double dp_E = 0, dp_F = 0, dp_G = 0, dp_H = 0, dp_I = 0;
      static thread_local double dp_total = 0;

#ifdef CT2_WITH_CUDA
      auto dp_sync_now = [&]() -> std::chrono::steady_clock::time_point {
        if (dp_enabled && device == Device::CUDA && !cuda::g_cuda_graph_capturing)
          synchronize_stream(device);
        return std::chrono::steady_clock::now();
      };
      auto dp_ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      auto dp_t_total_start = dp_sync_now();
      // Phase A: offsets CPU read + min_step loop
      auto dp_t0 = dp_t_total_start;
#endif

      // Read min_step from pre-computed scalar in batch_state (set by CB engine),
      // or fall back to computing from offsets on CPU.
      StorageView offsets_cpu(DataType::INT32);
      dim_t min_step = 0;
      {
        const auto min_step_it = state.find("min_step");
        if (min_step_it != state.end() && min_step_it->second) {
          min_step = min_step_it->second.at<int32_t>(0);
        } else {
          // Fallback: read offsets on CPU.
          const auto step_offsets_cpu_it = state.find("step_offsets_cpu");
          if (step_offsets_cpu_it != state.end())
            offsets_cpu.shallow_copy(step_offsets_cpu_it->second);
          else if (step_offsets.device() != Device::CPU)
            offsets_cpu.copy_from(step_offsets.to(Device::CPU));
          else
            offsets_cpu.shallow_copy(const_cast<StorageView&>(step_offsets));
          const dim_t batch_size_raw = offsets_cpu.size();
          min_step = offsets_cpu.at<int32_t>(0);
          for (dim_t i = 1; i < batch_size_raw; ++i)
            min_step = std::min(min_step, dim_t(offsets_cpu.at<int32_t>(i)));
        }
      }
      // Ensure offsets_cpu is available for position encoder (if not already set).
      if (!offsets_cpu) {
        const auto step_offsets_cpu_it = state.find("step_offsets_cpu");
        if (step_offsets_cpu_it != state.end())
          offsets_cpu.shallow_copy(step_offsets_cpu_it->second);
        else if (step_offsets.device() != Device::CPU)
          offsets_cpu.copy_from(step_offsets.to(Device::CPU));
        else
          offsets_cpu.shallow_copy(const_cast<StorageView&>(step_offsets));
      }

#ifdef CT2_WITH_CUDA
      auto dp_t1 = dp_sync_now();  // End Phase A
#endif

      // Pre-allocate ping-pong buffers for stable GPU addresses (CUDA graph support).
      const dim_t total_batch = ids.dim(0);
      ensure_cb_buffers(total_batch, dtype, device);

      // cur tracks which ping-pong buffer holds the current data.
      int cur = 0;
      StorageView (&buf)[2] = _cb_buffers.buf;

      _embeddings(ids, buf[cur]);
      // _start_from_zero_embedding and _embeddings_scale are not used by Whisper.
      // For generality, apply them based on min_step.
      if (_start_from_zero_embedding)
        zero_first_timestep(buf[cur], min_step);
      if (_embeddings_scale && (!_start_from_zero_embedding || min_step != 0))
        ops::Mul()(buf[cur], *_embeddings_scale, buf[cur]);
      if (_project_in) {
        (*_project_in)(buf[cur], buf[1 - cur]);
        cur = 1 - cur;
      }
      if (buf[cur].rank() == 2)
        buf[cur].expand_dims(1);

#ifdef CT2_WITH_CUDA
      auto dp_t2 = dp_sync_now();  // End Phase B (embedding + scale + project_in)
#endif

      // Per-element position encoding.
      if (_position_encoder) {
#ifdef CT2_WITH_CUDA
        if (cuda::g_cuda_graph_capturing) {
          // During graph capture: pass GPU offsets so the captured kernel reads from
          // the GPU buffer (updated each step) instead of a baked CPU pointer offset.
          (*_position_encoder)(buf[cur], step_offsets);
        } else
#endif
        {
          // Normal path: pass CPU offsets to avoid GPU→CPU sync inside position encoder.
          (*_position_encoder)(buf[cur], static_cast<const StorageView&>(offsets_cpu));
        }
      }

#ifdef CT2_WITH_CUDA
      auto dp_t3 = dp_sync_now();  // End Phase C (position encoding)
#endif

      if (_layernorm_embedding)
        (*_layernorm_embedding)(buf[cur], buf[cur]);

#ifdef CT2_WITH_CUDA
      auto dp_t4 = dp_sync_now();  // End Phase D (layernorm_embedding)
#endif

      const dim_t batch_size = buf[cur].dim(0);
      const dim_t max_time = buf[cur].dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths_val;
      std::unique_ptr<const StorageView> input_lengths_mask;

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      // Build per-element attention mask using cache_lengths from state if available.
      const bool skip_self_mask = state.count("no_self_attn_mask") > 0;
      const auto cache_lengths_it = state.find("cache_lengths");
      if (cache_lengths_it != state.end() && !skip_self_mask) {
        dim_t num_heads = _num_heads;
        if (_tensor_parallel)
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());

        const StorageView& cl = cache_lengths_it->second;
        // Use pre-allocated attn_lengths buffer (stable address for CUDA graph).
        StorageView& attn_lengths = _cb_buffers.attn_lengths;

#ifdef CT2_WITH_CUDA
        if (cl.device() == Device::CUDA) {
          // GPU kernel: attn_lengths[i] = cl[i] + max_time.
          cuda::add_scalar_int32_gpu(attn_lengths.data<int32_t>(),
                                     cl.data<int32_t>(),
                                     static_cast<int32_t>(max_time),
                                     static_cast<int>(batch_size));
        } else
#endif
        {
          for (dim_t b = 0; b < batch_size; ++b)
            attn_lengths.at<int32_t>(b) = cl.at<int32_t>(b) + max_time;
        }

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          attn_lengths,
          num_heads,
          max_time,
          /*mask_future=*/false,
          multi_query);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

#ifdef CT2_WITH_CUDA
      auto dp_t5 = dp_sync_now();  // End Phase E (attention mask)
#endif

      // Access encoder memory when any element needs cross-attention projection.
      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        // In continuous batching, memory is needed when any element has step == 0.
        if (min_step <= 0 && state.count("memory")) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          dim_t num_heads = _num_heads;
          if (_tensor_parallel)
            num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

#ifdef CT2_WITH_CUDA
      auto dp_t6 = dp_sync_now();  // End Phase F (memory setup)
#endif

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      // Reuse pre-allocated position_bias (empty for Whisper, stable address).
      StorageView& position_bias = _cb_buffers.position_bias;

      // Read cache_write_positions for scatter-based cache updates (mid-decode slot insertion).
      const auto wp_it = state.find("cache_write_positions");
      const StorageView* write_pos = (wp_it != state.end() && wp_it->second)
                                     ? &wp_it->second : nullptr;

      // --- Per-layer profiling (CT2_LAYER_PROFILE env var) ---
      static const bool layer_profile = std::getenv("CT2_LAYER_PROFILE") != nullptr;
      static thread_local size_t lp_call_count = 0;
      static thread_local double lp_total_layer_ms = 0;
      static thread_local std::vector<double> lp_layer_ms;
      if (layer_profile && lp_layer_ms.size() != _layers.size())
        lp_layer_ms.assign(_layers.size(), 0.0);

      // No sliding window chunking — Whisper doesn't use it.
      // Ping-pong pattern: alternate buf[cur] (input) and buf[1-cur] (output).
      for (size_t l = 0; l < _layers.size(); ++l) {
        StorageView* cached_self_attn_keys = nullptr;
        StorageView* cached_self_attn_values = nullptr;
        StorageView* cached_attn_keys = nullptr;
        StorageView* cached_attn_values = nullptr;

        // In continuous batching all elements use cache (all offsets >= 0).
        const std::string l_str = std::to_string(l);
        cached_self_attn_keys = &state.at("self_keys_" + l_str);
        cached_self_attn_values = &state.at("self_values_" + l_str);
        if (_with_encoder_attention) {
          cached_attn_keys = &state.at("memory_keys_" + l_str);
          cached_attn_values = &state.at("memory_values_" + l_str);
        }

        // Set scatter write positions on self-attention (only for self-attention caches).
        _layers[l]->get_self_attention().set_cache_write_positions(write_pos);

        std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
        std::unique_ptr<StorageView> layer_attention;
        if (attention && heads_to_select)
          layer_attention = std::make_unique<StorageView>(device);

#ifdef CT2_WITH_CUDA
        if (layer_profile && device == Device::CUDA && !cuda::g_cuda_graph_capturing)
          synchronize_stream(device);
        auto lp_t0 = std::chrono::steady_clock::now();
#endif

        // offset=0 for per-element (position already encoded via per-element offsets).
        (*_layers[l])(buf[cur],
                      input_lengths_mask.get(),
                      memory,
                      memory_lengths_mask.get(),
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      cached_attn_keys,
                      cached_attn_values,
                      buf[1 - cur],
                      layer_attention.get(),
                      input_padder.get(),
                      memory_padder.get(),
                      return_normalized_attention(),
                      &position_bias,
                      /*offset=*/0);

#ifdef CT2_WITH_CUDA
        if (layer_profile && device == Device::CUDA && !cuda::g_cuda_graph_capturing) {
          synchronize_stream(device);
          auto lp_t1 = std::chrono::steady_clock::now();
          double ms = std::chrono::duration<double, std::milli>(lp_t1 - lp_t0).count();
          lp_layer_ms[l] += ms;
          lp_total_layer_ms += ms;
        }
#endif

        cur = 1 - cur;  // Output is now the current buffer.

        // Clear scatter positions after each layer call.
        _layers[l]->get_self_attention().set_cache_write_positions(nullptr);

        if (layer_attention) {
          // Use layer_attention's actual dtype (may differ from output_type() in mixed precision).
          alignment_heads.emplace_back(layer_attention->dtype(), device);
          ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
        }
      }

#ifdef CT2_WITH_CUDA
      auto dp_t7 = dp_sync_now();  // End Phase G (layer loop)
#endif

      // Print per-layer profile every 500 calls.
      if (layer_profile) {
        ++lp_call_count;
        if (lp_call_count % 500 == 0) {
          fprintf(stderr, "[LAYER PROFILE] calls=%zu total=%.1fms per_call=%.2fms\n",
                  lp_call_count, lp_total_layer_ms, lp_total_layer_ms / lp_call_count);
          for (size_t l = 0; l < lp_layer_ms.size(); ++l) {
            fprintf(stderr, "  layer[%zu] = %.1fms (%.2fms/call)\n",
                    l, lp_layer_ms[l], lp_layer_ms[l] / lp_call_count);
          }
        }
      }

      // In continuous batching mode, do not erase memory.
      if (retain_memory) {
        state["_retain_memory"] = StorageView();  // marker
      }

      if (attention && !alignment_heads.empty()) {
        // Ensure attention output has the correct dtype/device from the alignment heads.
        const auto heads_dtype = alignment_heads.front().dtype();
        const auto heads_device = alignment_heads.front().device();
        if (attention->dtype() != heads_dtype || attention->device() != heads_device)
          *attention = StorageView(heads_dtype, heads_device);

        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);
        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);
          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

#ifdef CT2_WITH_CUDA
      auto dp_t8 = dp_sync_now();  // End Phase H (attention concat)
#endif

      // After the layer loop, buf[cur] holds the final layer output.
      if (outputs) {
        if (_output_norm)
          (*_output_norm)(buf[cur], buf[cur]);
        if (_project_out) {
          (*_project_out)(buf[cur], buf[1 - cur]);
          cur = 1 - cur;
        }

        if (_outputs_scale)
          ops::Mul()(buf[cur], *_outputs_scale, buf[cur]);

        if (return_logits)
          _proj(buf[cur], *outputs);
        else
          *outputs = std::move(buf[cur]);

        if (!is_sequence)
          outputs->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*outputs);
      }

#ifdef CT2_WITH_CUDA
      auto dp_t9 = dp_sync_now();  // End Phase I (output_norm + LM head)

      if (dp_enabled && device == Device::CUDA) {
        dp_A += dp_ms(dp_t0, dp_t1);
        dp_B += dp_ms(dp_t1, dp_t2);
        dp_C += dp_ms(dp_t2, dp_t3);
        dp_D += dp_ms(dp_t3, dp_t4);
        dp_E += dp_ms(dp_t4, dp_t5);
        dp_F += dp_ms(dp_t5, dp_t6);
        dp_G += dp_ms(dp_t6, dp_t7);
        dp_H += dp_ms(dp_t7, dp_t8);
        dp_I += dp_ms(dp_t8, dp_t9);
        dp_total += dp_ms(dp_t_total_start, dp_t9);
        ++dp_calls;

        if (dp_calls % 500 == 0) {
          auto pct = [&](double v) { return dp_total > 0 ? 100.0 * v / dp_total : 0.0; };
          fprintf(stderr,
            "[DECODE PHASE CB] calls=%zu  total=%.1fms (%.2fms/call)\n"
            "  A offsets_cpu:    %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  B embed+scale:    %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  C pos_encoding:   %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  D layernorm_emb:  %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  E attn_mask:      %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  F memory_setup:   %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  G layer_loop:     %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  H attn_concat:    %8.1fms  %5.2fms/call  %5.1f%%\n"
            "  I norm+lm_head:   %8.1fms  %5.2fms/call  %5.1f%%\n",
            dp_calls, dp_total, dp_total / dp_calls,
            dp_A, dp_A / dp_calls, pct(dp_A),
            dp_B, dp_B / dp_calls, pct(dp_B),
            dp_C, dp_C / dp_calls, pct(dp_C),
            dp_D, dp_D / dp_calls, pct(dp_D),
            dp_E, dp_E / dp_calls, pct(dp_E),
            dp_F, dp_F / dp_calls, pct(dp_F),
            dp_G, dp_G / dp_calls, pct(dp_G),
            dp_H, dp_H / dp_calls, pct(dp_H),
            dp_I, dp_I / dp_calls, pct(dp_I));
        }
      }
#endif
    }

  }
}
