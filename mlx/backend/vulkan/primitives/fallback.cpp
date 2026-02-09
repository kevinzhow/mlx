// Copyright © 2026 MLX Vulkan Backend

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/transforms_impl.h"

namespace {

inline void prepare_inputs_for_cpu_fallback(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::Stream stream) {
  for (const auto& in : inputs) {
    auto& mutable_in = const_cast<mlx::core::array&>(in);
    if (mutable_in.status() == mlx::core::array::Status::unscheduled) {
      mutable_in.eval();
      continue;
    }

    if (mutable_in.event().valid()) {
      if (mutable_in.event().is_signaled()) {
        mutable_in.detach_event();
      } else if (mutable_in.event().stream() != stream) {
        mutable_in.event().wait(stream);
      }
    } else {
      mutable_in.wait();
    }
  }
}

inline bool is_row_contiguous_materialized(const mlx::core::array& arr) {
  return arr.flags().row_contiguous && arr.data_size() == arr.size();
}

inline float encode_push_constant_u32(uint32_t value) {
  float encoded = 0.0f;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&encoded, &value, sizeof(uint32_t));
  return encoded;
}

inline bool can_use_native_affine_bf16_quantized_matmul(
    const std::vector<mlx::core::array>& inputs,
    const mlx::core::array& out,
    int group_size,
    int bits,
    bool transpose,
    mlx::core::QuantizationMode mode) {
  if (inputs.size() != 4 || out.size() == 0) {
    return false;
  }
  if (mode != mlx::core::QuantizationMode::Affine || bits != 4 ||
      group_size != 128 || !transpose) {
    return false;
  }
  if (out.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      (out.size() % 2) != 0) {
    return false;
  }

  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& biases = inputs[3];

  if (x.dtype() != mlx::core::bfloat16 || w.dtype() != mlx::core::uint32 ||
      scales.dtype() != mlx::core::bfloat16 ||
      biases.dtype() != mlx::core::bfloat16 ||
      out.dtype() != mlx::core::bfloat16) {
    return false;
  }

  if (!is_row_contiguous_materialized(x) ||
      !is_row_contiguous_materialized(w) ||
      !is_row_contiguous_materialized(scales) ||
      !is_row_contiguous_materialized(biases) ||
      !out.flags().row_contiguous) {
    return false;
  }

  if (x.ndim() < 1 || w.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2) {
    return false;
  }

  int k = x.shape(-1);
  int n = out.shape(-1);
  if (k <= 0 || n <= 0 || (k % group_size) != 0) {
    return false;
  }

  int groups_per_col = k / group_size;
  if (groups_per_col <= 0) {
    return false;
  }
  if (w.shape(-2) != n || scales.shape(-2) != n || biases.shape(-2) != n) {
    return false;
  }
  if (scales.shape(-1) != groups_per_col || biases.shape(-1) != groups_per_col) {
    return false;
  }

  constexpr int values_per_u32 = 8; // bits=4
  if (w.shape(-1) * values_per_u32 != k) {
    return false;
  }

  int rows = static_cast<int>(x.size() / static_cast<size_t>(k));
  if (rows <= 0 || static_cast<size_t>(rows) * static_cast<size_t>(n) != out.size()) {
    return false;
  }

  // Shader reads bf16 via packed uint words, so require even element counts.
  if ((x.size() % 2) != 0 || (scales.size() % 2) != 0 || (biases.size() % 2) != 0) {
    return false;
  }

  return true;
}

inline bool can_use_native_rmsnorm_bf16(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    uint32_t& n_rows,
    uint32_t& axis_size,
    uint32_t& w_stride) {
  if (inputs.size() != 2 || outputs.size() != 1 || outputs[0].size() == 0) {
    return false;
  }
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& out = outputs[0];

  if (x.dtype() != mlx::core::bfloat16 || w.dtype() != mlx::core::bfloat16 ||
      out.dtype() != mlx::core::bfloat16) {
    return false;
  }
  if (!is_row_contiguous_materialized(x) || !out.flags().row_contiguous ||
      x.shape() != out.shape()) {
    return false;
  }
  if (x.ndim() < 1) {
    return false;
  }

  int64_t axis = x.shape(-1);
  if (axis <= 0 || (axis % 2) != 0 ||
      axis > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  if (x.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      x.size() != out.size() || (x.size() % static_cast<size_t>(axis)) != 0) {
    return false;
  }

  if (w.ndim() == 0) {
    if (!is_row_contiguous_materialized(w) || w.size() != 1) {
      return false;
    }
    w_stride = 0u;
  } else if (w.ndim() == 1) {
    if (!is_row_contiguous_materialized(w) || w.shape(0) != axis ||
        w.strides()[0] != 1) {
      return false;
    }
    w_stride = 1u;
  } else {
    return false;
  }

  axis_size = static_cast<uint32_t>(axis);
  n_rows = static_cast<uint32_t>(x.size() / static_cast<size_t>(axis));
  return n_rows > 0;
}

inline bool read_scalar_offset_i32(const mlx::core::array& offset, int32_t& out) {
  if (offset.size() != 1) {
    return false;
  }
  switch (offset.dtype()) {
    case mlx::core::int32:
      out = offset.data<int32_t>()[0];
      return true;
    case mlx::core::int64: {
      auto v = offset.data<int64_t>()[0];
      if (v < std::numeric_limits<int32_t>::min() ||
          v > std::numeric_limits<int32_t>::max()) {
        return false;
      }
      out = static_cast<int32_t>(v);
      return true;
    }
    default:
      return false;
  }
}

inline bool can_use_native_rope_bf16(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    int dims,
    bool traditional,
    float base,
    bool& with_freqs,
    uint32_t& n_rows,
    uint32_t& half_dims,
    uint32_t& row_stride,
    uint32_t& t_size) {
  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      outputs[0].size() == 0) {
    return false;
  }
  if (traditional) {
    return false;
  }
  with_freqs = inputs.size() == 3;
  if (!with_freqs && base <= 0.0f) {
    return false;
  }

  const auto& in = inputs[0];
  const auto& offset = inputs[1];
  const auto& out = outputs[0];
  if (in.dtype() != mlx::core::bfloat16 || out.dtype() != mlx::core::bfloat16 ||
      in.shape() != out.shape()) {
    return false;
  }
  if (!is_row_contiguous_materialized(in) || !out.flags().row_contiguous ||
      in.ndim() < 2) {
    return false;
  }

  int64_t d = in.shape(-1);
  int64_t t = in.shape(-2);
  if (t <= 0 || d <= 0 || dims <= 0 || dims != d || (dims % 2) != 0) {
    return false;
  }
  if (in.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      t > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  int32_t offset_value = 0;
  if (!read_scalar_offset_i32(offset, offset_value)) {
    return false;
  }
  (void)offset_value;

  row_stride = static_cast<uint32_t>(d);
  half_dims = static_cast<uint32_t>(dims / 2);
  if (with_freqs) {
    const auto& freqs = inputs[2];
    if (freqs.dtype() != mlx::core::float32 || freqs.ndim() != 1 ||
        freqs.shape(0) != static_cast<int64_t>(half_dims) ||
        !is_row_contiguous_materialized(freqs) || freqs.strides()[0] != 1) {
      return false;
    }
  }
  n_rows = static_cast<uint32_t>(in.size() / static_cast<size_t>(d));
  t_size = static_cast<uint32_t>(t);
  return n_rows > 0;
}

inline void materialize_and_share_fast_outputs(
    const std::vector<mlx::core::array>& fallback_outputs,
    std::vector<mlx::core::array>& outputs) {
  if (fallback_outputs.size() != outputs.size()) {
    throw std::runtime_error("[Vulkan fast] Fallback output arity mismatch.");
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& src = fallback_outputs[i];
    auto& mutable_src = const_cast<mlx::core::array&>(src);
    if (mutable_src.status() == mlx::core::array::Status::unscheduled) {
      mutable_src.eval();
    } else {
      mutable_src.wait();
    }
    outputs[i].copy_shared_buffer(src);
  }
}

inline void collect_keepalive_buffers(
    const mlx::core::array& arr,
    std::unordered_set<std::shared_ptr<mlx::core::array::Data>>& buffers) {
  if (auto data = arr.data_shared_ptr()) {
    buffers.insert(std::move(data));
  }
  for (const auto& sib : arr.siblings()) {
    if (auto sib_data = sib.data_shared_ptr()) {
      buffers.insert(std::move(sib_data));
    }
  }
}

template <typename OutputCollector>
inline void finalize_cpu_fallback(
    const std::vector<mlx::core::array>& inputs,
    OutputCollector&& collect_outputs) {
  auto cpu_stream = mlx::core::default_stream(mlx::core::Device::cpu);
  auto& encoder = mlx::core::cpu::get_command_encoder(cpu_stream);

  std::unordered_set<std::shared_ptr<mlx::core::array::Data>> buffers;
  for (const auto& in : inputs) {
    collect_keepalive_buffers(in, buffers);
  }
  collect_outputs(buffers);

  // Mirror cpu::eval() keepalive semantics for fallback-dispatched CPU tasks.
  encoder.dispatch(
      [buffers = std::move(buffers),
       temps = std::move(encoder.temporaries())]() mutable {});
  mlx::core::synchronize(cpu_stream);
}

template <typename EvalFn>
inline void run_cpu_fallback_single(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    EvalFn&& eval_fn) {
  prepare_inputs_for_cpu_fallback(inputs, out.primitive().stream());
  std::forward<EvalFn>(eval_fn)();
  finalize_cpu_fallback(inputs, [&](auto& buffers) {
    collect_keepalive_buffers(out, buffers);
  });
}

template <typename EvalFn>
inline void run_cpu_fallback_multi(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs,
    EvalFn&& eval_fn) {
  auto stream = outputs.empty() ? mlx::core::default_stream(mlx::core::default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  std::forward<EvalFn>(eval_fn)();
  finalize_cpu_fallback(inputs, [&](auto& buffers) {
    for (const auto& out : outputs) {
      collect_keepalive_buffers(out, buffers);
    }
  });
}

} // namespace

#define VULKAN_CPU_FALLBACK_MULTI(func)                               \
  void func::eval_gpu(                                                \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    run_cpu_fallback_multi(inputs, outputs, [&]() { eval_cpu(inputs, outputs); }); \
  }

#define VULKAN_CPU_FALLBACK(func)                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    run_cpu_fallback_single(inputs, out, [&]() { eval_cpu(inputs, out); }); \
  }

#define VULKAN_NO_GPU_MULTI(func)                                     \
  void func::eval_gpu(                                                \
      const std::vector<array>&, std::vector<array>&) {              \
    throw std::runtime_error(#func " has no Vulkan GPU implementation."); \
  }

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto stream = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  if (can_use_native_affine_bf16_quantized_matmul(
          inputs, out, group_size_, bits_, transpose_, mode_)) {
    try {
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto x_tensor = device.get_tensor(inputs[0]);
      auto w_tensor = device.get_tensor(inputs[1]);
      auto scales_tensor = device.get_tensor(inputs[2]);
      auto biases_tensor = device.get_tensor(inputs[3]);
      auto out_tensor = device.get_tensor(out);

      const uint32_t out_elems = static_cast<uint32_t>(out.size());
      const uint32_t n = static_cast<uint32_t>(out.shape(-1));
      const uint32_t k = static_cast<uint32_t>(inputs[0].shape(-1));
      const uint32_t groups_per_col =
          static_cast<uint32_t>(k / static_cast<uint32_t>(group_size_));
      const uint32_t w_words_per_col =
          static_cast<uint32_t>(inputs[1].shape(-1));
      const uint32_t out_words = out_elems / 2u;
      const uint32_t groups_x = std::max<uint32_t>(1, (out_words + 63u) / 64u);

      const std::vector<float> push_consts{
          encode_push_constant_u32(out_elems),
          encode_push_constant_u32(n),
          encode_push_constant_u32(k),
          encode_push_constant_u32(groups_per_col),
          encode_push_constant_u32(w_words_per_col)};

      encoder.record_tensor_sync_device(
          {x_tensor, w_tensor, scales_tensor, biases_tensor, out_tensor});
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128,
          {x_tensor, w_tensor, scales_tensor, biases_tensor, out_tensor},
          {groups_x, 1, 1},
          push_consts);
      encoder.record_tensor_sync_local({out_tensor});
      synchronize(stream);

      std::memcpy(out.data<void>(), out_tensor->rawData(), out.nbytes());
      return;
    } catch (const std::exception&) {
      // Fall through to CPU fallback.
    }
  }

  run_cpu_fallback_single(inputs, out, [&]() { eval_cpu(inputs, out); });
}

VULKAN_CPU_FALLBACK(Abs)
VULKAN_CPU_FALLBACK(AddMM)
VULKAN_CPU_FALLBACK(Arange)
VULKAN_CPU_FALLBACK(ArcCos)
VULKAN_CPU_FALLBACK(ArcCosh)
VULKAN_CPU_FALLBACK(ArcSin)
VULKAN_CPU_FALLBACK(ArcSinh)
VULKAN_CPU_FALLBACK(ArcTan)
VULKAN_CPU_FALLBACK(ArcTan2)
VULKAN_CPU_FALLBACK(ArcTanh)
VULKAN_CPU_FALLBACK(ArgPartition)
VULKAN_CPU_FALLBACK(ArgReduce)
VULKAN_CPU_FALLBACK(ArgSort)
VULKAN_CPU_FALLBACK(BitwiseBinary)
VULKAN_CPU_FALLBACK(BitwiseInvert)
VULKAN_CPU_FALLBACK(BlockMaskedMM)
VULKAN_CPU_FALLBACK(Ceil)
VULKAN_CPU_FALLBACK(Cholesky)
VULKAN_CPU_FALLBACK(Conjugate)
VULKAN_CPU_FALLBACK(Convolution)
// VULKAN_CPU_FALLBACK(Cos)  // Now has native Vulkan implementation
VULKAN_CPU_FALLBACK(Cosh)
VULKAN_CPU_FALLBACK(Equal)
VULKAN_CPU_FALLBACK(Erf)
VULKAN_CPU_FALLBACK(ErfInv)
VULKAN_CPU_FALLBACK(Exp)
VULKAN_CPU_FALLBACK(Expm1)
VULKAN_CPU_FALLBACK(FFT)
VULKAN_CPU_FALLBACK(Floor)
VULKAN_CPU_FALLBACK(Gather)
VULKAN_CPU_FALLBACK(GatherAxis)
VULKAN_CPU_FALLBACK(GatherMM)
VULKAN_CPU_FALLBACK(GatherQMM)
VULKAN_CPU_FALLBACK(Greater)
VULKAN_CPU_FALLBACK(GreaterEqual)
VULKAN_CPU_FALLBACK(Hadamard)
VULKAN_CPU_FALLBACK(Imag)
VULKAN_CPU_FALLBACK(Inverse)
VULKAN_CPU_FALLBACK(Less)
VULKAN_CPU_FALLBACK(LessEqual)
VULKAN_CPU_FALLBACK(Load)
VULKAN_CPU_FALLBACK(Log)
VULKAN_CPU_FALLBACK(Log1p)
VULKAN_CPU_FALLBACK(LogicalNot)
VULKAN_CPU_FALLBACK(LogicalAnd)
VULKAN_CPU_FALLBACK(LogicalOr)
VULKAN_CPU_FALLBACK(LogAddExp)
VULKAN_CPU_FALLBACK(LogSumExp)
VULKAN_CPU_FALLBACK(MaskedScatter)
VULKAN_CPU_FALLBACK(Matmul)
VULKAN_CPU_FALLBACK(Maximum)
VULKAN_CPU_FALLBACK(Minimum)
VULKAN_CPU_FALLBACK(Negative)
VULKAN_CPU_FALLBACK(NotEqual)
VULKAN_CPU_FALLBACK(Partition)
VULKAN_CPU_FALLBACK(Power)
VULKAN_CPU_FALLBACK(QQMatmul)
VULKAN_CPU_FALLBACK(RandomBits)
VULKAN_CPU_FALLBACK(Real)
VULKAN_CPU_FALLBACK(Reduce)
VULKAN_CPU_FALLBACK(Remainder)
VULKAN_CPU_FALLBACK(Round)
VULKAN_CPU_FALLBACK(Scan)
VULKAN_CPU_FALLBACK(Scatter)
VULKAN_CPU_FALLBACK(ScatterAxis)
VULKAN_CPU_FALLBACK(SegmentedMM)
VULKAN_CPU_FALLBACK(Select)
VULKAN_CPU_FALLBACK(Sigmoid)
VULKAN_CPU_FALLBACK(Sign)
// VULKAN_CPU_FALLBACK(Sin)  // Native Vulkan implementation in unary.cpp
VULKAN_CPU_FALLBACK(Sinh)
VULKAN_CPU_FALLBACK(Softmax)
VULKAN_CPU_FALLBACK(Sort)
VULKAN_CPU_FALLBACK(Sqrt)
VULKAN_CPU_FALLBACK(Square)
VULKAN_CPU_FALLBACK(Tan)
VULKAN_CPU_FALLBACK(Tanh)

VULKAN_CPU_FALLBACK_MULTI(Compiled)
VULKAN_CPU_FALLBACK_MULTI(DivMod)
VULKAN_CPU_FALLBACK_MULTI(Eig)
VULKAN_CPU_FALLBACK_MULTI(Eigh)
VULKAN_CPU_FALLBACK_MULTI(LUF)
VULKAN_CPU_FALLBACK_MULTI(QRF)
VULKAN_CPU_FALLBACK_MULTI(SVD)

bool fast::LayerNorm::use_fallback(Stream) {
  return true;
}

bool fast::RMSNorm::use_fallback(Stream stream) {
  return stream.device == Device::cpu || detail::in_tracing();
}

bool fast::RoPE::use_fallback(Stream stream) {
  return stream.device == Device::cpu || detail::in_tracing();
}

bool fast::ScaledDotProductAttention::use_fallback(
    const array&,
    const array&,
    const array&,
    bool,
    bool,
    bool,
    bool,
    bool,
    Stream) {
  return true;
}

bool fast::ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

bool fast::ScaledDotProductAttentionVJP::use_fallback(const array&, Stream) {
  return true;
}

VULKAN_NO_GPU_MULTI(fast::LayerNorm)
VULKAN_NO_GPU_MULTI(fast::LayerNormVJP)

void fast::RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  uint32_t n_rows = 0;
  uint32_t axis_size = 0;
  uint32_t w_stride = 0;
  if (can_use_native_rmsnorm_bf16(
          inputs, outputs, n_rows, axis_size, w_stride)) {
    try {
      auto& out = outputs[0];
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto x_tensor = device.get_tensor(inputs[0]);
      auto w_tensor = device.get_tensor(inputs[1]);
      auto out_tensor = device.get_tensor(out);

      const std::vector<float> push_consts{
          encode_push_constant_u32(n_rows),
          encode_push_constant_u32(axis_size),
          encode_push_constant_u32(w_stride),
          eps_};

      encoder.record_tensor_sync_device({x_tensor, w_tensor, out_tensor});
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::RMSNORM_BF16,
          {x_tensor, w_tensor, out_tensor},
          {n_rows, 1, 1},
          push_consts);
      encoder.record_tensor_sync_local({out_tensor});
      synchronize(stream);

      std::memcpy(out.data<void>(), out_tensor->rawData(), out.nbytes());
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  uint32_t n_rows = 0;
  uint32_t half_dims = 0;
  uint32_t row_stride = 0;
  uint32_t t_size = 0;
  bool with_freqs = false;
  if (can_use_native_rope_bf16(
          inputs,
          outputs,
          dims_,
          traditional_,
          base_,
          with_freqs,
          n_rows,
          half_dims,
          row_stride,
          t_size)) {
    int32_t offset_value = 0;
    if (read_scalar_offset_i32(inputs[1], offset_value)) {
      try {
        auto& out = outputs[0];
        if (!out.data_shared_ptr()) {
          out.set_data(allocator::malloc(out.nbytes()));
        }

        auto& device = vulkan::device(stream.device);
        auto& encoder = device.get_command_encoder(stream.index);
        encoder.begin_encoding();

        auto in_tensor = device.get_tensor(inputs[0]);
        auto out_tensor = device.get_tensor(out);

        const uint32_t offset_bits = static_cast<uint32_t>(offset_value);
        const uint32_t forward_flag = forward_ ? 1u : 0u;
        if (with_freqs) {
          auto freqs_tensor = device.get_tensor(inputs[2]);
          const std::vector<float> push_consts{
              encode_push_constant_u32(n_rows),
              encode_push_constant_u32(half_dims),
              encode_push_constant_u32(row_stride),
              encode_push_constant_u32(offset_bits),
              encode_push_constant_u32(t_size),
              scale_,
              encode_push_constant_u32(forward_flag)};

          encoder.record_tensor_sync_device({in_tensor, out_tensor, freqs_tensor});
          encoder.record_algo_dispatch(
              vulkan::KernelRegistry::ROPE_BF16_FREQS,
              {in_tensor, out_tensor, freqs_tensor},
              {n_rows, 1, 1},
              push_consts);
        } else {
          const std::vector<float> push_consts{
              encode_push_constant_u32(n_rows),
              encode_push_constant_u32(half_dims),
              encode_push_constant_u32(row_stride),
              encode_push_constant_u32(offset_bits),
              encode_push_constant_u32(t_size),
              scale_,
              std::log2(base_),
              encode_push_constant_u32(forward_flag)};

          encoder.record_tensor_sync_device({in_tensor, out_tensor});
          encoder.record_algo_dispatch(
              vulkan::KernelRegistry::ROPE_BF16_T1,
              {in_tensor, out_tensor},
              {n_rows, 1, 1},
              push_consts);
        }
        encoder.record_tensor_sync_local({out_tensor});
        synchronize(stream);

        std::memcpy(out.data<void>(), out_tensor->rawData(), out.nbytes());
        return;
      } catch (const std::exception&) {
        // Fall through to fallback path.
      }
    }
  }

  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_NO_GPU_MULTI(fast::ScaledDotProductAttention)
VULKAN_NO_GPU_MULTI(fast::ScaledDotProductAttentionVJP)
VULKAN_CPU_FALLBACK_MULTI(fast::ConvertFP8)
VULKAN_NO_GPU_MULTI(fast::CustomKernel)

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_CPU_FALLBACK_MULTI(distributed::AllReduce)
VULKAN_CPU_FALLBACK_MULTI(distributed::AllGather)
VULKAN_CPU_FALLBACK_MULTI(distributed::Send)
VULKAN_CPU_FALLBACK_MULTI(distributed::Recv)
VULKAN_CPU_FALLBACK_MULTI(distributed::ReduceScatter)

} // namespace mlx::core
