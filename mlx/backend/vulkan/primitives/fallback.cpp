// Copyright © 2026 MLX Vulkan Backend

#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

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
VULKAN_CPU_FALLBACK(QuantizedMatmul)
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

bool fast::RMSNorm::use_fallback(Stream) {
  return true;
}

bool fast::RoPE::use_fallback(Stream) {
  return true;
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
VULKAN_NO_GPU_MULTI(fast::RMSNorm)
VULKAN_NO_GPU_MULTI(fast::RMSNormVJP)
VULKAN_NO_GPU_MULTI(fast::RoPE)
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
  auto fallback_outputs = fallback_(inputs);
  if (fallback_outputs.size() != outputs.size()) {
    throw std::runtime_error(
        "[Vulkan fast::Quantize] Fallback output arity mismatch.");
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& src = fallback_outputs[i];
    auto& mutable_src = const_cast<array&>(src);
    if (mutable_src.status() == array::Status::unscheduled) {
      mutable_src.eval();
    } else {
      mutable_src.wait();
    }
    outputs[i].copy_shared_buffer(src);
  }
}

VULKAN_CPU_FALLBACK_MULTI(distributed::AllReduce)
VULKAN_CPU_FALLBACK_MULTI(distributed::AllGather)
VULKAN_CPU_FALLBACK_MULTI(distributed::Send)
VULKAN_CPU_FALLBACK_MULTI(distributed::Recv)
VULKAN_CPU_FALLBACK_MULTI(distributed::ReduceScatter)

} // namespace mlx::core
