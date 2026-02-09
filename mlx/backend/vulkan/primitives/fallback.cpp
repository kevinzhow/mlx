// Copyright © 2026 MLX Vulkan Backend

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include "mlx/backend/vulkan/op_profiler.h"
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

inline void sync_inputs_to_host_if_needed(
    const std::vector<mlx::core::array>& inputs) {
  auto& device = mlx::core::vulkan::device(mlx::core::Device::gpu);
  for (const auto& in : inputs) {
    device.sync_array_to_host_if_needed(in);
  }
}

inline bool is_row_contiguous_materialized(const mlx::core::array& arr) {
  return arr.flags().row_contiguous && arr.data_size() == arr.size();
}

inline bool rope_debug_reject_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_ROPE_REJECT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline void log_rope_reject(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    int dims,
    bool traditional,
    float base,
    const char* reason) {
  if (!rope_debug_reject_enabled()) {
    return;
  }

  auto shape_string = [](const mlx::core::array& a) {
    std::string s = "[";
    for (int i = 0; i < a.ndim(); ++i) {
      if (i > 0) {
        s += ",";
      }
      s += std::to_string(a.shape(i));
    }
    s += "]";
    return s;
  };
  auto strides_string = [](const mlx::core::array& a) {
    std::string s = "[";
    const auto& st = a.strides();
    for (size_t i = 0; i < st.size(); ++i) {
      if (i > 0) {
        s += ",";
      }
      s += std::to_string(st[i]);
    }
    s += "]";
    return s;
  };

  std::cerr << "[VulkanRoPEReject] reason=" << (reason ? reason : "unknown")
            << " dims=" << dims
            << " traditional=" << (traditional ? 1 : 0)
            << " base=" << base;
  if (!inputs.empty()) {
    std::cerr << " in.dtype=" << inputs[0].dtype()
              << " in.shape=" << shape_string(inputs[0]);
    std::cerr << " in.strides=" << strides_string(inputs[0]);
    std::cerr << " in.row=" << (inputs[0].flags().row_contiguous ? 1 : 0);
  }
  if (inputs.size() > 1) {
    std::cerr << " offset.dtype=" << inputs[1].dtype()
              << " offset.shape=" << shape_string(inputs[1]);
  }
  if (inputs.size() > 2) {
    std::cerr << " freqs.dtype=" << inputs[2].dtype()
              << " freqs.shape=" << shape_string(inputs[2]);
  }
  if (!outputs.empty()) {
    std::cerr << " out.dtype=" << outputs[0].dtype()
              << " out.shape=" << shape_string(outputs[0]);
    std::cerr << " out.strides=" << strides_string(outputs[0]);
    std::cerr << " out.row=" << (outputs[0].flags().row_contiguous ? 1 : 0);
  }
  std::cerr << "\n";
}

inline kp::Tensor::TensorDataTypes to_kompute_dtype(mlx::core::Dtype dtype) {
  switch (dtype) {
    case mlx::core::bool_:
      return kp::Tensor::TensorDataTypes::eBool;
    case mlx::core::uint8:
    case mlx::core::uint16:
    case mlx::core::uint32:
    case mlx::core::uint64:
      return kp::Tensor::TensorDataTypes::eUnsignedInt;
    case mlx::core::int8:
    case mlx::core::int16:
    case mlx::core::int32:
    case mlx::core::int64:
      return kp::Tensor::TensorDataTypes::eInt;
    case mlx::core::float16:
    case mlx::core::float32:
    case mlx::core::bfloat16:
    case mlx::core::complex64:
      return kp::Tensor::TensorDataTypes::eFloat;
    case mlx::core::float64:
      return kp::Tensor::TensorDataTypes::eDouble;
  }
  return kp::Tensor::TensorDataTypes::eFloat;
}

struct CachedQmmConstTensorEntry {
  std::weak_ptr<kp::Tensor> tensor;
  const void* data_ptr{nullptr};
  size_t nbytes{0};
  mlx::core::Dtype dtype{mlx::core::float32};
  bool uploaded{false};
};

struct QmmConstTensorRef {
  std::uintptr_t key{0};
  std::shared_ptr<kp::Tensor> tensor;
  bool needs_sync{true};
  bool cacheable{false};
};

std::mutex& qmm_const_tensor_cache_mutex() {
  static std::mutex mtx;
  return mtx;
}

std::unordered_map<std::uintptr_t, CachedQmmConstTensorEntry>&
qmm_const_tensor_cache() {
  static std::unordered_map<std::uintptr_t, CachedQmmConstTensorEntry> cache;
  return cache;
}

inline QmmConstTensorRef get_qmm_const_tensor(
    const mlx::core::array& arr,
    mlx::core::vulkan::Device& device) {
  // Cache only leaf arrays (model weights/scales/biases). Dynamic graph
  // temporaries may mutate and should remain uncached.
  if (arr.has_primitive()) {
    return {0, device.get_tensor(arr), true, false};
  }

  auto manager = device.kompute_manager();
  if (!manager) {
    return {0, device.get_tensor(arr), true, false};
  }

  const auto key = arr.id();
  const void* ptr = arr.data<void>();
  const size_t nbytes = arr.nbytes();
  const auto dtype = arr.dtype();

  std::lock_guard<std::mutex> lock(qmm_const_tensor_cache_mutex());
  auto& cache = qmm_const_tensor_cache();
  auto it = cache.find(key);
  if (it != cache.end()) {
    const bool same_meta =
        it->second.data_ptr == ptr && it->second.nbytes == nbytes &&
        it->second.dtype == dtype;
    if (same_meta) {
      if (auto tensor = it->second.tensor.lock()) {
        return {key, tensor, !it->second.uploaded, true};
      }
    }
    cache.erase(it);
  }

  auto tensor = manager->tensor(
      const_cast<void*>(ptr),
      static_cast<uint32_t>(arr.size()),
      static_cast<uint32_t>(arr.itemsize()),
      to_kompute_dtype(dtype));
  cache[key] = CachedQmmConstTensorEntry{
      tensor, ptr, nbytes, dtype, false};
  return {key, tensor, true, true};
}

inline void mark_qmm_const_tensor_uploaded(std::uintptr_t key) {
  std::lock_guard<std::mutex> lock(qmm_const_tensor_cache_mutex());
  auto& cache = qmm_const_tensor_cache();
  auto it = cache.find(key);
  if (it != cache.end()) {
    it->second.uploaded = true;
  }
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
    uint32_t& t_size,
    uint32_t& rows_per_batch,
    uint32_t& offset_is_vector,
    const char** reject_reason = nullptr) {
  auto reject = [&](const char* reason) {
    if (reject_reason) {
      *reject_reason = reason;
    }
    return false;
  };

  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      outputs[0].size() == 0) {
    return reject("inputs_or_outputs_shape");
  }
  with_freqs = inputs.size() == 3;
  if (!with_freqs && base <= 0.0f) {
    return reject("invalid_base");
  }

  const auto& in = inputs[0];
  const auto& offset = inputs[1];
  const auto& out = outputs[0];
  if (in.dtype() != mlx::core::bfloat16 || out.dtype() != mlx::core::bfloat16 ||
      in.shape() != out.shape()) {
    return reject("dtype_or_shape_mismatch");
  }
  if (!is_row_contiguous_materialized(in) || !out.flags().row_contiguous ||
      in.ndim() < 2) {
    return reject("in_or_out_layout");
  }

  int64_t d = in.shape(-1);
  int64_t t = in.shape(-2);
  if (t <= 0 || d <= 0 || dims <= 0 || dims != d || (dims % 2) != 0) {
    return reject("dims_constraints");
  }
  if (in.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      t > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return reject("size_overflow");
  }

  const int64_t batch = in.shape(0);
  if (batch <= 0 || batch > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return reject("batch_constraints");
  }

  row_stride = static_cast<uint32_t>(d);
  half_dims = static_cast<uint32_t>(dims / 2);
  n_rows = static_cast<uint32_t>(in.size() / static_cast<size_t>(d));
  if (n_rows == 0) {
    return reject("zero_rows");
  }
  if (with_freqs) {
    const auto& freqs = inputs[2];
    if (freqs.dtype() != mlx::core::float32 || freqs.ndim() != 1 ||
        freqs.shape(0) != static_cast<int64_t>(half_dims) ||
        !is_row_contiguous_materialized(freqs) || freqs.strides()[0] != 1) {
      return reject("freqs_layout_or_dtype");
    }
  }

  if (offset.size() == 1) {
    int32_t offset_value = 0;
    if (!read_scalar_offset_i32(offset, offset_value) ||
        !is_row_contiguous_materialized(offset)) {
      return reject("scalar_offset_constraints");
    }
    (void)offset_value;
    rows_per_batch = 1u;
    offset_is_vector = 0u;
  } else {
    if (offset.dtype() != mlx::core::int32 || offset.ndim() != 1 ||
        offset.shape(0) != batch || !is_row_contiguous_materialized(offset) ||
        offset.strides()[0] != 1) {
      return reject("vector_offset_constraints");
    }
    const uint32_t batch_u32 = static_cast<uint32_t>(batch);
    if (batch_u32 == 0 || (n_rows % batch_u32) != 0) {
      return reject("rows_per_batch_divisibility");
    }
    rows_per_batch = n_rows / batch_u32;
    if (rows_per_batch == 0) {
      return reject("zero_rows_per_batch");
    }
    offset_is_vector = 1u;
  }

  t_size = static_cast<uint32_t>(t);
  if (reject_reason) {
    *reject_reason = nullptr;
  }
  return n_rows > 0;
}

inline bool can_use_native_sdpa_bf16_decode_q1(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    bool do_causal,
    bool has_sinks,
    bool output_logsumexp,
    uint32_t& batch_size,
    uint32_t& n_q_heads,
    uint32_t& n_kv_heads,
    uint32_t& k_len,
    uint32_t& qk_dim,
    uint32_t& v_dim) {
  if (has_sinks || output_logsumexp || inputs.size() != 3 ||
      outputs.size() != 1 || outputs[0].size() == 0) {
    return false;
  }

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const auto& out = outputs[0];
  if (q.dtype() != mlx::core::bfloat16 || k.dtype() != mlx::core::bfloat16 ||
      v.dtype() != mlx::core::bfloat16 || out.dtype() != mlx::core::bfloat16) {
    return false;
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
      !is_row_contiguous_materialized(q) || !is_row_contiguous_materialized(k) ||
      !is_row_contiguous_materialized(v) || !out.flags().row_contiguous) {
    return false;
  }

  if (q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(3) != k.shape(3) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2) || q.shape(2) != 1) {
    return false;
  }
  if (do_causal && q.shape(2) != 1) {
    return false;
  }

  const int64_t b = q.shape(0);
  const int64_t hq = q.shape(1);
  const int64_t hkv = k.shape(1);
  const int64_t lk = k.shape(2);
  const int64_t dq = q.shape(3);
  const int64_t dv = v.shape(3);
  if (b <= 0 || hq <= 0 || hkv <= 0 || lk <= 0 || dq <= 0 || dv <= 0 ||
      (hq % hkv) != 0) {
    return false;
  }
  if (lk > 8) {
    return false;
  }
  if (dq > 256 || dv > 256) {
    return false;
  }
  if ((q.size() % 2) != 0 || (k.size() % 2) != 0 || (v.size() % 2) != 0 ||
      (out.size() % 2) != 0) {
    return false;
  }
  if (b > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      hq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      hkv > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      lk > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      dq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      dv > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  if (out.ndim() != 4 || out.shape(0) != b || out.shape(1) != hq ||
      out.shape(2) != 1 || out.shape(3) != dv) {
    return false;
  }

  batch_size = static_cast<uint32_t>(b);
  n_q_heads = static_cast<uint32_t>(hq);
  n_kv_heads = static_cast<uint32_t>(hkv);
  k_len = static_cast<uint32_t>(lk);
  qk_dim = static_cast<uint32_t>(dq);
  v_dim = static_cast<uint32_t>(dv);
  return true;
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
    OutputCollector&& collect_outputs,
    mlx::core::vulkan::OpProfileScope* profile) {
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
  if (profile) {
    profile->mark_sync();
  }
}

template <typename EvalFn>
inline void run_cpu_fallback_single(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    EvalFn&& eval_fn,
    mlx::core::vulkan::OpProfileScope* profile) {
  auto stream = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  sync_inputs_to_host_if_needed(inputs);
  std::forward<EvalFn>(eval_fn)();
  mlx::core::vulkan::device(mlx::core::Device::gpu).invalidate_tensor(out);
  finalize_cpu_fallback(
      inputs,
      [&](auto& buffers) { collect_keepalive_buffers(out, buffers); },
      profile);
}

template <typename EvalFn>
inline void run_cpu_fallback_multi(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs,
    EvalFn&& eval_fn,
    mlx::core::vulkan::OpProfileScope* profile) {
  auto stream = outputs.empty() ? mlx::core::default_stream(mlx::core::default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  sync_inputs_to_host_if_needed(inputs);
  std::forward<EvalFn>(eval_fn)();
  auto& device = mlx::core::vulkan::device(mlx::core::Device::gpu);
  for (const auto& out : outputs) {
    device.invalidate_tensor(out);
  }
  finalize_cpu_fallback(
      inputs,
      [&](auto& buffers) {
        for (const auto& out : outputs) {
          collect_keepalive_buffers(out, buffers);
        }
      },
      profile);
}

} // namespace

#define VULKAN_CPU_FALLBACK_MULTI(func)                               \
  void func::eval_gpu(                                                \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    vulkan::OpProfileScope profile(#func);                            \
    profile.mark_fallback();                                          \
    run_cpu_fallback_multi(                                           \
        inputs, outputs, [&]() { eval_cpu(inputs, outputs); }, &profile); \
  }

#define VULKAN_CPU_FALLBACK(func)                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    vulkan::OpProfileScope profile(#func);                            \
    profile.mark_fallback();                                          \
    run_cpu_fallback_single(                                          \
        inputs, out, [&]() { eval_cpu(inputs, out); }, &profile);     \
  }

#define VULKAN_NO_GPU_MULTI(func)                                     \
  void func::eval_gpu(                                                \
      const std::vector<array>&, std::vector<array>&) {              \
    throw std::runtime_error(#func " has no Vulkan GPU implementation."); \
  }

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("QuantizedMatmul");
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
      auto w_cached = get_qmm_const_tensor(inputs[1], device);
      auto scales_cached = get_qmm_const_tensor(inputs[2], device);
      auto biases_cached = get_qmm_const_tensor(inputs[3], device);
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

      // Output tensor is fully overwritten by the kernel; no need to upload it.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors{x_tensor};
      if (w_cached.needs_sync) {
        sync_tensors.push_back(w_cached.tensor);
      }
      if (scales_cached.needs_sync) {
        sync_tensors.push_back(scales_cached.tensor);
      }
      if (biases_cached.needs_sync) {
        sync_tensors.push_back(biases_cached.tensor);
      }
      encoder.record_tensor_sync_device(sync_tensors);
      if (w_cached.cacheable && w_cached.needs_sync) {
        mark_qmm_const_tensor_uploaded(w_cached.key);
      }
      if (scales_cached.cacheable && scales_cached.needs_sync) {
        mark_qmm_const_tensor_uploaded(scales_cached.key);
      }
      if (biases_cached.cacheable && biases_cached.needs_sync) {
        mark_qmm_const_tensor_uploaded(biases_cached.key);
      }
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128,
          {x_tensor,
           w_cached.tensor,
           scales_cached.tensor,
           biases_cached.tensor,
           out_tensor},
          {groups_x, 1, 1},
          push_consts);
      if (out.data<void>() != out_tensor->rawData()) {
        device.mark_tensor_host_dirty(out, stream.index);
      }
      return;
    } catch (const std::exception&) {
      // Fall through to CPU fallback.
    }
  }

  profile.mark_fallback();
  run_cpu_fallback_single(inputs, out, [&]() { eval_cpu(inputs, out); }, &profile);
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
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream stream) {
  if (stream.device == Device::cpu || detail::in_tracing() || has_mask ||
      do_causal || is_training || output_logsumexp) {
    return true;
  }

  if (q.dtype() != mlx::core::bfloat16 || k.dtype() != mlx::core::bfloat16 ||
      v.dtype() != mlx::core::bfloat16) {
    return true;
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    return true;
  }
  if (!q.flags().row_contiguous || !k.flags().row_contiguous ||
      !v.flags().row_contiguous) {
    return true;
  }

  if (q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(3) != k.shape(3) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2) || q.shape(2) != 1) {
    return true;
  }
  if (k.shape(1) <= 0 || (q.shape(1) % k.shape(1)) != 0) {
    return true;
  }
  if (k.shape(2) <= 0 || k.shape(2) > 8 || q.shape(3) <= 0 || q.shape(3) > 256 ||
      v.shape(3) <= 0 || v.shape(3) > 256) {
    return true;
  }
  if ((q.size() % 2) != 0 || (k.size() % 2) != 0 || (v.size() % 2) != 0) {
    return true;
  }

  return false;
}

bool fast::ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

bool fast::ScaledDotProductAttentionVJP::use_fallback(const array&, Stream) {
  return true;
}

VULKAN_NO_GPU_MULTI(fast::LayerNorm)
VULKAN_NO_GPU_MULTI(fast::LayerNormVJP)

void fast::RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RMSNorm");
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

      // Output tensor is write-only in this dispatch.
      encoder.record_tensor_sync_device({x_tensor, w_tensor});
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::RMSNORM_BF16,
          {x_tensor, w_tensor, out_tensor},
          {n_rows, 1, 1},
          push_consts);
      if (out.data<void>() != out_tensor->rawData()) {
        device.mark_tensor_host_dirty(out, stream.index);
      }
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  profile.mark_fallback();
  sync_inputs_to_host_if_needed(inputs);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RMSNormVJP");
  profile.mark_fallback();
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  sync_inputs_to_host_if_needed(inputs);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RoPE");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  uint32_t n_rows = 0;
  uint32_t half_dims = 0;
  uint32_t row_stride = 0;
  uint32_t t_size = 0;
  uint32_t rows_per_batch = 0;
  uint32_t offset_is_vector = 0;
  bool with_freqs = false;
  const char* rope_reject_reason = nullptr;
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
          t_size,
          rows_per_batch,
          offset_is_vector,
          &rope_reject_reason)) {
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
      auto offset_tensor = device.get_tensor(inputs[1]);

      const uint32_t forward_flag = forward_ ? 1u : 0u;
      const uint32_t traditional_flag = traditional_ ? 1u : 0u;
      if (with_freqs) {
        auto freqs_tensor = device.get_tensor(inputs[2]);
        const std::vector<float> push_consts{
            encode_push_constant_u32(n_rows),
            encode_push_constant_u32(half_dims),
            encode_push_constant_u32(row_stride),
            encode_push_constant_u32(t_size),
            encode_push_constant_u32(rows_per_batch),
            encode_push_constant_u32(offset_is_vector),
            encode_push_constant_u32(traditional_flag),
            scale_,
            encode_push_constant_u32(forward_flag)};

        // Output tensor is written by the kernel; skip redundant upload.
        encoder.record_tensor_sync_device(
            {in_tensor, freqs_tensor, offset_tensor});
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::ROPE_BF16_FREQS,
            {in_tensor, out_tensor, freqs_tensor, offset_tensor},
            {n_rows, 1, 1},
            push_consts);
      } else {
        const std::vector<float> push_consts{
            encode_push_constant_u32(n_rows),
            encode_push_constant_u32(half_dims),
            encode_push_constant_u32(row_stride),
            encode_push_constant_u32(t_size),
            encode_push_constant_u32(rows_per_batch),
            encode_push_constant_u32(offset_is_vector),
            encode_push_constant_u32(traditional_flag),
            scale_,
            std::log2(base_),
            encode_push_constant_u32(forward_flag)};

        // Output tensor is written by the kernel; skip redundant upload.
        encoder.record_tensor_sync_device(
            {in_tensor, offset_tensor});
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::ROPE_BF16_T1,
            {in_tensor, out_tensor, offset_tensor},
            {n_rows, 1, 1},
            push_consts);
      }
      if (out.data<void>() != out_tensor->rawData()) {
        device.mark_tensor_host_dirty(out, stream.index);
      }
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  log_rope_reject(
      inputs, outputs, dims_, traditional_, base_, rope_reject_reason);
  profile.mark_fallback();
  sync_inputs_to_host_if_needed(inputs);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::ScaledDotProductAttention");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  uint32_t batch_size = 0;
  uint32_t n_q_heads = 0;
  uint32_t n_kv_heads = 0;
  uint32_t k_len = 0;
  uint32_t qk_dim = 0;
  uint32_t v_dim = 0;
  if (can_use_native_sdpa_bf16_decode_q1(
          inputs,
          outputs,
          do_causal_,
          has_sinks_,
          output_logsumexp_,
          batch_size,
          n_q_heads,
          n_kv_heads,
          k_len,
          qk_dim,
          v_dim)) {
    try {
      auto& out = outputs[0];
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto q_tensor = device.get_tensor(inputs[0]);
      auto k_tensor = device.get_tensor(inputs[1]);
      auto v_tensor = device.get_tensor(inputs[2]);
      auto out_tensor = device.get_tensor(out);

      const uint32_t n_work = batch_size * n_q_heads;
      const std::vector<float> push_consts{
          encode_push_constant_u32(batch_size),
          encode_push_constant_u32(n_q_heads),
          encode_push_constant_u32(n_kv_heads),
          encode_push_constant_u32(k_len),
          encode_push_constant_u32(qk_dim),
          encode_push_constant_u32(v_dim),
          scale_};

      // Output tensor is write-only in this decode kernel.
      encoder.record_tensor_sync_device({q_tensor, k_tensor, v_tensor});
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1,
          {q_tensor, k_tensor, v_tensor, out_tensor},
          {n_work, 1, 1},
          push_consts);
      if (out.data<void>() != out_tensor->rawData()) {
        device.mark_tensor_host_dirty(out, stream.index);
      }
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  profile.mark_fallback();
  sync_inputs_to_host_if_needed(inputs);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_NO_GPU_MULTI(fast::ScaledDotProductAttentionVJP)
VULKAN_CPU_FALLBACK_MULTI(fast::ConvertFP8)
VULKAN_NO_GPU_MULTI(fast::CustomKernel)

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::Quantize");
  profile.mark_fallback();
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  sync_inputs_to_host_if_needed(inputs);
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_CPU_FALLBACK_MULTI(distributed::AllReduce)
VULKAN_CPU_FALLBACK_MULTI(distributed::AllGather)
VULKAN_CPU_FALLBACK_MULTI(distributed::Send)
VULKAN_CPU_FALLBACK_MULTI(distributed::Recv)
VULKAN_CPU_FALLBACK_MULTI(distributed::ReduceScatter)

} // namespace mlx::core
