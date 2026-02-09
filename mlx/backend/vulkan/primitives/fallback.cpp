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

// Accept dense row-major views even if data_size != size (e.g. KV cache views).
// This keeps layout constraints strict while allowing sliced/cached buffers.
inline bool is_dense_row_major_view(const mlx::core::array& arr) {
  if (!arr.flags().row_contiguous) {
    return false;
  }
  if (arr.ndim() == 0) {
    return true;
  }

  int64_t expected = 1;
  for (int i = arr.ndim() - 1; i >= 0; --i) {
    const int64_t dim = arr.shape(i);
    const int64_t stride = arr.strides(i);
    if (dim <= 0) {
      return false;
    }
    if (dim != 1 && stride != expected) {
      return false;
    }
    if (expected > std::numeric_limits<int64_t>::max() / dim) {
      return false;
    }
    expected *= dim;
  }
  return true;
}

inline bool is_rope_head_seq_transposed_layout(const mlx::core::array& arr) {
  if (arr.ndim() != 4) {
    return false;
  }
  const int64_t heads = arr.shape(1);
  const int64_t t_size = arr.shape(2);
  const int64_t d_size = arr.shape(3);
  if (heads <= 0 || t_size <= 0 || d_size <= 0) {
    return false;
  }
  const auto& st = arr.strides();
  return st[0] == t_size * heads * d_size && st[1] == d_size &&
      st[2] == heads * d_size && st[3] == 1;
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

inline bool sdpa_debug_reject_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_SDPA_REJECT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool env_flag_default_true(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return true;
  }
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "off") == 0) {
    return false;
  }
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
      std::strcmp(v, "on") == 0;
}

inline bool env_flag_default_false(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return false;
  }
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "off") == 0) {
    return false;
  }
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
      std::strcmp(v, "on") == 0;
}

inline bool native_qmm_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE");
  return enabled;
}

inline bool native_rmsnorm_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_RMSNORM_NATIVE");
  return enabled;
}

inline bool native_rope_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_ROPE_NATIVE");
  return enabled;
}

inline bool native_sdpa_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_SDPA_NATIVE");
  return enabled;
}

inline uint32_t native_sdpa_max_k_len() {
  static const uint32_t max_k_len = []() -> uint32_t {
    constexpr uint32_t kDefault = 8u;
    const char* v = std::getenv("MLX_VK_SDPA_MAX_K_LEN");
    if (!v || v[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v || *end != '\0' || parsed == 0ul) {
      return kDefault;
    }
    if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
      return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(parsed);
  }();
  return max_k_len;
}

inline uint32_t parse_env_u32(const char* name, uint32_t default_value) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0') {
    return default_value;
  }
  if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(parsed);
}

inline uint32_t native_sdpa_splitk_min_k_len() {
  static const uint32_t value =
      std::max<uint32_t>(2u, parse_env_u32("MLX_VK_SDPA_SPLITK_MIN_K_LEN", 16u));
  return value;
}

inline uint32_t native_sdpa_splitk_target_chunk() {
  static const uint32_t value = std::max<uint32_t>(
      1u, parse_env_u32("MLX_VK_SDPA_SPLITK_TARGET_CHUNK", 16u));
  return value;
}

inline uint32_t native_sdpa_splitk_max_parts() {
  static const uint32_t value =
      std::max<uint32_t>(1u, parse_env_u32("MLX_VK_SDPA_SPLITK_MAX_PARTS", 8u));
  return value;
}

inline uint32_t native_sdpa_splitk_forced_parts() {
  static const uint32_t value = parse_env_u32("MLX_VK_SDPA_SPLIT_K", 0u);
  return value;
}

inline uint32_t select_sdpa_split_k(uint32_t k_len) {
  const uint32_t forced = native_sdpa_splitk_forced_parts();
  if (forced > 0u) {
    return std::min<uint32_t>(k_len, std::max<uint32_t>(1u, forced));
  }
  if (k_len < native_sdpa_splitk_min_k_len()) {
    return 1u;
  }
  const uint32_t target_chunk = native_sdpa_splitk_target_chunk();
  uint32_t split_k = (k_len + target_chunk - 1u) / target_chunk;
  split_k = std::max<uint32_t>(2u, split_k);
  split_k = std::min<uint32_t>(split_k, native_sdpa_splitk_max_parts());
  split_k = std::min<uint32_t>(split_k, k_len);
  return std::max<uint32_t>(1u, split_k);
}

inline bool native_rope_hs_transposed_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_ENABLE_ROPE_HS_TRANSPOSED");
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

inline void log_sdpa_reject(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const char* reason,
    bool has_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp) {
  if (!sdpa_debug_reject_enabled()) {
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

  std::cerr << "[VulkanSDPAReject] reason=" << (reason ? reason : "unknown")
            << " has_mask=" << (has_mask ? 1 : 0)
            << " do_causal=" << (do_causal ? 1 : 0)
            << " training=" << (is_training ? 1 : 0)
            << " logsumexp=" << (output_logsumexp ? 1 : 0)
            << " q.dtype=" << q.dtype() << " q.shape=" << shape_string(q)
            << " q.strides=" << strides_string(q)
            << " q.row=" << (q.flags().row_contiguous ? 1 : 0)
            << " k.dtype=" << k.dtype() << " k.shape=" << shape_string(k)
            << " k.strides=" << strides_string(k)
            << " k.row=" << (k.flags().row_contiguous ? 1 : 0)
            << " v.dtype=" << v.dtype() << " v.shape=" << shape_string(v)
            << " v.strides=" << strides_string(v)
            << " v.row=" << (v.flags().row_contiguous ? 1 : 0)
            << "\n";
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

inline uint32_t encode_push_constant_u32(uint32_t value) {
  return value;
}

inline uint32_t encode_push_constant_f32(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&bits, &value, sizeof(uint32_t));
  return bits;
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
    uint32_t& input_hs_transposed,
    uint32_t& input_batch_stride,
    uint32_t& input_head_stride,
    uint32_t& input_t_stride,
    uint32_t& n_heads,
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
  const bool in_row_contiguous = is_row_contiguous_materialized(in);
  const bool allow_hs_transposed = native_rope_hs_transposed_enabled();
  const bool in_hs_transposed =
      allow_hs_transposed && !in_row_contiguous && in.data_size() == in.size() &&
      is_rope_head_seq_transposed_layout(in);
  if ((!in_row_contiguous && !in_hs_transposed) || !out.flags().row_contiguous ||
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
  const uint32_t batch_u32 = static_cast<uint32_t>(batch);

  row_stride = static_cast<uint32_t>(d);
  half_dims = static_cast<uint32_t>(dims / 2);
  n_rows = static_cast<uint32_t>(in.size() / static_cast<size_t>(d));
  if (n_rows == 0) {
    return reject("zero_rows");
  }
  if (batch_u32 == 0 || (n_rows % batch_u32) != 0) {
    return reject("rows_per_batch_divisibility");
  }
  rows_per_batch = n_rows / batch_u32;
  if (rows_per_batch == 0) {
    return reject("zero_rows_per_batch");
  }

  input_hs_transposed = in_hs_transposed ? 1u : 0u;
  input_batch_stride = 0u;
  input_head_stride = 0u;
  input_t_stride = 0u;
  n_heads = 1u;
  if (input_hs_transposed != 0u) {
    const auto& st = in.strides();
    if (in.shape(1) <= 0 ||
        in.shape(1) > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[0] <= 0 || st[1] <= 0 || st[2] <= 0 ||
        st[0] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[1] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[2] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return reject("hs_transpose_strides_or_heads");
    }
    n_heads = static_cast<uint32_t>(in.shape(1));
    input_batch_stride = static_cast<uint32_t>(st[0]);
    input_head_stride = static_cast<uint32_t>(st[1]);
    input_t_stride = static_cast<uint32_t>(st[2]);
    if (rows_per_batch != n_heads * static_cast<uint32_t>(t)) {
      return reject("hs_transpose_rows_shape_mismatch");
    }
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
    offset_is_vector = 0u;
  } else {
    if (offset.dtype() != mlx::core::int32 || offset.ndim() != 1 ||
        offset.shape(0) != batch || !is_row_contiguous_materialized(offset) ||
        offset.strides()[0] != 1) {
      return reject("vector_offset_constraints");
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
    uint32_t& v_dim,
    const char** reject_reason = nullptr) {
  auto reject = [&](const char* reason) {
    if (reject_reason) {
      *reject_reason = reason;
    }
    return false;
  };

  if (has_sinks || output_logsumexp || inputs.size() != 3 ||
      outputs.size() != 1 || outputs[0].size() == 0) {
    return reject("shape_or_aux");
  }

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const auto& out = outputs[0];
  if (q.dtype() != mlx::core::bfloat16 || k.dtype() != mlx::core::bfloat16 ||
      v.dtype() != mlx::core::bfloat16 || out.dtype() != mlx::core::bfloat16) {
    return reject("dtype");
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || out.ndim() != 4) {
    return reject("ndim");
  }
  if (!is_dense_row_major_view(q)) {
    return reject("q_layout");
  }
  if (!is_dense_row_major_view(k)) {
    return reject("k_layout");
  }
  if (!is_dense_row_major_view(v)) {
    return reject("v_layout");
  }
  if (!is_dense_row_major_view(out)) {
    return reject("out_layout");
  }

  if (q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(3) != k.shape(3) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2) || q.shape(2) != 1) {
    return reject("shape_contract");
  }
  if (do_causal && q.shape(2) != 1) {
    return reject("causal_q_len");
  }

  const int64_t b = q.shape(0);
  const int64_t hq = q.shape(1);
  const int64_t hkv = k.shape(1);
  const int64_t lk = k.shape(2);
  const int64_t dq = q.shape(3);
  const int64_t dv = v.shape(3);
  if (b <= 0 || hq <= 0 || hkv <= 0 || lk <= 0 || dq <= 0 || dv <= 0 ||
      (hq % hkv) != 0) {
    return reject("dim_nonpositive_or_head_repeat");
  }
  if (lk > static_cast<int64_t>(native_sdpa_max_k_len())) {
    return reject("k_len_cap");
  }
  if (dq > 256 || dv > 256) {
    return reject("dim_bounds");
  }
  if ((q.size() % 2) != 0 || (k.size() % 2) != 0 || (v.size() % 2) != 0 ||
      (out.size() % 2) != 0) {
    return reject("bf16_pack");
  }
  if (b > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      hq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      hkv > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      lk > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      dq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      dv > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return reject("u32_overflow");
  }

  if (out.ndim() != 4 || out.shape(0) != b || out.shape(1) != hq ||
      out.shape(2) != 1 || out.shape(3) != dv) {
    return reject("output_shape");
  }

  batch_size = static_cast<uint32_t>(b);
  n_q_heads = static_cast<uint32_t>(hq);
  n_kv_heads = static_cast<uint32_t>(hkv);
  k_len = static_cast<uint32_t>(lk);
  qk_dim = static_cast<uint32_t>(dq);
  v_dim = static_cast<uint32_t>(dv);
  if (reject_reason) {
    *reject_reason = nullptr;
  }
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

  if (native_qmm_enabled() && can_use_native_affine_bf16_quantized_matmul(
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

      const std::vector<uint32_t> push_consts{
          encode_push_constant_u32(out_elems),
          encode_push_constant_u32(n),
          encode_push_constant_u32(k),
          encode_push_constant_u32(groups_per_col),
          encode_push_constant_u32(w_words_per_col)};

      // Output tensor is fully overwritten by the kernel; no need to upload it.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
      if (device.tensor_needs_sync_device(inputs[0])) {
        sync_tensors.push_back(x_tensor);
      }
      if (w_cached.needs_sync) {
        sync_tensors.push_back(w_cached.tensor);
      }
      if (scales_cached.needs_sync) {
        sync_tensors.push_back(scales_cached.tensor);
      }
      if (biases_cached.needs_sync) {
        sync_tensors.push_back(biases_cached.tensor);
      }
      if (!sync_tensors.empty()) {
        encoder.record_tensor_sync_device(sync_tensors);
      }
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
      device.mark_tensor_host_dirty(out, stream.index);
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
  auto reject = [&](const char* reason) {
    log_sdpa_reject(
        q, k, v, reason, has_mask, do_causal, is_training, output_logsumexp);
    return true;
  };

  if (!native_sdpa_enabled()) {
    return reject("native_disabled");
  }

  if (stream.device == Device::cpu || detail::in_tracing() || has_mask ||
      do_causal || is_training || output_logsumexp) {
    return reject("global_gate");
  }

  if (q.dtype() != mlx::core::bfloat16 || k.dtype() != mlx::core::bfloat16 ||
      v.dtype() != mlx::core::bfloat16) {
    return reject("dtype");
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    return reject("ndim");
  }
  if (!is_dense_row_major_view(q) ||
      !is_dense_row_major_view(k) ||
      !is_dense_row_major_view(v)) {
    return reject("row_contiguous");
  }

  if (q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(3) != k.shape(3) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2) || q.shape(2) != 1) {
    return reject("shape_contract");
  }
  if (k.shape(1) <= 0 || (q.shape(1) % k.shape(1)) != 0) {
    return reject("head_repeat");
  }
  if (k.shape(2) <= 0) {
    return reject("k_len_nonpositive");
  }
  if (k.shape(2) > static_cast<int64_t>(native_sdpa_max_k_len())) {
    return reject("k_len_cap");
  }
  if (q.shape(3) <= 0 || q.shape(3) > 256 ||
      v.shape(3) <= 0 || v.shape(3) > 256) {
    return reject("dim_bounds");
  }
  if ((q.size() % 2) != 0 || (k.size() % 2) != 0 || (v.size() % 2) != 0) {
    return reject("bf16_pack");
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

  uint32_t n_rows = 0;
  uint32_t axis_size = 0;
  uint32_t w_stride = 0;
  if (native_rmsnorm_enabled() && can_use_native_rmsnorm_bf16(
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

      const std::vector<uint32_t> push_consts{
          encode_push_constant_u32(n_rows),
          encode_push_constant_u32(axis_size),
          encode_push_constant_u32(w_stride),
          encode_push_constant_f32(eps_)};

      // Output tensor is write-only in this dispatch.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
      if (device.tensor_needs_sync_device(inputs[0])) {
        sync_tensors.push_back(x_tensor);
      }
      if (device.tensor_needs_sync_device(inputs[1])) {
        sync_tensors.push_back(w_tensor);
      }
      if (!sync_tensors.empty()) {
        encoder.record_tensor_sync_device(sync_tensors);
      }
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::RMSNORM_BF16,
          {x_tensor, w_tensor, out_tensor},
          {n_rows, 1, 1},
          push_consts);
      device.mark_tensor_host_dirty(out, stream.index);
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RMSNormVJP");
  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RoPE");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();

  uint32_t n_rows = 0;
  uint32_t half_dims = 0;
  uint32_t row_stride = 0;
  uint32_t t_size = 0;
  uint32_t rows_per_batch = 0;
  uint32_t offset_is_vector = 0;
  uint32_t input_hs_transposed = 0;
  uint32_t input_batch_stride = 0;
  uint32_t input_head_stride = 0;
  uint32_t input_t_stride = 0;
  uint32_t n_heads = 0;
  bool with_freqs = false;
  const char* rope_reject_reason = nullptr;
  if (native_rope_enabled() && can_use_native_rope_bf16(
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
          input_hs_transposed,
          input_batch_stride,
          input_head_stride,
          input_t_stride,
          n_heads,
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
        const std::vector<uint32_t> push_consts{
            encode_push_constant_u32(n_rows),
            encode_push_constant_u32(half_dims),
            encode_push_constant_u32(row_stride),
            encode_push_constant_u32(t_size),
            encode_push_constant_u32(rows_per_batch),
            encode_push_constant_u32(offset_is_vector),
            encode_push_constant_u32(traditional_flag),
            encode_push_constant_u32(input_hs_transposed),
            encode_push_constant_u32(input_batch_stride),
            encode_push_constant_u32(input_head_stride),
            encode_push_constant_u32(input_t_stride),
            encode_push_constant_u32(n_heads),
            encode_push_constant_f32(scale_),
            encode_push_constant_u32(forward_flag)};

        // Output tensor is written by the kernel; skip redundant upload.
        std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
        if (device.tensor_needs_sync_device(inputs[0])) {
          sync_tensors.push_back(in_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[2])) {
          sync_tensors.push_back(freqs_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[1])) {
          sync_tensors.push_back(offset_tensor);
        }
        if (!sync_tensors.empty()) {
          encoder.record_tensor_sync_device(sync_tensors);
        }
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::ROPE_BF16_FREQS,
            {in_tensor, out_tensor, freqs_tensor, offset_tensor},
            {n_rows, 1, 1},
            push_consts);
      } else {
        const std::vector<uint32_t> push_consts{
            encode_push_constant_u32(n_rows),
            encode_push_constant_u32(half_dims),
            encode_push_constant_u32(row_stride),
            encode_push_constant_u32(t_size),
            encode_push_constant_u32(rows_per_batch),
            encode_push_constant_u32(offset_is_vector),
            encode_push_constant_u32(traditional_flag),
            encode_push_constant_u32(input_hs_transposed),
            encode_push_constant_u32(input_batch_stride),
            encode_push_constant_u32(input_head_stride),
            encode_push_constant_u32(input_t_stride),
            encode_push_constant_u32(n_heads),
            encode_push_constant_f32(scale_),
            encode_push_constant_f32(std::log2(base_)),
            encode_push_constant_u32(forward_flag)};

        // Output tensor is written by the kernel; skip redundant upload.
        std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
        if (device.tensor_needs_sync_device(inputs[0])) {
          sync_tensors.push_back(in_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[1])) {
          sync_tensors.push_back(offset_tensor);
        }
        if (!sync_tensors.empty()) {
          encoder.record_tensor_sync_device(sync_tensors);
        }
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::ROPE_BF16_T1,
            {in_tensor, out_tensor, offset_tensor},
            {n_rows, 1, 1},
            push_consts);
      }
      device.mark_tensor_host_dirty(out, stream.index);
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  log_rope_reject(
      inputs, outputs, dims_, traditional_, base_, rope_reject_reason);
  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::ScaledDotProductAttention");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();

  uint32_t batch_size = 0;
  uint32_t n_q_heads = 0;
  uint32_t n_kv_heads = 0;
  uint32_t k_len = 0;
  uint32_t qk_dim = 0;
  uint32_t v_dim = 0;
  uint32_t split_k = 1;
  const char* native_reject_reason = nullptr;
  if (native_sdpa_enabled() && can_use_native_sdpa_bf16_decode_q1(
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
          v_dim,
          &native_reject_reason)) {
    split_k = select_sdpa_split_k(k_len);
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

      // Output tensor is write-only in this decode kernel.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
      if (device.tensor_needs_sync_device(inputs[0])) {
        sync_tensors.push_back(q_tensor);
      }
      if (device.tensor_needs_sync_device(inputs[1])) {
        sync_tensors.push_back(k_tensor);
      }
      if (device.tensor_needs_sync_device(inputs[2])) {
        sync_tensors.push_back(v_tensor);
      }
      if (!sync_tensors.empty()) {
        encoder.record_tensor_sync_device(sync_tensors);
      }

      const uint64_t n_rows_u64 =
          static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(n_q_heads);
      if (n_rows_u64 == 0u ||
          n_rows_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("[Vulkan SDPA] row shape overflow.");
      }
      const uint32_t n_rows = static_cast<uint32_t>(n_rows_u64);
      if (split_k <= 1u) {
        const std::vector<uint32_t> push_consts{
            encode_push_constant_u32(batch_size),
            encode_push_constant_u32(n_q_heads),
            encode_push_constant_u32(n_kv_heads),
            encode_push_constant_u32(k_len),
            encode_push_constant_u32(qk_dim),
            encode_push_constant_u32(v_dim),
            encode_push_constant_f32(scale_)};

        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1,
            {q_tensor, k_tensor, v_tensor, out_tensor},
            {n_rows, 1, 1},
            push_consts);
      } else {
        auto manager = device.kompute_manager();
        if (!manager) {
          throw std::runtime_error("[Vulkan SDPA] Missing Kompute manager.");
        }

        const uint64_t partial_rows_u64 =
            n_rows_u64 * static_cast<uint64_t>(split_k);
        const uint64_t partial_o_elems_u64 =
            partial_rows_u64 * static_cast<uint64_t>(v_dim);
        if (partial_rows_u64 == 0u ||
            partial_rows_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            partial_rows_u64 > std::numeric_limits<size_t>::max() ||
            partial_o_elems_u64 == 0u ||
            partial_o_elems_u64 > std::numeric_limits<size_t>::max()) {
          throw std::runtime_error("[Vulkan SDPA] split-k temporary shape overflow.");
        }

        const size_t partial_rows = static_cast<size_t>(partial_rows_u64);
        const size_t partial_o_elems = static_cast<size_t>(partial_o_elems_u64);
        auto partial_o_tensor = manager->tensor(std::vector<float>(partial_o_elems, 0.0f));
        auto partial_m_tensor = manager->tensor(std::vector<float>(partial_rows, 0.0f));
        auto partial_l_tensor = manager->tensor(std::vector<float>(partial_rows, 0.0f));

        const std::vector<uint32_t> stage1_push_consts{
            encode_push_constant_u32(batch_size),
            encode_push_constant_u32(n_q_heads),
            encode_push_constant_u32(n_kv_heads),
            encode_push_constant_u32(k_len),
            encode_push_constant_u32(qk_dim),
            encode_push_constant_u32(v_dim),
            encode_push_constant_u32(split_k),
            encode_push_constant_f32(scale_)};

        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::SDPA_BF16_DECODE_SPLITK_STAGE1,
            {q_tensor,
             k_tensor,
             v_tensor,
             partial_o_tensor,
             partial_m_tensor,
             partial_l_tensor},
            {static_cast<uint32_t>(partial_rows_u64), 1, 1},
            stage1_push_consts);

        const std::vector<uint32_t> reduce_push_consts{
            encode_push_constant_u32(batch_size),
            encode_push_constant_u32(n_q_heads),
            encode_push_constant_u32(v_dim),
            encode_push_constant_u32(split_k)};

        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::SDPA_BF16_DECODE_SPLITK_REDUCE,
            {partial_o_tensor, partial_m_tensor, partial_l_tensor, out_tensor},
            {n_rows, 1, 1},
            reduce_push_consts);
      }
      device.mark_tensor_host_dirty(out, stream.index);
      return;
    } catch (const std::exception& e) {
      if (sdpa_debug_reject_enabled() ||
          env_flag_default_false("MLX_VK_DEBUG_SDPA_ERROR")) {
        std::cerr << "[VulkanSDPAError] native path failed: " << e.what()
                  << " split_k=" << split_k << " k_len=" << k_len
                  << " qk_dim=" << qk_dim << " v_dim=" << v_dim << "\n";
      }
      // Fall through to fallback path.
    }
  }

  if (native_sdpa_enabled() &&
      (sdpa_debug_reject_enabled() ||
       env_flag_default_false("MLX_VK_DEBUG_SDPA_ERROR")) &&
      native_reject_reason) {
    std::cerr << "[VulkanSDPANativeReject] reason=" << native_reject_reason
              << "\n";
  }

  profile.mark_fallback();
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
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_CPU_FALLBACK_MULTI(distributed::AllReduce)
VULKAN_CPU_FALLBACK_MULTI(distributed::AllGather)
VULKAN_CPU_FALLBACK_MULTI(distributed::Send)
VULKAN_CPU_FALLBACK_MULTI(distributed::Recv)
VULKAN_CPU_FALLBACK_MULTI(distributed::ReduceScatter)

} // namespace mlx::core
