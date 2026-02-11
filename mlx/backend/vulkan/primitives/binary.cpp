// Copyright © 2026 MLX Vulkan Backend
// Binary operations using Kompute - Aligned with Metal

#include "mlx/backend/common/binary.h"
#include "mlx/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include "mlx/backend/vulkan/op_profiler.h"
#include "mlx/primitives.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/dtype_utils.h"
#include "mlx/stream.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <mutex>

namespace mlx::core {

namespace {

inline void prepare_inputs_for_cpu_fallback(
    const std::vector<array>& inputs,
    Stream stream) {
  for (const auto& in : inputs) {
    auto& mutable_in = const_cast<array&>(in);
    if (mutable_in.status() == array::Status::unscheduled) {
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

inline void sync_inputs_to_host_if_needed(const std::vector<array>& inputs) {
  auto& device = vulkan::device(Device::gpu);
  for (const auto& in : inputs) {
    device.sync_array_to_host_if_needed(in);
  }
}

inline bool is_row_contiguous_materialized(const array& arr) {
  return arr.flags().row_contiguous && arr.data_size() == arr.size();
}

inline bool can_use_native_binary_f32(
    const array& a,
    const array& b,
    const array& out) {
  return out.size() > 1 &&
      a.dtype() == float32 && b.dtype() == float32 && out.dtype() == float32 &&
      a.size() == b.size() && a.size() == out.size() &&
      a.shape() == b.shape() && a.shape() == out.shape() &&
      is_row_contiguous_materialized(a) &&
      is_row_contiguous_materialized(b) && out.flags().row_contiguous;
}

inline bool can_use_native_binary_f32_scalar_rhs(
    const array& a,
    const array& b,
    const array& out) {
  const bool rhs_is_scalar =
      (b.size() == 1 && is_row_contiguous_materialized(b)) ||
      (b.size() > 1 && b.data_size() == 1 && !b.strides().empty() &&
       std::all_of(
           b.strides().begin(),
           b.strides().end(),
           [](int64_t stride) { return stride == 0; }));
  return out.size() > 1 &&
      a.dtype() == float32 && b.dtype() == float32 && out.dtype() == float32 &&
      rhs_is_scalar && a.size() == out.size() && a.shape() == out.shape() &&
      is_row_contiguous_materialized(a) && out.flags().row_contiguous;
}

inline bool can_use_native_binary_bf16(
    const array& a,
    const array& b,
    const array& out) {
  return out.size() > 0 &&
      a.dtype() == bfloat16 && b.dtype() == bfloat16 &&
      out.dtype() == bfloat16 && a.size() == b.size() &&
      a.size() == out.size() && a.shape() == b.shape() &&
      a.shape() == out.shape() && is_row_contiguous_materialized(a) &&
      is_row_contiguous_materialized(b) && out.flags().row_contiguous;
}

inline bool can_use_native_binary_bf16_scalar_rhs(
    const array& a,
    const array& b,
    const array& out) {
  const bool rhs_is_scalar =
      (b.size() == 1 && is_row_contiguous_materialized(b)) ||
      (b.size() > 1 && b.data_size() == 1 && !b.strides().empty() &&
       std::all_of(
           b.strides().begin(),
           b.strides().end(),
           [](int64_t stride) { return stride == 0; }));
  return out.size() > 0 &&
      a.dtype() == bfloat16 && b.dtype() == bfloat16 &&
      out.dtype() == bfloat16 && rhs_is_scalar &&
      a.size() == out.size() && a.shape() == out.shape() &&
      is_row_contiguous_materialized(a) && out.flags().row_contiguous;
}

inline bool can_use_native_binary_bf16_bcast_rhs(
    const array& a,
    const array& b,
    const array& out) {
  constexpr uint64_t kU32Max =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  if (!(out.size() > 0 && a.dtype() == bfloat16 && b.dtype() == bfloat16 &&
        out.dtype() == bfloat16 && a.size() == b.size() &&
        a.size() == out.size() && a.shape() == b.shape() &&
        a.shape() == out.shape() && is_row_contiguous_materialized(a) &&
        out.flags().row_contiguous && b.ndim() > 0 && b.ndim() <= 4)) {
    return false;
  }
  for (int i = 0; i < b.ndim(); ++i) {
    const int64_t stride = b.strides(i);
    if (stride < 0 || static_cast<uint64_t>(stride) > kU32Max) {
      return false;
    }
    if (static_cast<uint64_t>(b.shape(i)) > kU32Max) {
      return false;
    }
  }
  return true;
}

inline std::vector<uint32_t> build_bf16_bcast_push_consts(const array& b) {
  std::array<uint32_t, 4> dims{1u, 1u, 1u, 1u};
  std::array<uint32_t, 4> strides{0u, 0u, 0u, 0u};
  const int n = b.ndim();
  const int base = 4 - n;
  for (int i = 0; i < n; ++i) {
    dims[base + i] = static_cast<uint32_t>(b.shape(i));
    strides[base + i] = static_cast<uint32_t>(b.strides(i));
  }
  return {
      dims[0],
      dims[1],
      dims[2],
      dims[3],
      strides[0],
      strides[1],
      strides[2],
      strides[3]};
}

inline uint32_t encode_push_constant_u32(uint32_t value) {
  return value;
}

inline bool native_add_f32_enabled() {
  const char* env = std::getenv("MLX_VK_ENABLE_ADD_F32");
  if (!env) {
    return false;
  }
  return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
      std::strcmp(env, "on") == 0;
}

inline bool env_flag_default_true(const char* name) {
  const char* env = std::getenv(name);
  if (!env) {
    return true;
  }
  if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
      std::strcmp(env, "off") == 0) {
    return false;
  }
  return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
      std::strcmp(env, "on") == 0;
}

inline bool native_add_bf16_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_ADD_BF16");
  return enabled;
}

inline bool native_mul_bf16_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_MUL_BF16");
  return enabled;
}

inline bool native_sub_bf16_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_SUB_BF16");
  return enabled;
}

inline bool native_sub_f32_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_SUB_F32");
  return enabled;
}

inline bool debug_subtract_fallback_enabled() {
  const char* env = std::getenv("MLX_VK_DEBUG_SUBTRACT_FALLBACK");
  if (!env) {
    return false;
  }
  return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
      std::strcmp(env, "on") == 0;
}

inline std::string shape_string(const array& arr) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < arr.shape().size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << arr.shape()[i];
  }
  os << "]";
  return os.str();
}

inline std::string strides_string(const array& arr) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < arr.strides().size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << arr.strides()[i];
  }
  os << "]";
  return os.str();
}

inline void log_subtract_path(
    const char* path,
    const std::vector<array>& inputs,
    const array& out) {
  if (!debug_subtract_fallback_enabled() || inputs.size() < 2) {
    return;
  }
  std::cerr << "[VulkanSubtractPath] path=" << path
            << " a.dtype=" << dtype_to_string(inputs[0].dtype())
            << " a.shape=" << shape_string(inputs[0])
            << " a.strides=" << strides_string(inputs[0])
            << " a.size=" << inputs[0].size()
            << " a.data_size=" << inputs[0].data_size()
            << " a.row=" << (inputs[0].flags().row_contiguous ? 1 : 0)
            << " b.dtype=" << dtype_to_string(inputs[1].dtype())
            << " b.shape=" << shape_string(inputs[1])
            << " b.strides=" << strides_string(inputs[1])
            << " b.size=" << inputs[1].size()
            << " b.data_size=" << inputs[1].data_size()
            << " b.row=" << (inputs[1].flags().row_contiguous ? 1 : 0)
            << " out.dtype=" << dtype_to_string(out.dtype())
            << " out.shape=" << shape_string(out)
            << " out.strides=" << strides_string(out)
            << " out.size=" << out.size()
            << " out.data_size=" << out.data_size()
            << " out.row=" << (out.flags().row_contiguous ? 1 : 0) << "\n";
}

inline bool debug_binary_fallback_enabled() {
  const char* env = std::getenv("MLX_VK_DEBUG_BINARY_FALLBACK");
  if (!env) {
    return false;
  }
  return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
      std::strcmp(env, "on") == 0;
}

inline void log_binary_fallback_once(
    const char* op_name,
    const std::vector<array>& inputs,
    const array& out,
    const char* reason) {
  if (!debug_binary_fallback_enabled() || inputs.size() < 2) {
    return;
  }
  static std::mutex mtx;
  static std::unordered_set<std::string> seen;
  std::ostringstream key;
  key << op_name << "|reason=" << (reason ? reason : "unknown")
      << "|a_dtype=" << dtype_to_string(inputs[0].dtype())
      << "|a_shape=" << shape_string(inputs[0])
      << "|a_strides=" << strides_string(inputs[0])
      << "|a_size=" << inputs[0].size()
      << "|a_data_size=" << inputs[0].data_size()
      << "|a_row=" << (inputs[0].flags().row_contiguous ? 1 : 0)
      << "|b_dtype=" << dtype_to_string(inputs[1].dtype())
      << "|b_shape=" << shape_string(inputs[1])
      << "|b_strides=" << strides_string(inputs[1])
      << "|b_size=" << inputs[1].size()
      << "|b_data_size=" << inputs[1].data_size()
      << "|b_row=" << (inputs[1].flags().row_contiguous ? 1 : 0)
      << "|out_dtype=" << dtype_to_string(out.dtype())
      << "|out_shape=" << shape_string(out)
      << "|out_strides=" << strides_string(out)
      << "|out_size=" << out.size()
      << "|out_data_size=" << out.data_size()
      << "|out_row=" << (out.flags().row_contiguous ? 1 : 0);
  const std::string signature = key.str();
  std::lock_guard<std::mutex> lock(mtx);
  if (!seen.insert(signature).second) {
    return;
  }
  std::cerr << "[VulkanBinaryFallback] op=" << op_name
            << " reason=" << (reason ? reason : "unknown")
            << " a.dtype=" << dtype_to_string(inputs[0].dtype())
            << " a.shape=" << shape_string(inputs[0])
            << " a.strides=" << strides_string(inputs[0])
            << " a.size=" << inputs[0].size()
            << " a.data_size=" << inputs[0].data_size()
            << " a.row=" << (inputs[0].flags().row_contiguous ? 1 : 0)
            << " b.dtype=" << dtype_to_string(inputs[1].dtype())
            << " b.shape=" << shape_string(inputs[1])
            << " b.strides=" << strides_string(inputs[1])
            << " b.size=" << inputs[1].size()
            << " b.data_size=" << inputs[1].data_size()
            << " b.row=" << (inputs[1].flags().row_contiguous ? 1 : 0)
            << " out.dtype=" << dtype_to_string(out.dtype())
            << " out.shape=" << shape_string(out)
            << " out.strides=" << strides_string(out)
            << " out.size=" << out.size()
            << " out.data_size=" << out.data_size()
            << " out.row=" << (out.flags().row_contiguous ? 1 : 0)
            << "\n";
}

inline bool dispatch_native_binary(
    Stream stream,
    const array& a,
    const array& b,
    array& out,
    const char* kernel_name,
    uint32_t work_items,
    const std::vector<uint32_t>& extra_push_consts,
    vulkan::OpProfileScope* profile) {
  try {
    if (!out.data_shared_ptr()) {
      out.set_data(allocator::malloc(out.nbytes()));
    }

    auto& device = vulkan::device(stream.device);
    auto& encoder = device.get_command_encoder(stream.index);
    encoder.begin_encoding();

    auto a_tensor = device.get_tensor(a);
    auto b_tensor = device.get_tensor(b);
    auto out_tensor = device.get_tensor(out);

    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t groups_x = std::max<uint32_t>(1, (work_items + 255u) / 256u);
    std::vector<uint32_t> push_consts;
    push_consts.reserve(1 + extra_push_consts.size());
    push_consts.push_back(encode_push_constant_u32(n));
    push_consts.insert(
        push_consts.end(), extra_push_consts.begin(), extra_push_consts.end());

    // Output tensor is write-only. Upload only host-authoritative inputs.
    std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
    if (device.tensor_needs_sync_device(a)) {
      sync_tensors.push_back(a_tensor);
    }
    if (device.tensor_needs_sync_device(b)) {
      sync_tensors.push_back(b_tensor);
    }
    if (!sync_tensors.empty()) {
      encoder.record_tensor_sync_device(sync_tensors);
    }
    encoder.record_algo_dispatch(
        kernel_name,
        {a_tensor, b_tensor, out_tensor},
        {groups_x, 1, 1},
        push_consts);
    device.mark_tensor_host_dirty(out, stream.index);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

} // namespace

// ============================================================================
// Add Implementation with Kompute
// ============================================================================

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("Add");

  // Check Vulkan availability
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  auto s = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, s);
  const bool can_bf16 = can_use_native_binary_bf16(inputs[0], inputs[1], out);
  const bool can_bf16_scalar =
      can_use_native_binary_bf16_scalar_rhs(inputs[0], inputs[1], out);
  const bool can_bf16_bcast =
      can_use_native_binary_bf16_bcast_rhs(inputs[0], inputs[1], out);

  if (native_add_f32_enabled() && can_use_native_binary_f32(inputs[0], inputs[1], out) &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::ADD_F32,
            n,
            {},
            &profile)) {
      return;
    }
  }

  if (native_add_bf16_enabled() &&
      can_bf16 &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::ADD_BF16,
            n_words,
            {},
            &profile)) {
      return;
    }
  }

  if (native_add_bf16_enabled() &&
      can_bf16_scalar &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::ADD_BF16_SCALAR,
            n_words,
            {},
            &profile)) {
      return;
    }
  }

  if (native_add_bf16_enabled() &&
      can_bf16_bcast &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    const auto bcast_push_consts = build_bf16_bcast_push_consts(inputs[1]);
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::ADD_BF16_BCAST,
            n_words,
            bcast_push_consts,
            &profile)) {
      return;
    }
  }

  // Route through CPU implementation until the Vulkan tensor path is stable.
  const char* add_reason = "bf16_shape_or_layout";
  if (!native_add_bf16_enabled()) {
    add_reason = "native_gate_off";
  } else if (out.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    add_reason = "size_over_u32";
  } else if (can_bf16 || can_bf16_scalar || can_bf16_bcast) {
    add_reason = "dispatch_exception";
  }
  log_binary_fallback_once("Add", inputs, out, add_reason);
  profile.mark_fallback();
  sync_inputs_to_host_if_needed(inputs);
  eval_cpu(inputs, out);
  vulkan::device(Device::gpu).invalidate_tensor(out);
  synchronize(default_stream(Device::cpu));
  profile.mark_sync();
}

// ============================================================================
// Other Binary Operations (placeholders)
// ============================================================================

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("Multiply");

  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }

  auto s = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, s);
  const bool can_bf16 = can_use_native_binary_bf16(inputs[0], inputs[1], out);
  const bool can_bf16_scalar =
      can_use_native_binary_bf16_scalar_rhs(inputs[0], inputs[1], out);
  const bool can_bf16_bcast =
      can_use_native_binary_bf16_bcast_rhs(inputs[0], inputs[1], out);

  if (native_mul_bf16_enabled() &&
      can_bf16 &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::MUL_BF16,
            n_words,
            {},
            &profile)) {
      return;
    }
  }

  if (native_mul_bf16_enabled() &&
      can_bf16_scalar &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::MUL_BF16_SCALAR,
            n_words,
            {},
            &profile)) {
      return;
    }
  }

  if (native_mul_bf16_enabled() &&
      can_bf16_bcast &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    const auto bcast_push_consts = build_bf16_bcast_push_consts(inputs[1]);
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::MUL_BF16_BCAST,
            n_words,
            bcast_push_consts,
            &profile)) {
      return;
    }
  }

  const char* mul_reason = "bf16_shape_or_layout";
  if (!native_mul_bf16_enabled()) {
    mul_reason = "native_gate_off";
  } else if (out.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    mul_reason = "size_over_u32";
  } else if (can_bf16 || can_bf16_scalar || can_bf16_bcast) {
    mul_reason = "dispatch_exception";
  }
  log_binary_fallback_once("Multiply", inputs, out, mul_reason);
  profile.mark_fallback();
  sync_inputs_to_host_if_needed(inputs);
  eval_cpu(inputs, out);
  vulkan::device(Device::gpu).invalidate_tensor(out);
  synchronize(default_stream(Device::cpu));
  profile.mark_sync();
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("Subtract");
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }

  auto s = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, s);

  if (native_sub_f32_enabled() &&
      can_use_native_binary_f32(inputs[0], inputs[1], out) &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::SUB_F32,
            n,
            {},
            &profile)) {
      log_subtract_path("native_sub_f32", inputs, out);
      return;
    }
  }

  if (native_sub_f32_enabled() &&
      can_use_native_binary_f32_scalar_rhs(inputs[0], inputs[1], out) &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::SUB_F32_SCALAR,
            n,
            {},
            &profile)) {
      log_subtract_path("native_sub_f32_scalar_rhs", inputs, out);
      return;
    }
  }

  if (native_sub_bf16_enabled() &&
      can_use_native_binary_bf16(inputs[0], inputs[1], out) &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::SUB_BF16,
            n_words,
            {},
            &profile)) {
      log_subtract_path("native_sub_bf16", inputs, out);
      return;
    }
  }

  if (native_sub_bf16_enabled() &&
      can_use_native_binary_bf16_scalar_rhs(inputs[0], inputs[1], out) &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t n_words = (n + 1u) / 2u;
    if (dispatch_native_binary(
            s,
            inputs[0],
            inputs[1],
            out,
            vulkan::KernelRegistry::SUB_BF16_SCALAR,
            n_words,
            {},
            &profile)) {
      log_subtract_path("native_sub_bf16_scalar_rhs", inputs, out);
      return;
    }
  }

  log_subtract_path("fallback", inputs, out);
  profile.mark_fallback();
  sync_inputs_to_host_if_needed(inputs);
  eval_cpu(inputs, out);
  vulkan::device(Device::gpu).invalidate_tensor(out);
  synchronize(default_stream(Device::cpu));
  profile.mark_sync();
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("Divide");
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }

  auto s = out.primitive().stream();
  profile.mark_fallback();
  prepare_inputs_for_cpu_fallback(inputs, s);
  sync_inputs_to_host_if_needed(inputs);
  eval_cpu(inputs, out);
  vulkan::device(Device::gpu).invalidate_tensor(out);
  synchronize(default_stream(Device::cpu));
  profile.mark_sync();
}

} // namespace mlx::core
