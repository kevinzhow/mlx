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
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>

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

inline bool dispatch_native_binary(
    Stream stream,
    const array& a,
    const array& b,
    array& out,
    const char* kernel_name,
    uint32_t work_items,
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
    const std::vector<uint32_t> push_consts{encode_push_constant_u32(n)};

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
            &profile)) {
      return;
    }
  }

  if (native_add_bf16_enabled() &&
      can_use_native_binary_bf16(inputs[0], inputs[1], out) &&
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
            &profile)) {
      return;
    }
  }

  // Route through CPU implementation until the Vulkan tensor path is stable.
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

  if (native_mul_bf16_enabled() &&
      can_use_native_binary_bf16(inputs[0], inputs[1], out) &&
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
            &profile)) {
      return;
    }
  }

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
