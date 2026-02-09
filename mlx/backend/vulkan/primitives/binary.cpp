// Copyright © 2026 MLX Vulkan Backend
// Binary operations using Kompute - Aligned with Metal

#include "mlx/backend/common/binary.h"
#include "mlx/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include "mlx/primitives.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/stream.h"

#include <algorithm>
#include <cstring>
#include <limits>

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

inline bool can_use_native_add_f32(
    const array& a,
    const array& b,
    const array& out) {
  return out.size() > 0 && 
      a.dtype() == float32 && b.dtype() == float32 && out.dtype() == float32 &&
      a.size() == b.size() && a.size() == out.size() &&
      a.shape() == b.shape() && a.shape() == out.shape() &&
      a.flags().row_contiguous && b.flags().row_contiguous &&
      a.data_size() == a.size() && b.data_size() == b.size();
}

inline float encode_push_constant_u32(uint32_t value) {
  float encoded = 0.0f;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&encoded, &value, sizeof(uint32_t));
  return encoded;
}

} // namespace

// ============================================================================
// Add Implementation with Kompute
// ============================================================================

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Check Vulkan availability
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  auto s = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, s);

  if (can_use_native_add_f32(inputs[0], inputs[1], out) &&
      out.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    try {
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(s.device);
      auto& encoder = device.get_command_encoder(s.index);
      encoder.begin_encoding();

      auto a_tensor = device.get_tensor(inputs[0]);
      auto b_tensor = device.get_tensor(inputs[1]);
      auto out_tensor = device.get_tensor(out);

      const uint32_t n = static_cast<uint32_t>(out.size());
      const uint32_t groups_x = std::max<uint32_t>(1, (n + 255u) / 256u);
      const std::vector<float> push_consts{encode_push_constant_u32(n)};

      encoder.record_tensor_sync_device({a_tensor, b_tensor, out_tensor});
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::ADD_F32,
          {a_tensor, b_tensor, out_tensor},
          {groups_x, 1, 1},
          push_consts);
      encoder.record_tensor_sync_local({out_tensor});
      synchronize(s);
      
      std::memcpy(out.data<void>(), out_tensor->rawData(), out.nbytes());
      return;
    } catch (const std::exception& e) {
      // Fall through to CPU fallback path.
    }
  }

  // Route through CPU implementation until the Vulkan tensor path is stable.
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
}

// ============================================================================
// Other Binary Operations (placeholders)
// ============================================================================

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  prepare_inputs_for_cpu_fallback(inputs, out.primitive().stream());
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  prepare_inputs_for_cpu_fallback(inputs, out.primitive().stream());
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  prepare_inputs_for_cpu_fallback(inputs, out.primitive().stream());
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
}

} // namespace mlx::core
