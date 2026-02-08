// Copyright © 2026 MLX Vulkan Backend
// Binary operations using Kompute - Aligned with Metal

#include "mlx/backend/common/binary.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/stream.h"

#include <iostream>

namespace mlx::core {

namespace {

inline void prepare_inputs_for_cpu_fallback(const std::vector<array>& inputs) {
  for (const auto& in : inputs) {
    auto& mutable_in = const_cast<array&>(in);
    if (mutable_in.status() == array::Status::unscheduled) {
      mutable_in.eval();
    } else {
      mutable_in.wait();
    }
  }
}

} // namespace

// ============================================================================
// Add Implementation with Kompute
// ============================================================================

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  std::cerr << "[Add::eval_gpu] BEGIN" << std::endl;
  
  // Check Vulkan availability
  if (!vulkan::is_available()) {
    std::cerr << "[Add::eval_gpu] Vulkan not available, throwing error" << std::endl;
    throw std::runtime_error("Vulkan not available");
  }
  
  // Route through CPU implementation until the Vulkan tensor path is stable.
  prepare_inputs_for_cpu_fallback(inputs);
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
  
  std::cerr << "[Add::eval_gpu] END" << std::endl;
}

// ============================================================================
// Other Binary Operations (placeholders)
// ============================================================================

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  std::cerr << "[Multiply::eval_gpu] BEGIN" << std::endl;
  
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  prepare_inputs_for_cpu_fallback(inputs);
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
  
  std::cerr << "[Multiply::eval_gpu] END" << std::endl;
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  prepare_inputs_for_cpu_fallback(inputs);
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  prepare_inputs_for_cpu_fallback(inputs);
  eval_cpu(inputs, out);
  synchronize(default_stream(Device::cpu));
}

} // namespace mlx::core
