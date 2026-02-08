// Copyright © 2026 MLX Vulkan Backend
// Binary operations using Kompute - Aligned with Metal

#include "mlx/backend/common/binary.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/buffer.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include "mlx/primitives.h"
#include "mlx/backend/cpu/eval.h"

#include <iostream>
#include <cstring>

namespace mlx::core {

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
  
  if (out.size() == 0) {
    return;
  }
  
  // Check input types
  auto& a = inputs[0];
  auto& b = inputs[1];
  
  if (a.dtype() != b.dtype() || a.dtype() != out.dtype()) {
    std::cerr << "[Add::eval_gpu] Type mismatch" << std::endl;
    throw std::runtime_error("Type mismatch in Add::eval_gpu");
  }
  
  // For now, support only float32
  if (a.dtype() != float32) {
    std::cerr << "[Add::eval_gpu] Non-float32 type not yet supported" << std::endl;
    throw std::runtime_error("Non-float32 type not yet supported");
  }
  
  try {
    auto& device = vulkan::device(mlx::core::Device::gpu);
    auto s = out.primitive().stream();
    
    // Get tensors for inputs and output
    auto a_tensor = device.get_tensor(a);
    auto b_tensor = device.get_tensor(b);
    auto c_tensor = device.get_tensor(out);
    
    std::cerr << "[Add::eval_gpu] Got tensors" << std::endl;
    
    // Get command encoder
    auto& encoder = device.get_command_encoder(s.index);
    
    // Begin encoding if not already started
    encoder.begin_encoding();
    
    // Track inputs/outputs
    encoder.set_input_array(a, 0);
    encoder.set_input_array(b, 1);
    encoder.set_output_array(out, 2);
    
    // Sync input tensors to device
    encoder.record_tensor_sync_device({a_tensor, b_tensor});
    
    // Dispatch algorithm
    uint32_t threads_per_group = 256;
    uint32_t nthreads = (out.size() + threads_per_group - 1) / threads_per_group;
    std::vector<uint32_t> workgroup = {nthreads, 1, 1};
    
    encoder.record_algo_dispatch("add_f32", {a_tensor, b_tensor, c_tensor}, workgroup);
    
    // Sync output tensor to host
    encoder.record_tensor_sync_local({c_tensor});
    
    std::cerr << "[Add::eval_gpu] Recorded operations" << std::endl;
    
    // Check if we need to commit
    if (device.command_buffer_needs_commit(s.index)) {
      std::cerr << "[Add::eval_gpu] Committing command buffer" << std::endl;
      device.commit_command_buffer(s.index);
    }
    
  } catch (const std::exception& e) {
    std::cerr << "[Add::eval_gpu] Exception: " << e.what() << std::endl;
    throw;
  }
  
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
  
  if (out.size() == 0) {
    return;
  }
  
  // For now, fall back to CPU implementation
  // TODO: Implement using Kompute
  std::cerr << "[Multiply::eval_gpu] Not implemented, using CPU fallback" << std::endl;
  eval_cpu(inputs, out);
  
  std::cerr << "[Multiply::eval_gpu] END" << std::endl;
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  if (out.size() == 0) {
    return;
  }
  
  // TODO: Implement using Kompute
  std::cerr << "[Subtract::eval_gpu] Not implemented, using CPU fallback" << std::endl;
  eval_cpu(inputs, out);
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  
  if (out.size() == 0) {
    return;
  }
  
  // TODO: Implement using Kompute
  std::cerr << "[Divide::eval_gpu] Not implemented, using CPU fallback" << std::endl;
  eval_cpu(inputs, out);
}

} // namespace mlx::core
