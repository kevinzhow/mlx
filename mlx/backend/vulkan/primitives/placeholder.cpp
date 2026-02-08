// Copyright © 2026 MLX Vulkan Backend
// Placeholder primitive using Kompute

#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/buffer.h"
#include "mlx/primitives.h"
#include "mlx/backend/cpu/eval.h"

#include <iostream>
#include <cstring>

namespace mlx::core {

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  std::cerr << "[Copy] eval_gpu: BEGIN" << std::endl;
  
  if (!vulkan::is_available()) {
    std::cerr << "[Copy] Vulkan not available, using CPU fallback" << std::endl;
    eval_cpu(inputs, out);
    return;
  }
  
  if (out.size() == 0) {
    return;
  }
  
  try {
    auto& device = vulkan::device(mlx::core::Device::gpu);
    auto s = out.primitive().stream();
    
    // Get tensors
    auto in_tensor = device.get_tensor(inputs[0]);
    auto out_tensor = device.get_tensor(out);
    
    // Get command encoder
    auto& encoder = device.get_command_encoder(s.index);
    encoder.begin_encoding();
    
    // For copy, we sync input to device and immediately sync output to host
    // In a real implementation, we'd use a copy shader
    encoder.record_tensor_sync_device({in_tensor});
    
    // TODO: Use a proper copy algorithm instead of eval_cpu
    // For now, fall back to CPU after sync
    encoder.record_tensor_sync_local({in_tensor});
    
    // Commit immediately since we need the data on CPU
    device.commit_command_buffer(s.index);
    
    // Now do CPU copy
    eval_cpu(inputs, out);
    
  } catch (const std::exception& e) {
    std::cerr << "[Copy] Exception: " << e.what() << std::endl;
    eval_cpu(inputs, out);
  }
  
  std::cerr << "[Copy] eval_gpu: END" << std::endl;
}

} // namespace mlx::core
