// Copyright © 2026 MLX Vulkan Backend
// GPU interface - Aligned with metal/eval.cpp

#include <memory>
#include <sstream>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    vulkan::device(stream.device).new_queue(stream.index);
  }
}

inline void check_error(vk::Result result) {
  if (result != vk::Result::eSuccess) {
    std::ostringstream msg;
    msg << "[VULKAN] Command buffer execution failed: " << vk::to_string(result);
    throw std::runtime_error(msg.str());
  }
}

void eval(array& arr) {
  auto s = arr.primitive().stream();
  auto& d = vulkan::device(s.device);
  auto sequence = d.get_sequence(s.index);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    // Call the primitive's GPU evaluation
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  
  // Collect buffers for lifetime management
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& sib : arr.siblings()) {
    buffers.insert(sib.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    
    // Commit and set completion handler
    d.commit_command_buffer(s.index);
    scheduler::notify_task_completion(s);
    
    // Get new sequence
    d.get_sequence(s.index);
  }
}

void finalize(Stream s) {
  if (s.device == mlx::core::Device::gpu) {
    auto& d = vulkan::device(s.device);
    d.end_encoding(s.index);
    d.commit_command_buffer(s.index);
    d.get_sequence(s.index);
  }
}

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::gpu) {
    auto& d = vulkan::device(s.device);
    d.wait_for_stream(s.index);
    d.sync_dirty_tensors_for_stream(s.index);
    
    // Get sequence to force completion
    d.get_sequence(s.index);
  }
}

// Additional helper for buffer management
void register_input(array& arr, int idx, int64_t offset) {
  auto s = arr.primitive().stream();
  auto& d = vulkan::device(s.device);
  auto& enc = d.get_command_encoder(s.index);
  enc.set_input_array(arr, idx, offset);
}

void register_output(array& arr, int idx, int64_t offset) {
  auto s = arr.primitive().stream();
  auto& d = vulkan::device(s.device);
  auto& enc = d.get_command_encoder(s.index);
  enc.set_output_array(arr, idx, offset);
}

} // namespace mlx::core::gpu
