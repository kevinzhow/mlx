// Copyright © 2026 MLX Vulkan Backend
// Placeholder primitive using Kompute

#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"
#include "mlx/backend/cpu/eval.h"

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

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }
  if (out.size() == 0) {
    return;
  }
  prepare_inputs_for_cpu_fallback(inputs);
  eval_cpu(inputs, out);
}

} // namespace mlx::core
