// Copyright © 2026 MLX Vulkan Backend

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/device.h"

namespace mlx::core::gpu {

bool is_available() {
  return vulkan::is_available();
}

int device_count() {
  return vulkan::device_count();
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;
  if (device_index < 0 || device_index >= device_count()) {
    return empty;
  }

  static auto info = []() {
    std::unordered_map<std::string, std::variant<std::string, size_t>> out;
    std::string device_name = "Vulkan GPU";
    try {
      device_name = vulkan::device(mlx::core::Device::gpu).get_device_name();
    } catch (...) {
      // Keep fallback values if querying the Vulkan device fails.
    }
    out.emplace("device_name", std::move(device_name));
    out.emplace("architecture", std::string("vulkan"));
    return out;
  }();

  return info;
}

} // namespace mlx::core::gpu
