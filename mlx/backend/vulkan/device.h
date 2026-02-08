// Copyright © 2026 MLX Vulkan Backend
// Device implementation using Kompute - Aligned with Metal architecture

#pragma once

#include <kompute/Kompute.hpp>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/backend/vulkan/buffer.h"

namespace mlx::core::vulkan {

// Forward declarations
struct DeviceStream;

// ============================================================================
// CommandEncoder - Vulkan equivalent of MTL::ComputeCommandEncoder
// Uses Kompute but provides Metal-like interface for architecture alignment
// ============================================================================

struct MLX_API CommandEncoder {
  explicit CommandEncoder(DeviceStream& stream);
  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  // Buffer binding (equivalent to setBuffer)
  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void register_output_array(const array& a);
  void bind_buffer(const std::shared_ptr<Buffer>& buffer, int idx, bool is_output);

  // Algorithm/Pipeline state
  void set_compute_pipeline(const std::string& kernel_name);

  // Record operations for lazy evaluation
  void record_tensor_sync_device(const std::vector<std::shared_ptr<kp::Tensor>>& tensors);
  void record_tensor_sync_local(const std::vector<std::shared_ptr<kp::Tensor>>& tensors);
  void record_algo_dispatch(
      const std::string& kernel_name,
      const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
      const std::vector<uint32_t>& workgroup = {256, 1, 1},
      const std::vector<float>& push_consts = {});

  // Dispatch (triggers actual recording to sequence)
  void dispatch_threadgroups(uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1);

  // Barrier management
  void maybeInsertBarrier();
  void barrier();

  // Destructor
  ~CommandEncoder();

  // Accessors for tracking
  std::unordered_set<const void*>& inputs() { return all_inputs_; }
  std::unordered_set<const void*>& outputs() { return all_outputs_; }
  int buffer_ops() const { return buffer_ops_; }

  // Begin/End encoding
  void begin_encoding();
  void end_encoding();

 private:
  DeviceStream& stream_;
  std::string current_kernel_;
  bool needs_barrier_{false};
  int buffer_ops_{0};
  
  // Resource tracking for barrier insertion
  std::unordered_set<std::shared_ptr<kp::Tensor>> prev_outputs_;
  std::unordered_set<std::shared_ptr<kp::Tensor>> next_outputs_;
  std::unordered_set<const void*> all_inputs_;
  std::unordered_set<const void*> all_outputs_;
  
  // Track if encoding has begun
  bool encoding_begun_{false};
};

// ============================================================================
// Fence - Vulkan equivalent of MTL::Fence
// ============================================================================

struct Fence {
  Fence(std::shared_ptr<kp::Sequence> sequence);
  ~Fence();
  
  void wait();
  void reset();
  bool is_done() const;
  
 private:
  std::shared_ptr<kp::Sequence> sequence_;
  bool signaled_{false};
};

// ============================================================================
// DeviceStream - Aligned with Metal's DeviceStream
// ============================================================================

struct DeviceStream {
  DeviceStream(std::shared_ptr<kp::Manager> manager, uint32_t queue_index);
  ~DeviceStream();
  
  // Queue index
  uint32_t queue_index;
  
  // Fence tracking (equivalent to outputs map in Metal)
  std::unordered_map<const void*, std::shared_ptr<Fence>> outputs;
  std::mutex fence_mtx;

  // Kompute sequence for command recording
  std::shared_ptr<kp::Sequence> sequence;
  
  // Command buffer state (equivalent to Metal's buffer tracking)
  int buffer_ops{0};
  size_t buffer_sizes{0};

  // Encoder and temporaries
  std::unique_ptr<CommandEncoder> encoder{nullptr};
  std::shared_ptr<Fence> fence;
  std::vector<array> temporaries;
  
  // Device reference
  std::shared_ptr<kp::Manager> manager;
  
  // Reset sequence for new recording
  void reset_sequence();
};

// ============================================================================
// Device - Aligned with Metal's Device class using Kompute
// ============================================================================

class MLX_API Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  // Kompute manager access
  std::shared_ptr<kp::Manager> kompute_manager() { return manager_; }
  
  // Buffer management
  std::shared_ptr<Buffer> get_buffer(const array& arr);
  std::shared_ptr<Buffer> create_buffer(size_t size);
  void register_array_buffer(const array& arr, std::shared_ptr<Buffer> buffer);
  
  // Tensor management (Kompute)
  std::shared_ptr<kp::Tensor> get_tensor(const array& arr);
  std::shared_ptr<kp::Tensor> create_tensor(size_t size);
  
  // Queue management
  void new_queue(int index);
  
  // Command buffer management (aligned with Metal)
  std::shared_ptr<kp::Sequence> get_sequence(int index);
  bool command_buffer_needs_commit(int index);
  void commit_command_buffer(int index);
  CommandEncoder& get_command_encoder(int index);
  void end_encoding(int index);
  
  // Algorithm/Shader management
  std::shared_ptr<kp::Algorithm> get_algorithm(
      const std::string& kernel_name,
      const std::vector<uint32_t>& spirv_code);
  
  void clear_algorithm_cache();
  
  // Temporaries
  void add_temporary(array arr, int index);
  void add_temporaries(std::vector<array> arrays, int index);
  
  // Info
  std::string get_device_name() const;
  bool supports_unified_memory() const;

 private:
  DeviceStream& get_stream_(int index);
  void create_sequence_(DeviceStream& stream);
  
  // Kompute manager
  std::shared_ptr<kp::Manager> manager_;
  
  // Stream management
  std::unordered_map<int32_t, std::unique_ptr<DeviceStream>> stream_map_;
  std::mutex stream_mutex_;
  
  // Algorithm cache (equivalent to pipeline cache)
  std::shared_mutex algorithm_mutex_;
  std::unordered_map<std::string, std::shared_ptr<kp::Algorithm>> algorithm_cache_;
  
  // Configuration
  int max_ops_per_buffer_ = 100;
  int max_mb_per_buffer_ = 50;
  
  // Buffer manager reference
  bool initialized_buffer_manager_ = false;
};

// Singleton accessor
MLX_API Device& device(mlx::core::Device);

// Check availability
bool is_available();
int device_count();

} // namespace mlx::core::vulkan
