// Copyright © 2026 MLX Vulkan Backend
// Device implementation using Kompute - Aligned with metal/device.cpp

#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include <iostream>
#include <set>
#include <sstream>

namespace mlx::core::vulkan {

// ============================================================================
// CommandEncoder Implementation
// ============================================================================

CommandEncoder::CommandEncoder(DeviceStream& stream) 
    : stream_(stream),
      current_kernel_(""),
      needs_barrier_(false),
      buffer_ops_(0),
      encoding_begun_(false) {
}

CommandEncoder::~CommandEncoder() {
  end_encoding();
}

void CommandEncoder::begin_encoding() {
  if (encoding_begun_) return;
  encoding_begun_ = true;
}

void CommandEncoder::end_encoding() {
  if (!encoding_begun_) return;
  encoding_begun_ = false;
}

void CommandEncoder::set_input_array(const array& a, int idx, int64_t offset) {
  all_inputs_.insert(a.data<void>());
  auto& device = vulkan::device(mlx::core::Device::gpu);
  auto tensor = device.get_tensor(a);
  
  // Track tensor for potential barrier
  if (prev_outputs_.find(tensor) != prev_outputs_.end()) {
    needs_barrier_ = true;
  }
}

void CommandEncoder::set_output_array(array& a, int idx, int64_t offset) {
  all_outputs_.insert(a.data<void>());
  auto& device = vulkan::device(mlx::core::Device::gpu);
  auto tensor = device.get_tensor(a);
  
  // Track as output for barrier logic
  next_outputs_.insert(tensor);
}

void CommandEncoder::register_output_array(const array& a) {
  all_outputs_.insert(a.data<void>());
}

void CommandEncoder::bind_buffer(const std::shared_ptr<Buffer>& buffer, int idx, bool is_output) {
  if (!buffer || !buffer->tensor()) return;
  
  auto tensor = buffer->tensor();
  
  if (is_output) {
    if (prev_outputs_.find(tensor) != prev_outputs_.end() ||
        all_inputs_.find(buffer->mapped_ptr()) != all_inputs_.end()) {
      needs_barrier_ = true;
    }
    next_outputs_.insert(tensor);
  } else {
    if (prev_outputs_.find(tensor) != prev_outputs_.end()) {
      needs_barrier_ = true;
    }
  }
}

void CommandEncoder::set_compute_pipeline(const std::string& kernel_name) {
  current_kernel_ = kernel_name;
}

void CommandEncoder::record_tensor_sync_device(const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
  if (!encoding_begun_) begin_encoding();
  
  for (auto& tensor : tensors) {
    stream_.sequence->record<kp::OpTensorSyncDevice>({tensor});
    buffer_ops_++;
    stream_.buffer_ops++;
  }
}

void CommandEncoder::record_tensor_sync_local(const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
  if (!encoding_begun_) begin_encoding();
  
  for (auto& tensor : tensors) {
    stream_.sequence->record<kp::OpTensorSyncLocal>({tensor});
    buffer_ops_++;
    stream_.buffer_ops++;
  }
}

void CommandEncoder::record_algo_dispatch(
    const std::string& kernel_name,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
    const std::vector<uint32_t>& workgroup) {
  if (!encoding_begun_) begin_encoding();
  
  // Insert barrier if needed
  maybeInsertBarrier();
  
  // Get algorithm from registry
  auto& registry = KernelRegistry::instance();
  kp::Workgroup wg{workgroup[0], workgroup[1], workgroup[2]};
  auto algo = registry.get_algorithm(
      kernel_name, 
      *stream_.manager, 
      tensors, 
      wg);
  
  // Record algorithm execution
  stream_.sequence->record<kp::OpAlgoDispatch>(algo);
  
  // Update tracking
  buffer_ops_++;
  stream_.buffer_ops++;
  
  // Update prev_outputs for next operation
  prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  next_outputs_.clear();
}

void CommandEncoder::dispatch_threadgroups(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
  // In Kompute, workgroup dispatch is handled by OpAlgoDispatch
  // This function is kept for API compatibility with Metal
  if (!encoding_begun_) begin_encoding();
}

void CommandEncoder::maybeInsertBarrier() {
  if (needs_barrier_) {
    barrier();
    needs_barrier_ = false;
  }
}

void CommandEncoder::barrier() {
  // Kompute handles barriers internally between operations
  // But we can force a barrier if needed for correctness
  // For now, this is a placeholder as Kompute's sequence ordering provides implicit barriers
}

// ============================================================================
// Fence Implementation
// ============================================================================

Fence::Fence(std::shared_ptr<kp::Sequence> sequence)
    : sequence_(sequence), signaled_(false) {}

Fence::~Fence() {
  if (signaled_ && sequence_) {
    // Wait for completion
    sequence_->eval();
  }
}

void Fence::wait() {
  if (sequence_) {
    sequence_->eval();
    signaled_ = false;
  }
}

void Fence::reset() {
  signaled_ = false;
}

bool Fence::is_done() const {
  // Kompute sequences don't have a direct "is done" query
  // We'd need to implement this with fences if needed
  return !signaled_;
}

// ============================================================================
// DeviceStream Implementation
// ============================================================================

DeviceStream::DeviceStream(std::shared_ptr<kp::Manager> manager, uint32_t queue_index)
    : manager(manager),
      queue_index(queue_index),
      buffer_ops(0),
      buffer_sizes(0) {
  // Create initial sequence
  reset_sequence();
}

DeviceStream::~DeviceStream() {
  // Ensure all operations complete
  if (sequence && buffer_ops > 0) {
    sequence->eval();
  }
}

void DeviceStream::reset_sequence() {
  sequence = manager->sequence();
  buffer_ops = 0;
  buffer_sizes = 0;
}

// ============================================================================
// Device Implementation
// ============================================================================

Device::Device() {
  std::cerr << "[Vulkan Device] Initializing with Kompute..." << std::endl;
  
  // Create Kompute manager with default GPU
  manager_ = std::make_shared<kp::Manager>();
  
  std::cerr << "[Vulkan Device] Kompute Manager created" << std::endl;
  
  // Initialize buffer manager
  BufferManager::instance().initialize(manager_);
  initialized_buffer_manager_ = true;
  
  std::cerr << "[Vulkan Device] Initialized successfully" << std::endl;
  std::cerr << "[Vulkan Device] Unified memory support: " 
            << (supports_unified_memory() ? "yes" : "no") << std::endl;
}

Device::~Device() {
  // Cleanup
  stream_map_.clear();
  
  if (initialized_buffer_manager_) {
    BufferManager::instance().shutdown();
  }
  
  clear_algorithm_cache();
  
  // Manager cleanup is handled by shared_ptr
  manager_.reset();
}

std::shared_ptr<Buffer> Device::get_buffer(const array& arr) {
  return BufferManager::instance().get_buffer(arr);
}

std::shared_ptr<Buffer> Device::create_buffer(size_t size) {
  return BufferManager::instance().create_temp_buffer(size);
}

std::shared_ptr<kp::Tensor> Device::get_tensor(const array& arr) {
  auto buffer = get_buffer(arr);
  if (buffer && buffer->tensor()) {
    return buffer->tensor();
  }
  
  // Create tensor from array data
  auto tensor = manager_->tensor(
      const_cast<void*>(arr.data<void>()),
      arr.size(),
      arr.itemsize(),
      kp::Tensor::TensorDataTypes::eFloat);  // TODO: Map MLX dtype to Kompute type
  
  return tensor;
}

std::shared_ptr<kp::Tensor> Device::create_tensor(size_t size) {
  std::vector<float> initial_data(size / sizeof(float), 0.0f);
  return manager_->tensor(initial_data);
}

void Device::register_array_buffer(const array& arr, std::shared_ptr<Buffer> buffer) {
  // Store in buffer manager
  // This is handled internally by BufferManager
}

void Device::new_queue(int index) {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  
  if (stream_map_.find(index) != stream_map_.end()) {
    throw std::runtime_error("Queue already exists: " + std::to_string(index));
  }
  
  // Create DeviceStream with Kompute manager
  auto stream = std::make_unique<DeviceStream>(manager_, 0);  // Use queue 0
  stream_map_[index] = std::move(stream);
  
  std::cerr << "[Vulkan Device] new_queue: index=" << index << std::endl;
}

std::shared_ptr<kp::Sequence> Device::get_sequence(int index) {
  DeviceStream& stream = get_stream_(index);
  return stream.sequence;
}

bool Device::command_buffer_needs_commit(int index) {
  DeviceStream& stream = get_stream_(index);
  return stream.buffer_ops >= max_ops_per_buffer_ || 
         stream.buffer_sizes >= (max_mb_per_buffer_ * 1024 * 1024);
}

void Device::commit_command_buffer(int index) {
  DeviceStream& stream = get_stream_(index);
  
  // End encoding if active
  if (stream.encoder) {
    stream.encoder->end_encoding();
    stream.encoder.reset();
  }
  
  // Evaluate sequence (submit to GPU) only when there is recorded work.
  if (stream.sequence && stream.buffer_ops > 0) {
    stream.sequence->eval();
  }
  
  // Reset sequence for new operations
  stream.reset_sequence();
  
  std::cerr << "[Vulkan Device] commit_command_buffer: index=" << index << std::endl;
}

CommandEncoder& Device::get_command_encoder(int index) {
  DeviceStream& stream = get_stream_(index);
  
  if (!stream.encoder) {
    stream.encoder = std::make_unique<CommandEncoder>(stream);
  }
  
  return *stream.encoder;
}

void Device::end_encoding(int index) {
  DeviceStream& stream = get_stream_(index);
  if (stream.encoder) {
    stream.encoder->end_encoding();
    stream.encoder.reset();
  }
}

std::shared_ptr<kp::Algorithm> Device::get_algorithm(
    const std::string& kernel_name,
    const std::vector<uint32_t>& spirv_code) {
  
  std::shared_lock<std::shared_mutex> lock(algorithm_mutex_);
  
  auto it = algorithm_cache_.find(kernel_name);
  if (it != algorithm_cache_.end()) {
    return it->second;
  }
  
  lock.unlock();
  std::unique_lock<std::shared_mutex> unique_lock(algorithm_mutex_);
  
  // Double-check
  it = algorithm_cache_.find(kernel_name);
  if (it != algorithm_cache_.end()) {
    return it->second;
  }
  
  // Create algorithm (this would need proper tensor spec)
  // For now, return nullptr - actual creation happens in KernelRegistry
  return nullptr;
}

void Device::clear_algorithm_cache() {
  std::unique_lock<std::shared_mutex> lock(algorithm_mutex_);
  algorithm_cache_.clear();
}

void Device::add_temporary(array arr, int index) {
  DeviceStream& stream = get_stream_(index);
  stream.temporaries.push_back(std::move(arr));
}

void Device::add_temporaries(std::vector<array> arrays, int index) {
  DeviceStream& stream = get_stream_(index);
  stream.temporaries.insert(
      stream.temporaries.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

std::string Device::get_device_name() const {
  if (manager_) {
    // Kompute doesn't expose device name directly
    // We'd need to access the Vulkan device properties
    return "Vulkan GPU (Kompute)";
  }
  return "Unknown";
}

bool Device::supports_unified_memory() const {
  if (manager_) {
    return vulkan::supports_unified_memory(manager_);
  }
  return false;
}

DeviceStream& Device::get_stream_(int index) {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  auto it = stream_map_.find(index);
  if (it == stream_map_.end() || !it->second) {
    throw std::invalid_argument("Stream not found: " + std::to_string(index));
  }
  return *it->second;
}

// ============================================================================
// Global Functions
// ============================================================================

Device& device(mlx::core::Device) {
  static Device instance;
  return instance;
}

bool is_available() {
  try {
    // Try to create a temporary manager
    auto manager = std::make_shared<kp::Manager>();
    return true;
  } catch (...) {
    return false;
  }
}

int device_count() {
  // Kompute doesn't expose device count directly
  // For now, return 1 if available, 0 otherwise
  return is_available() ? 1 : 0;
}

} // namespace mlx::core::vulkan
