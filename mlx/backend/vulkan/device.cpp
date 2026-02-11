// Copyright © 2026 MLX Vulkan Backend
// Device implementation using Kompute - Aligned with metal/device.cpp

#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vulkan/vulkan.h>

namespace mlx::core::vulkan {

namespace {

struct VulkanAvailabilityProbe {
  bool available{false};
  int count{0};

  VulkanAvailabilityProbe() {
    VkInstance instance = VK_NULL_HANDLE;

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "mlx_vulkan_probe";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "mlx";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
      return;
    }

    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) ==
            VK_SUCCESS &&
        device_count > 0) {
      available = true;
      count = static_cast<int>(device_count);
    }

    vkDestroyInstance(instance, nullptr);
  }
};

const VulkanAvailabilityProbe& vulkan_availability() {
  static const VulkanAvailabilityProbe kProbe{};
  return kProbe;
}

inline kp::Tensor::TensorDataTypes to_kompute_dtype(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return kp::Tensor::TensorDataTypes::eBool;
    case uint8:
    case uint16:
    case uint32:
    case uint64:
      return kp::Tensor::TensorDataTypes::eUnsignedInt;
    case int8:
    case int16:
    case int32:
    case int64:
      return kp::Tensor::TensorDataTypes::eInt;
    case float16:
    case float32:
    case bfloat16:
    case complex64:
      return kp::Tensor::TensorDataTypes::eFloat;
    case float64:
      return kp::Tensor::TensorDataTypes::eDouble;
  }
  return kp::Tensor::TensorDataTypes::eFloat;
}

inline size_t tensor_element_count(const array& arr) {
  return arr.data_size() > 0 ? arr.data_size() : arr.size();
}

inline int parse_env_int_clamped(
    const char* name,
    int default_value,
    int min_value,
    int max_value) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return default_value;
  }
  if (parsed < static_cast<long>(min_value)) {
    return min_value;
  }
  if (parsed > static_cast<long>(max_value)) {
    return max_value;
  }
  return static_cast<int>(parsed);
}

} // namespace

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

  std::vector<std::shared_ptr<kp::Tensor>> unique_tensors;
  unique_tensors.reserve(tensors.size());
  std::unordered_set<kp::Tensor*> seen;
  seen.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (!tensor) {
      continue;
    }
    if (seen.insert(tensor.get()).second) {
      unique_tensors.push_back(tensor);
    }
  }
  if (unique_tensors.empty()) {
    return;
  }

  stream_.sequence->record<kp::OpTensorSyncDevice>(unique_tensors);
  buffer_ops_++;
  stream_.buffer_ops++;
}

void CommandEncoder::record_tensor_sync_local(const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
  if (!encoding_begun_) begin_encoding();

  std::vector<std::shared_ptr<kp::Tensor>> unique_tensors;
  unique_tensors.reserve(tensors.size());
  std::unordered_set<kp::Tensor*> seen;
  seen.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (!tensor) {
      continue;
    }
    if (seen.insert(tensor.get()).second) {
      unique_tensors.push_back(tensor);
    }
  }
  if (unique_tensors.empty()) {
    return;
  }

  stream_.sequence->record<kp::OpTensorSyncLocal>(unique_tensors);
  buffer_ops_++;
  stream_.buffer_ops++;
}

void CommandEncoder::record_algo_dispatch(
    const std::string& kernel_name,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
    const std::vector<uint32_t>& workgroup,
    const std::vector<uint32_t>& push_consts) {
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
      wg,
      push_consts);
  
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
  for (auto& in_flight : inflight_sequences) {
    if (in_flight) {
      in_flight->evalAwait();
    }
  }
  inflight_sequences.clear();
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
  max_ops_per_buffer_ = parse_env_int_clamped(
      "MLX_VK_MAX_OPS_PER_BUFFER", max_ops_per_buffer_, 1, 100000);
  max_mb_per_buffer_ = parse_env_int_clamped(
      "MLX_VK_MAX_MB_PER_BUFFER", max_mb_per_buffer_, 1, 4096);
  max_inflight_sequences_ = parse_env_int_clamped(
      "MLX_VK_MAX_INFLIGHT_SEQUENCES", max_inflight_sequences_, 1, 64);

  // Create Kompute manager with default GPU
  manager_ = std::make_shared<kp::Manager>();

  // Initialize buffer manager
  BufferManager::instance().initialize(manager_);
  initialized_buffer_manager_ = true;
}

Device::~Device() {
  // Cleanup
  stream_map_.clear();
  {
    std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
    tensor_cache_.clear();
    tensor_storage_index_.clear();
    dirty_tensors_by_stream_.clear();
  }
  
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

Device::TensorStorageKey Device::make_tensor_storage_key_(
    const void* data_ptr,
    const std::shared_ptr<array::Data>& data_ref,
    Dtype dtype) {
  return TensorStorageKey{data_ptr, data_ref.get(), dtype};
}

bool Device::tensor_entry_matches_request_(
    const TensorCacheEntry& entry,
    const TensorStorageKey& key,
    size_t min_elem_count) const {
  return entry.data_ptr == key.data_ptr && entry.data_owner == key.data_owner &&
      entry.dtype == key.dtype && entry.elem_count >= min_elem_count &&
      !entry.data_ref.expired();
}

void Device::add_tensor_index_locked_(
    const TensorStorageKey& storage_key,
    std::uintptr_t key) {
  auto& keys = tensor_storage_index_[storage_key];
  if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
    keys.push_back(key);
  }
}

void Device::remove_tensor_index_locked_(
    const TensorStorageKey& storage_key,
    std::uintptr_t key) {
  auto it = tensor_storage_index_.find(storage_key);
  if (it == tensor_storage_index_.end()) {
    return;
  }
  auto& keys = it->second;
  keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
  if (keys.empty()) {
    tensor_storage_index_.erase(it);
  }
}

void Device::erase_tensor_entry_locked_(
    std::unordered_map<std::uintptr_t, TensorCacheEntry>::iterator it) {
  if (it == tensor_cache_.end()) {
    return;
  }
  const auto storage_key =
      TensorStorageKey{it->second.data_ptr, it->second.data_owner, it->second.dtype};
  if (it->second.host_dirty && it->second.dirty_stream_index >= 0) {
    untrack_dirty_tensor_(it->second.dirty_stream_index, it->first);
  }
  remove_tensor_index_locked_(storage_key, it->first);
  tensor_cache_.erase(it);
}

std::unordered_map<std::uintptr_t, Device::TensorCacheEntry>::iterator
Device::find_tensor_entry_locked_(
    std::uintptr_t key,
    const TensorStorageKey& storage_key,
    size_t min_elem_count) {
  auto it = tensor_cache_.find(key);
  if (it != tensor_cache_.end()) {
    if (tensor_entry_matches_request_(it->second, storage_key, min_elem_count)) {
      return it;
    }
    erase_tensor_entry_locked_(it);
  }

  auto index_it = tensor_storage_index_.find(storage_key);
  if (index_it == tensor_storage_index_.end()) {
    return tensor_cache_.end();
  }

  std::uintptr_t best_key = 0;
  bool has_best = false;
  size_t best_elem_count = 0;
  const auto candidate_keys = index_it->second;
  for (auto candidate_key : candidate_keys) {
    auto candidate_it = tensor_cache_.find(candidate_key);
    if (candidate_it == tensor_cache_.end()) {
      remove_tensor_index_locked_(storage_key, candidate_key);
      continue;
    }
    if (!tensor_entry_matches_request_(
            candidate_it->second, storage_key, min_elem_count)) {
      if (candidate_it->second.data_ref.expired() ||
          candidate_it->second.data_owner != storage_key.data_owner ||
          candidate_it->second.data_ptr != storage_key.data_ptr ||
          candidate_it->second.dtype != storage_key.dtype) {
        erase_tensor_entry_locked_(candidate_it);
      }
      continue;
    }
    if (candidate_it->second.elem_count >= best_elem_count) {
      best_elem_count = candidate_it->second.elem_count;
      best_key = candidate_key;
      has_best = true;
    }
  }

  if (!has_best) {
    return tensor_cache_.end();
  }
  return tensor_cache_.find(best_key);
}

std::shared_ptr<kp::Tensor> Device::get_tensor(const array& arr) {
  const auto key = arr.id();
  const auto data_ptr = arr.data<void>();
  const auto nbytes = arr.nbytes();
  const auto elem_count = tensor_element_count(arr);
  const auto dtype = arr.dtype();
  const auto data_ref = arr.data_shared_ptr();
  const auto storage_key = make_tensor_storage_key_(data_ptr, data_ref, dtype);

  {
    std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
    auto it = find_tensor_entry_locked_(key, storage_key, elem_count);
    if (it != tensor_cache_.end()) {
      if (auto tensor = it->second.tensor.lock()) {
        return tensor;
      }
      erase_tensor_entry_locked_(it);
    }
  }

  if (elem_count >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(
        "[Vulkan Device] Tensor element count exceeds uint32 range.");
  }

  auto tensor = manager_->tensor(
      const_cast<void*>(data_ptr),
      static_cast<uint32_t>(elem_count),
      static_cast<uint32_t>(arr.itemsize()),
      to_kompute_dtype(dtype));

  {
    std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
    tensor_cache_[key] = TensorCacheEntry{
        tensor,
        nullptr,
        data_ref,
        data_ptr,
        data_ref.get(),
        nbytes,
        elem_count,
        dtype,
        false,
        -1};
    add_tensor_index_locked_(storage_key, key);
  }

  return tensor;
}

std::shared_ptr<kp::Tensor> Device::create_tensor(size_t size) {
  std::vector<float> initial_data(size / sizeof(float), 0.0f);
  return manager_->tensor(initial_data);
}

void Device::invalidate_tensor(const array& arr) {
  const auto data_ref = arr.data_shared_ptr();
  const auto storage_key =
      make_tensor_storage_key_(arr.data<void>(), data_ref, arr.dtype());
  std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
  auto it =
      find_tensor_entry_locked_(arr.id(), storage_key, tensor_element_count(arr));
  if (it == tensor_cache_.end()) {
    return;
  }
  erase_tensor_entry_locked_(it);
}

void Device::mark_tensor_host_dirty(const array& arr, int stream_index) {
  const auto key = arr.id();
  const auto data_ptr = arr.data<void>();
  const auto elem_count = tensor_element_count(arr);
  const auto dtype = arr.dtype();
  const auto data_ref = arr.data_shared_ptr();
  const auto storage_key = make_tensor_storage_key_(data_ptr, data_ref, dtype);

  std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
  auto it = find_tensor_entry_locked_(key, storage_key, elem_count);
  if (it == tensor_cache_.end()) {
    return;
  }
  auto tensor = it->second.tensor.lock();
  if (!tensor) {
    erase_tensor_entry_locked_(it);
    return;
  }
  if (it->second.host_dirty && it->second.dirty_stream_index >= 0 &&
      it->second.dirty_stream_index != stream_index) {
    untrack_dirty_tensor_(it->second.dirty_stream_index, it->first);
  }
  it->second.host_dirty = true;
  it->second.dirty_stream_index = stream_index;
  // Keep tensor alive until host sync copies back dirty data.
  it->second.pinned_tensor = std::move(tensor);
  track_dirty_tensor_(stream_index, it->first);
}

bool Device::tensor_needs_sync_device(const array& arr) {
  const auto key = arr.id();
  const auto data_ptr = arr.data<void>();
  const auto elem_count = tensor_element_count(arr);
  const auto dtype = arr.dtype();
  const auto data_ref = arr.data_shared_ptr();
  const auto storage_key = make_tensor_storage_key_(data_ptr, data_ref, dtype);

  std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
  auto it = find_tensor_entry_locked_(key, storage_key, elem_count);
  if (it == tensor_cache_.end()) {
    return true;
  }

  auto tensor = it->second.pinned_tensor ? it->second.pinned_tensor
                                         : it->second.tensor.lock();
  if (!tensor) {
    erase_tensor_entry_locked_(it);
    return true;
  }

  // host_dirty=true means device has the newest contents; uploading host memory
  // would overwrite fresh device results with stale host data.
  return !it->second.host_dirty;
}

void Device::sync_array_to_host_if_needed(const array& arr) {
  const auto key = arr.id();
  const auto data_ptr = arr.data<void>();
  const auto nbytes = arr.nbytes();
  const auto elem_count = tensor_element_count(arr);
  const auto dtype = arr.dtype();
  const auto data_ref = arr.data_shared_ptr();
  const auto storage_key = make_tensor_storage_key_(data_ptr, data_ref, dtype);

  std::shared_ptr<kp::Tensor> tensor;
  int dirty_stream_index = -1;
  std::uintptr_t matched_key = 0;
  {
    std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
    auto it = find_tensor_entry_locked_(key, storage_key, elem_count);
    if (it == tensor_cache_.end()) {
      return;
    }

    matched_key = it->first;
    if (!it->second.host_dirty) {
      return;
    }
    dirty_stream_index = it->second.dirty_stream_index;
    tensor = it->second.pinned_tensor ? it->second.pinned_tensor
                                      : it->second.tensor.lock();
    if (!tensor) {
      erase_tensor_entry_locked_(it);
      return;
    }
  }

  // Ensure pending work on the producing stream is submitted before host sync.
  if (dirty_stream_index >= 0) {
    wait_for_stream(dirty_stream_index);
  }

  auto seq = manager_->sequence();
  seq->record<kp::OpTensorSyncLocal>({tensor});
  seq->eval();

  if (data_ptr != tensor->rawData()) {
    std::memcpy(const_cast<void*>(data_ptr), tensor->rawData(), nbytes);
  }

  std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
  auto it = tensor_cache_.find(matched_key);
  if (it != tensor_cache_.end()) {
    if (auto cur = it->second.tensor.lock(); cur && cur == tensor) {
      it->second.host_dirty = false;
      it->second.dirty_stream_index = -1;
      it->second.pinned_tensor.reset();
      if (dirty_stream_index >= 0) {
        untrack_dirty_tensor_(dirty_stream_index, matched_key);
      }
    }
  }
}

void Device::sync_dirty_tensors_for_stream(int stream_index) {
  struct PendingCopy {
    std::uintptr_t key{0};
    std::shared_ptr<kp::Tensor> tensor;
    std::shared_ptr<array::Data> data_ref;
    const void* data_ptr{nullptr};
    size_t nbytes{0};
  };

  std::vector<PendingCopy> pending;
  {
    std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
    auto tracker_it = dirty_tensors_by_stream_.find(stream_index);
    if (tracker_it == dirty_tensors_by_stream_.end()) {
      return;
    }
    std::vector<std::uintptr_t> dirty_keys = std::move(tracker_it->second.keys);
    dirty_tensors_by_stream_.erase(tracker_it);

    for (auto key : dirty_keys) {
      auto it = tensor_cache_.find(key);
      if (it == tensor_cache_.end()) {
        continue;
      }
      auto& entry = it->second;
      if (!entry.host_dirty || entry.dirty_stream_index != stream_index) {
        continue;
      }

      auto tensor = entry.pinned_tensor ? entry.pinned_tensor : entry.tensor.lock();
      auto data_ref = entry.data_ref.lock();
      if (!tensor || !data_ref) {
        erase_tensor_entry_locked_(it);
        continue;
      }

      pending.push_back(PendingCopy{
          it->first, tensor, data_ref, entry.data_ptr, entry.nbytes});
    }
  }

  if (pending.empty()) {
    return;
  }

  auto seq = manager_->sequence();
  std::vector<std::shared_ptr<kp::Tensor>> local_sync_tensors;
  local_sync_tensors.reserve(pending.size());
  std::unordered_set<kp::Tensor*> seen_tensors;
  seen_tensors.reserve(pending.size());
  for (const auto& item : pending) {
    if (item.tensor && seen_tensors.insert(item.tensor.get()).second) {
      local_sync_tensors.push_back(item.tensor);
    }
  }
  if (!local_sync_tensors.empty()) {
    seq->record<kp::OpTensorSyncLocal>(local_sync_tensors);
  }
  seq->eval();

  for (const auto& item : pending) {
    if (item.data_ptr != item.tensor->rawData()) {
      std::memcpy(
          const_cast<void*>(item.data_ptr), item.tensor->rawData(), item.nbytes);
    }
  }

  std::lock_guard<std::mutex> lock(tensor_cache_mutex_);
  for (const auto& item : pending) {
    auto it = tensor_cache_.find(item.key);
    if (it == tensor_cache_.end()) {
      continue;
    }
    if (auto cur = it->second.tensor.lock(); cur && cur == item.tensor) {
      it->second.host_dirty = false;
      it->second.dirty_stream_index = -1;
      it->second.pinned_tensor.reset();
      untrack_dirty_tensor_(stream_index, item.key);
    }
  }
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
    stream.sequence->evalAsync();
    stream.inflight_sequences.push_back(stream.sequence);
    await_inflight_sequences_(
        stream, static_cast<size_t>(max_inflight_sequences_));
  }
  
  // Reset sequence for new operations
  stream.reset_sequence();
}

void Device::wait_for_stream(int index) {
  // Submit currently recorded work first (if any), then wait for all in-flight
  // submissions on this stream.
  end_encoding(index);
  commit_command_buffer(index);
  DeviceStream& stream = get_stream_(index);
  await_inflight_sequences_(stream, 0);
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

void Device::await_inflight_sequences_(DeviceStream& stream, size_t keep_pending) {
  while (stream.inflight_sequences.size() > keep_pending) {
    auto seq = stream.inflight_sequences.front();
    stream.inflight_sequences.pop_front();
    if (seq) {
      seq->evalAwait();
    }
  }
}

void Device::track_dirty_tensor_(int stream_index, std::uintptr_t key) {
  auto& tracker = dirty_tensors_by_stream_[stream_index];
  if (tracker.key_set.insert(key).second) {
    tracker.keys.push_back(key);
  }
}

void Device::untrack_dirty_tensor_(int stream_index, std::uintptr_t key) {
  auto it = dirty_tensors_by_stream_.find(stream_index);
  if (it == dirty_tensors_by_stream_.end()) {
    return;
  }
  it->second.key_set.erase(key);
  if (it->second.key_set.empty()) {
    dirty_tensors_by_stream_.erase(it);
  }
}

// ============================================================================
// Global Functions
// ============================================================================

Device& device(mlx::core::Device) {
  static Device instance;
  return instance;
}

bool is_available() {
  return vulkan_availability().available;
}

int device_count() {
  return vulkan_availability().count;
}

} // namespace mlx::core::vulkan
