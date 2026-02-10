// Copyright © 2026 MLX Vulkan Backend

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernel_registry.h"
#include "mlx/backend/vulkan/op_profiler.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/transforms_impl.h"

namespace {

inline void prepare_inputs_for_cpu_fallback(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::Stream stream) {
  for (const auto& in : inputs) {
    auto& mutable_in = const_cast<mlx::core::array&>(in);
    if (mutable_in.status() == mlx::core::array::Status::unscheduled) {
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

inline void sync_inputs_to_host_if_needed(
    const std::vector<mlx::core::array>& inputs) {
  auto& device = mlx::core::vulkan::device(mlx::core::Device::gpu);
  for (const auto& in : inputs) {
    device.sync_array_to_host_if_needed(in);
  }
}

inline bool is_row_contiguous_materialized(const mlx::core::array& arr) {
  return arr.flags().row_contiguous && arr.data_size() == arr.size();
}

// Accept dense row-major views even if data_size != size (e.g. KV cache views).
// This keeps layout constraints strict while allowing sliced/cached buffers.
inline bool is_dense_row_major_view(const mlx::core::array& arr) {
  if (!arr.flags().row_contiguous) {
    return false;
  }
  if (arr.ndim() == 0) {
    return true;
  }

  int64_t expected = 1;
  for (int i = arr.ndim() - 1; i >= 0; --i) {
    const int64_t dim = arr.shape(i);
    const int64_t stride = arr.strides(i);
    if (dim <= 0) {
      return false;
    }
    if (dim != 1 && stride != expected) {
      return false;
    }
    if (expected > std::numeric_limits<int64_t>::max() / dim) {
      return false;
    }
    expected *= dim;
  }
  return true;
}

inline bool is_rope_head_seq_transposed_layout(const mlx::core::array& arr) {
  if (arr.ndim() != 4) {
    return false;
  }
  const int64_t heads = arr.shape(1);
  const int64_t t_size = arr.shape(2);
  const int64_t d_size = arr.shape(3);
  if (heads <= 0 || t_size <= 0 || d_size <= 0) {
    return false;
  }
  const auto& st = arr.strides();
  return st[0] == t_size * heads * d_size && st[1] == d_size &&
      st[2] == heads * d_size && st[3] == 1;
}

inline std::string shape_string(const mlx::core::array& arr) {
  std::string s = "[";
  for (int i = 0; i < arr.ndim(); ++i) {
    if (i > 0) {
      s += ",";
    }
    s += std::to_string(arr.shape(i));
  }
  s += "]";
  return s;
}

inline std::string strides_string(const mlx::core::array& arr) {
  std::string s = "[";
  for (int i = 0; i < arr.ndim(); ++i) {
    if (i > 0) {
      s += ",";
    }
    s += std::to_string(arr.strides(i));
  }
  s += "]";
  return s;
}

inline bool rope_debug_reject_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_ROPE_REJECT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool sdpa_debug_reject_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_SDPA_REJECT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool sdpa_debug_hit_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_SDPA_HIT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool logsumexp_debug_reject_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_LOGSUMEXP_REJECT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool compiled_profile_detail_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_PROFILE_COMPILED_DETAIL");
    if (!v) {
      return false;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
        std::strcmp(v, "off") == 0) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool compiled_debug_detail_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_COMPILED_DETAIL");
    if (!v) {
      return false;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
        std::strcmp(v, "off") == 0) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool env_flag_default_true(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return true;
  }
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "off") == 0) {
    return false;
  }
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
      std::strcmp(v, "on") == 0;
}

inline bool env_flag_default_false(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return false;
  }
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "off") == 0) {
    return false;
  }
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
      std::strcmp(v, "on") == 0;
}

inline bool native_compiled_sigmoid_mul_mul_bf16_enabled() {
  static const bool enabled = env_flag_default_true(
      "MLX_VK_ENABLE_COMPILED_SIGMOID_MUL_MUL_BF16");
  return enabled;
}

inline bool sdpa_stats_enabled() {
  static const bool enabled = env_flag_default_false("MLX_VK_SDPA_STATS");
  return enabled;
}

inline bool qmm_stats_enabled() {
  static const bool enabled = env_flag_default_false("MLX_VK_QMM_STATS");
  return enabled;
}

inline uint32_t clamp_len_hint(int64_t len) {
  if (len <= 0) {
    return 0u;
  }
  if (len > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(len);
}

inline std::pair<uint32_t, uint32_t> sdpa_qk_len_hint(
    const std::vector<mlx::core::array>& inputs) {
  if (inputs.size() < 2) {
    return {0u, 0u};
  }
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const uint32_t q_len = (q.ndim() >= 3) ? clamp_len_hint(q.shape(2)) : 0u;
  const uint32_t k_len = (k.ndim() >= 3) ? clamp_len_hint(k.shape(2)) : 0u;
  return {q_len, k_len};
}

inline uint32_t sdpa_qk_len_hint_from_array(const mlx::core::array& arr) {
  return (arr.ndim() >= 3) ? clamp_len_hint(arr.shape(2)) : 0u;
}

inline const char* sdpa_len_bucket(uint32_t len) {
  if (len <= 1u) {
    return "1";
  }
  if (len <= 4u) {
    return "2-4";
  }
  if (len <= 8u) {
    return "5-8";
  }
  if (len <= 16u) {
    return "9-16";
  }
  if (len <= 32u) {
    return "17-32";
  }
  if (len <= 64u) {
    return "33-64";
  }
  return "65+";
}

inline const char* sdpa_splitk_bucket(uint32_t split_k) {
  if (split_k <= 2u) {
    return "2";
  }
  if (split_k <= 4u) {
    return "3-4";
  }
  if (split_k <= 8u) {
    return "5-8";
  }
  if (split_k <= 16u) {
    return "9-16";
  }
  return "17+";
}

inline const char* sdpa_stage_name(uint32_t q_len) {
  return q_len <= 1u ? "decode" : "prefill";
}

enum class SdpaNativePathKind { Decode, Prefill };

inline SdpaNativePathKind sdpa_native_path_kind(uint32_t q_len) {
  return q_len <= 1u ? SdpaNativePathKind::Decode
                     : SdpaNativePathKind::Prefill;
}

inline const char* sdpa_native_path_name(SdpaNativePathKind path_kind) {
  return path_kind == SdpaNativePathKind::Decode ? "decode" : "prefill";
}

inline bool native_sdpa_decode_d128_kernel_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_SDPA_DECODE_D128");
  return enabled;
}

inline bool native_sdpa_decode_d128_k32_kernel_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_SDPA_DECODE_D128_K32");
  return enabled;
}

inline bool native_sdpa_decode_d128_k64_kernel_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_SDPA_DECODE_D128_K64");
  return enabled;
}

inline bool native_sdpa_decode_d128_k128_kernel_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_SDPA_DECODE_D128_K128");
  return enabled;
}

inline bool native_sdpa_decode_splitk_reduce_l32_kernel_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_SDPA_DECODE_SPLITK_REDUCE_L32");
  return enabled;
}

inline bool native_sdpa_decode_splitk_reduce_subgroup_kernel_enabled() {
  static const bool enabled =
      env_flag_default_false("MLX_VK_ENABLE_SDPA_DECODE_SPLITK_REDUCE_SUBGROUP");
  return enabled;
}

inline std::string sdpa_bucket_key(uint32_t q_len, uint32_t k_len) {
  return std::string("stage=") + sdpa_stage_name(q_len) + " q=" +
      sdpa_len_bucket(q_len) + " k=" + sdpa_len_bucket(k_len);
}

inline const char* sdpa_native_direct_kernel(
    SdpaNativePathKind path_kind,
    uint32_t qk_dim,
    uint32_t v_dim,
    uint32_t k_len,
    uint32_t mask_mode,
    uint32_t split_k) {
  if (path_kind == SdpaNativePathKind::Decode && split_k <= 1u &&
      qk_dim == 128u && v_dim == 128u &&
      native_sdpa_decode_d128_kernel_enabled()) {
    if (k_len <= 32u && mask_mode == 0u &&
        native_sdpa_decode_d128_k32_kernel_enabled()) {
      return mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1_D128_K32;
    }
    if (k_len <= 64u && mask_mode == 0u &&
        native_sdpa_decode_d128_k64_kernel_enabled()) {
      return mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1_D128_K64;
    }
    if (k_len <= 128u && mask_mode == 0u &&
        native_sdpa_decode_d128_k128_kernel_enabled()) {
      return mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1_D128_K128;
    }
    return mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1_D128;
  }
  return path_kind == SdpaNativePathKind::Decode
      ? mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_Q1
      : mlx::core::vulkan::KernelRegistry::SDPA_BF16_PREFILL_Q1;
}

inline bool prefer_sdpa_decode_direct_d128(
    SdpaNativePathKind path_kind,
    uint32_t qk_dim,
    uint32_t v_dim,
    uint32_t k_len,
    uint32_t mask_mode) {
  return path_kind == SdpaNativePathKind::Decode &&
      qk_dim == 128u &&
      v_dim == 128u &&
      k_len <= 128u &&
      mask_mode == 0u &&
      native_sdpa_decode_d128_kernel_enabled() &&
      native_sdpa_decode_d128_k128_kernel_enabled();
}

inline const char* sdpa_native_splitk_stage1_kernel(
    SdpaNativePathKind path_kind) {
  return path_kind == SdpaNativePathKind::Decode
      ? mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_SPLITK_STAGE1
      : mlx::core::vulkan::KernelRegistry::SDPA_BF16_PREFILL_SPLITK_STAGE1;
}

inline const char* sdpa_native_splitk_reduce_kernel(
    SdpaNativePathKind path_kind,
    uint32_t split_k) {
  if (path_kind == SdpaNativePathKind::Decode && split_k > 1u &&
      native_sdpa_decode_splitk_reduce_subgroup_kernel_enabled()) {
    return mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_SPLITK_REDUCE_SUBGROUP;
  }
  if (path_kind == SdpaNativePathKind::Decode && split_k <= 16u &&
      native_sdpa_decode_splitk_reduce_l32_kernel_enabled()) {
    return mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_SPLITK_REDUCE_L32;
  }
  return path_kind == SdpaNativePathKind::Decode
      ? mlx::core::vulkan::KernelRegistry::SDPA_BF16_DECODE_SPLITK_REDUCE
      : mlx::core::vulkan::KernelRegistry::SDPA_BF16_PREFILL_SPLITK_REDUCE;
}

struct SdpaStatsState {
  std::mutex mtx;
  uint64_t use_fallback_calls{0};
  uint64_t use_fallback_rejects{0};
  uint64_t native_gate_calls{0};
  uint64_t native_gate_rejects{0};
  uint64_t native_hits{0};
  uint64_t native_dispatch_success{0};
  uint64_t native_dispatch_fail{0};
  uint64_t final_fallbacks{0};
  std::unordered_map<std::string, uint64_t> use_fallback_reason_counts;
  std::unordered_map<std::string, uint64_t> use_fallback_reason_bucket_counts;
  std::unordered_map<std::string, uint64_t> native_reject_reason_counts;
  std::unordered_map<std::string, uint64_t> native_reject_reason_bucket_counts;
  std::unordered_map<std::string, uint64_t> native_hit_bucket_counts;
  std::unordered_map<std::string, uint64_t> splitk_reduce_kernel_counts;
  std::unordered_map<std::string, uint64_t> splitk_parts_bucket_counts;
};

inline SdpaStatsState& sdpa_stats_state() {
  static SdpaStatsState state;
  return state;
}

inline std::vector<std::pair<std::string, uint64_t>> sort_stats_counts(
    const std::unordered_map<std::string, uint64_t>& counts) {
  std::vector<std::pair<std::string, uint64_t>> out(counts.begin(), counts.end());
  std::sort(
      out.begin(),
      out.end(),
      [](const auto& a, const auto& b) {
        if (a.second != b.second) {
          return a.second > b.second;
        }
        return a.first < b.first;
      });
  return out;
}

struct SdpaStatsReporter;
inline SdpaStatsReporter& sdpa_stats_reporter();

inline void sdpa_stats_record_use_fallback_call() {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  std::lock_guard<std::mutex> lock(state.mtx);
  state.use_fallback_calls++;
}

inline void sdpa_stats_record_use_fallback_reject(
    const char* reason,
    uint32_t q_len,
    uint32_t k_len) {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  const std::string reason_str = reason ? reason : "unknown";
  const std::string bucket = sdpa_bucket_key(q_len, k_len);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.use_fallback_rejects++;
  state.use_fallback_reason_counts[reason_str]++;
  state.use_fallback_reason_bucket_counts[reason_str + " | " + bucket]++;
}

inline void sdpa_stats_record_native_gate_call() {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  std::lock_guard<std::mutex> lock(state.mtx);
  state.native_gate_calls++;
}

inline void sdpa_stats_record_native_reject(
    const char* reason,
    uint32_t q_len,
    uint32_t k_len) {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  const std::string reason_str = reason ? reason : "unknown";
  const std::string bucket = sdpa_bucket_key(q_len, k_len);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.native_gate_rejects++;
  state.native_reject_reason_counts[reason_str]++;
  state.native_reject_reason_bucket_counts[reason_str + " | " + bucket]++;
}

inline void sdpa_stats_record_native_hit(uint32_t q_len, uint32_t k_len) {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  const std::string bucket = sdpa_bucket_key(q_len, k_len);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.native_hits++;
  state.native_hit_bucket_counts[bucket]++;
}

inline void sdpa_stats_record_dispatch_result(bool success) {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  std::lock_guard<std::mutex> lock(state.mtx);
  if (success) {
    state.native_dispatch_success++;
  } else {
    state.native_dispatch_fail++;
  }
}

inline void sdpa_stats_record_final_fallback() {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  std::lock_guard<std::mutex> lock(state.mtx);
  state.final_fallbacks++;
}

inline void sdpa_stats_record_splitk_reduce_dispatch(
    const char* kernel,
    uint32_t split_k,
    uint32_t q_len,
    uint32_t k_len) {
  if (!sdpa_stats_enabled() || split_k <= 1u) {
    return;
  }
  auto& state = sdpa_stats_state();
  (void)sdpa_stats_reporter();
  const std::string kernel_name =
      (kernel && kernel[0] != '\0') ? kernel : "unknown";
  const std::string kernel_key =
      std::string("stage=") + sdpa_stage_name(q_len) + " kernel=" + kernel_name;
  const std::string splitk_bucket_key =
      std::string("stage=") + sdpa_stage_name(q_len) + " split_k=" +
      sdpa_splitk_bucket(split_k) + " k=" + sdpa_len_bucket(k_len);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.splitk_reduce_kernel_counts[kernel_key]++;
  state.splitk_parts_bucket_counts[splitk_bucket_key]++;
}

inline void sdpa_stats_dump() {
  if (!sdpa_stats_enabled()) {
    return;
  }
  auto& state = sdpa_stats_state();
  std::lock_guard<std::mutex> lock(state.mtx);
  if (state.use_fallback_calls == 0 && state.native_gate_calls == 0 &&
      state.final_fallbacks == 0 && state.native_dispatch_success == 0 &&
      state.native_dispatch_fail == 0) {
    return;
  }

  std::cerr << "[VulkanSDPAStats] use_fallback_calls=" << state.use_fallback_calls
            << " use_fallback_rejects=" << state.use_fallback_rejects
            << " native_gate_calls=" << state.native_gate_calls
            << " native_gate_rejects=" << state.native_gate_rejects
            << " native_hits=" << state.native_hits
            << " dispatch_success=" << state.native_dispatch_success
            << " dispatch_fail=" << state.native_dispatch_fail
            << " final_fallbacks=" << state.final_fallbacks
            << " use_reason_keys=" << state.use_fallback_reason_counts.size()
            << " native_reject_reason_keys="
            << state.native_reject_reason_counts.size()
            << " hit_bucket_keys=" << state.native_hit_bucket_counts.size()
            << " splitk_reduce_kernel_keys="
            << state.splitk_reduce_kernel_counts.size()
            << " splitk_parts_bucket_keys="
            << state.splitk_parts_bucket_counts.size()
            << "\n";

  for (const auto& [key, count] :
       sort_stats_counts(state.use_fallback_reason_counts)) {
    std::cerr << "[VulkanSDPAStats][UseFallbackReason] reason=" << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.native_reject_reason_counts)) {
    std::cerr << "[VulkanSDPAStats][NativeRejectReason] reason=" << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.native_hit_bucket_counts)) {
    std::cerr << "[VulkanSDPAStats][NativeHitBucket] " << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.use_fallback_reason_bucket_counts)) {
    std::cerr << "[VulkanSDPAStats][UseFallbackReasonBucket] " << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.native_reject_reason_bucket_counts)) {
    std::cerr << "[VulkanSDPAStats][NativeRejectReasonBucket] " << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.splitk_reduce_kernel_counts)) {
    std::cerr << "[VulkanSDPAStats][SplitKReduceKernel] " << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.splitk_parts_bucket_counts)) {
    std::cerr << "[VulkanSDPAStats][SplitKPartsBucket] " << key
              << " count=" << count << "\n";
  }
}

struct SdpaStatsReporter {
  ~SdpaStatsReporter() {
    sdpa_stats_dump();
  }
};

inline SdpaStatsReporter& sdpa_stats_reporter() {
  static SdpaStatsReporter reporter;
  return reporter;
}

inline const char* qmm_rows_bucket(uint32_t rows) {
  if (rows <= 1u) {
    return "1";
  }
  if (rows <= 2u) {
    return "2";
  }
  if (rows <= 4u) {
    return "3-4";
  }
  if (rows <= 8u) {
    return "5-8";
  }
  if (rows <= 16u) {
    return "9-16";
  }
  return "17+";
}

struct QmmStatsState {
  std::mutex mtx;
  uint64_t native_dispatch_success{0};
  uint64_t native_dispatch_fail{0};
  uint64_t final_fallbacks{0};
  std::unordered_map<std::string, uint64_t> native_kernel_counts;
  std::unordered_map<std::string, uint64_t> native_rows_bucket_counts;
  std::unordered_map<std::string, uint64_t> fallback_rows_bucket_counts;
};

inline QmmStatsState& qmm_stats_state() {
  static QmmStatsState state;
  return state;
}

struct QmmStatsReporter;
inline QmmStatsReporter& qmm_stats_reporter();

inline void qmm_stats_record_native_dispatch_success(
    uint32_t rows,
    const char* kernel) {
  if (!qmm_stats_enabled()) {
    return;
  }
  auto& state = qmm_stats_state();
  (void)qmm_stats_reporter();
  const std::string kernel_name =
      (kernel && kernel[0] != '\0') ? kernel : "unknown";
  const std::string rows_key =
      std::string("rows=") + qmm_rows_bucket(rows);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.native_dispatch_success++;
  state.native_kernel_counts[kernel_name]++;
  state.native_rows_bucket_counts[rows_key]++;
}

inline void qmm_stats_record_native_dispatch_fail(
    uint32_t rows,
    const char* kernel) {
  if (!qmm_stats_enabled()) {
    return;
  }
  auto& state = qmm_stats_state();
  (void)qmm_stats_reporter();
  const std::string kernel_name =
      (kernel && kernel[0] != '\0') ? kernel : "unknown";
  const std::string rows_key =
      std::string("rows=") + qmm_rows_bucket(rows);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.native_dispatch_fail++;
  state.native_kernel_counts[std::string("fail:") + kernel_name]++;
  state.fallback_rows_bucket_counts[rows_key]++;
}

inline void qmm_stats_record_final_fallback(uint32_t rows) {
  if (!qmm_stats_enabled()) {
    return;
  }
  auto& state = qmm_stats_state();
  (void)qmm_stats_reporter();
  const std::string rows_key =
      std::string("rows=") + qmm_rows_bucket(rows);
  std::lock_guard<std::mutex> lock(state.mtx);
  state.final_fallbacks++;
  state.fallback_rows_bucket_counts[rows_key]++;
}

inline void qmm_stats_dump() {
  if (!qmm_stats_enabled()) {
    return;
  }
  auto& state = qmm_stats_state();
  std::lock_guard<std::mutex> lock(state.mtx);
  if (state.native_dispatch_success == 0 && state.native_dispatch_fail == 0 &&
      state.final_fallbacks == 0) {
    return;
  }

  std::cerr << "[VulkanQMMStats] native_dispatch_success="
            << state.native_dispatch_success
            << " native_dispatch_fail=" << state.native_dispatch_fail
            << " final_fallbacks=" << state.final_fallbacks
            << " native_kernel_keys=" << state.native_kernel_counts.size()
            << " native_rows_bucket_keys="
            << state.native_rows_bucket_counts.size()
            << " fallback_rows_bucket_keys="
            << state.fallback_rows_bucket_counts.size()
            << "\n";

  for (const auto& [key, count] :
       sort_stats_counts(state.native_kernel_counts)) {
    std::cerr << "[VulkanQMMStats][NativeKernel] kernel=" << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.native_rows_bucket_counts)) {
    std::cerr << "[VulkanQMMStats][NativeRowsBucket] " << key
              << " count=" << count << "\n";
  }
  for (const auto& [key, count] :
       sort_stats_counts(state.fallback_rows_bucket_counts)) {
    std::cerr << "[VulkanQMMStats][FallbackRowsBucket] " << key
              << " count=" << count << "\n";
  }
}

struct QmmStatsReporter {
  ~QmmStatsReporter() {
    qmm_stats_dump();
  }
};

inline QmmStatsReporter& qmm_stats_reporter() {
  static QmmStatsReporter reporter;
  return reporter;
}

inline bool native_qmm_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE");
  return enabled;
}

inline bool native_qmm_m1_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE_M1");
  return enabled;
}

inline bool native_qmm_m1_reduce_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE");
  return enabled;
}

inline bool native_qmm_m16_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE_M16");
  return enabled;
}

inline bool native_qmm_m2_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE_M2");
  return enabled;
}

inline bool native_qmm_m4_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE_M4");
  return enabled;
}

inline bool native_qmm_m8_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_QMM_NATIVE_M8");
  return enabled;
}

inline bool native_rmsnorm_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_RMSNORM_NATIVE");
  return enabled;
}

inline bool native_rope_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_ROPE_NATIVE");
  return enabled;
}

inline bool native_sdpa_enabled() {
  static const bool enabled = env_flag_default_true("MLX_VK_ENABLE_SDPA_NATIVE");
  return enabled;
}

inline bool native_logsumexp_f32_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_LOGSUMEXP_F32");
  return enabled;
}

inline bool native_logsumexp_bf16_row1_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_LOGSUMEXP_BF16_ROW1");
  return enabled;
}

inline bool native_argreduce_argmax_lastdim_enabled() {
  static const bool enabled =
      env_flag_default_true("MLX_VK_ENABLE_ARGREDUCE_ARGMAX_LASTDIM");
  return enabled;
}

inline bool argreduce_debug_reject_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_DEBUG_ARGREDUCE_REJECT");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline bool env_has_nonempty_value(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] != '\0';
}

inline uint32_t parse_env_u32(const char* name, uint32_t default_value);

inline uint32_t native_sdpa_max_k_len_global() {
  static const uint32_t max_k_len = []() -> uint32_t {
    // Default widened after K-cap A/B (13/14/16): hit-rate improves on early
    // decode steps while throughput remains neutral in 10/40-token workloads.
    constexpr uint32_t kDefault = 16u;
    const char* v = std::getenv("MLX_VK_SDPA_MAX_K_LEN");
    if (!v || v[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v || *end != '\0' || parsed == 0ul) {
      return kDefault;
    }
    if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
      return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(parsed);
  }();
  return max_k_len;
}

inline uint32_t native_sdpa_max_q_len() {
  static const uint32_t max_q_len = []() -> uint32_t {
    // Keep Q cap aligned with current default K cap so causal prefill up to
    // the native K window can hit Vulkan SDPA without extra env toggles.
    constexpr uint32_t kDefault = 16u;
    const char* v = std::getenv("MLX_VK_SDPA_MAX_Q_LEN");
    if (!v || v[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v || *end != '\0' || parsed == 0ul) {
      return kDefault;
    }
    if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
      return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(parsed);
  }();
  return max_q_len;
}

inline uint32_t native_sdpa_max_q_len_decode() {
  static const uint32_t value = parse_env_u32("MLX_VK_SDPA_MAX_Q_LEN_DECODE", 1u);
  return value;
}

inline uint32_t native_sdpa_max_q_len_prefill() {
  static const uint32_t value = parse_env_u32(
      "MLX_VK_SDPA_MAX_Q_LEN_PREFILL", native_sdpa_max_q_len());
  return value;
}

inline uint32_t native_sdpa_max_q_len_for_q_len(uint32_t q_len) {
  return q_len <= 1u ? native_sdpa_max_q_len_decode()
                     : native_sdpa_max_q_len_prefill();
}

inline uint32_t parse_env_u32(const char* name, uint32_t default_value) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0') {
    return default_value;
  }
  if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(parsed);
}

inline bool sdpa_len_within_cap_or_unlimited(uint32_t len, uint32_t cap) {
  // cap==0 is treated as unlimited for staged rollouts.
  return cap == 0u || len <= cap;
}

inline uint32_t native_sdpa_max_k_len_decode() {
  static const uint32_t value = []() -> uint32_t {
    // Decode path uses split-k and specialized kernels; keep default unbounded
    // and let dispatch heuristics decide partitioning. Use *_DECODE env to cap.
    constexpr uint32_t kDecodeDefault = 0u; // 0 => unlimited
    if (env_has_nonempty_value("MLX_VK_SDPA_MAX_K_LEN")) {
      return parse_env_u32(
          "MLX_VK_SDPA_MAX_K_LEN_DECODE", native_sdpa_max_k_len_global());
    }
    return parse_env_u32("MLX_VK_SDPA_MAX_K_LEN_DECODE", kDecodeDefault);
  }();
  return value;
}

inline uint32_t native_sdpa_max_k_len_prefill() {
  static const uint32_t value = parse_env_u32(
      "MLX_VK_SDPA_MAX_K_LEN_PREFILL", native_sdpa_max_k_len_global());
  return value;
}

inline uint32_t native_sdpa_max_k_len_for_q_len(uint32_t q_len) {
  return q_len <= 1u ? native_sdpa_max_k_len_decode()
                     : native_sdpa_max_k_len_prefill();
}

inline uint32_t native_sdpa_splitk_min_k_len() {
  static const uint32_t value =
      std::max<uint32_t>(2u, parse_env_u32("MLX_VK_SDPA_SPLITK_MIN_K_LEN", 16u));
  return value;
}

inline uint32_t native_sdpa_splitk_target_chunk() {
  static const uint32_t value = std::max<uint32_t>(
      1u, parse_env_u32("MLX_VK_SDPA_SPLITK_TARGET_CHUNK", 16u));
  return value;
}

inline uint32_t native_sdpa_splitk_max_parts() {
  static const uint32_t value =
      std::max<uint32_t>(1u, parse_env_u32("MLX_VK_SDPA_SPLITK_MAX_PARTS", 8u));
  return value;
}

inline uint32_t native_sdpa_splitk_forced_parts() {
  static const uint32_t value = parse_env_u32("MLX_VK_SDPA_SPLIT_K", 0u);
  return value;
}

inline bool sdpa_splitk_debug_enabled() {
  static const bool enabled = env_flag_default_false("MLX_VK_DEBUG_SDPA_SPLITK");
  return enabled;
}

inline uint32_t native_sdpa_splitk_min_k_len_decode() {
  static const uint32_t value = std::max<uint32_t>(
      2u,
      parse_env_u32(
          "MLX_VK_SDPA_SPLITK_MIN_K_LEN_DECODE",
          std::max<uint32_t>(24u, native_sdpa_splitk_min_k_len())));
  return value;
}

inline uint32_t native_sdpa_splitk_target_chunk_decode() {
  static const uint32_t value = std::max<uint32_t>(
      1u,
      parse_env_u32(
          "MLX_VK_SDPA_SPLITK_TARGET_CHUNK_DECODE",
          std::max<uint32_t>(32u, native_sdpa_splitk_target_chunk())));
  return value;
}

inline uint32_t native_sdpa_splitk_max_parts_decode() {
  static const uint32_t value = std::max<uint32_t>(
      1u,
      parse_env_u32(
          "MLX_VK_SDPA_SPLITK_MAX_PARTS_DECODE",
          std::max<uint32_t>(16u, native_sdpa_splitk_max_parts())));
  return value;
}

inline uint32_t native_sdpa_splitk_min_k_len_prefill() {
  static const uint32_t value = std::max<uint32_t>(
      2u,
      parse_env_u32(
          "MLX_VK_SDPA_SPLITK_MIN_K_LEN_PREFILL",
          std::max<uint32_t>(32u, native_sdpa_splitk_min_k_len())));
  return value;
}

inline uint32_t native_sdpa_splitk_target_chunk_prefill() {
  static const uint32_t value = std::max<uint32_t>(
      1u,
      parse_env_u32(
          "MLX_VK_SDPA_SPLITK_TARGET_CHUNK_PREFILL",
          std::max<uint32_t>(32u, native_sdpa_splitk_target_chunk())));
  return value;
}

inline uint32_t native_sdpa_splitk_max_parts_prefill() {
  static const uint32_t value = std::max<uint32_t>(
      1u,
      parse_env_u32(
          "MLX_VK_SDPA_SPLITK_MAX_PARTS_PREFILL",
          std::min<uint32_t>(4u, native_sdpa_splitk_max_parts())));
  return value;
}

inline uint32_t native_sdpa_splitk_target_workgroups_decode() {
  static const uint32_t value = std::max<uint32_t>(
      1u, parse_env_u32("MLX_VK_SDPA_SPLITK_TARGET_WG_DECODE", 128u));
  return value;
}

inline uint32_t native_sdpa_splitk_target_workgroups_prefill() {
  static const uint32_t value = std::max<uint32_t>(
      1u, parse_env_u32("MLX_VK_SDPA_SPLITK_TARGET_WG_PREFILL", 128u));
  return value;
}

struct SdpaSplitKConfig {
  uint32_t min_k_len;
  uint32_t target_chunk;
  uint32_t target_workgroups;
  uint32_t max_parts;
  const char* stage;
};

inline SdpaSplitKConfig select_sdpa_splitk_config(uint32_t q_len) {
  if (q_len <= 1u) {
    return {
        native_sdpa_splitk_min_k_len_decode(),
        native_sdpa_splitk_target_chunk_decode(),
        native_sdpa_splitk_target_workgroups_decode(),
        native_sdpa_splitk_max_parts_decode(),
        "decode"};
  }
  return {
      native_sdpa_splitk_min_k_len_prefill(),
      native_sdpa_splitk_target_chunk_prefill(),
      native_sdpa_splitk_target_workgroups_prefill(),
      native_sdpa_splitk_max_parts_prefill(),
      "prefill"};
}

inline uint32_t select_sdpa_split_k(uint32_t k_len, uint32_t q_len, uint32_t n_rows) {
  const uint32_t forced = native_sdpa_splitk_forced_parts();
  const auto cfg = select_sdpa_splitk_config(q_len);
  uint32_t split_k = 1u;
  if (forced > 0u) {
    split_k = std::min<uint32_t>(k_len, std::max<uint32_t>(1u, forced));
    if (sdpa_splitk_debug_enabled()) {
      std::cerr << "[VulkanSDPASplitK] "
                << "stage=" << cfg.stage
                << " q_len=" << q_len
                << " k_len=" << k_len
                << " n_rows=" << n_rows
                << " forced=" << forced
                << " split_k=" << split_k
                << "\n";
    }
    return split_k;
  }
  if (k_len < cfg.min_k_len) {
    if (sdpa_splitk_debug_enabled()) {
      std::cerr << "[VulkanSDPASplitK] "
                << "stage=" << cfg.stage
                << " q_len=" << q_len
                << " k_len=" << k_len
                << " n_rows=" << n_rows
                << " min_k_len=" << cfg.min_k_len
                << " split_k=1"
                << "\n";
    }
    return 1u;
  }
  uint32_t requested_split_k =
      std::max<uint32_t>(1u, (k_len + cfg.target_chunk - 1u) / cfg.target_chunk);
  if (n_rows > 0u) {
    const uint32_t requested_by_parallel = std::max<uint32_t>(
        1u, (cfg.target_workgroups + n_rows - 1u) / n_rows);
    const uint32_t boosted_limit =
        requested_split_k > (std::numeric_limits<uint32_t>::max() / 2u)
        ? std::numeric_limits<uint32_t>::max()
        : requested_split_k * 2u;
    requested_split_k = std::max<uint32_t>(
        requested_split_k, std::min<uint32_t>(requested_by_parallel, boosted_limit));
  }
  if (requested_split_k <= 1u) {
    if (sdpa_splitk_debug_enabled()) {
      std::cerr << "[VulkanSDPASplitK] "
                << "stage=" << cfg.stage
                << " q_len=" << q_len
                << " k_len=" << k_len
                << " n_rows=" << n_rows
                << " min_k_len=" << cfg.min_k_len
                << " target_chunk=" << cfg.target_chunk
                << " target_workgroups=" << cfg.target_workgroups
                << " max_parts=" << cfg.max_parts
                << " split_k=1"
                << "\n";
    }
    return 1u;
  }
  split_k = requested_split_k;
  split_k = std::min<uint32_t>(split_k, cfg.max_parts);
  split_k = std::min<uint32_t>(split_k, k_len);
  split_k = std::max<uint32_t>(1u, split_k);
  if (sdpa_splitk_debug_enabled()) {
    std::cerr << "[VulkanSDPASplitK] "
              << "stage=" << cfg.stage
              << " q_len=" << q_len
              << " k_len=" << k_len
              << " n_rows=" << n_rows
              << " min_k_len=" << cfg.min_k_len
              << " target_chunk=" << cfg.target_chunk
              << " target_workgroups=" << cfg.target_workgroups
              << " max_parts=" << cfg.max_parts
              << " split_k=" << split_k
              << "\n";
  }
  return split_k;
}

inline bool native_rope_hs_transposed_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_VK_ENABLE_ROPE_HS_TRANSPOSED");
    if (!v) {
      return false;
    }
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "on") == 0;
  }();
  return enabled;
}

inline void log_rope_reject(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    int dims,
    bool traditional,
    float base,
    const char* reason) {
  if (!rope_debug_reject_enabled()) {
    return;
  }

  auto shape_string = [](const mlx::core::array& a) {
    std::string s = "[";
    for (int i = 0; i < a.ndim(); ++i) {
      if (i > 0) {
        s += ",";
      }
      s += std::to_string(a.shape(i));
    }
    s += "]";
    return s;
  };
  auto strides_string = [](const mlx::core::array& a) {
    std::string s = "[";
    const auto& st = a.strides();
    for (size_t i = 0; i < st.size(); ++i) {
      if (i > 0) {
        s += ",";
      }
      s += std::to_string(st[i]);
    }
    s += "]";
    return s;
  };

  std::cerr << "[VulkanRoPEReject] reason=" << (reason ? reason : "unknown")
            << " dims=" << dims
            << " traditional=" << (traditional ? 1 : 0)
            << " base=" << base;
  if (!inputs.empty()) {
    std::cerr << " in.dtype=" << inputs[0].dtype()
              << " in.shape=" << shape_string(inputs[0]);
    std::cerr << " in.strides=" << strides_string(inputs[0]);
    std::cerr << " in.row=" << (inputs[0].flags().row_contiguous ? 1 : 0);
  }
  if (inputs.size() > 1) {
    std::cerr << " offset.dtype=" << inputs[1].dtype()
              << " offset.shape=" << shape_string(inputs[1]);
  }
  if (inputs.size() > 2) {
    std::cerr << " freqs.dtype=" << inputs[2].dtype()
              << " freqs.shape=" << shape_string(inputs[2]);
  }
  if (!outputs.empty()) {
    std::cerr << " out.dtype=" << outputs[0].dtype()
              << " out.shape=" << shape_string(outputs[0]);
    std::cerr << " out.strides=" << strides_string(outputs[0]);
    std::cerr << " out.row=" << (outputs[0].flags().row_contiguous ? 1 : 0);
  }
  std::cerr << "\n";
}

inline void log_sdpa_reject(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const char* reason,
    bool has_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp) {
  if (!sdpa_debug_reject_enabled()) {
    return;
  }

  auto shape_string = [](const mlx::core::array& a) {
    std::string s = "[";
    for (int i = 0; i < a.ndim(); ++i) {
      if (i > 0) {
        s += ",";
      }
      s += std::to_string(a.shape(i));
    }
    s += "]";
    return s;
  };
  auto strides_string = [](const mlx::core::array& a) {
    std::string s = "[";
    const auto& st = a.strides();
    for (size_t i = 0; i < st.size(); ++i) {
      if (i > 0) {
        s += ",";
      }
      s += std::to_string(st[i]);
    }
    s += "]";
    return s;
  };

  std::cerr << "[VulkanSDPAReject] reason=" << (reason ? reason : "unknown")
            << " has_mask=" << (has_mask ? 1 : 0)
            << " do_causal=" << (do_causal ? 1 : 0)
            << " training=" << (is_training ? 1 : 0)
            << " logsumexp=" << (output_logsumexp ? 1 : 0)
            << " q.dtype=" << q.dtype() << " q.shape=" << shape_string(q)
            << " q.strides=" << strides_string(q)
            << " q.row=" << (q.flags().row_contiguous ? 1 : 0)
            << " k.dtype=" << k.dtype() << " k.shape=" << shape_string(k)
            << " k.strides=" << strides_string(k)
            << " k.row=" << (k.flags().row_contiguous ? 1 : 0)
            << " v.dtype=" << v.dtype() << " v.shape=" << shape_string(v)
            << " v.strides=" << strides_string(v)
            << " v.row=" << (v.flags().row_contiguous ? 1 : 0)
            << "\n";
}

inline kp::Tensor::TensorDataTypes to_kompute_dtype(mlx::core::Dtype dtype) {
  switch (dtype) {
    case mlx::core::bool_:
      return kp::Tensor::TensorDataTypes::eBool;
    case mlx::core::uint8:
    case mlx::core::uint16:
    case mlx::core::uint32:
    case mlx::core::uint64:
      return kp::Tensor::TensorDataTypes::eUnsignedInt;
    case mlx::core::int8:
    case mlx::core::int16:
    case mlx::core::int32:
    case mlx::core::int64:
      return kp::Tensor::TensorDataTypes::eInt;
    case mlx::core::float16:
    case mlx::core::float32:
    case mlx::core::bfloat16:
    case mlx::core::complex64:
      return kp::Tensor::TensorDataTypes::eFloat;
    case mlx::core::float64:
      return kp::Tensor::TensorDataTypes::eDouble;
  }
  return kp::Tensor::TensorDataTypes::eFloat;
}

struct CachedQmmConstTensorEntry {
  std::weak_ptr<kp::Tensor> tensor;
  const void* data_ptr{nullptr};
  size_t nbytes{0};
  mlx::core::Dtype dtype{mlx::core::float32};
  bool uploaded{false};
};

struct QmmConstTensorRef {
  std::uintptr_t key{0};
  std::shared_ptr<kp::Tensor> tensor;
  bool needs_sync{true};
  bool cacheable{false};
};

struct SdpaSplitKTempTensors {
  std::shared_ptr<kp::Tensor> partial_o_tensor;
  std::shared_ptr<kp::Tensor> partial_m_tensor;
  std::shared_ptr<kp::Tensor> partial_l_tensor;
  size_t partial_o_capacity{0};
  size_t partial_rows_capacity{0};
};

struct SdpaSplitKTempTensorRef {
  std::shared_ptr<kp::Tensor> partial_o_tensor;
  std::shared_ptr<kp::Tensor> partial_m_tensor;
  std::shared_ptr<kp::Tensor> partial_l_tensor;
};

std::mutex& qmm_const_tensor_cache_mutex() {
  static std::mutex mtx;
  return mtx;
}

std::unordered_map<std::uintptr_t, CachedQmmConstTensorEntry>&
qmm_const_tensor_cache() {
  static std::unordered_map<std::uintptr_t, CachedQmmConstTensorEntry> cache;
  return cache;
}

std::mutex& sdpa_splitk_temp_cache_mutex() {
  static std::mutex mtx;
  return mtx;
}

std::unordered_map<std::uint64_t, SdpaSplitKTempTensors>&
sdpa_splitk_temp_tensor_cache() {
  static std::unordered_map<std::uint64_t, SdpaSplitKTempTensors> cache;
  return cache;
}

inline std::uint64_t sdpa_splitk_cache_key(const mlx::core::Stream& stream) {
  const std::uint64_t device_type =
      static_cast<std::uint64_t>(stream.device.type);
  const std::uint64_t device_index =
      static_cast<std::uint64_t>(std::max(stream.device.index, 0));
  const std::uint64_t stream_index =
      static_cast<std::uint64_t>(std::max(stream.index, 0));
  return (device_type << 60u) ^ (device_index << 32u) ^ stream_index;
}

inline SdpaSplitKTempTensorRef get_sdpa_splitk_temp_tensors(
    mlx::core::vulkan::Device& device,
    const mlx::core::Stream& stream,
    size_t partial_rows,
    size_t partial_o_elems) {
  auto manager = device.kompute_manager();
  if (!manager) {
    throw std::runtime_error("[Vulkan SDPA] Missing Kompute manager.");
  }

  std::lock_guard<std::mutex> lock(sdpa_splitk_temp_cache_mutex());
  auto& cache = sdpa_splitk_temp_tensor_cache();
  auto& slot = cache[sdpa_splitk_cache_key(stream)];

  if (!slot.partial_o_tensor || slot.partial_o_capacity < partial_o_elems) {
    slot.partial_o_tensor = manager->tensor(
        std::vector<float>(partial_o_elems, 0.0f));
    slot.partial_o_capacity = partial_o_elems;
  }
  if (!slot.partial_m_tensor || !slot.partial_l_tensor ||
      slot.partial_rows_capacity < partial_rows) {
    slot.partial_m_tensor = manager->tensor(std::vector<float>(partial_rows, 0.0f));
    slot.partial_l_tensor = manager->tensor(std::vector<float>(partial_rows, 0.0f));
    slot.partial_rows_capacity = partial_rows;
  }

  return {
      slot.partial_o_tensor,
      slot.partial_m_tensor,
      slot.partial_l_tensor};
}

inline QmmConstTensorRef get_qmm_const_tensor(
    const mlx::core::array& arr,
    mlx::core::vulkan::Device& device) {
  // Cache only leaf arrays (model weights/scales/biases). Dynamic graph
  // temporaries may mutate and should remain uncached.
  if (arr.has_primitive()) {
    return {0, device.get_tensor(arr), true, false};
  }

  auto manager = device.kompute_manager();
  if (!manager) {
    return {0, device.get_tensor(arr), true, false};
  }

  const auto key = arr.id();
  const void* ptr = arr.data<void>();
  const size_t nbytes = arr.nbytes();
  const auto dtype = arr.dtype();

  std::lock_guard<std::mutex> lock(qmm_const_tensor_cache_mutex());
  auto& cache = qmm_const_tensor_cache();
  auto it = cache.find(key);
  if (it != cache.end()) {
    const bool same_meta =
        it->second.data_ptr == ptr && it->second.nbytes == nbytes &&
        it->second.dtype == dtype;
    if (same_meta) {
      if (auto tensor = it->second.tensor.lock()) {
        return {key, tensor, !it->second.uploaded, true};
      }
    }
    cache.erase(it);
  }

  auto tensor = manager->tensor(
      const_cast<void*>(ptr),
      static_cast<uint32_t>(arr.size()),
      static_cast<uint32_t>(arr.itemsize()),
      to_kompute_dtype(dtype));
  cache[key] = CachedQmmConstTensorEntry{
      tensor, ptr, nbytes, dtype, false};
  return {key, tensor, true, true};
}

inline void mark_qmm_const_tensor_uploaded(std::uintptr_t key) {
  std::lock_guard<std::mutex> lock(qmm_const_tensor_cache_mutex());
  auto& cache = qmm_const_tensor_cache();
  auto it = cache.find(key);
  if (it != cache.end()) {
    it->second.uploaded = true;
  }
}

inline uint32_t encode_push_constant_u32(uint32_t value) {
  return value;
}

inline uint32_t encode_push_constant_f32(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&bits, &value, sizeof(uint32_t));
  return bits;
}

inline bool can_use_native_affine_bf16_quantized_matmul(
    const std::vector<mlx::core::array>& inputs,
    const mlx::core::array& out,
    int group_size,
    int bits,
    bool transpose,
    mlx::core::QuantizationMode mode) {
  if (inputs.size() != 4 || out.size() == 0) {
    return false;
  }
  if (mode != mlx::core::QuantizationMode::Affine || bits != 4 ||
      group_size != 128 || !transpose) {
    return false;
  }
  if (out.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      (out.size() % 2) != 0) {
    return false;
  }

  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& biases = inputs[3];

  if (x.dtype() != mlx::core::bfloat16 || w.dtype() != mlx::core::uint32 ||
      scales.dtype() != mlx::core::bfloat16 ||
      biases.dtype() != mlx::core::bfloat16 ||
      out.dtype() != mlx::core::bfloat16) {
    return false;
  }

  if (!is_row_contiguous_materialized(x) ||
      !is_row_contiguous_materialized(w) ||
      !is_row_contiguous_materialized(scales) ||
      !is_row_contiguous_materialized(biases) ||
      !out.flags().row_contiguous) {
    return false;
  }

  if (x.ndim() < 1 || w.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2) {
    return false;
  }

  int k = x.shape(-1);
  int n = out.shape(-1);
  if (k <= 0 || n <= 0 || (k % group_size) != 0) {
    return false;
  }

  int groups_per_col = k / group_size;
  if (groups_per_col <= 0) {
    return false;
  }
  if (w.shape(-2) != n || scales.shape(-2) != n || biases.shape(-2) != n) {
    return false;
  }
  if (scales.shape(-1) != groups_per_col || biases.shape(-1) != groups_per_col) {
    return false;
  }

  constexpr int values_per_u32 = 8; // bits=4
  if (w.shape(-1) * values_per_u32 != k) {
    return false;
  }

  int rows = static_cast<int>(x.size() / static_cast<size_t>(k));
  if (rows <= 0 || static_cast<size_t>(rows) * static_cast<size_t>(n) != out.size()) {
    return false;
  }

  // Shader reads bf16 via packed uint words, so require even element counts.
  if ((x.size() % 2) != 0 || (scales.size() % 2) != 0 || (biases.size() % 2) != 0) {
    return false;
  }

  return true;
}

inline bool can_use_native_rmsnorm_bf16(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    uint32_t& n_rows,
    uint32_t& axis_size,
    uint32_t& w_stride) {
  if (inputs.size() != 2 || outputs.size() != 1 || outputs[0].size() == 0) {
    return false;
  }
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& out = outputs[0];

  if (x.dtype() != mlx::core::bfloat16 || w.dtype() != mlx::core::bfloat16 ||
      out.dtype() != mlx::core::bfloat16) {
    return false;
  }
  if (!is_row_contiguous_materialized(x) || !out.flags().row_contiguous ||
      x.shape() != out.shape()) {
    return false;
  }
  if (x.ndim() < 1) {
    return false;
  }

  int64_t axis = x.shape(-1);
  if (axis <= 0 || (axis % 2) != 0 ||
      axis > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  if (x.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      x.size() != out.size() || (x.size() % static_cast<size_t>(axis)) != 0) {
    return false;
  }

  if (w.ndim() == 0) {
    if (!is_row_contiguous_materialized(w) || w.size() != 1) {
      return false;
    }
    w_stride = 0u;
  } else if (w.ndim() == 1) {
    if (!is_row_contiguous_materialized(w) || w.shape(0) != axis ||
        w.strides()[0] != 1) {
      return false;
    }
    w_stride = 1u;
  } else {
    return false;
  }

  axis_size = static_cast<uint32_t>(axis);
  n_rows = static_cast<uint32_t>(x.size() / static_cast<size_t>(axis));
  return n_rows > 0;
}

inline bool read_scalar_offset_i32(const mlx::core::array& offset, int32_t& out) {
  if (offset.size() != 1) {
    return false;
  }
  switch (offset.dtype()) {
    case mlx::core::int32:
      out = offset.data<int32_t>()[0];
      return true;
    case mlx::core::int64: {
      auto v = offset.data<int64_t>()[0];
      if (v < std::numeric_limits<int32_t>::min() ||
          v > std::numeric_limits<int32_t>::max()) {
        return false;
      }
      out = static_cast<int32_t>(v);
      return true;
    }
    default:
      return false;
  }
}

inline bool can_use_native_rope_bf16(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    int dims,
    bool traditional,
    float base,
    bool& with_freqs,
    uint32_t& n_rows,
    uint32_t& half_dims,
    uint32_t& row_stride,
    uint32_t& t_size,
    uint32_t& rows_per_batch,
    uint32_t& offset_is_vector,
    uint32_t& input_hs_transposed,
    uint32_t& input_batch_stride,
    uint32_t& input_head_stride,
    uint32_t& input_t_stride,
    uint32_t& n_heads,
    const char** reject_reason = nullptr) {
  auto reject = [&](const char* reason) {
    if (reject_reason) {
      *reject_reason = reason;
    }
    return false;
  };

  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      outputs[0].size() == 0) {
    return reject("inputs_or_outputs_shape");
  }
  with_freqs = inputs.size() == 3;
  if (!with_freqs && base <= 0.0f) {
    return reject("invalid_base");
  }

  const auto& in = inputs[0];
  const auto& offset = inputs[1];
  const auto& out = outputs[0];
  if (in.dtype() != mlx::core::bfloat16 || out.dtype() != mlx::core::bfloat16 ||
      in.shape() != out.shape()) {
    return reject("dtype_or_shape_mismatch");
  }
  const bool in_row_contiguous = is_row_contiguous_materialized(in);
  const bool allow_hs_transposed = native_rope_hs_transposed_enabled();
  const bool in_hs_transposed =
      allow_hs_transposed && !in_row_contiguous && in.data_size() == in.size() &&
      is_rope_head_seq_transposed_layout(in);
  if ((!in_row_contiguous && !in_hs_transposed) || !out.flags().row_contiguous ||
      in.ndim() < 2) {
    return reject("in_or_out_layout");
  }

  int64_t d = in.shape(-1);
  int64_t t = in.shape(-2);
  if (t <= 0 || d <= 0 || dims <= 0 || dims != d || (dims % 2) != 0) {
    return reject("dims_constraints");
  }
  if (in.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      t > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return reject("size_overflow");
  }

  const int64_t batch = in.shape(0);
  if (batch <= 0 || batch > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return reject("batch_constraints");
  }
  const uint32_t batch_u32 = static_cast<uint32_t>(batch);

  row_stride = static_cast<uint32_t>(d);
  half_dims = static_cast<uint32_t>(dims / 2);
  n_rows = static_cast<uint32_t>(in.size() / static_cast<size_t>(d));
  if (n_rows == 0) {
    return reject("zero_rows");
  }
  if (batch_u32 == 0 || (n_rows % batch_u32) != 0) {
    return reject("rows_per_batch_divisibility");
  }
  rows_per_batch = n_rows / batch_u32;
  if (rows_per_batch == 0) {
    return reject("zero_rows_per_batch");
  }

  input_hs_transposed = in_hs_transposed ? 1u : 0u;
  input_batch_stride = 0u;
  input_head_stride = 0u;
  input_t_stride = 0u;
  n_heads = 1u;
  if (input_hs_transposed != 0u) {
    const auto& st = in.strides();
    if (in.shape(1) <= 0 ||
        in.shape(1) > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[0] <= 0 || st[1] <= 0 || st[2] <= 0 ||
        st[0] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[1] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[2] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return reject("hs_transpose_strides_or_heads");
    }
    n_heads = static_cast<uint32_t>(in.shape(1));
    input_batch_stride = static_cast<uint32_t>(st[0]);
    input_head_stride = static_cast<uint32_t>(st[1]);
    input_t_stride = static_cast<uint32_t>(st[2]);
    if (rows_per_batch != n_heads * static_cast<uint32_t>(t)) {
      return reject("hs_transpose_rows_shape_mismatch");
    }
  }

  if (with_freqs) {
    const auto& freqs = inputs[2];
    if (freqs.dtype() != mlx::core::float32 || freqs.ndim() != 1 ||
        freqs.shape(0) != static_cast<int64_t>(half_dims) ||
        !is_row_contiguous_materialized(freqs) || freqs.strides()[0] != 1) {
      return reject("freqs_layout_or_dtype");
    }
  }

  if (offset.size() == 1) {
    int32_t offset_value = 0;
    if (!read_scalar_offset_i32(offset, offset_value) ||
        !is_row_contiguous_materialized(offset)) {
      return reject("scalar_offset_constraints");
    }
    (void)offset_value;
    offset_is_vector = 0u;
  } else {
    if (offset.dtype() != mlx::core::int32 || offset.ndim() != 1 ||
        offset.shape(0) != batch || !is_row_contiguous_materialized(offset) ||
        offset.strides()[0] != 1) {
      return reject("vector_offset_constraints");
    }
    offset_is_vector = 1u;
  }

  t_size = static_cast<uint32_t>(t);
  if (reject_reason) {
    *reject_reason = nullptr;
  }
  return n_rows > 0;
}

inline bool can_use_native_sdpa_bf16_decode_q1(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs,
    bool do_causal,
    bool has_sinks,
    bool output_logsumexp,
    uint32_t& batch_size,
    uint32_t& n_q_heads,
    uint32_t& n_kv_heads,
    uint32_t& q_len,
    uint32_t& k_len,
    uint32_t& qk_dim,
    uint32_t& v_dim,
    uint32_t& k_head_stride,
    uint32_t& k_seq_stride,
    uint32_t& v_head_stride,
    uint32_t& v_seq_stride,
    uint32_t& mask_mode,
    uint32_t& mask_batch_stride,
    uint32_t& mask_head_stride,
    uint32_t& mask_q_stride,
    uint32_t& mask_k_stride,
    const char** reject_reason = nullptr) {
  const auto [q_len_hint, k_len_hint] = sdpa_qk_len_hint(inputs);
  sdpa_stats_record_native_gate_call();

  auto reject = [&](const char* reason) {
    sdpa_stats_record_native_reject(reason, q_len_hint, k_len_hint);
    if (reject_reason) {
      *reject_reason = reason;
    }
    return false;
  };
  auto reject_layout = [&](const char* reason, const mlx::core::array& arr) {
    if (sdpa_debug_reject_enabled()) {
      std::cerr << "[VulkanSDPALayoutReject] reason=" << reason
                << " shape=" << shape_string(arr)
                << " strides=" << strides_string(arr)
                << " row=" << (arr.flags().row_contiguous ? 1 : 0)
                << " data_size=" << arr.data_size()
                << " size=" << arr.size() << "\n";
    }
    return reject(reason);
  };

  if (has_sinks || output_logsumexp ||
      (inputs.size() != 3 && inputs.size() != 4) ||
      outputs.size() != 1 || outputs[0].size() == 0) {
    return reject("shape_or_aux");
  }

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const bool has_arr_mask = inputs.size() == 4;
  const auto* mask = has_arr_mask ? &inputs[3] : nullptr;
  const auto& out = outputs[0];
  if (q.dtype() != mlx::core::bfloat16 || k.dtype() != mlx::core::bfloat16 ||
      v.dtype() != mlx::core::bfloat16 || out.dtype() != mlx::core::bfloat16) {
    return reject("dtype");
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || out.ndim() != 4) {
    return reject("ndim");
  }
  if (!is_dense_row_major_view(q)) {
    return reject_layout("q_layout", q);
  }
  if (!is_dense_row_major_view(out)) {
    return reject_layout("out_layout", out);
  }

  if (q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(3) != k.shape(3) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2)) {
    return reject("shape_contract");
  }

  const int64_t b = q.shape(0);
  const int64_t hq = q.shape(1);
  const int64_t hkv = k.shape(1);
  const int64_t lq = q.shape(2);
  const int64_t lk = k.shape(2);
  const int64_t dq = q.shape(3);
  const int64_t dv = v.shape(3);
  if (b <= 0 || hq <= 0 || hkv <= 0 || lq <= 0 || lk <= 0 || dq <= 0 ||
      dv <= 0 ||
      (hq % hkv) != 0) {
    return reject("dim_nonpositive_or_head_repeat");
  }
  if (lq > static_cast<int64_t>(native_sdpa_max_q_len_for_q_len(
                   static_cast<uint32_t>(lq)))) {
    return reject("q_len_cap");
  }
  if (!sdpa_len_within_cap_or_unlimited(
          static_cast<uint32_t>(lk),
          native_sdpa_max_k_len_for_q_len(static_cast<uint32_t>(lq)))) {
    return reject("k_len_cap");
  }
  if (do_causal && lq > lk) {
    return reject("causal_qk_len");
  }
  if (dq > 256 || dv > 256) {
    return reject("dim_bounds");
  }
  if ((q.size() % 2) != 0 || (k.size() % 2) != 0 || (v.size() % 2) != 0 ||
      (out.size() % 2) != 0) {
    return reject("bf16_pack");
  }
  if (b > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      hq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      hkv > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      lq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      lk > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      dq > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      dv > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return reject("u32_overflow");
  }

  if (out.ndim() != 4 || out.shape(0) != b || out.shape(1) != hq ||
      out.shape(2) != lq || out.shape(3) != dv) {
    return reject("output_shape");
  }

  auto parse_kv_layout = [&](
                             const mlx::core::array& arr,
                             int64_t expected_heads,
                             int64_t expected_len,
                             int64_t expected_dim,
                             uint32_t& head_stride_out,
                             uint32_t& seq_stride_out,
                             const char* layout_reason) {
    if (arr.ndim() != 4 || arr.shape(1) != expected_heads ||
        arr.shape(2) != expected_len || arr.shape(3) != expected_dim) {
      return reject(layout_reason);
    }

    const auto& st = arr.strides();
    if (st[0] <= 0 || st[1] <= 0 || st[2] <= 0 || st[3] != 1) {
      return reject_layout(layout_reason, arr);
    }
    if (arr.shape(0) > 1 && arr.shape(1) > 1 &&
        st[0] != st[1] * arr.shape(1)) {
      return reject_layout(layout_reason, arr);
    }

    const int64_t head_stride_i64 = (arr.shape(1) == 1) ? st[0] : st[1];
    const int64_t seq_stride_i64 = st[2];
    if (head_stride_i64 <= 0 || seq_stride_i64 <= 0 ||
        head_stride_i64 >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        seq_stride_i64 >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return reject_layout(layout_reason, arr);
    }

    const uint64_t max_elem_u64 =
        ((static_cast<uint64_t>(b) - 1u) * static_cast<uint64_t>(expected_heads) +
         (static_cast<uint64_t>(expected_heads) - 1u)) *
            static_cast<uint64_t>(head_stride_i64) +
        (static_cast<uint64_t>(expected_len) - 1u) *
            static_cast<uint64_t>(seq_stride_i64) +
        (static_cast<uint64_t>(expected_dim) - 1u);
    if (arr.data_size() == 0 ||
        max_elem_u64 >= static_cast<uint64_t>(arr.data_size())) {
      return reject_layout(layout_reason, arr);
    }

    head_stride_out = static_cast<uint32_t>(head_stride_i64);
    seq_stride_out = static_cast<uint32_t>(seq_stride_i64);
    return true;
  };

  if (!parse_kv_layout(
          k, hkv, lk, dq, k_head_stride, k_seq_stride, "k_layout")) {
    return false;
  }
  if (!parse_kv_layout(
          v, hkv, lk, dv, v_head_stride, v_seq_stride, "v_layout")) {
    return false;
  }

  if (has_arr_mask) {
    if (mask->dtype() != mlx::core::bfloat16 &&
        mask->dtype() != mlx::core::uint32) {
      return reject("mask_dtype");
    }
    if (mask->ndim() != 4 || mask->shape(0) != b || mask->shape(1) != hq ||
        mask->shape(2) != lq || mask->shape(3) != lk) {
      return reject("mask_shape");
    }
    const auto& st = mask->strides();
    if (st[0] < 0 || st[1] < 0 || st[2] < 0 || st[3] < 0) {
      return reject_layout("mask_layout", *mask);
    }
    if (st[0] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[1] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[2] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        st[3] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return reject_layout("mask_layout", *mask);
    }
    const uint64_t max_elem_u64 =
        (static_cast<uint64_t>(b) - 1u) * static_cast<uint64_t>(st[0]) +
        (static_cast<uint64_t>(hq) - 1u) * static_cast<uint64_t>(st[1]) +
        (static_cast<uint64_t>(lq) - 1u) * static_cast<uint64_t>(st[2]) +
        (static_cast<uint64_t>(lk) - 1u) * static_cast<uint64_t>(st[3]);
    if (mask->data_size() == 0 ||
        max_elem_u64 >= static_cast<uint64_t>(mask->data_size())) {
      return reject_layout("mask_layout", *mask);
    }
    mask_mode = (mask->dtype() == mlx::core::bfloat16) ? 1u : 2u;
    mask_batch_stride = static_cast<uint32_t>(st[0]);
    mask_head_stride = static_cast<uint32_t>(st[1]);
    mask_q_stride = static_cast<uint32_t>(st[2]);
    mask_k_stride = static_cast<uint32_t>(st[3]);
  } else {
    mask_mode = 0u;
    mask_batch_stride = 0u;
    mask_head_stride = 0u;
    mask_q_stride = 0u;
    mask_k_stride = 0u;
  }

  batch_size = static_cast<uint32_t>(b);
  n_q_heads = static_cast<uint32_t>(hq);
  n_kv_heads = static_cast<uint32_t>(hkv);
  q_len = static_cast<uint32_t>(lq);
  k_len = static_cast<uint32_t>(lk);
  qk_dim = static_cast<uint32_t>(dq);
  v_dim = static_cast<uint32_t>(dv);
  sdpa_stats_record_native_hit(q_len, k_len);
  if (sdpa_debug_hit_enabled()) {
    std::cerr << "[VulkanSDPAHit] "
              << "q_len=" << q_len << " k_len=" << k_len
              << " qk_dim=" << qk_dim
              << " v_dim=" << v_dim
              << " q.shape=" << shape_string(q)
              << " q.strides=" << strides_string(q)
              << " q.data_size=" << q.data_size()
              << " q.size=" << q.size()
              << " k.shape=" << shape_string(k)
              << " k.strides=" << strides_string(k)
              << " k.data_size=" << k.data_size()
              << " k.size=" << k.size()
              << " v.shape=" << shape_string(v)
              << " v.strides=" << strides_string(v)
              << " v.data_size=" << v.data_size()
              << " v.size=" << v.size()
              << " mask_mode=" << mask_mode;
    if (has_arr_mask) {
      std::cerr << " mask.shape=" << shape_string(*mask)
                << " mask.strides=" << strides_string(*mask)
                << " mask.data_size=" << mask->data_size()
                << " mask.size=" << mask->size();
    }
    std::cerr
              << "\n";
  }
  if (reject_reason) {
    *reject_reason = nullptr;
  }
  return true;
}

inline void materialize_and_share_fast_outputs(
    const std::vector<mlx::core::array>& fallback_outputs,
    std::vector<mlx::core::array>& outputs) {
  if (fallback_outputs.size() != outputs.size()) {
    throw std::runtime_error("[Vulkan fast] Fallback output arity mismatch.");
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& src = fallback_outputs[i];
    auto& mutable_src = const_cast<mlx::core::array&>(src);
    if (mutable_src.status() == mlx::core::array::Status::unscheduled) {
      mutable_src.eval();
    } else {
      mutable_src.wait();
    }
    outputs[i].copy_shared_buffer(src);
  }
}

inline void collect_keepalive_buffers(
    const mlx::core::array& arr,
    std::unordered_set<std::shared_ptr<mlx::core::array::Data>>& buffers) {
  if (auto data = arr.data_shared_ptr()) {
    buffers.insert(std::move(data));
  }
  for (const auto& sib : arr.siblings()) {
    if (auto sib_data = sib.data_shared_ptr()) {
      buffers.insert(std::move(sib_data));
    }
  }
}

template <typename OutputCollector>
inline void finalize_cpu_fallback(
    const std::vector<mlx::core::array>& inputs,
    OutputCollector&& collect_outputs,
    mlx::core::vulkan::OpProfileScope* profile) {
  auto cpu_stream = mlx::core::default_stream(mlx::core::Device::cpu);
  auto& encoder = mlx::core::cpu::get_command_encoder(cpu_stream);

  std::unordered_set<std::shared_ptr<mlx::core::array::Data>> buffers;
  for (const auto& in : inputs) {
    collect_keepalive_buffers(in, buffers);
  }
  collect_outputs(buffers);

  // Mirror cpu::eval() keepalive semantics for fallback-dispatched CPU tasks.
  encoder.dispatch(
      [buffers = std::move(buffers),
       temps = std::move(encoder.temporaries())]() mutable {});
  mlx::core::synchronize(cpu_stream);
  if (profile) {
    profile->mark_sync();
  }
}

template <typename EvalFn>
inline void run_cpu_fallback_single(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    EvalFn&& eval_fn,
    mlx::core::vulkan::OpProfileScope* profile) {
  auto stream = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  sync_inputs_to_host_if_needed(inputs);
  std::forward<EvalFn>(eval_fn)();
  mlx::core::vulkan::device(mlx::core::Device::gpu).invalidate_tensor(out);
  finalize_cpu_fallback(
      inputs,
      [&](auto& buffers) { collect_keepalive_buffers(out, buffers); },
      profile);
}

template <typename EvalFn>
inline void run_cpu_fallback_multi(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs,
    EvalFn&& eval_fn,
    mlx::core::vulkan::OpProfileScope* profile) {
  auto stream = outputs.empty() ? mlx::core::default_stream(mlx::core::default_device())
                                : outputs.front().primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);
  sync_inputs_to_host_if_needed(inputs);
  std::forward<EvalFn>(eval_fn)();
  auto& device = mlx::core::vulkan::device(mlx::core::Device::gpu);
  for (const auto& out : outputs) {
    device.invalidate_tensor(out);
  }
  finalize_cpu_fallback(
      inputs,
      [&](auto& buffers) {
        for (const auto& out : outputs) {
          collect_keepalive_buffers(out, buffers);
        }
      },
      profile);
}

} // namespace

#define VULKAN_CPU_FALLBACK_MULTI(func)                               \
  void func::eval_gpu(                                                \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    vulkan::OpProfileScope profile(#func);                            \
    profile.mark_fallback();                                          \
    run_cpu_fallback_multi(                                           \
        inputs, outputs, [&]() { eval_cpu(inputs, outputs); }, &profile); \
  }

#define VULKAN_CPU_FALLBACK(func)                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    vulkan::OpProfileScope profile(#func);                            \
    profile.mark_fallback();                                          \
    run_cpu_fallback_single(                                          \
        inputs, out, [&]() { eval_cpu(inputs, out); }, &profile);     \
  }

#define VULKAN_NO_GPU_MULTI(func)                                     \
  void func::eval_gpu(                                                \
      const std::vector<array>&, std::vector<array>&) {              \
    throw std::runtime_error(#func " has no Vulkan GPU implementation."); \
  }

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("QuantizedMatmul");
  auto stream = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  uint32_t rows_hint = 0u;
  if (out.ndim() > 0) {
    const int64_t n_hint = out.shape(-1);
    if (n_hint > 0) {
      rows_hint = static_cast<uint32_t>(
          out.size() / static_cast<size_t>(n_hint));
    }
  }
  const char* qmm_kernel_for_stats = "none";

  if (native_qmm_enabled() && can_use_native_affine_bf16_quantized_matmul(
          inputs, out, group_size_, bits_, transpose_, mode_)) {
    try {
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto x_tensor = device.get_tensor(inputs[0]);
      auto w_cached = get_qmm_const_tensor(inputs[1], device);
      auto scales_cached = get_qmm_const_tensor(inputs[2], device);
      auto biases_cached = get_qmm_const_tensor(inputs[3], device);
      auto out_tensor = device.get_tensor(out);

      const uint32_t out_elems = static_cast<uint32_t>(out.size());
      const uint32_t n = static_cast<uint32_t>(out.shape(-1));
      const uint32_t k = static_cast<uint32_t>(inputs[0].shape(-1));
      const uint32_t rows = static_cast<uint32_t>(
          inputs[0].size() / static_cast<size_t>(k));
      const uint32_t groups_per_col =
          static_cast<uint32_t>(k / static_cast<uint32_t>(group_size_));
      const uint32_t w_words_per_col =
          static_cast<uint32_t>(inputs[1].shape(-1));
      const uint32_t out_words = (out_elems + 1u) / 2u;
      const bool use_m1_kernel = native_qmm_m1_enabled() && (rows == 1u);
      const bool use_m1_reduce_kernel =
          native_qmm_m1_reduce_enabled() && (rows == 1u);
      const bool use_m16_kernel =
          native_qmm_m16_enabled() && (rows > 8u) && (rows <= 16u);
      const bool use_m2_kernel = native_qmm_m2_enabled() && (rows == 2u);
      const bool use_m4_kernel = native_qmm_m4_enabled() && (rows == 4u);
      const bool use_m8_kernel = native_qmm_m8_enabled() && (rows == 8u);
      const uint32_t wg_size_x = use_m1_kernel ? 128u : 64u;
      const uint32_t groups_x =
          use_m1_reduce_kernel
          ? std::max<uint32_t>(1, out_words)
          : std::max<uint32_t>(1, (out_words + wg_size_x - 1u) / wg_size_x);
      const char* qmm_kernel =
          use_m1_reduce_kernel
          ? vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128_M1_REDUCE
          : (use_m1_kernel
          ? vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128_M1
          : (use_m16_kernel
                 ? vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128_M16
                 : (use_m2_kernel
                 ? vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128_M2
                 : (use_m4_kernel
                        ? vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128_M4
                        : (use_m8_kernel
                               ? vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128_M8
                               : vulkan::KernelRegistry::QMM_AFFINE_BF16_T4_G128)))));
      qmm_kernel_for_stats = qmm_kernel;

      const std::vector<uint32_t> push_consts{
          encode_push_constant_u32(out_elems),
          encode_push_constant_u32(n),
          encode_push_constant_u32(k),
          encode_push_constant_u32(groups_per_col),
          encode_push_constant_u32(w_words_per_col)};

      // Output tensor is fully overwritten by the kernel; no need to upload it.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
      if (device.tensor_needs_sync_device(inputs[0])) {
        sync_tensors.push_back(x_tensor);
      }
      if (w_cached.needs_sync) {
        sync_tensors.push_back(w_cached.tensor);
      }
      if (scales_cached.needs_sync) {
        sync_tensors.push_back(scales_cached.tensor);
      }
      if (biases_cached.needs_sync) {
        sync_tensors.push_back(biases_cached.tensor);
      }
      if (!sync_tensors.empty()) {
        encoder.record_tensor_sync_device(sync_tensors);
      }
      if (w_cached.cacheable && w_cached.needs_sync) {
        mark_qmm_const_tensor_uploaded(w_cached.key);
      }
      if (scales_cached.cacheable && scales_cached.needs_sync) {
        mark_qmm_const_tensor_uploaded(scales_cached.key);
      }
      if (biases_cached.cacheable && biases_cached.needs_sync) {
        mark_qmm_const_tensor_uploaded(biases_cached.key);
      }
      encoder.record_algo_dispatch(
          qmm_kernel,
          {x_tensor,
           w_cached.tensor,
           scales_cached.tensor,
           biases_cached.tensor,
           out_tensor},
          {groups_x, 1, 1},
          push_consts);
      device.mark_tensor_host_dirty(out, stream.index);
      qmm_stats_record_native_dispatch_success(rows, qmm_kernel);
      return;
    } catch (const std::exception&) {
      qmm_stats_record_native_dispatch_fail(rows_hint, qmm_kernel_for_stats);
      // Fall through to CPU fallback.
    }
  }

  profile.mark_fallback();
  qmm_stats_record_final_fallback(rows_hint);
  run_cpu_fallback_single(inputs, out, [&]() { eval_cpu(inputs, out); }, &profile);
}

VULKAN_CPU_FALLBACK(Abs)
VULKAN_CPU_FALLBACK(AddMM)
VULKAN_CPU_FALLBACK(Arange)
VULKAN_CPU_FALLBACK(ArcCos)
VULKAN_CPU_FALLBACK(ArcCosh)
VULKAN_CPU_FALLBACK(ArcSin)
VULKAN_CPU_FALLBACK(ArcSinh)
VULKAN_CPU_FALLBACK(ArcTan)
VULKAN_CPU_FALLBACK(ArcTan2)
VULKAN_CPU_FALLBACK(ArcTanh)
VULKAN_CPU_FALLBACK(ArgPartition)
VULKAN_CPU_FALLBACK(ArgSort)
VULKAN_CPU_FALLBACK(BitwiseBinary)
VULKAN_CPU_FALLBACK(BitwiseInvert)
VULKAN_CPU_FALLBACK(BlockMaskedMM)
VULKAN_CPU_FALLBACK(Ceil)
VULKAN_CPU_FALLBACK(Cholesky)
VULKAN_CPU_FALLBACK(Conjugate)
VULKAN_CPU_FALLBACK(Convolution)
// VULKAN_CPU_FALLBACK(Cos)  // Now has native Vulkan implementation
VULKAN_CPU_FALLBACK(Cosh)
VULKAN_CPU_FALLBACK(Equal)
VULKAN_CPU_FALLBACK(Erf)
VULKAN_CPU_FALLBACK(ErfInv)
VULKAN_CPU_FALLBACK(Exp)
VULKAN_CPU_FALLBACK(Expm1)
VULKAN_CPU_FALLBACK(FFT)
VULKAN_CPU_FALLBACK(Floor)
VULKAN_CPU_FALLBACK(Gather)
VULKAN_CPU_FALLBACK(GatherAxis)
VULKAN_CPU_FALLBACK(GatherMM)
VULKAN_CPU_FALLBACK(GatherQMM)
VULKAN_CPU_FALLBACK(Greater)
VULKAN_CPU_FALLBACK(GreaterEqual)
VULKAN_CPU_FALLBACK(Hadamard)
VULKAN_CPU_FALLBACK(Imag)
VULKAN_CPU_FALLBACK(Inverse)
VULKAN_CPU_FALLBACK(Less)
VULKAN_CPU_FALLBACK(LessEqual)
VULKAN_CPU_FALLBACK(Load)
VULKAN_CPU_FALLBACK(Log)
VULKAN_CPU_FALLBACK(Log1p)
VULKAN_CPU_FALLBACK(LogicalNot)
VULKAN_CPU_FALLBACK(LogicalAnd)
VULKAN_CPU_FALLBACK(LogicalOr)
VULKAN_CPU_FALLBACK(LogAddExp)

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("ArgReduce");

  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }

  auto stream = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  const char* reject_reason = nullptr;
  auto reject = [&](const char* reason) {
    reject_reason = reason;
    return false;
  };

  uint32_t n_rows = 0u;
  uint32_t axis_size = 0u;
  bool native_success = false;

  if (inputs.size() == 1) {
    const auto& in = inputs[0];
    bool common_ok = true;

    if (!native_argreduce_argmax_lastdim_enabled()) {
      common_ok = reject("gate_disabled");
    } else if (reduce_type_ != ArgReduce::ArgMax) {
      common_ok = reject("reduce_type");
    } else if (in.ndim() < 1) {
      common_ok = reject("ndim");
    } else if (axis_ < 0 || axis_ != (in.ndim() - 1)) {
      common_ok = reject("axis_lastdim");
    } else if (!is_row_contiguous_materialized(in)) {
      common_ok = reject("in_layout");
    } else if (!out.flags().row_contiguous) {
      common_ok = reject("out_layout");
    } else if (out.dtype() != mlx::core::uint32) {
      common_ok = reject("out_dtype");
    } else if (in.strides()[in.ndim() - 1] != 1) {
      common_ok = reject("last_stride");
    }

    if (common_ok) {
      const int64_t axis_size_i64 = in.shape(in.ndim() - 1);
      if (axis_size_i64 <= 0 ||
          axis_size_i64 >
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        common_ok = reject("axis_size");
      } else {
        axis_size = static_cast<uint32_t>(axis_size_i64);
      }
    }

    if (common_ok) {
      const size_t in_size = in.size();
      if ((in_size % axis_size) != 0) {
        common_ok = reject("size_mod_axis");
      } else {
        const size_t rows_size_t = in_size / axis_size;
        if (rows_size_t == 0 || rows_size_t != out.size() ||
            rows_size_t >
                static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
          common_ok = reject("rows_shape");
        } else {
          n_rows = static_cast<uint32_t>(rows_size_t);
        }
      }
    }

    if (common_ok) {
      const char* kernel_name = nullptr;
      if (in.dtype() == mlx::core::float32) {
        kernel_name = vulkan::KernelRegistry::ARGMAX_F32_LASTDIM;
      } else if (in.dtype() == mlx::core::bfloat16) {
        kernel_name = vulkan::KernelRegistry::ARGMAX_BF16_LASTDIM;
      } else {
        reject("in_dtype");
      }

      if (kernel_name) {
        try {
          if (!out.data_shared_ptr()) {
            out.set_data(allocator::malloc(out.nbytes()));
          }

          auto& device = vulkan::device(stream.device);
          auto& encoder = device.get_command_encoder(stream.index);
          encoder.begin_encoding();

          auto in_tensor = device.get_tensor(in);
          auto out_tensor = device.get_tensor(out);
          if (device.tensor_needs_sync_device(in)) {
            encoder.record_tensor_sync_device({in_tensor});
          }

          const std::vector<uint32_t> push_consts{
              encode_push_constant_u32(n_rows),
              encode_push_constant_u32(axis_size)};
          encoder.record_algo_dispatch(
              kernel_name,
              {in_tensor, out_tensor},
              {n_rows, 1, 1},
              push_consts);
          device.mark_tensor_host_dirty(out, stream.index);
          native_success = true;
          return;
        } catch (const std::exception&) {
          reject("dispatch_error");
        }
      }
    }
  } else {
    reject("arity");
  }

  if (!native_success && argreduce_debug_reject_enabled() && inputs.size() == 1) {
    const auto& in = inputs[0];
    std::cerr << "[VulkanArgReduceReject] reason="
              << (reject_reason ? reject_reason : "unknown")
              << " reduce_type=" << (reduce_type_ == ArgReduce::ArgMax ? "argmax" : "argmin")
              << " axis=" << axis_
              << " in_dtype=" << in.dtype()
              << " out_dtype=" << out.dtype()
              << " in_shape=" << shape_string(in)
              << " in_strides=" << strides_string(in)
              << " in_row=" << (in.flags().row_contiguous ? 1 : 0)
              << " in_data_size=" << in.data_size()
              << " in_size=" << in.size()
              << " out_shape=" << shape_string(out)
              << " out_strides=" << strides_string(out)
              << " out_row=" << (out.flags().row_contiguous ? 1 : 0)
              << " out_data_size=" << out.data_size()
              << " out_size=" << out.size() << "\n";
  }

  profile.mark_fallback();
  run_cpu_fallback_single(inputs, out, [&]() { eval_cpu(inputs, out); }, &profile);
}

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::OpProfileScope profile("LogSumExp");

  if (!vulkan::is_available()) {
    throw std::runtime_error("Vulkan not available");
  }

  auto stream = out.primitive().stream();
  prepare_inputs_for_cpu_fallback(inputs, stream);

  const char* reject_reason = nullptr;
  auto reject = [&](const char* reason) {
    reject_reason = reason;
    return false;
  };

  uint32_t n_rows = 0u;
  uint32_t axis_size = 0u;
  bool native_success = false;
  if (inputs.size() == 1) {
    const auto& in = inputs[0];
    bool common_ok = true;
    if (in.ndim() < 1) {
      common_ok = reject("ndim");
    } else if (!in.flags().row_contiguous || in.data_size() != in.size()) {
      common_ok = reject("in_layout");
    } else if (!out.flags().row_contiguous) {
      common_ok = reject("out_layout");
    } else if (in.strides()[in.ndim() - 1] != 1) {
      common_ok = reject("last_stride");
    }

    if (common_ok) {
      const int64_t axis_size_i64 = in.shape(in.ndim() - 1);
      if (axis_size_i64 <= 0 ||
          axis_size_i64 >
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        common_ok = reject("axis_size");
      } else {
        axis_size = static_cast<uint32_t>(axis_size_i64);
      }
    }

    if (common_ok) {
      const size_t in_size = in.size();
      if ((in_size % axis_size) != 0) {
        common_ok = reject("size_mod_axis");
      } else {
        const size_t rows_size_t = in_size / axis_size;
        if (rows_size_t == 0 || rows_size_t != out.size() ||
            rows_size_t >
                static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
          common_ok = reject("rows_shape");
        } else {
          n_rows = static_cast<uint32_t>(rows_size_t);
        }
      }
    }

    if (common_ok) {
      const char* kernel_name = nullptr;
      uint32_t dispatch_x = 0u;
      std::vector<uint32_t> push_consts;

      if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float32) {
        if (!native_logsumexp_f32_enabled()) {
          reject("gate_f32_disabled");
        } else {
          kernel_name = vulkan::KernelRegistry::LOGSUMEXP_F32;
          dispatch_x = n_rows;
          push_consts = {
              encode_push_constant_u32(n_rows),
              encode_push_constant_u32(axis_size)};
        }
      } else if (
          in.dtype() == mlx::core::bfloat16 &&
          out.dtype() == mlx::core::bfloat16) {
        if (!native_logsumexp_bf16_row1_enabled()) {
          reject("gate_bf16_row1_disabled");
        } else if (n_rows != 1u) {
          reject("bf16_row1_nrows");
        } else {
          kernel_name = vulkan::KernelRegistry::LOGSUMEXP_BF16_ROW1;
          dispatch_x = 1u;
          push_consts = {encode_push_constant_u32(axis_size)};
        }
      } else {
        reject("dtype");
      }

      if (kernel_name) {
        try {
          if (!out.data_shared_ptr()) {
            out.set_data(allocator::malloc(out.nbytes()));
          }

          auto& device = vulkan::device(stream.device);
          auto& encoder = device.get_command_encoder(stream.index);
          encoder.begin_encoding();

          auto in_tensor = device.get_tensor(in);
          auto out_tensor = device.get_tensor(out);
          if (device.tensor_needs_sync_device(in)) {
            encoder.record_tensor_sync_device({in_tensor});
          }

          encoder.record_algo_dispatch(
              kernel_name,
              {in_tensor, out_tensor},
              {dispatch_x, 1, 1},
              push_consts);
          device.mark_tensor_host_dirty(out, stream.index);
          native_success = true;
          return;
        } catch (const std::exception&) {
          reject("dispatch_error");
        }
      }
    }
  } else {
    reject("arity");
  }

  if (!native_success && logsumexp_debug_reject_enabled() && inputs.size() == 1) {
    const auto& in = inputs[0];
    std::cerr << "[VulkanLogSumExpReject] reason="
              << (reject_reason ? reject_reason : "unknown")
              << " in_dtype=" << in.dtype()
              << " out_dtype=" << out.dtype()
              << " in_shape=" << shape_string(in)
              << " in_strides=" << strides_string(in)
              << " in_row=" << (in.flags().row_contiguous ? 1 : 0)
              << " in_data_size=" << in.data_size()
              << " in_size=" << in.size()
              << " out_shape=" << shape_string(out)
              << " out_strides=" << strides_string(out)
              << " out_row=" << (out.flags().row_contiguous ? 1 : 0)
              << " out_data_size=" << out.data_size()
              << " out_size=" << out.size() << "\n";
  }

  profile.mark_fallback();
  run_cpu_fallback_single(inputs, out, [&]() { eval_cpu(inputs, out); }, &profile);
}

VULKAN_CPU_FALLBACK(MaskedScatter)
VULKAN_CPU_FALLBACK(Matmul)
VULKAN_CPU_FALLBACK(Maximum)
VULKAN_CPU_FALLBACK(Minimum)
VULKAN_CPU_FALLBACK(Negative)
VULKAN_CPU_FALLBACK(NotEqual)
VULKAN_CPU_FALLBACK(Partition)
VULKAN_CPU_FALLBACK(Power)
VULKAN_CPU_FALLBACK(QQMatmul)
VULKAN_CPU_FALLBACK(RandomBits)
VULKAN_CPU_FALLBACK(Real)
VULKAN_CPU_FALLBACK(Reduce)
VULKAN_CPU_FALLBACK(Remainder)
VULKAN_CPU_FALLBACK(Round)
VULKAN_CPU_FALLBACK(Scan)
VULKAN_CPU_FALLBACK(Scatter)
VULKAN_CPU_FALLBACK(ScatterAxis)
VULKAN_CPU_FALLBACK(SegmentedMM)
VULKAN_CPU_FALLBACK(Select)
VULKAN_CPU_FALLBACK(Sigmoid)
VULKAN_CPU_FALLBACK(Sign)
// VULKAN_CPU_FALLBACK(Sin)  // Native Vulkan implementation in unary.cpp
VULKAN_CPU_FALLBACK(Sinh)
VULKAN_CPU_FALLBACK(Softmax)
VULKAN_CPU_FALLBACK(Sort)
VULKAN_CPU_FALLBACK(Sqrt)
VULKAN_CPU_FALLBACK(Square)
VULKAN_CPU_FALLBACK(Tan)
VULKAN_CPU_FALLBACK(Tanh)

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();
  std::string detailed_name;
  const char* profile_name = "Compiled";
  if (compiled_profile_detail_enabled()) {
    detailed_name = std::string("Compiled::") + name();
    profile_name = detailed_name.c_str();
  }

  if (compiled_debug_detail_enabled()) {
    static std::mutex debug_mtx;
    static std::unordered_set<std::string> printed;
    std::lock_guard<std::mutex> lock(debug_mtx);
    const std::string debug_key = std::string(name()) + "|" + lib_name();
    if (printed.insert(debug_key).second) {
      std::cerr << "[VulkanCompiledDetail] name=" << name()
                << " lib=" << lib_name()
                << " inputs=" << inputs.size()
                << " outputs=" << outputs.size() << "\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& in = inputs[i];
        std::cerr << "  in[" << i << "] dtype=" << in.dtype()
                  << " shape=" << shape_string(in)
                  << " strides=" << strides_string(in)
                  << " row=" << (in.flags().row_contiguous ? 1 : 0)
                  << " size=" << in.size()
                  << " data_size=" << in.data_size() << "\n";
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& out = outputs[i];
        std::cerr << "  out[" << i << "] dtype=" << out.dtype()
                  << " shape=" << shape_string(out)
                  << " strides=" << strides_string(out)
                  << " row=" << (out.flags().row_contiguous ? 1 : 0)
                  << " size=" << out.size()
                  << " data_size=" << out.data_size() << "\n";
      }
    }
  }

  vulkan::OpProfileScope profile(profile_name);

  bool native_success = false;
  if (native_compiled_sigmoid_mul_mul_bf16_enabled() &&
      std::strcmp(name(), "CompiledSigmoidMultiplyMultiply") == 0) {
    const char* reject_reason = nullptr;
    auto reject = [&](const char* reason) {
      reject_reason = reason;
      return false;
    };

    bool common_ok = true;
    if (inputs.size() != 2 || outputs.size() != 1) {
      common_ok = reject("arity");
    } else if (
        inputs[0].dtype() != bfloat16 || inputs[1].dtype() != bfloat16 ||
        outputs[0].dtype() != bfloat16) {
      common_ok = reject("dtype");
    } else if (
        inputs[0].shape() != inputs[1].shape() ||
        inputs[0].shape() != outputs[0].shape()) {
      common_ok = reject("shape");
    } else if (
        !is_row_contiguous_materialized(inputs[0]) ||
        !is_row_contiguous_materialized(inputs[1]) ||
        !outputs[0].flags().row_contiguous) {
      common_ok = reject("layout");
    } else if (
        outputs[0].size() == 0 ||
        outputs[0].size() >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      common_ok = reject("size");
    }

    if (common_ok) {
      try {
        auto& out = outputs[0];
        if (!out.data_shared_ptr()) {
          out.set_data(allocator::malloc(out.nbytes()));
        }

        prepare_inputs_for_cpu_fallback(inputs, stream);

        auto& device = vulkan::device(stream.device);
        auto& encoder = device.get_command_encoder(stream.index);
        encoder.begin_encoding();

        auto x_tensor = device.get_tensor(inputs[0]);
        auto y_tensor = device.get_tensor(inputs[1]);
        auto out_tensor = device.get_tensor(out);

        std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
        if (device.tensor_needs_sync_device(inputs[0])) {
          sync_tensors.push_back(x_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[1])) {
          sync_tensors.push_back(y_tensor);
        }
        if (!sync_tensors.empty()) {
          encoder.record_tensor_sync_device(sync_tensors);
        }

        const uint32_t n = static_cast<uint32_t>(out.size());
        const uint32_t n_words = (n + 1u) / 2u;
        const uint32_t groups_x = std::max<uint32_t>(1, (n_words + 255u) / 256u);
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::SILU_MUL_BF16,
            {x_tensor, y_tensor, out_tensor},
            {groups_x, 1, 1},
            {encode_push_constant_u32(n)});

        device.mark_tensor_host_dirty(out, stream.index);
        native_success = true;
      } catch (const std::exception&) {
        reject("dispatch_error");
      }
    }

    if (!native_success && compiled_debug_detail_enabled()) {
      std::cerr << "[VulkanCompiledReject] name=" << name()
                << " reason=" << (reject_reason ? reject_reason : "unknown")
                << "\n";
    }
  }

  if (native_success) {
    return;
  }

  profile.mark_fallback();
  run_cpu_fallback_multi(
      inputs, outputs, [&]() { eval_cpu(inputs, outputs); }, &profile);
}

VULKAN_CPU_FALLBACK_MULTI(DivMod)
VULKAN_CPU_FALLBACK_MULTI(Eig)
VULKAN_CPU_FALLBACK_MULTI(Eigh)
VULKAN_CPU_FALLBACK_MULTI(LUF)
VULKAN_CPU_FALLBACK_MULTI(QRF)
VULKAN_CPU_FALLBACK_MULTI(SVD)

bool fast::LayerNorm::use_fallback(Stream) {
  return true;
}

bool fast::RMSNorm::use_fallback(Stream stream) {
  return stream.device == Device::cpu || detail::in_tracing();
}

bool fast::RoPE::use_fallback(Stream stream) {
  return stream.device == Device::cpu || detail::in_tracing();
}

bool fast::ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream stream) {
  const uint32_t q_len_hint = sdpa_qk_len_hint_from_array(q);
  const uint32_t k_len_hint = sdpa_qk_len_hint_from_array(k);
  sdpa_stats_record_use_fallback_call();

  auto reject = [&](const char* reason) {
    sdpa_stats_record_use_fallback_reject(reason, q_len_hint, k_len_hint);
    log_sdpa_reject(
        q, k, v, reason, has_mask, do_causal, is_training, output_logsumexp);
    return true;
  };

  if (!native_sdpa_enabled()) {
    return reject("native_disabled");
  }

  if (stream.device == Device::cpu || detail::in_tracing() || is_training ||
      output_logsumexp) {
    return reject("global_gate");
  }
  // Decode path supports:
  // - implicit causal (`mask="causal"`)
  // - explicit array mask (`mask_mode="array"`)
  // Any other mask mode remains fallback.
  if ((has_mask && !do_causal && !has_arr_mask) ||
      (do_causal && has_arr_mask)) {
    return reject("mask_mode");
  }

  if (q.dtype() != mlx::core::bfloat16 || k.dtype() != mlx::core::bfloat16 ||
      v.dtype() != mlx::core::bfloat16) {
    return reject("dtype");
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    return reject("ndim");
  }
  if (!is_dense_row_major_view(q) ||
      !is_dense_row_major_view(k) ||
      !is_dense_row_major_view(v)) {
    return reject("row_contiguous");
  }

  if (q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(3) != k.shape(3) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2)) {
    return reject("shape_contract");
  }
  if (q.shape(2) <= 0 || q.shape(2) > static_cast<int64_t>(native_sdpa_max_q_len())) {
    return reject("q_len_cap");
  }
  if (k.shape(1) <= 0 || (q.shape(1) % k.shape(1)) != 0) {
    return reject("head_repeat");
  }
  if (k.shape(2) <= 0) {
    return reject("k_len_nonpositive");
  }
  if (do_causal && q.shape(2) > k.shape(2)) {
    return reject("causal_qk_len");
  }
  if (!sdpa_len_within_cap_or_unlimited(
          static_cast<uint32_t>(k.shape(2)),
          native_sdpa_max_k_len_for_q_len(static_cast<uint32_t>(q.shape(2))))) {
    return reject("k_len_cap");
  }
  if (q.shape(3) <= 0 || q.shape(3) > 256 ||
      v.shape(3) <= 0 || v.shape(3) > 256) {
    return reject("dim_bounds");
  }
  if ((q.size() % 2) != 0 || (k.size() % 2) != 0 || (v.size() % 2) != 0) {
    return reject("bf16_pack");
  }

  return false;
}

bool fast::ScaledDotProductAttention::supports_bool_mask() {
  // Vulkan decode path supports bool masks via native mask_mode=2.
  return true;
}

bool fast::ScaledDotProductAttentionVJP::use_fallback(const array&, Stream) {
  return true;
}

VULKAN_NO_GPU_MULTI(fast::LayerNorm)
VULKAN_NO_GPU_MULTI(fast::LayerNormVJP)

void fast::RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RMSNorm");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();

  uint32_t n_rows = 0;
  uint32_t axis_size = 0;
  uint32_t w_stride = 0;
  if (native_rmsnorm_enabled() && can_use_native_rmsnorm_bf16(
          inputs, outputs, n_rows, axis_size, w_stride)) {
    try {
      auto& out = outputs[0];
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto x_tensor = device.get_tensor(inputs[0]);
      auto w_tensor = device.get_tensor(inputs[1]);
      auto out_tensor = device.get_tensor(out);

      const std::vector<uint32_t> push_consts{
          encode_push_constant_u32(n_rows),
          encode_push_constant_u32(axis_size),
          encode_push_constant_u32(w_stride),
          encode_push_constant_f32(eps_)};

      // Output tensor is write-only in this dispatch.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
      if (device.tensor_needs_sync_device(inputs[0])) {
        sync_tensors.push_back(x_tensor);
      }
      if (device.tensor_needs_sync_device(inputs[1])) {
        sync_tensors.push_back(w_tensor);
      }
      if (!sync_tensors.empty()) {
        encoder.record_tensor_sync_device(sync_tensors);
      }
      encoder.record_algo_dispatch(
          vulkan::KernelRegistry::RMSNORM_BF16,
          {x_tensor, w_tensor, out_tensor},
          {n_rows, 1, 1},
          push_consts);
      device.mark_tensor_host_dirty(out, stream.index);
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RMSNormVJP");
  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::RoPE");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();

  uint32_t n_rows = 0;
  uint32_t half_dims = 0;
  uint32_t row_stride = 0;
  uint32_t t_size = 0;
  uint32_t rows_per_batch = 0;
  uint32_t offset_is_vector = 0;
  uint32_t input_hs_transposed = 0;
  uint32_t input_batch_stride = 0;
  uint32_t input_head_stride = 0;
  uint32_t input_t_stride = 0;
  uint32_t n_heads = 0;
  bool with_freqs = false;
  const char* rope_reject_reason = nullptr;
  if (native_rope_enabled() && can_use_native_rope_bf16(
          inputs,
          outputs,
          dims_,
          traditional_,
          base_,
          with_freqs,
          n_rows,
          half_dims,
          row_stride,
          t_size,
          rows_per_batch,
          offset_is_vector,
          input_hs_transposed,
          input_batch_stride,
          input_head_stride,
          input_t_stride,
          n_heads,
          &rope_reject_reason)) {
    try {
      auto& out = outputs[0];
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto in_tensor = device.get_tensor(inputs[0]);
      auto out_tensor = device.get_tensor(out);
      auto offset_tensor = device.get_tensor(inputs[1]);

      const uint32_t forward_flag = forward_ ? 1u : 0u;
      const uint32_t traditional_flag = traditional_ ? 1u : 0u;
      if (with_freqs) {
        auto freqs_tensor = device.get_tensor(inputs[2]);
        const std::vector<uint32_t> push_consts{
            encode_push_constant_u32(n_rows),
            encode_push_constant_u32(half_dims),
            encode_push_constant_u32(row_stride),
            encode_push_constant_u32(t_size),
            encode_push_constant_u32(rows_per_batch),
            encode_push_constant_u32(offset_is_vector),
            encode_push_constant_u32(traditional_flag),
            encode_push_constant_u32(input_hs_transposed),
            encode_push_constant_u32(input_batch_stride),
            encode_push_constant_u32(input_head_stride),
            encode_push_constant_u32(input_t_stride),
            encode_push_constant_u32(n_heads),
            encode_push_constant_f32(scale_),
            encode_push_constant_u32(forward_flag)};

        // Output tensor is written by the kernel; skip redundant upload.
        std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
        if (device.tensor_needs_sync_device(inputs[0])) {
          sync_tensors.push_back(in_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[2])) {
          sync_tensors.push_back(freqs_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[1])) {
          sync_tensors.push_back(offset_tensor);
        }
        if (!sync_tensors.empty()) {
          encoder.record_tensor_sync_device(sync_tensors);
        }
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::ROPE_BF16_FREQS,
            {in_tensor, out_tensor, freqs_tensor, offset_tensor},
            {n_rows, 1, 1},
            push_consts);
      } else {
        const std::vector<uint32_t> push_consts{
            encode_push_constant_u32(n_rows),
            encode_push_constant_u32(half_dims),
            encode_push_constant_u32(row_stride),
            encode_push_constant_u32(t_size),
            encode_push_constant_u32(rows_per_batch),
            encode_push_constant_u32(offset_is_vector),
            encode_push_constant_u32(traditional_flag),
            encode_push_constant_u32(input_hs_transposed),
            encode_push_constant_u32(input_batch_stride),
            encode_push_constant_u32(input_head_stride),
            encode_push_constant_u32(input_t_stride),
            encode_push_constant_u32(n_heads),
            encode_push_constant_f32(scale_),
            encode_push_constant_f32(std::log2(base_)),
            encode_push_constant_u32(forward_flag)};

        // Output tensor is written by the kernel; skip redundant upload.
        std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
        if (device.tensor_needs_sync_device(inputs[0])) {
          sync_tensors.push_back(in_tensor);
        }
        if (device.tensor_needs_sync_device(inputs[1])) {
          sync_tensors.push_back(offset_tensor);
        }
        if (!sync_tensors.empty()) {
          encoder.record_tensor_sync_device(sync_tensors);
        }
        encoder.record_algo_dispatch(
            vulkan::KernelRegistry::ROPE_BF16_T1,
            {in_tensor, out_tensor, offset_tensor},
            {n_rows, 1, 1},
            push_consts);
      }
      device.mark_tensor_host_dirty(out, stream.index);
      return;
    } catch (const std::exception&) {
      // Fall through to fallback path.
    }
  }

  log_rope_reject(
      inputs, outputs, dims_, traditional_, base_, rope_reject_reason);
  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

void fast::ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::ScaledDotProductAttention");
  auto stream = outputs.empty() ? default_stream(default_device())
                                : outputs.front().primitive().stream();

  uint32_t batch_size = 0;
  uint32_t n_q_heads = 0;
  uint32_t n_kv_heads = 0;
  uint32_t q_len = 0;
  uint32_t k_len = 0;
  uint32_t qk_dim = 0;
  uint32_t v_dim = 0;
  uint32_t k_head_stride = 0;
  uint32_t k_seq_stride = 0;
  uint32_t v_head_stride = 0;
  uint32_t v_seq_stride = 0;
  uint32_t mask_mode = 0;
  uint32_t mask_batch_stride = 0;
  uint32_t mask_head_stride = 0;
  uint32_t mask_q_stride = 0;
  uint32_t mask_k_stride = 0;
  const char* native_reject_reason = nullptr;
  auto dispatch_native_sdpa = [&](
                                    SdpaNativePathKind path_kind,
                                    const std::vector<array>& native_inputs,
                                    uint32_t batch_size_local,
                                    uint32_t n_q_heads_local,
                                    uint32_t n_kv_heads_local,
                                    uint32_t q_len_local,
                                    uint32_t k_len_local,
                                    uint32_t qk_dim_local,
                                    uint32_t v_dim_local,
                                    uint32_t k_head_stride_local,
                                    uint32_t k_seq_stride_local,
                                    uint32_t v_head_stride_local,
                                    uint32_t v_seq_stride_local,
                                    uint32_t mask_mode_local,
                                    uint32_t mask_batch_stride_local,
                                    uint32_t mask_head_stride_local,
                                    uint32_t mask_q_stride_local,
                                    uint32_t mask_k_stride_local,
                                    uint32_t split_k_local) -> bool {
    try {
      auto& out = outputs[0];
      if (!out.data_shared_ptr()) {
        out.set_data(allocator::malloc(out.nbytes()));
      }

      auto& device = vulkan::device(stream.device);
      auto& encoder = device.get_command_encoder(stream.index);
      encoder.begin_encoding();

      auto q_tensor = device.get_tensor(native_inputs[0]);
      auto k_tensor = device.get_tensor(native_inputs[1]);
      auto v_tensor = device.get_tensor(native_inputs[2]);
      auto out_tensor = device.get_tensor(out);
      auto mask_tensor = q_tensor;
      const bool has_mask_tensor = native_inputs.size() > 3;
      if (has_mask_tensor) {
        mask_tensor = device.get_tensor(native_inputs[3]);
      }

      // Output tensor is write-only in this decode kernel.
      std::vector<std::shared_ptr<kp::Tensor>> sync_tensors;
      if (device.tensor_needs_sync_device(native_inputs[0])) {
        sync_tensors.push_back(q_tensor);
      }
      if (device.tensor_needs_sync_device(native_inputs[1])) {
        sync_tensors.push_back(k_tensor);
      }
      if (device.tensor_needs_sync_device(native_inputs[2])) {
        sync_tensors.push_back(v_tensor);
      }
      if (has_mask_tensor && device.tensor_needs_sync_device(native_inputs[3])) {
        sync_tensors.push_back(mask_tensor);
      }
      if (!sync_tensors.empty()) {
        encoder.record_tensor_sync_device(sync_tensors);
      }

      const uint64_t n_rows_u64 =
          static_cast<uint64_t>(batch_size_local) *
          static_cast<uint64_t>(n_q_heads_local) *
          static_cast<uint64_t>(q_len_local);
      if (n_rows_u64 == 0u ||
          n_rows_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("[Vulkan SDPA] row shape overflow.");
      }
      const uint32_t n_rows = static_cast<uint32_t>(n_rows_u64);
      if (split_k_local <= 1u) {
        const std::vector<uint32_t> push_consts{
            encode_push_constant_u32(batch_size_local),
            encode_push_constant_u32(n_q_heads_local),
            encode_push_constant_u32(n_kv_heads_local),
            encode_push_constant_u32(q_len_local),
            encode_push_constant_u32(k_len_local),
            encode_push_constant_u32(qk_dim_local),
            encode_push_constant_u32(v_dim_local),
            encode_push_constant_u32(k_head_stride_local),
            encode_push_constant_u32(k_seq_stride_local),
            encode_push_constant_u32(v_head_stride_local),
            encode_push_constant_u32(v_seq_stride_local),
            encode_push_constant_u32(mask_mode_local),
            encode_push_constant_u32(mask_batch_stride_local),
            encode_push_constant_u32(mask_head_stride_local),
            encode_push_constant_u32(mask_q_stride_local),
            encode_push_constant_u32(mask_k_stride_local),
            encode_push_constant_u32(do_causal_ ? 1u : 0u),
            encode_push_constant_f32(scale_)};

        encoder.record_algo_dispatch(
            sdpa_native_direct_kernel(
                path_kind,
                qk_dim_local,
                v_dim_local,
                k_len_local,
                mask_mode_local,
                split_k_local),
            {q_tensor, k_tensor, v_tensor, out_tensor, mask_tensor},
            {n_rows, 1, 1},
            push_consts);
      } else {
        const uint64_t partial_rows_u64 =
            n_rows_u64 * static_cast<uint64_t>(split_k_local);
        const uint64_t partial_o_elems_u64 =
            partial_rows_u64 * static_cast<uint64_t>(v_dim_local);
        if (partial_rows_u64 == 0u ||
            partial_rows_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            partial_rows_u64 > std::numeric_limits<size_t>::max() ||
            partial_o_elems_u64 == 0u ||
            partial_o_elems_u64 > std::numeric_limits<size_t>::max()) {
          throw std::runtime_error("[Vulkan SDPA] split-k temporary shape overflow.");
        }

        const size_t partial_rows = static_cast<size_t>(partial_rows_u64);
        const size_t partial_o_elems = static_cast<size_t>(partial_o_elems_u64);
        auto splitk_tensors = get_sdpa_splitk_temp_tensors(
            device, stream, partial_rows, partial_o_elems);

        const std::vector<uint32_t> stage1_push_consts{
            encode_push_constant_u32(batch_size_local),
            encode_push_constant_u32(n_q_heads_local),
            encode_push_constant_u32(n_kv_heads_local),
            encode_push_constant_u32(q_len_local),
            encode_push_constant_u32(k_len_local),
            encode_push_constant_u32(qk_dim_local),
            encode_push_constant_u32(v_dim_local),
            encode_push_constant_u32(k_head_stride_local),
            encode_push_constant_u32(k_seq_stride_local),
            encode_push_constant_u32(v_head_stride_local),
            encode_push_constant_u32(v_seq_stride_local),
            encode_push_constant_u32(mask_mode_local),
            encode_push_constant_u32(mask_batch_stride_local),
            encode_push_constant_u32(mask_head_stride_local),
            encode_push_constant_u32(mask_q_stride_local),
            encode_push_constant_u32(mask_k_stride_local),
            encode_push_constant_u32(do_causal_ ? 1u : 0u),
            encode_push_constant_u32(split_k_local),
            encode_push_constant_f32(scale_)};

        encoder.record_algo_dispatch(
            sdpa_native_splitk_stage1_kernel(path_kind),
            {q_tensor,
             k_tensor,
             v_tensor,
             splitk_tensors.partial_o_tensor,
             splitk_tensors.partial_m_tensor,
             splitk_tensors.partial_l_tensor,
             mask_tensor},
            {static_cast<uint32_t>(partial_rows_u64), 1, 1},
            stage1_push_consts);

        const std::vector<uint32_t> reduce_push_consts{
            encode_push_constant_u32(batch_size_local),
            encode_push_constant_u32(n_q_heads_local),
            encode_push_constant_u32(q_len_local),
            encode_push_constant_u32(v_dim_local),
            encode_push_constant_u32(split_k_local)};

        const char* reduce_kernel =
            sdpa_native_splitk_reduce_kernel(path_kind, split_k_local);
        sdpa_stats_record_splitk_reduce_dispatch(
            reduce_kernel, split_k_local, q_len_local, k_len_local);
        encoder.record_algo_dispatch(
            reduce_kernel,
            {splitk_tensors.partial_o_tensor,
             splitk_tensors.partial_m_tensor,
             splitk_tensors.partial_l_tensor,
             out_tensor},
            {n_rows, 1, 1},
            reduce_push_consts);
      }
      device.mark_tensor_host_dirty(out, stream.index);
      sdpa_stats_record_dispatch_result(true);
      return true;
    } catch (const std::exception& e) {
      if (sdpa_debug_reject_enabled() ||
          env_flag_default_false("MLX_VK_DEBUG_SDPA_ERROR")) {
        std::cerr << "[VulkanSDPAError] native path failed: " << e.what()
                  << " path=" << sdpa_native_path_name(path_kind)
                  << " q_len=" << q_len_local
                  << " split_k=" << split_k_local << " k_len=" << k_len_local
                  << " qk_dim=" << qk_dim_local << " v_dim=" << v_dim_local
                  << " k_head_stride=" << k_head_stride_local
                  << " k_seq_stride=" << k_seq_stride_local
                  << " v_head_stride=" << v_head_stride_local
                  << " v_seq_stride=" << v_seq_stride_local
                  << " mask_mode=" << mask_mode_local
                  << " mask_batch_stride=" << mask_batch_stride_local
                  << " mask_head_stride=" << mask_head_stride_local
                  << " mask_q_stride=" << mask_q_stride_local
                  << " mask_k_stride=" << mask_k_stride_local
                  << "\n";
      }
      sdpa_stats_record_dispatch_result(false);
      return false;
    }
  };

  auto prepare_native_inputs = [&](const std::vector<array>& src_inputs) {
    std::vector<array> native_inputs = src_inputs;
    if (native_inputs.size() <= 3 || native_inputs[3].dtype() != mlx::core::bool_) {
      return native_inputs;
    }

    try {
      // Repack bool mask to uint32 for shader-side native bool path
      // (mask_mode=2). This avoids front-end bool->additive conversion.
      native_inputs[3] = astype(native_inputs[3], mlx::core::uint32, stream);
      auto& mask_u32 = native_inputs[3];
      if (mask_u32.status() == array::Status::unscheduled) {
        mask_u32.eval();
      }
    } catch (const std::exception& e) {
      if (sdpa_debug_reject_enabled() ||
          env_flag_default_false("MLX_VK_DEBUG_SDPA_ERROR")) {
        std::cerr << "[VulkanSDPABoolMaskCastError] " << e.what() << "\n";
      }
      return src_inputs;
    }
    return native_inputs;
  };

  auto native_inputs = prepare_native_inputs(inputs);

  if (native_sdpa_enabled() && can_use_native_sdpa_bf16_decode_q1(
          native_inputs,
          outputs,
          do_causal_,
          has_sinks_,
          output_logsumexp_,
          batch_size,
          n_q_heads,
          n_kv_heads,
          q_len,
          k_len,
          qk_dim,
          v_dim,
          k_head_stride,
          k_seq_stride,
          v_head_stride,
          v_seq_stride,
          mask_mode,
          mask_batch_stride,
          mask_head_stride,
          mask_q_stride,
          mask_k_stride,
          &native_reject_reason)) {
    const SdpaNativePathKind path_kind = sdpa_native_path_kind(q_len);
    const uint64_t n_rows_u64 =
        static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(n_q_heads) *
        static_cast<uint64_t>(q_len);
    const uint32_t n_rows =
        n_rows_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        ? std::numeric_limits<uint32_t>::max()
        : static_cast<uint32_t>(n_rows_u64);
    uint32_t split_k = select_sdpa_split_k(k_len, q_len, n_rows);
    if (prefer_sdpa_decode_direct_d128(
            path_kind, qk_dim, v_dim, k_len, mask_mode)) {
      split_k = 1u;
    }
    if (dispatch_native_sdpa(
            path_kind,
            native_inputs,
            batch_size,
            n_q_heads,
            n_kv_heads,
            q_len,
            k_len,
            qk_dim,
            v_dim,
            k_head_stride,
            k_seq_stride,
            v_head_stride,
            v_seq_stride,
            mask_mode,
            mask_batch_stride,
            mask_head_stride,
            mask_q_stride,
            mask_k_stride,
            split_k)) {
      return;
    }
  }

  auto is_layout_reason = [](const char* reason) {
    return reason &&
        (std::strcmp(reason, "q_layout") == 0 ||
         std::strcmp(reason, "k_layout") == 0 ||
         std::strcmp(reason, "v_layout") == 0 ||
         std::strcmp(reason, "mask_layout") == 0 ||
         std::strcmp(reason, "out_layout") == 0);
  };

  if (native_sdpa_enabled() && is_layout_reason(native_reject_reason)) {
    std::vector<array> packed_inputs = inputs;
    bool repacked_any = false;
    for (size_t i = 0; i < std::min<size_t>(packed_inputs.size(), 3); ++i) {
      if (!is_dense_row_major_view(packed_inputs[i])) {
        packed_inputs[i] = copy(packed_inputs[i], stream);
        auto& packed = packed_inputs[i];
        if (packed.status() == array::Status::unscheduled) {
          packed.eval();
        } else {
          packed.wait();
        }
        repacked_any = true;
      }
    }
    if (packed_inputs.size() > 3 && native_reject_reason &&
        std::strcmp(native_reject_reason, "mask_layout") == 0) {
      packed_inputs[3] = copy(packed_inputs[3], stream);
      auto& packed_mask = packed_inputs[3];
      if (packed_mask.status() == array::Status::unscheduled) {
        packed_mask.eval();
      } else {
        packed_mask.wait();
      }
      repacked_any = true;
    }

    if (repacked_any) {
      auto packed_native_inputs = prepare_native_inputs(packed_inputs);
      uint32_t packed_batch = 0;
      uint32_t packed_q_heads = 0;
      uint32_t packed_kv_heads = 0;
      uint32_t packed_q_len = 0;
      uint32_t packed_k_len = 0;
      uint32_t packed_qk_dim = 0;
      uint32_t packed_v_dim = 0;
      uint32_t packed_k_head_stride = 0;
      uint32_t packed_k_seq_stride = 0;
      uint32_t packed_v_head_stride = 0;
      uint32_t packed_v_seq_stride = 0;
      uint32_t packed_mask_mode = 0;
      uint32_t packed_mask_batch_stride = 0;
      uint32_t packed_mask_head_stride = 0;
      uint32_t packed_mask_q_stride = 0;
      uint32_t packed_mask_k_stride = 0;
      const char* packed_reject_reason = nullptr;
      if (can_use_native_sdpa_bf16_decode_q1(
              packed_native_inputs,
              outputs,
              do_causal_,
              has_sinks_,
              output_logsumexp_,
              packed_batch,
              packed_q_heads,
              packed_kv_heads,
              packed_q_len,
              packed_k_len,
              packed_qk_dim,
              packed_v_dim,
              packed_k_head_stride,
              packed_k_seq_stride,
              packed_v_head_stride,
              packed_v_seq_stride,
              packed_mask_mode,
              packed_mask_batch_stride,
              packed_mask_head_stride,
              packed_mask_q_stride,
              packed_mask_k_stride,
              &packed_reject_reason)) {
        const SdpaNativePathKind packed_path_kind =
            sdpa_native_path_kind(packed_q_len);
        const uint64_t packed_n_rows_u64 =
            static_cast<uint64_t>(packed_batch) *
            static_cast<uint64_t>(packed_q_heads) *
            static_cast<uint64_t>(packed_q_len);
        const uint32_t packed_n_rows =
            packed_n_rows_u64 >
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? std::numeric_limits<uint32_t>::max()
            : static_cast<uint32_t>(packed_n_rows_u64);
        uint32_t packed_split_k =
            select_sdpa_split_k(packed_k_len, packed_q_len, packed_n_rows);
        if (prefer_sdpa_decode_direct_d128(
                packed_path_kind,
                packed_qk_dim,
                packed_v_dim,
                packed_k_len,
                packed_mask_mode)) {
          packed_split_k = 1u;
        }
        if (dispatch_native_sdpa(
                packed_path_kind,
                packed_native_inputs,
                packed_batch,
                packed_q_heads,
                packed_kv_heads,
                packed_q_len,
                packed_k_len,
                packed_qk_dim,
                packed_v_dim,
                packed_k_head_stride,
                packed_k_seq_stride,
                packed_v_head_stride,
                packed_v_seq_stride,
                packed_mask_mode,
                packed_mask_batch_stride,
                packed_mask_head_stride,
                packed_mask_q_stride,
                packed_mask_k_stride,
                packed_split_k)) {
          return;
        }
      } else if (sdpa_debug_reject_enabled()) {
        std::cerr << "[VulkanSDPARepackReject] reason="
                  << (packed_reject_reason ? packed_reject_reason : "unknown")
                  << "\n";
      }
    }
  }

  if (native_sdpa_enabled() &&
      (sdpa_debug_reject_enabled() ||
       env_flag_default_false("MLX_VK_DEBUG_SDPA_ERROR")) &&
      native_reject_reason) {
    std::cerr << "[VulkanSDPANativeReject] reason=" << native_reject_reason
              << "\n";
  }

  sdpa_stats_record_final_fallback();
  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_NO_GPU_MULTI(fast::ScaledDotProductAttentionVJP)
VULKAN_CPU_FALLBACK_MULTI(fast::ConvertFP8)
VULKAN_NO_GPU_MULTI(fast::CustomKernel)

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::OpProfileScope profile("fast::Quantize");
  profile.mark_fallback();
  materialize_and_share_fast_outputs(fallback_(inputs), outputs);
}

VULKAN_CPU_FALLBACK_MULTI(distributed::AllReduce)
VULKAN_CPU_FALLBACK_MULTI(distributed::AllGather)
VULKAN_CPU_FALLBACK_MULTI(distributed::Send)
VULKAN_CPU_FALLBACK_MULTI(distributed::Recv)
VULKAN_CPU_FALLBACK_MULTI(distributed::ReduceScatter)

} // namespace mlx::core
