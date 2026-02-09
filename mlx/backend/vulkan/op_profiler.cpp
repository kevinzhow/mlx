// Copyright © 2026 MLX Vulkan Backend

#include "mlx/backend/vulkan/op_profiler.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx::core::vulkan {

namespace {

struct OpStats {
  uint64_t calls{0};
  uint64_t fallback_calls{0};
  uint64_t sync_calls{0};
  uint64_t copy_bytes{0};
  uint64_t total_ns{0};
};

std::mutex& stats_mutex() {
  static std::mutex mtx;
  return mtx;
}

std::unordered_map<std::string, OpStats>& stats_map() {
  static std::unordered_map<std::string, OpStats> map;
  return map;
}

bool parse_env_flag(const char* value) {
  if (!value) {
    return false;
  }
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  return v == "1" || v == "true" || v == "on" || v == "yes";
}

bool profile_print_each() {
  static const bool enabled =
      parse_env_flag(std::getenv("MLX_VK_PROFILE_PRINT_EACH"));
  return enabled;
}

int parse_top_n() {
  const char* value = std::getenv("MLX_VK_PROFILE_TOP");
  if (!value) {
    return 20;
  }
  int n = std::atoi(value);
  return n > 0 ? n : 20;
}

void dump_profile_report() {
  if (!profile_enabled()) {
    return;
  }

  std::vector<std::pair<std::string, OpStats>> rows;
  {
    std::lock_guard<std::mutex> lock(stats_mutex());
    rows.reserve(stats_map().size());
    for (const auto& [name, stats] : stats_map()) {
      rows.emplace_back(name, stats);
    }
  }

  if (rows.empty()) {
    return;
  }

  std::sort(
      rows.begin(),
      rows.end(),
      [](const auto& a, const auto& b) { return a.second.total_ns > b.second.total_ns; });

  uint64_t total_ns = 0;
  uint64_t total_calls = 0;
  uint64_t total_fallback = 0;
  uint64_t total_sync = 0;
  uint64_t total_copy = 0;
  for (const auto& [_, stats] : rows) {
    total_ns += stats.total_ns;
    total_calls += stats.calls;
    total_fallback += stats.fallback_calls;
    total_sync += stats.sync_calls;
    total_copy += stats.copy_bytes;
  }

  const int top_n = std::min<int>(parse_top_n(), static_cast<int>(rows.size()));
  std::cerr << "[VulkanProfile] ===== Op Summary (Top " << top_n << ") =====\n";
  std::cerr << std::fixed << std::setprecision(3);
  for (int i = 0; i < top_n; ++i) {
    const auto& [name, stats] = rows[i];
    const double total_ms = static_cast<double>(stats.total_ns) / 1e6;
    const double avg_us =
        stats.calls > 0 ? (static_cast<double>(stats.total_ns) / stats.calls) / 1e3 : 0.0;
    const double copy_mb = static_cast<double>(stats.copy_bytes) / (1024.0 * 1024.0);
    const double fallback_pct =
        stats.calls > 0 ? (100.0 * static_cast<double>(stats.fallback_calls) / stats.calls) : 0.0;
    std::cerr << "[VulkanProfile] "
              << std::setw(2) << (i + 1) << ". "
              << name
              << " | calls=" << stats.calls
              << " fallback=" << stats.fallback_calls << " (" << fallback_pct << "%)"
              << " sync=" << stats.sync_calls
              << " copyMB=" << copy_mb
              << " total_ms=" << total_ms
              << " avg_us=" << avg_us
              << "\n";
  }
  std::cerr << "[VulkanProfile] TOTAL"
            << " | calls=" << total_calls
            << " fallback=" << total_fallback
            << " sync=" << total_sync
            << " copyMB=" << (static_cast<double>(total_copy) / (1024.0 * 1024.0))
            << " total_ms=" << (static_cast<double>(total_ns) / 1e6)
            << "\n";
}

void record_sample(
    const char* op_name,
    uint64_t elapsed_ns,
    bool fallback,
    uint64_t sync_count,
    uint64_t copy_bytes) {
  if (!profile_enabled() || !op_name) {
    return;
  }

  if (profile_print_each()) {
    std::cerr << "[VulkanProfileSample] op=" << op_name
              << " ns=" << elapsed_ns
              << " fallback=" << (fallback ? 1 : 0)
              << " sync=" << sync_count
              << " copy_bytes=" << copy_bytes << "\n";
  }

  std::lock_guard<std::mutex> lock(stats_mutex());
  auto& stats = stats_map()[op_name];
  stats.calls += 1;
  stats.fallback_calls += fallback ? 1 : 0;
  stats.sync_calls += sync_count;
  stats.copy_bytes += copy_bytes;
  stats.total_ns += elapsed_ns;
}

} // namespace

bool profile_enabled() {
  static const bool enabled = [] {
    bool v = parse_env_flag(std::getenv("MLX_VK_PROFILE"));
    if (v) {
      std::atexit(dump_profile_report);
      std::cerr << "[VulkanProfile] enabled\n";
    }
    return v;
  }();
  return enabled;
}

OpProfileScope::OpProfileScope(const char* op_name)
    : op_name_(op_name),
      enabled_(profile_enabled()),
      start_time_(std::chrono::steady_clock::now()) {}

OpProfileScope::~OpProfileScope() {
  if (!enabled_) {
    return;
  }
  auto elapsed = std::chrono::steady_clock::now() - start_time_;
  uint64_t elapsed_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  record_sample(op_name_, elapsed_ns, fallback_, sync_count_, copy_bytes_);
}

void OpProfileScope::mark_fallback() {
  fallback_ = true;
}

void OpProfileScope::mark_sync(uint64_t count) {
  sync_count_ += count;
}

void OpProfileScope::add_copy_bytes(size_t bytes) {
  copy_bytes_ += static_cast<uint64_t>(bytes);
}

} // namespace mlx::core::vulkan
