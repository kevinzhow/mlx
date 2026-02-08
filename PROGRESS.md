# PROGRESS

更新日期: 2026-02-09

## 目标

- 对标 Metal Backend 机制，实现基于 Kompute `v0.9.0` 的 Vulkan Backend。
- 先保证机制对齐与稳定性（stream/eval/finalize/synchronize），再逐步替换 CPU fallback。

## 已完成进展

1. Vulkan 基础链路补齐并可编译链接:
- 补齐 `device_info` / `event` / `fence` / `gpu_fallback` / `primitives/fallback` 等缺失实现。
- 修复重复符号与关键链接缺口，`cmake --build build --target tests` 可通过。

2. 机制稳定化修复:
- 修复 `Device::commit_command_buffer` 与 sequence 生命周期问题，避免重复提交。
- CPU encoder 支持 GPU stream 回退到 CPU 默认 stream，避免 fallback 调度崩溃。
- 多处 Vulkan fallback 增加输入就绪保障（先 `eval/wait` 再 `eval_cpu`）。
- 修复二元算子 fallback 的 0-size 早退问题（避免输出未 materialize 导致崩溃）。
- 为 `array::unsafe_weak_copy` 增加防御性检查，避免空 data 指针直接段错误。

3. 本轮稳定性与性能修复:
- `vulkan::is_available()` 改为进程级一次探测缓存，避免高频重复创建/销毁 `kp::Manager`。
- 清理 Vulkan runtime 高频调试输出（`device/gpu_interface/binary/placeholder`），移除 I/O 干扰。
- 保留 CPU fallback 的同步语义（`synchronize(default_stream(Device::cpu))`），消除竞态崩溃。
- 调整 `scheduler.cpp` 判断顺序，仅在 GPU 分支触发 `gpu::is_available()`。

4. 测试里程碑:
- `test quantize dequantize` 通过。
- `test scheduler races` 在 Vulkan 下恢复稳定，通过 20 次连续复测。
- `ctest --test-dir build --stop-on-failure --output-on-failure` 全量通过:
  - `226/226` Passed
  - Total Test time (real) = `23.28 sec`

## 当前阻塞

- 功能性阻塞: 暂无（当前全量测试通过）。
- 工程性阻塞:
  - 核心算子仍大量依赖 CPU fallback，Vulkan 原生算子覆盖率不足，尚未达到最终目标的能力对齐。

## 下一步计划

1. 优先替换基础高频 fallback:
- 从 `Copy` 开始提供 Vulkan 原生实现，替换 placeholder CPU fallback。
- 在 `binary/unary` 中优先落地高频算子原生路径（如 `Add` / `Multiply` / `Exp`）。

2. 保持机制对齐:
- 持续验证 `stream/eval/finalize/synchronize` 行为与 Metal 一致。
- 每引入一个原生算子后，执行单项回归 + 全量回归。

3. 验证门禁:
- 单项:
  - `ctest --test-dir build -R "test scheduler races" --output-on-failure --timeout 120`
- 全量:
  - `ctest --test-dir build --stop-on-failure --output-on-failure`

## 维护规则

- 每次有实质进展（修复、发现新阻塞、测试里程碑）必须更新本文件。
- 进入下一轮工作前，先以本文件中的“当前阻塞 + 下一步计划”为执行入口。
