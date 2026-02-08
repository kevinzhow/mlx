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
- `vulkan::is_available()` 改为原生 Vulkan 物理设备探测 + 进程级缓存，避免高频重复创建/销毁 `kp::Manager`。
- 清理 Vulkan runtime 高频调试输出（`device/gpu_interface/binary/placeholder`），移除 I/O 干扰。
- 保留 CPU fallback 的同步语义（`synchronize(default_stream(Device::cpu))`），消除竞态崩溃。
- 调整 `scheduler.cpp` 判断顺序，仅在 GPU 分支触发 `gpu::is_available()`。
- 清理测试枚举阶段的 Kompute 日志副作用，`ctest -N` 测试列表恢复干净。
- 修复 `Buffer::from_array` 数据桥接:
  - 从“复制到 `std::vector<float>`”改为直接基于 array 原始内存创建 Kompute tensor。
  - 补齐 dtype 到 Kompute 数据类型映射。
  - `Buffer::upload/download/sync` 改为基于 raw bytes 指针处理，避免 float-only 假设。
- **修复 Algorithm 缓存逻辑**: 在 cache key 中包含 push constants（十六进制编码），解决不同 size 算子复用旧 Algorithm 导致 push constants 失效的问题。
- **解决内存复用导致的数据不一致**: 暂时关闭 `BufferManager` 按 raw pointer 缓存 Buffer 的逻辑，避免 MLX allocator 复用地址时拿到包含旧数据的 Tensor。
- **修复 Add 原生算子**:
  - 增加 `out.size() > 0` 检查，避免 Kompute 处理 0-sized buffer 报错。
  - 增加输出 tensor 的同步指令，确保 GPU 写入位点正确。
  - 验证通过 `Add` 的 float32 contiguous 路径。

4. 测试里程碑:
- `test arithmetic binary ops` 全量通过（包含 native Add 路径）。
- `test quantize dequantize` 通过。
- `test scheduler races` 在 Vulkan 下恢复稳定，通过 20 次连续复测。
- `ctest --test-dir build --stop-on-failure --output-on-failure` 全量通过:
  - `223/223` Passed
  - Total Test time (real) = `16.21 sec`

## 当前阻塞

- 工程性阻塞:
  - 核心算子仍大量依赖 CPU fallback，Vulkan 原生算子覆盖率不足，尚未达到最终目标的能力对齐。
  - `BufferManager` 缺乏与 `array::Data` 生命周期的深度绑定，导致当前必须关闭缓存以维持正确性，存在性能优化空间。

## 下一步计划

1. 扩展原生算子覆盖:
- 按照 `Add` 的成功经验，继续推进 `Multiply` / `Subtract` / `Divide` 等二元算子的原生路径。
- 开始引入 `Unary` 算子（如 `Exp`, `Log`）。

2. 优化内存管理:
- 探索将 `vulkan::Buffer` 与 `array::Data` 绑定的更优方案，以恢复缓存并减少 Tensor 创建开销。

3. 保持机制对齐:
- 持续验证 `stream/eval/finalize/synchronize` 行为与 Metal 一致。
- 每引入一个原生算子后，执行单项回归 + 全量回归。

## 下一步计划

1. 优先替换基础高频 fallback:
- 先收敛 `Add` native 路径正确性（优先检查 command sequence 提交/同步时机与 shader 参数契约）。
- 若短期无法稳定，先回退该实验分支，恢复全绿，再以更小步重试（先单 kernel 最小闭环验证）。
- 在 `Add` 稳定后继续推进 `binary/unary` 高频算子原生路径（`Multiply` / `Exp`）。

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
