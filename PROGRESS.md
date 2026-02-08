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

3. 测试推进结果:
- `test arithmetic unary ops` 已从稳定崩溃修复为通过。
- `ctest --stop-on-failure --output-on-failure` 可稳定推进到 `174/235` 通过。

## 当前阻塞

- 首个失败测试:
  - `test quantize dequantize` (`CTest #175`)
  - 现象: `SIGSEGV`
- 关键栈信息（最近一次 gdb）:
  - `mlx::core::array::size() const`
  - `mlx::core::array::nbytes() const`
  - `mlx::core::fast::Quantize::eval_cpu(...)`
  - `mlx::core::fast::Quantize::eval_gpu(...)`
- 结论:
  - 崩溃发生在 fast::Quantize 多输出路径，当前 Vulkan fallback 在该路径仍存在输出对象未正确对齐/物化的问题。

## 下一步计划

1. 定点修复 `fast::Quantize::eval_gpu` 的 Vulkan fallback:
- 明确多输出（特别是 dequantize 与非 dequantize 模式）在 fallback 中的输出契约。
- 确保 `outputs[i]` 在进入 CPU eval 或 fallback 结果复制前形状/数据状态合法。

2. 复现与验证:
- 单测: `ctest --test-dir build -R "test quantize dequantize" --output-on-failure`
- 必要时 gdb:
  - `gdb -q -batch -ex "run --test-case=\"test quantize dequantize\"" -ex "bt" --args build/tests/tests`

3. 回归推进:
- 修复后重新跑:
  - `ctest --test-dir build --stop-on-failure --output-on-failure`
- 记录新的首个失败点并继续迭代。

## 维护规则

- 每次有实质进展（修复、发现新阻塞、测试里程碑）必须更新本文件。
- 进入下一轮工作前，先以本文件中的“当前阻塞 + 下一步计划”为执行入口。
