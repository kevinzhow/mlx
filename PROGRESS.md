# PROGRESS

更新日期: 2026-02-11（精简版）

## 目标
- 对标 Metal backend 运行时契约，在 MLX 现有 GPU 路径内完成 Vulkan backend（Kompute v0.9.0）主线化。
- 优先级保持不变：先稳定与正确，再扩原生覆盖，最后做架构级性能提升。
- 开发流程约束：先做命中分析，再做优化；避免在低命中路径上投入主优化成本。

## 当前快照
- 构建状态：`MLX_BUILD_VULKAN=ON` 下 `mlx` 与 `tests` 均可稳定构建链接。
- 回归状态：`ctest` 全量 `223/223` 通过；`python/tests/test_fast_sdpa.py` 通过（`20 passed, 1 skipped`）。
- 模型正确性：Qwen3-0.6B-MLX-4bit（EN/ZH，10 token）输出正常，无乱码/`!!!!!!!!!!` 回归。
- 流程约束（新增）：Qwen3 `mlx_lm generate` 正确性/性能口径需串行执行；任意 C++/Vulkan/shader 变更后，Python 侧验证前必须先执行 `python3 setup.py build_ext --inplace`，避免旧扩展产物导致误判。
- 近期吞吐（实卡 Vulkan，串行口径）：
  - 10-token（EN）：约 `4.63 tok/s`（`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE=1`）
  - 40-token（EN）：约 `3.65 tok/s`（`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP=1`，默认 ON）
  - 80-token（EN）：约 `2.97 tok/s`（同口径）
- decode 主线 fallback 占比（同口径 profile）：
  - 由 `18.40%` 降到 `13.50%`
- RoPE 回退状态（Qwen3 EN 40-token 样本）：
  - `MLX_VK_ENABLE_ROPE_HS_TRANSPOSED=1` 时 `VulkanRoPEReject` 由 `55` 次降到 `0`。

## 历史完成摘要（已压缩）
1. Vulkan 基础链路与运行时契约已打通  
- 设备识别、stream/eval/sync/fallback 生命周期问题已完成关键修复。  
- 修复过往 async 死锁、host/device 脏数据覆盖、alias cache 不一致等关键问题。

2. 正确性主阻塞已解除  
- Qwen 生成异常（含 `!!!!!!!!!!`）已定位并修复。  
- 当前默认门禁配置下，中英文生成输出稳定。

3. 原生覆盖持续扩展  
- 已落地主线原生路径（含部分窄门禁）：QMM、RMSNorm、RoPE、SDPA（分阶段）、ArgReduce（argmax last-dim）、若干 binary/unary。

4. 可观测性与工程闭环完善  
- 已建立 Vulkan profile/debug 门禁体系。  
- Shader 更新流程已固化为 `.comp -> .spv -> *_spv.h`（同轮闭环）。

5. 文档与运行参数固化  
- `AGENTS.md` 已记录实卡运行环境、标准正确性命令、常用 gate/debug 开关。

## 最近里程碑

### D-9.1：ArgReduce decode 热点 native 化（已完成）
- 目标：移除 decode 主链路 `ArgReduce` CPU fallback 热点。
- 结果：`ArgReduce(argmax, axis=last)` 命中 native 且回归通过。
- 收益：10/40 token 吞吐均有小幅稳定正收益（约 `~1%`）。

### D-9.2：Compiled 热点拆解 + `Sigmoid*Mul*Mul` native 化（已完成）
- 目标：拆解 `Compiled` 高占比来源，并优先移除单一高频融合子图热点。
- 归因：主热点为 `CompiledSigmoidMultiplyMultiply`（bf16，同形状行连续）。
- 变更：新增 `silu_mul_bf16` kernel，并在 `Compiled::eval_gpu` 增加该子图窄门禁 native 桥接。
- profile 收益（10-token 同口径）：
  - `Compiled`：`~660ms -> ~30ms`
  - 总体：`fallback_pct 18.40% -> 13.50%`
- 吞吐 A/B（gate on/off，串行）：
  - 10-token：`3.233 vs 3.192 tok/s`（`+1.3%`）
  - 40-token：`2.701 vs 2.486 tok/s`（`+8.6%`）

### D-9.3：QMM decode `M=1` 专核（已完成）
- 目标：优先吃下 decode 高频形状（`rows==1`）的 `QuantizedMatmul`，降低 QMM 热点指令开销。
- Metal/Ollama 对照启发：
  - Metal 路线强调“高频形状专核 + 最小分支调度”。
  - Ollama/ggml Vulkan 路线强调“解码小 batch 专项优化 + 内核特化”。
  - 本轮采用同类策略：仅对 decode 主形状开窄门禁专核，不改通用路径行为。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1.comp`（`rows==1` 专核，`local_size_x=128`）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1`。
  - 在 `QuantizedMatmul::eval_gpu` 增加 `rows==1` 动态派发与 gate：`MLX_VK_ENABLE_QMM_NATIVE_M1`（默认 ON）。
  - 按 kernel 工作组大小动态计算 `groups_x`，并修正 `out_words` 为 ceil 形式。
- 回归：
  - `ctest -R "quantize dequantize|arithmetic binary ops|scheduler races|arg reduce"` 通过。
  - `python/tests/test_fast_sdpa.py -v` 通过（`20 passed, 1 skipped`）。
- 吞吐 A/B（Qwen3 EN，实卡 Vulkan，串行）：
  - 40-token：`3.348 vs 2.652 tok/s`（`+26.2%`）
  - 10-token：`4.102 vs 3.201 tok/s`（`+28.1%`）

### E-9.3a：`M=1` 双列同线程累加实验（已回退）
- 假设：同一线程同时累加相邻两列，复用 `x` 解包，可进一步降指令数。
- 实测：`M1=1` 时 40-token 吞吐约 `3.33 -> 2.86 tok/s`（显著回退）。
- 结论：该实现引入了额外分支与寄存器压力，抵消了访存复用收益；已回退到 D-9.3 稳定版本。

### D-9.4：QMM `M>1` 形状分桶（阶段1：`M=2/4/8/16` 已完成）
- 目标：延续 Metal/Ollama 的“高频形状专核”路线，把 QMM 从单点 `M=1` 扩展到小 batch 桶。
- 本轮变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m2.comp`（`rows==2` 专核）。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m4.comp`（`rows==4` 专核）。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m8.comp`（`rows==8` 专核）。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m16.comp`（`9<=rows<=16` 专核）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M2`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M4`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M8`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M16`。
  - `QuantizedMatmul::eval_gpu` 增加 `rows==2/4/8` 与 `9<=rows<=16` 动态派发与 gate：
    - `MLX_VK_ENABLE_QMM_NATIVE_M2=1`（默认 ON）
    - `MLX_VK_ENABLE_QMM_NATIVE_M4=1`（默认 ON）
    - `MLX_VK_ENABLE_QMM_NATIVE_M8=1`（默认 ON）
    - `MLX_VK_ENABLE_QMM_NATIVE_M16=1`（默认 ON）
  - 新增 QMM 运行统计：`MLX_VK_QMM_STATS=1`
    - 进程退出时打印 native kernel 命中计数（`m1/m2/m4/m8/generic`）与 `rows` 桶分布、fallback 桶分布。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h` 同轮完成并构建。
- 验证：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3（EN 40-token）维持高位：约 `3.32~3.43 tok/s`（无正确性回归）。
- 结论：
  - `M=2/4/8/16` 路径已打通，形状分桶框架成立；下一步进入向量化/子组化分支。

### D-9.5：QMM 命中分布可观测性（已完成）
- 目标：在继续做 QMM 架构优化前，先拿到真实命中桶，避免“优化了未命中路径”。
- 变更：
  - `QuantizedMatmul::eval_gpu` 增加 QMM 统计埋点与退出打印。
  - 新增环境变量：`MLX_VK_QMM_STATS=1`。
  - 指标包含：native success/fail、kernel 命中计数、native/fallback 的 rows 桶分布。
- 观测（Qwen3 EN 40-token，实卡 Vulkan）：
  - 优化前：
    - `qmm_affine_bf16_t4_g128_m1`: `8077`
    - `qmm_affine_bf16_t4_g128`(generic): `191`
    - rows 桶：`rows=1` 主导，另有 `rows=9-16` 命中 generic。
  - 优化后（新增 `m16`）：
    - `qmm_affine_bf16_t4_g128_m1`: `8077`
    - `qmm_affine_bf16_t4_g128_m16`: `191`
    - rows 桶仍为 `rows=1` + `rows=9-16`，但 `9-16` 已迁移到专核。
- 结论：
  - 当前工作负载主命中为 `M1`，次热点为 `rows=9-16`，且两者均已专核化。
  - 下一阶段聚焦 `M1` 主核向量化/子组化，避免继续扩低命中桶。

### D-9.6：QMM `M1` 并行归约路径（已完成，默认 ON）
- 目标：直接优化 decode 主命中路径（`rows=1`），提升每 token 生成吞吐。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce.comp`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE`。
  - `QuantizedMatmul::eval_gpu` 增加 `rows==1` 归约路径派发。
  - gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE`（默认 ON，可关闭回退旧 `M1` 路径）。
- 命中验证（`MLX_VK_QMM_STATS=1`，Qwen3 EN 40-token）：
  - `qmm_affine_bf16_t4_g128_m1_reduce`: `8077`
  - `qmm_affine_bf16_t4_g128_m16`: `191`
  - 说明：主命中 `rows=1` 已切换到新归约核，`rows=9-16` 维持 `m16`。
- 吞吐 A/B（串行）：
  - 40-token：`3.62 / 3.58 tok/s`（`M1_REDUCE=1`） vs `3.34 / 3.35 tok/s`（`M1_REDUCE=0`）
  - 平均提升约 `+8~9%`。
- 稳定性：
  - `ctest` 全量 `223/223` 通过。
  - `python/tests/test_fast_sdpa.py` 通过（`20 passed, 1 skipped`）。
  - Qwen3 EN/ZH 输出正确，无乱码回归。

### E-9.5a：QMM `M=1` 多累加器展开实验（已回退）
- 假设：通过多累加器 + 最内层显式展开降低依赖链，提升 decode 吞吐。
- 实测：Qwen3 EN 40-token 吞吐无稳定正收益（约 `3.32~3.33 tok/s`，未优于基线）。
- 结论：已回退该实现，保持当前稳定核。

### D-9.7：流程护栏修正（已完成）
- 背景：出现过“源码已更新但 Python 扩展未重编，导致运行仍落在旧产物”的问题，干扰性能/正确性判断。
- 本轮修正：
  - 将“Qwen3 `mlx_lm generate` 口径必须串行执行（不并行）”写入 `AGENTS.md` 强制规则。
  - 明确该串行约束仅针对 Qwen3 生成测试，不约束 `ctest`/常规 Python 单测。
  - 将“Python 侧验证前必须先 `python3 setup.py build_ext --inplace`”写入 `AGENTS.md` 强制规则。
  - 将该流程护栏同步记录到 `PROGRESS.md` 当前快照，作为后续默认执行前置条件。
- 影响：
  - 避免并行执行导致的瞬时缓存/产物竞争噪声。
  - 避免旧扩展产物造成的假结论（尤其是 shader/C++ 改动后）。

### D-9.8：RoPE head/seq 转置布局默认放开（已完成）
- 背景：
  - Qwen3 decode 样本中 RoPE fallback 主因集中为 `in_or_out_layout`。
  - 典型输入布局：`shape=[1,8|16,12,128]`，`strides=[B,128,H*128,1]`，即 head/seq 转置视图。
- 变更：
  - 将 `MLX_VK_ENABLE_ROPE_HS_TRANSPOSED` 从默认 OFF 改为默认 ON（仍可 env 关闭）。
  - 不改变 `MLX_VK_ENABLE_ROPE_NATIVE` 总开关语义。
- 验证：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN 40-token（实卡 Vulkan，串行）：`VulkanRoPEReject` 由 `55` -> `0`。
  - 吞吐无负收益（40-token A/B 约持平，`~3.63 tok/s`）。
- 结论：
  - RoPE 高频布局回退已消除，后续优化重心继续放在 QMM 主耗时与剩余小算子回退尾部。

### D-9.9：QMM `M1_REDUCE` subgroup 归约路径（已完成，默认 ON）
- 背景（Metal/Ollama 对照）：
  - Metal 路线长期依赖 SIMD-group / wave 级归约减少 workgroup 级 barrier 开销。
  - Ollama/ggml Vulkan 路线同样在 decode 热核优先利用 subgroup/wave 原语，减少 shared-memory 归约树。
  - 当前 QMM `rows=1` 是绝对主命中（`~8k` 次/40-token），适合作为 subgroup 先行落点。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup.comp`（`subgroupAdd` 两级归约）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP`。
  - `QuantizedMatmul::eval_gpu` 新增派发策略：
    - 默认优先 `M1_REDUCE_SUBGROUP`。
    - 若 subgroup kernel dispatch 异常，进程内自动降级关闭该路径并回退到 `M1_REDUCE`，避免重复异常。
  - 新增 gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP`（默认 ON，可显式关闭）。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性：输出正常，无乱码，约 `4.648 tok/s`。
- 吞吐 A/B（实卡 Vulkan，串行，EN）：
  - 40-token：`3.646 tok/s`（subgroup ON） vs `3.611 tok/s`（subgroup OFF），`+1.0%`。
  - 80-token：`2.974 tok/s`（subgroup ON） vs `2.924 tok/s`（subgroup OFF），`+1.7%`。
  - 命中确认：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup` 覆盖 `rows=1` 主桶（`8077/15957`）。
- 结论：
  - subgroup 路径带来小幅稳定正收益，说明方向正确。
  - 但距离“性能蜕变”仍远，下一阶段必须进入更激进的架构级重构（不仅是微调）。

### D-9.9a：QMM subgroup 核 `scale/bias` 共享缓存（已完成）
- 背景（Metal/Ollama 对照）：
  - 二者的高性能路径都强调“减少重复全局访存”，尤其在 decode 小 batch 场景下避免同一参数被 lane 重复加载。
  - 原 subgroup 核中同一 `g` 的 `scale/bias` 被 `ww` 维度重复读取（最多 16x 重复）。
- 变更：
  - 在 `qmm_affine_bf16_t4_g128_m1_reduce_subgroup.comp` 引入 workgroup 级 `shared` 缓存：
    - `scale0/bias0/scale1/bias1` 按 `groups_per_col` 预加载一次，再在主循环复用。
  - 对 subgroup 核增加安全条件：仅 `groups_per_col <= 256` 走该路径，避免共享缓存越界。
  - 维持 dispatch 失败自动降级到 `M1_REDUCE` 的安全护栏。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest` 全量 `223/223` 通过。
  - `python/tests/test_fast_sdpa.py -v` 通过（`20 passed, 1 skipped`）。
  - Qwen3 ZH 10-token 正确性通过，无乱码。
- 结果：
  - 与 `M1_REDUCE` 对照仍是小幅正收益（约 `+1% ~ +1.7%`），说明访存复用方向有效，但尚不足以带来量级跃迁。

## 当前性能卡点（按优先级）
1. `QuantizedMatmul`  
- 仍是端到端主耗时大头；`M1_REDUCE_SUBGROUP` 仅带来 `~1%` 级增益，说明需要更深层内核重构（访存与解量化融合策略）。

2. `fast::RMSNorm / fast::ScaledDotProductAttention`
- 在 RoPE 回退清理后，二者成为次级原生耗时来源，需继续做 kernel 级效率优化。

3. 高频小算子 fallback 尾部  
- `Multiply`、`BitwiseBinary` 等仍有较高调用次数的 fallback。

4. SDPA 长上下文效率  
- 当前 decode-unlimited + split-k 已稳定，但 `k=65+` 的 stage1/reduce 仍有进一步优化空间。

## 下一步（精确执行入口）
1. QMM decode 主核架构升级（对标 Metal/Ollama）  
- 目标：从“微调归约”升级到“访存/解量化/归约一体化”内核，优先作用于 `rows=1` 主桶。
- 方向：`uvec4`/向量化加载、scale/bias 访问复用、减少重复解包与无效算术、压缩指令路径。

2. SDPA decode 次热点并行推进  
- 对照 Metal/Ollama 的 decode attention 内核拆分策略，继续优化 `k=33~128` 区间的 stage1/reduce 配置与核形态。

3. 清理 `Multiply` 高频 fallback  
- 继续沿 Metal/Ollama 的“高频小算子专核”路线，先做 decode 高频形态桶。

4. 维持命中优先与串行口径  
- 每轮必须先跑 `MLX_VK_QMM_STATS=1` / `MLX_VK_SDPA_STATS=1` 确认命中，再做优化与 A/B。

5. 门禁与评估口径保持一致  
- 继续用 EN/ZH `10/40/80` 串行压测 + profile 对照，确保“命中提升 = 吞吐提升”。

6. 研究流程约束  
- 每轮方案分析必须对照 Metal 与 Ollama 技术路径，并将结论与目标回写本文件。

## 标准验证命令（保留）
- C++ 全量：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`
- Python（SDPA 关键集）：
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`
- Qwen3 正确性（实卡 Vulkan）：
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 -m mlx_lm generate --model Qwen/Qwen3-0.6B-MLX-4bit --prompt "你好啊" --max-tokens 10 --temp 0`
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 -m mlx_lm generate --model Qwen/Qwen3-0.6B-MLX-4bit --prompt "Hi what is your name" --max-tokens 10 --temp 0`

## 说明
- 旧版超长历史日志已压缩为本摘要；详细过程以 git 历史与相关 commit 为准。
