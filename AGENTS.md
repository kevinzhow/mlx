# AGENTS.md

## Mission

Build and harden the Vulkan backend with Kompute, aligned to Metal backend mechanisms and runtime contracts.

## Progress-Driven Workflow (Mandatory)

- Before starting any new work, read `PROGRESS.md`.
- Decide what to do next primarily from:
  - current blockers
  - next-step plan
  - latest verification results
- If `PROGRESS.md` and code reality diverge, update `PROGRESS.md` first, then continue.
- After each meaningful progress point (fix, new failure found, verification milestone), append/update `PROGRESS.md` in the same working cycle.
- Before finishing a cycle, ensure `PROGRESS.md` records:
  - what changed
  - current status
  - exact next actions
- Keep `PROGRESS.md` as the single source of truth for handoff and continuation.

## Target State

- Vulkan backend is selected through the existing GPU path, not a side path.
- `gpu::new_stream`, `gpu::eval`, `gpu::finalize`, and `gpu::synchronize` match Metal semantics.
- `Device -> DeviceStream -> CommandEncoder` lifecycle is behaviorally aligned with Metal.
- Core operators run on Vulkan where implemented; temporary fallbacks remain correct and explicit.

## Alignment Rules

- Keep common orchestration in `mlx/backend/gpu/*`.
- Keep Vulkan-specific code in `mlx/backend/vulkan/*`.
- Do not change Metal behavior unless changing a shared cross-backend contract.
- Maintain lazy scheduling and command-buffer commit behavior consistent with Metal.
- Preserve buffer lifetime and donation safety behavior.
- Implement and maintain `gpu::is_available`, `gpu::device_count`, and `gpu::device_info` for Vulkan builds.

## Kompute Policy

- Use Kompute `v0.9.0` by default.
- Keep version configurable via CMake cache variable `MLX_KOMPUTE_TAG`.
- Prefer target `kompute::kompute` when available; keep compatibility fallback to `kompute`.

## Fallback Policy

- CPU fallback is temporary and must be explicit.
- Every fallback path must include a short TODO and remain stream-safe and correct.
- Replace fallback paths incrementally with Vulkan implementations.

## Priority Order

1. Fix build/link/runtime contract blockers.
2. Stabilize stream/eval/sync behavior.
3. Implement foundational ops used by common GPU primitives: copy, reshape, fill, concatenate, slicing.
4. Expand binary/unary/reduce/indexing coverage.
5. Optimize barriers, memory reuse, and dispatch efficiency.

## Build and Verification

- Configure:
  - `cmake -S . -B build -DMLX_BUILD_VULKAN=ON`
- Build core library:
  - `cmake --build build --target mlx -j`
- Build tests:
  - `cmake --build build --target tests -j`
- Verify:
  - No duplicate symbols or unresolved symbols.
  - GPU path links and initializes with Vulkan enabled.
  - Covered operators match CPU results.

## Runtime Parameters (Vulkan + Qwen3)

- Use this baseline runtime env for real-GPU Vulkan validation:
  - `LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH`
  - `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`
  - `MESA_VK_DEVICE_SELECT=1002:1900`
  - `PYTHONPATH=python`
  - `TARGET_DEVICE=gpu`
- Default native gates expected on current mainline:
  - `MLX_VK_ENABLE_QMM_NATIVE=1` (default ON)
  - `MLX_VK_ENABLE_RMSNORM_NATIVE=1` (default ON)
  - `MLX_VK_ENABLE_ROPE_NATIVE=1` (default ON)
  - `MLX_VK_ENABLE_SDPA_NATIVE=1` (default ON, still narrow gate in code)

### Standard Qwen3 correctness checks

- 10-token check (Chinese prompt):
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 -m mlx_lm generate --model Qwen/Qwen3-0.6B-MLX-4bit --prompt "你好啊" --max-tokens 10 --temp 0`
- 10-token check (English prompt):
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 -m mlx_lm generate --model Qwen/Qwen3-0.6B-MLX-4bit --prompt "Hi what is your name" --max-tokens 10 --temp 0`
- Split-prefill finite check helper:
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 /tmp/check_split_prefill.py`

### Optional debug env (only when bisecting)

- `MLX_VK_DEBUG_ROPE_REJECT=1`
- `MLX_VK_DEBUG_SDPA_REJECT=1`
- `MLX_VK_DEBUG_SDPA_SPLITK=1`
- `MLX_VK_SDPA_STATS=1` (进程退出时打印 SDPA 命中/回退分布与 `k_len_cap` 占比)
- Native gate toggles for isolation:
  - `MLX_VK_ENABLE_QMM_NATIVE=0|1`
  - `MLX_VK_ENABLE_RMSNORM_NATIVE=0|1`
  - `MLX_VK_ENABLE_ROPE_NATIVE=0|1`
  - `MLX_VK_ENABLE_SDPA_NATIVE=0|1`
  - `MLX_VK_ENABLE_ADD_BF16=0|1`
  - `MLX_VK_ENABLE_MUL_BF16=0|1`
- Split-k tuning knobs:
  - Global (existing): `MLX_VK_SDPA_SPLITK_MIN_K_LEN`, `MLX_VK_SDPA_SPLITK_TARGET_CHUNK`, `MLX_VK_SDPA_SPLITK_MAX_PARTS`, `MLX_VK_SDPA_SPLIT_K`
  - Decode overrides (`q_len==1`): `MLX_VK_SDPA_SPLITK_MIN_K_LEN_DECODE`, `MLX_VK_SDPA_SPLITK_TARGET_CHUNK_DECODE`, `MLX_VK_SDPA_SPLITK_MAX_PARTS_DECODE`
  - Prefill overrides (`q_len>1`): `MLX_VK_SDPA_SPLITK_MIN_K_LEN_PREFILL`, `MLX_VK_SDPA_SPLITK_TARGET_CHUNK_PREFILL`, `MLX_VK_SDPA_SPLITK_MAX_PARTS_PREFILL`

## Definition of Done

- `mlx` builds and links cleanly with `MLX_BUILD_VULKAN=ON`.
- `tests` links cleanly in the same configuration.
- Implemented Vulkan operators are correct and mechanism-aligned with Metal.
- Architectural changes are reflected in `mlx/backend/vulkan/ARCHITECTURE.md`.
