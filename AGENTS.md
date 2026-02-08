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

## Definition of Done

- `mlx` builds and links cleanly with `MLX_BUILD_VULKAN=ON`.
- `tests` links cleanly in the same configuration.
- Implemented Vulkan operators are correct and mechanism-aligned with Metal.
- Architectural changes are reflected in `mlx/backend/vulkan/ARCHITECTURE.md`.
