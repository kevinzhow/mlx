# Vulkan Backend CMakeLists.txt
# Aligned with metal/CMakeLists.txt structure
# Uses Kompute for simplified Vulkan compute

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

message(STATUS "Vulkan found: ${Vulkan_INCLUDE_DIRS}")

# ============================================================================
# Ensure fmt is available for Kompute
# ============================================================================

# Check if fmt is already being fetched by MLX
if(NOT TARGET fmt::fmt AND NOT TARGET fmt::fmt-header-only)
  # If not, fetch it ourselves
  include(FetchContent)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
  )
  FetchContent_MakeAvailable(fmt)
endif()

# Set fmt_DIR so Kompute can find it
if(fmt_BINARY_DIR)
  set(fmt_DIR "${fmt_BINARY_DIR}" CACHE PATH "fmt directory" FORCE)
elseif(FMT_BINARY_DIR)
  set(fmt_DIR "${FMT_BINARY_DIR}" CACHE PATH "fmt directory" FORCE)
endif()

# Also set CMAKE_PREFIX_PATH to include fmt
list(APPEND CMAKE_PREFIX_PATH "${fmt_BINARY_DIR}")

# ============================================================================
# Now fetch Kompute
# ============================================================================
include(FetchContent)

# Kompute version (can be overridden from top-level cmake command line)
set(MLX_KOMPUTE_TAG "v0.9.0" CACHE STRING "Kompute git tag for Vulkan backend")

if(NOT TARGET kompute::kompute)
  message(STATUS "Fetching Kompute (${MLX_KOMPUTE_TAG})...")
  
  FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
    GIT_TAG        ${MLX_KOMPUTE_TAG}
    GIT_SHALLOW    TRUE
  )
  
  # Kompute options
  set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(KOMPUTE_OPT_BUILD_SHADERS OFF CACHE BOOL "" FORCE)
  set(KOMPUTE_OPT_BUILD_SINGLE_HEADER OFF CACHE BOOL "" FORCE)
  set(KOMPUTE_OPT_USE_BUILT_IN_SPDLOG OFF CACHE BOOL "" FORCE)
  set(KOMPUTE_OPT_USE_BUILT_IN_FMT OFF CACHE BOOL "" FORCE)  # Use MLX's fmt
  set(KOMPUTE_OPT_LOG_LEVEL "Off" CACHE STRING "" FORCE)
  
  FetchContent_MakeAvailable(kompute)
  message(STATUS "Kompute fetched successfully")
endif()

if(MLX_BUILD_PYTHON_BINDINGS AND BUILD_SHARED_LIBS AND TARGET kompute)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(kompute PROPERTIES INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(kompute PROPERTIES INSTALL_RPATH "\$ORIGIN")
  endif()
  set_target_properties(kompute PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
endif()

# ============================================================================
# Shader Compilation
# ============================================================================
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

# Find glslc
find_program(GLSLC glslc)
if(NOT GLSLC)
  message(FATAL_ERROR "glslc not found. Please install Vulkan SDK.")
endif()

# Shader files
set(SHADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/binary_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_bf16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_rmsnorm_bf16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_bf16_scalar.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_bf16_bcast.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mul_bf16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mul_bf16_scalar.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mul_bf16_bcast.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lshift_u32_scalar.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rshift_u32_scalar.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/affine_dequantize_bf16_g128_b4.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gather_rows_words_i32_idx.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/silu_mul_bf16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sub_bf16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sub_bf16_scalar.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sub_f32.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sub_f32_scalar.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sin.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cos.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/argmax_f32_lastdim.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/argmax_bf16_lastdim.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/logsumexp_f32.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/logsumexp_bf16_row1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16_add_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24_add_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g32.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m4.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m8.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rmsnorm_bf16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rope_bf16_t1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rope_bf16_freqs.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_q1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_q1_d128.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_q1_d128_k32.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_q1_d128_k64.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_q1_d128_k128.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_splitk_stage1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_splitk_reduce.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_splitk_reduce_subgroup.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_splitk_reduce_l32.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_prefill_q1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_prefill_splitk_stage1.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_prefill_splitk_reduce.comp
)

# Shaders requiring subgroup ops must target Vulkan 1.1 so glslc emits a
# compatible SPIR-V version for subgroup instructions.
set(SUBGROUP_SHADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16_add_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24_add.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24_add_x2.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g32.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdpa_bf16_decode_splitk_reduce_subgroup.comp
)

# Compile shaders to SPIR-V
foreach(shader_src ${SHADER_FILES})
  get_filename_component(shader_name ${shader_src} NAME_WE)
  set(shader_spv ${SHADER_BIN_DIR}/${shader_name}.spv)
  set(shader_compile_args -fshader-stage=compute)
  if("${shader_src}" IN_LIST SUBGROUP_SHADER_FILES)
    list(APPEND shader_compile_args --target-env=vulkan1.1)
  endif()
  
  add_custom_command(
    OUTPUT ${shader_spv}
    COMMAND ${GLSLC} ${shader_compile_args} ${shader_src} -o ${shader_spv}
    DEPENDS ${shader_src}
    COMMENT "Compiling ${shader_name}.comp to SPIR-V"
  )
  
  list(APPEND SHADER_BINARIES ${shader_spv})
endforeach()

# Create shader target
add_custom_target(vulkan_shaders DEPENDS ${SHADER_BINARIES})

# ============================================================================
# Source Files
# ============================================================================
set(VULKAN_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../no_gpu/allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/buffer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_fallback.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_registry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/op_profiler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives/unary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives/fallback.cpp
)

target_sources(mlx PRIVATE ${VULKAN_SOURCES})

# ============================================================================
# Include Directories
# ============================================================================
target_include_directories(
  mlx 
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}
          ${kompute_SOURCE_DIR}/src/include
)

# ============================================================================
# Link Libraries
# ============================================================================
target_link_libraries(
  mlx 
  PRIVATE Vulkan::Vulkan
)

if(TARGET kompute::kompute)
  target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:kompute::kompute>)
elseif(TARGET kompute)
  target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:kompute>)
else()
  message(FATAL_ERROR "Kompute target not found after FetchContent_MakeAvailable")
endif()

# Add dependency on shader compilation
add_dependencies(mlx vulkan_shaders)

target_compile_definitions(mlx PRIVATE MLX_USE_VULKAN)

message(STATUS "Vulkan backend with Kompute configured successfully")
