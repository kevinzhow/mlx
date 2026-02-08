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
  
  FetchContent_MakeAvailable(kompute)
  message(STATUS "Kompute fetched successfully")
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
)

# Compile shaders to SPIR-V
foreach(shader_src ${SHADER_FILES})
  get_filename_component(shader_name ${shader_src} NAME_WE)
  set(shader_spv ${SHADER_BIN_DIR}/${shader_name}.spv)
  
  add_custom_command(
    OUTPUT ${shader_spv}
    COMMAND ${GLSLC} -fshader-stage=compute ${shader_src} -o ${shader_spv}
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
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives/binary.cpp
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
