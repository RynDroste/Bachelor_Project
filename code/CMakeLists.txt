cmake_minimum_required(VERSION 3.23)
project(BachelorProject LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect project sources under src/
file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Keep the primary app single-entry; auxiliary demos define their own main().
list(REMOVE_ITEM APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dam.cpp"
)

if(NOT APP_SOURCES)
    message(FATAL_ERROR "No source files found under ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c")
    message(FATAL_ERROR "GLAD source not found: external/glad/src/glad.c")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include/glad/glad.h")
    message(FATAL_ERROR "GLAD headers not found: external/glad/include/glad/glad.h")
endif()

add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/airy_cuda_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wave_decompose_gpu.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shallow_water_solver_gpu.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jw_transport_gpu.cu"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
# Do not default to "native": configuring without a GPU (or wrong GPU) yields a mismatched arch and
# Ampere/Ada then JITs older PTX → "PTX was compiled with an unsupported toolchain" on some setups.
# XX-real embeds cubin only for that SM (no forward-compat PTX), which avoids that JIT path.
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;80-real;75-real" CACHE STRING
        "CUDA targets, e.g. 86-real for RTX 30xx. Reconfigure if your GPU differs; update NVIDIA driver if errors persist.")
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    message(WARNING "CUDA: CMAKE_CUDA_ARCHITECTURES=native can pick the wrong SM if configure ran without your GPU; "
                    "use e.g. -DCMAKE_CUDA_ARCHITECTURES=86-real for RTX 30xx (Ampere).")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include"
)

find_package(OpenGL REQUIRED)

# cuFFT for Airy eWave (Algorithm 2); requires CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# GLFW: prefer system installation, fallback to bundled external/glfw
find_package(glfw3 CONFIG QUIET)
if(glfw3_FOUND)
    message(STATUS "Using system GLFW")
    set(GLFW_TARGET glfw)
else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/CMakeLists.txt")
        message(STATUS "System GLFW not found, using bundled external/glfw")
        add_subdirectory(external/glfw)
        set(GLFW_TARGET glfw)
    else()
        message(FATAL_ERROR "GLFW not found. Install system glfw3 or add it to external/glfw")
    endif()
endif()

# GLM: prefer system installation, fallback to bundled header-only external/glm
find_package(glm CONFIG QUIET)
if(glm_FOUND)
    message(STATUS "Using system GLM")
    set(GLM_TARGET glm::glm)
else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glm/glm/glm.hpp")
        message(STATUS "System GLM not found, using bundled external/glm")
        add_library(glm_local INTERFACE)
        target_include_directories(glm_local INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/external/glm")
        set(GLM_TARGET glm_local)
    else()
        message(FATAL_ERROR "GLM not found. Install system glm or add it to external/glm")
    endif()
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${GLFW_TARGET}
    ${GLM_TARGET}
    OpenGL::GL
    CUDA::cudart
    CUDA::cufft
)

# ---------------------------------------------------------------------------
# Pure SWE dam-break demo (no J&W split, no Airy CUDA)
# ---------------------------------------------------------------------------
add_executable(DamSWE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dam.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shallow_water_solver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shallow_water_solver_gpu.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c"
)

target_include_directories(DamSWE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include"
)

target_link_libraries(DamSWE PRIVATE
    ${GLFW_TARGET}
    ${GLM_TARGET}
    OpenGL::GL
    CUDA::cudart
)
