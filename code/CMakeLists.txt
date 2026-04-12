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

file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(REMOVE_ITEM APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/render/dam.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/transport.cpp"
)

if(NOT APP_SOURCES)
    message(FATAL_ERROR "No .cpp sources under ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c"
    OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include/glad/glad.h")
    message(FATAL_ERROR "GLAD missing: need external/glad/src/glad.c and external/glad/include/glad/glad.h")
endif()

# Dear ImGui: use external/imgui if you vendored it; otherwise fetch at configure time (needs Git + network once).
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui")
if(NOT EXISTS "${IMGUI_DIR}/imgui.cpp")
    include(FetchContent)
    FetchContent_Declare(
        imgui_fc
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(imgui_fc)
    if(NOT imgui_fc_POPULATED)
        message(STATUS "ImGui not in external/imgui — FetchContent downloading to build dir...")
        FetchContent_Populate(imgui_fc)
    endif()
    set(IMGUI_DIR "${imgui_fc_SOURCE_DIR}")
endif()

add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/airy_cuda_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/wave_decompose_gpu.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/shallow_water_solver_gpu.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/transport_gpu.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/pipeline_coupled_gpu.cu"
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Avoid default "native" arch (wrong SM when configuring off-GPU); XX-real embeds cubin for that SM only.
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;80-real;75-real" CACHE STRING
        "CUDA targets, e.g. 86-real for RTX 30xx. Reconfigure if your GPU differs.")
endif()
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    message(WARNING "CUDA: CMAKE_CUDA_ARCHITECTURES=native can pick the wrong SM; use e.g. -DCMAKE_CUDA_ARCHITECTURES=86-real for RTX 30xx.")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
    "${IMGUI_DIR}"
    "${IMGUI_DIR}/backends"
)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(glfw3 CONFIG QUIET)
if(glfw3_FOUND)
    set(GLFW_TARGET glfw)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/CMakeLists.txt")
    add_subdirectory(external/glfw)
    set(GLFW_TARGET glfw)
else()
    message(FATAL_ERROR "GLFW not found: install glfw3 or add external/glfw")
endif()

find_package(glm CONFIG QUIET)
if(glm_FOUND)
    set(GLM_TARGET glm::glm)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glm/glm/glm.hpp")
    add_library(glm_local INTERFACE)
    target_include_directories(glm_local INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/external/glm")
    set(GLM_TARGET glm_local)
else()
    message(FATAL_ERROR "GLM not found: install glm or add external/glm")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${GLFW_TARGET}
    ${GLM_TARGET}
    OpenGL::GL
    CUDA::cudart
    CUDA::cufft
)

add_executable(DamSWE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/render/dam.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/render/shader_file.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/shallow_water_solver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/gpu_terrain_h2d_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/shallow_water_solver_gpu.cu"
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

add_executable(DiffusionDispersionBench
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/diffusion_dispersion_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/wavedecomposer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/shallow_water_solver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/gpu_terrain_h2d_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_pipeline/wave_decompose_gpu.cu"
)
set_target_properties(DiffusionDispersionBench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(DiffusionDispersionBench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(DiffusionDispersionBench PRIVATE CUDA::cudart)

set(BP_SHADERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
string(REPLACE "\\" "/" BP_SHADERS_ROOT "${BP_SHADERS_ROOT}")
set(BP_SAND01_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/sand_01_1k")
string(REPLACE "\\" "/" BP_SAND01_ROOT "${BP_SAND01_ROOT}")
set(BP_TOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tools")
string(REPLACE "\\" "/" BP_TOOLS_ROOT "${BP_TOOLS_ROOT}")
set(BP_CAUSTIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/render")
string(REPLACE "\\" "/" BP_CAUSTIC_ROOT "${BP_CAUSTIC_ROOT}")
set(BP_SKYBOX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/skybox")
string(REPLACE "\\" "/" BP_SKYBOX_ROOT "${BP_SKYBOX_ROOT}")
foreach(_bp_target IN ITEMS ${PROJECT_NAME} DamSWE)
    target_compile_definitions(${_bp_target} PRIVATE "BP_SHADER_ROOT=\"${BP_SHADERS_ROOT}\"")
endforeach()
target_compile_definitions(${PROJECT_NAME} PRIVATE "BP_SAND01_ROOT=\"${BP_SAND01_ROOT}\"")
target_compile_definitions(${PROJECT_NAME} PRIVATE "BP_TOOLS_ROOT=\"${BP_TOOLS_ROOT}\"")
target_compile_definitions(${PROJECT_NAME} PRIVATE "BP_CAUSTIC_ROOT=\"${BP_CAUSTIC_ROOT}\"")
target_compile_definitions(${PROJECT_NAME} PRIVATE "BP_SKYBOX_ROOT=\"${BP_SKYBOX_ROOT}\"")

if(MSVC)
    foreach(_bp_target IN ITEMS ${PROJECT_NAME} DamSWE DiffusionDispersionBench)
        target_compile_options(
            ${_bp_target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        )
    endforeach()
endif()
