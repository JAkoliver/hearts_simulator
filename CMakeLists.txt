cmake_minimum_required(VERSION 3.14)
project(HeartsSimulator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG v5.0.0
)
FetchContent_MakeAvailable(ftxui)

# Torch: wired manually instead of find_package(Torch). The CUDA-enabled
# prebuilt libtorch's TorchConfig.cmake demands a full CUDA Toolkit install at
# configure time, but we only ever LINK against the import libs libtorch
# ships - we compile no .cu code. Works for both CPU-only and CUDA libtorch
# drops (the CUDA bits are added only when torch_cuda.lib is present).
set(TORCH_ROOT ${CMAKE_SOURCE_DIR}/libtorch)
add_library(torch_manual INTERFACE)
target_include_directories(torch_manual INTERFACE
  ${TORCH_ROOT}/include
  ${TORCH_ROOT}/include/torch/csrc/api/include
  # CUDA runtime HEADERS only (vendored from pip nvidia-cuda-runtime-cu12
  # 12.6.77; no toolkit install needed) - required by ATen/cuda/CUDAGraph.h.
  # All CUDA symbols still come from libtorch's own DLLs.
  ${CMAKE_SOURCE_DIR}/third_party/cuda_include)
target_link_directories(torch_manual INTERFACE ${TORCH_ROOT}/lib)
target_link_libraries(torch_manual INTERFACE torch torch_cpu c10)
if(MSVC)
  # dll-interface noise from torch headers (find_package(Torch) suppressed
  # these too, via its own flags)
  target_compile_options(torch_manual INTERFACE /wd4251 /wd4275)
endif()
if(EXISTS ${TORCH_ROOT}/lib/torch_cuda.lib)
  target_link_libraries(torch_manual INTERFACE torch_cuda c10_cuda)
  # MSVC dead-strips the torch_cuda.dll reference (we call no symbol from it
  # directly); anchor one so CUDA registers at runtime. Documented in the
  # PyTorch Windows FAQ.
  target_link_options(torch_manual INTERFACE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

add_executable(HeartsSimulator main.cpp)
target_link_libraries(HeartsSimulator PRIVATE ftxui::screen ftxui::dom ftxui::component torch_manual)
set_property(TARGET HeartsSimulator PROPERTY CXX_STANDARD 17)

# Decision-time search evaluation harness
add_executable(SearchEval search_eval.cpp)
target_link_libraries(SearchEval PRIVATE torch_manual)
set_property(TARGET SearchEval PROPERTY CXX_STANDARD 17)

# Expert-iteration self-play data generator
add_executable(SelfPlayGen selfplay_gen.cpp)
target_link_libraries(SelfPlayGen PRIVATE torch_manual)
set_property(TARGET SelfPlayGen PROPERTY CXX_STANDARD 17)

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Build the Python module (creates a .pyd on Windows or .so on Linux/Mac)
pybind11_add_module(hearts_env bindings.cpp)
