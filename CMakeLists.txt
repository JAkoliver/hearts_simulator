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

# Find Torch
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/libtorch)
find_package(Torch REQUIRED)

add_executable(HeartsSimulator main.cpp)
target_link_libraries(HeartsSimulator PRIVATE ftxui::screen ftxui::dom ftxui::component ${TORCH_LIBRARIES})
set_property(TARGET HeartsSimulator PROPERTY CXX_STANDARD 17)

# Decision-time search evaluation harness
add_executable(SearchEval search_eval.cpp)
target_link_libraries(SearchEval PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET SearchEval PROPERTY CXX_STANDARD 17)

# Expert-iteration self-play data generator
add_executable(SelfPlayGen selfplay_gen.cpp)
target_link_libraries(SelfPlayGen PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET SelfPlayGen PROPERTY CXX_STANDARD 17)

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Build the Python module (creates a .pyd on Windows or .so on Linux/Mac)
pybind11_add_module(hearts_env bindings.cpp)
