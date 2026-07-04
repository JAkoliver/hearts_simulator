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

add_executable(HeartsSimulator main.cpp)
target_link_libraries(HeartsSimulator PRIVATE ftxui::screen ftxui::dom ftxui::component)
