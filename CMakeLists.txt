cmake_minimum_required(VERSION 3.25)
project(kestrel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KESTREL_SANITIZE "Enable sanitizers" OFF)
set(KESTREL_SANITIZERS "address,undefined" CACHE STRING "Clang/GCC sanitizer list")

add_library(kestrel_core
  src/tensor.cpp
  src/config.cpp
  src/safetensors.cpp
  src/backends/cpu.cpp
  src/profiler.cpp)
target_include_directories(kestrel_core PUBLIC include third_party)
target_compile_options(kestrel_core PRIVATE -Wall -Wextra -Wpedantic)

if(KESTREL_SANITIZE)
  target_compile_options(kestrel_core PUBLIC -fsanitize=${KESTREL_SANITIZERS} -fno-omit-frame-pointer)
  target_link_options(kestrel_core PUBLIC -fsanitize=${KESTREL_SANITIZERS})
endif()
