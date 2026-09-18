cmake_minimum_required(VERSION 3.25)
project(kestrel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KESTREL_METAL "Build Apple Metal backend" OFF)
option(KESTREL_CUDA "Build CUDA backend" OFF)
option(KESTREL_SANITIZE "Enable sanitizers" OFF)
set(KESTREL_SANITIZERS "address,undefined" CACHE STRING "Clang/GCC sanitizer list")

add_library(kestrel_core
  src/tensor.cpp
  src/config.cpp
  src/safetensors.cpp
  src/backends/cpu.cpp
  src/cache.cpp
  src/model.cpp
  src/generation.cpp
  src/sampling.cpp
  src/tokenizer.cpp
  src/profiler.cpp)
target_include_directories(kestrel_core PUBLIC include third_party)
target_compile_options(kestrel_core PRIVATE -Wall -Wextra -Wpedantic)

if(KESTREL_SANITIZE AND KESTREL_CUDA)
  message(FATAL_ERROR "Use a CPU build for host sanitizers; use compute-sanitizer for CUDA")
endif()
if(KESTREL_SANITIZE)
  target_compile_options(kestrel_core PUBLIC -fsanitize=${KESTREL_SANITIZERS} -fno-omit-frame-pointer)
  target_link_options(kestrel_core PUBLIC -fsanitize=${KESTREL_SANITIZERS})
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_executable(kestrel_tests tests/test_main.cpp)
  target_link_libraries(kestrel_tests PRIVATE kestrel_core)
  add_test(NAME fixture COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/make_fixture.py ${CMAKE_BINARY_DIR}/fixture)
  set_tests_properties(fixture PROPERTIES FIXTURES_SETUP model_fixture)
  add_test(NAME unit COMMAND kestrel_tests ${CMAKE_BINARY_DIR}/fixture)
  set_tests_properties(unit PROPERTIES FIXTURES_REQUIRED model_fixture)
endif()

add_executable(kestrel src/main.cpp)
target_link_libraries(kestrel PRIVATE kestrel_core)
target_compile_definitions(kestrel PRIVATE KESTREL_TOKENIZER_HELPER="${CMAKE_SOURCE_DIR}/tools/tokenizer_bridge.py")

if(KESTREL_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "Metal requires macOS")
  endif()
  enable_language(OBJCXX)
  set_property(TARGET kestrel_core PROPERTY OBJCXX_STANDARD 20)
  file(READ ${CMAKE_SOURCE_DIR}/src/backends/kernels.metal METAL_SOURCE)
  file(WRITE ${CMAKE_BINARY_DIR}/generated/metal_source.hpp "#pragma once\ninline constexpr const char* kMetalSource = R\"METAL(${METAL_SOURCE})METAL\";\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/backends/kernels.metal)
  target_sources(kestrel_core PRIVATE src/backends/metal.mm)
  set_source_files_properties(src/backends/metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc;-Wall;-Wextra")
  target_include_directories(kestrel_core PRIVATE ${CMAKE_BINARY_DIR}/generated)
  target_compile_definitions(kestrel_core PUBLIC KESTREL_HAS_METAL)
  target_link_libraries(kestrel_core PRIVATE "-framework Metal" "-framework Foundation")
endif()

if(KESTREL_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12 REQUIRED)
  target_sources(kestrel_core PRIVATE src/backends/cuda.cu)
  set_target_properties(kestrel_core PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET kestrel_core PROPERTY CUDA_ARCHITECTURES 75 80 86 89)
  endif()
  target_compile_definitions(kestrel_core PUBLIC KESTREL_HAS_CUDA)
  target_compile_options(kestrel_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_link_libraries(kestrel_core PRIVATE CUDA::cudart)
endif()

if(BUILD_TESTING)
  foreach(BACKEND IN ITEMS metal cuda)
    string(TOUPPER ${BACKEND} UPPER_BACKEND)
    if(KESTREL_${UPPER_BACKEND})
      add_test(NAME unit_${BACKEND} COMMAND kestrel_tests ${CMAKE_BINARY_DIR}/fixture ${BACKEND})
      set_tests_properties(unit_${BACKEND} PROPERTIES FIXTURES_REQUIRED model_fixture)
    endif()
  endforeach()
endif()
