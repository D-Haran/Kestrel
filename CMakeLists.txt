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
  src/cache.cpp
  src/model.cpp
  src/generation.cpp
  src/sampling.cpp
  src/tokenizer.cpp
  src/profiler.cpp)
target_include_directories(kestrel_core PUBLIC include third_party)
target_compile_options(kestrel_core PRIVATE -Wall -Wextra -Wpedantic)

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
