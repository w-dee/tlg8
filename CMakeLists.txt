cmake_minimum_required(VERSION 3.16)

project(tlgconv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(NOT MSVC)
  foreach(flag_var CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE)
    if(DEFINED ${flag_var})
      string(REGEX REPLACE "-O[0-3]?[^ ]*" "" cleaned "${${flag_var}}")
      string(STRIP "${cleaned}" cleaned)
      set(${flag_var} "${cleaned} -O2" CACHE STRING "" FORCE)
    endif()
  endforeach()

  foreach(flag_var CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
    if(DEFINED ${flag_var})
      string(REGEX REPLACE "-O[0-3]?[^ ]*" "" cleaned "${${flag_var}}")
      string(STRIP "${cleaned}" cleaned)
      set(${flag_var} "${cleaned} -Og" CACHE STRING "" FORCE)
    endif()
  endforeach()
endif()

find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(tlgconv
    src/main.cpp
    src/png_io.cpp
    src/bmp_io.cpp
    src/tlg_io.cpp
    src/tlg_io_common.cpp
    src/tlg5_io.cpp
    src/tlg6_io.cpp
    src/tlg7_common.cpp
    src/tlg7_decoder.cpp
    src/tlg7_encoder.cpp
    src/tlg7_entropy_codec.cpp
)

target_include_directories(tlgconv PRIVATE src)

if(MSVC)
  target_compile_options(tlgconv PRIVATE /W4 /permissive-)
else()
  target_compile_options(tlgconv PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(tlgconv PRIVATE PNG::PNG ZLIB::ZLIB)

enable_testing()

add_test(
    NAME roundtrip
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/run_roundtrip.py
            --tlgconv $<TARGET_FILE:tlgconv>
            --images ${CMAKE_CURRENT_SOURCE_DIR}/test/images
)

set_tests_properties(roundtrip PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(roundtrip_tests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/run_roundtrip.py
            --tlgconv $<TARGET_FILE:tlgconv>
            --images ${CMAKE_CURRENT_SOURCE_DIR}/test/images
    DEPENDS tlgconv
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running tlgconv BMP round-trip tests"
    VERBATIM
)
