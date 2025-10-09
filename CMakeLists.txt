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
      string(REGEX REPLACE "-flto[^ ]*" "" cleaned "${cleaned}")
      string(REGEX REPLACE "-DNDEBUG" "" cleaned "${cleaned}")
      string(STRIP "${cleaned}" cleaned)
      set(${flag_var} "${cleaned} -O3 -DNDEBUG -flto" CACHE STRING "" FORCE)
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

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(TLG_PGO_MODE "off" CACHE STRING "PGO モード (off/generate/use)")
set_property(CACHE TLG_PGO_MODE PROPERTY STRINGS off generate use)
set(_tlg_pgo_modes off generate use)
set(_tlg_pgo_mode_lower "${TLG_PGO_MODE}")
string(TOLOWER "${_tlg_pgo_mode_lower}" _tlg_pgo_mode_lower)
if(NOT _tlg_pgo_mode_lower IN_LIST _tlg_pgo_modes)
  message(FATAL_ERROR "TLG_PGO_MODE は off/generate/use のいずれかを指定してください")
endif()
set(TLG_PGO_MODE "${_tlg_pgo_mode_lower}" CACHE STRING "PGO モード (off/generate/use)" FORCE)
unset(_tlg_pgo_modes)
unset(_tlg_pgo_mode_lower)

set(_tlg_default_pgo_dir "${CMAKE_BINARY_DIR}/pgo-profiles")
set(TLG_PGO_PROFILE_DIR "${_tlg_default_pgo_dir}" CACHE PATH "PGO プロファイルを配置するディレクトリ")
unset(_tlg_default_pgo_dir)

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
    src/tlg8_io.cpp
    src/tlg8_decode.cpp
    src/tlg8_color_filter.cpp
    src/tlg8_interleave.cpp
    src/tlg8_predictors.cpp
    src/tlg8_entropy.cpp
    src/tlg8_reorder.cpp
    src/tlg8_encode.cpp
)

target_include_directories(tlgconv PRIVATE src)

if(MSVC)
  target_compile_options(tlgconv PRIVATE /W4 /permissive-)
else()
  target_compile_options(tlgconv PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(tlgconv PRIVATE PNG::PNG ZLIB::ZLIB)

if(NOT MSVC)
  set(_tlg_pgo_compile_options)
  set(_tlg_pgo_link_options)
  if(TLG_PGO_MODE STREQUAL "generate")
    list(APPEND _tlg_pgo_compile_options "-fprofile-generate=${TLG_PGO_PROFILE_DIR}")
    list(APPEND _tlg_pgo_link_options "-fprofile-generate=${TLG_PGO_PROFILE_DIR}")
  elseif(TLG_PGO_MODE STREQUAL "use")
    list(APPEND _tlg_pgo_compile_options "-fprofile-use=${TLG_PGO_PROFILE_DIR}" "-fprofile-correction")
    list(APPEND _tlg_pgo_link_options "-fprofile-use=${TLG_PGO_PROFILE_DIR}")
  endif()
  if(_tlg_pgo_compile_options)
    # PGO に関するコンパイルオプション
    target_compile_options(tlgconv PRIVATE ${_tlg_pgo_compile_options})
  endif()
  if(_tlg_pgo_link_options)
    # PGO に関するリンクオプション
    target_link_options(tlgconv PRIVATE ${_tlg_pgo_link_options})
  endif()
  unset(_tlg_pgo_compile_options)
  unset(_tlg_pgo_link_options)
endif()

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
