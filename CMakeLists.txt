cmake_minimum_required(VERSION 3.16)

project(tlgconv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(tlgconv
    src/main.cpp
    src/png_io.cpp
    src/bmp_io.cpp
    src/tlg_io.cpp
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
