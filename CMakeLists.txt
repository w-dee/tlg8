cmake_minimum_required(VERSION 3.16)

project(tlgconv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)

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
