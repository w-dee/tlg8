cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tlg8_fastpath_common STATIC
)

# いまのところ共通実装はヘッダオンリーなのでダミーソースを追加
set_source_files_properties(dummy_common.cpp PROPERTIES GENERATED TRUE)
add_custom_command(OUTPUT dummy_common.cpp COMMAND ${CMAKE_COMMAND} -E touch dummy_common.cpp)
target_sources(tlg8_fastpath_common PRIVATE dummy_common.cpp)

target_include_directories(tlg8_fastpath_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nlohmann
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/miniz
)

add_executable(tlg8_labelcache tlg8_labelcache.cpp)
target_link_libraries(tlg8_labelcache PRIVATE tlg8_fastpath_common)

add_executable(tlg8_featcache tlg8_featcache.cpp)
target_link_libraries(tlg8_featcache PRIVATE tlg8_fastpath_common)
