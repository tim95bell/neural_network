
project(tnn_logic_gates_example VERSION 0.0.0 LANGUAGES CXX)

set(
    source_files
    src/main.cpp
)

set(include_files)

add_executable("${PROJECT_NAME}" ${source_files})

target_link_libraries("${PROJECT_NAME}" PUBLIC tnn)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${source_files} ${header_files})
