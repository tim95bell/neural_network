
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_C_STANDARD 11)

project(tnn VERSION 0.0.0 LANGUAGES CXX)

option(TNN_DEBUG "Debug build" TRUE)

set(source_files)

set(include_files
    include/tnn/tnn.hpp
)

add_subdirectory(modules)

add_library("${PROJECT_NAME}" INTERFACE)
target_link_libraries("${PROJECT_NAME}" INTERFACE tla)
target_include_directories("${PROJECT_NAME}" INTERFACE include)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${source_files} ${include_files})

add_subdirectory(examples)

if(APPLE)
    set_target_properties("${PROJECT_NAME}" PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH[variant=Debug] YES)
endif()
