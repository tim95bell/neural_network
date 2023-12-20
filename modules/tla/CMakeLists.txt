
project(tla VERSION 0.0.0 LANGUAGES CXX)

set(source_files)

set(include_files
    include/tla/tla.hpp
)

add_library("${PROJECT_NAME}" INTERFACE)
target_include_directories("${PROJECT_NAME}" INTERFACE include)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${source_files} ${include_files})

if(APPLE)
    set_target_properties("${PROJECT_NAME}" PROPERTIES XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH[variant=Debug] YES)
endif()
