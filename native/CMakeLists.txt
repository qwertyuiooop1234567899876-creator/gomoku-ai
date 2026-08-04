cmake_minimum_required(VERSION 3.16)
project(gomoku_native LANGUAGES CXX)

add_library(gomoku_native SHARED gomoku_native.cpp)
target_compile_features(gomoku_native PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(gomoku_native PRIVATE /O2 /EHsc /W4)
else()
    target_compile_options(gomoku_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

set_target_properties(gomoku_native PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(gomoku_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    )
endforeach()
