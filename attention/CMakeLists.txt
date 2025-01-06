cmake_minimum_required(VERSION 3.10)

project(llm_nodes)

enable_language(C)
enable_language(CXX)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89")
endif()
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_CFLAGS "-O0 -w -g")

add_subdirectory(pybind11)
add_subdirectory(bmengine)

file(GLOB_RECURSE BACKEND_SOURCES
    "backend/*.cpp"
    "backend/*.cu"
)

add_library(backend STATIC ${BACKEND_SOURCES})
set_property(TARGET backend PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(backend
    PUBLIC "pybind11/include"
    PUBLIC "bmengine/bmengine/include"
)

target_link_libraries(backend
    PUBLIC bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
)

pybind11_add_module(llm_nodes layer_feedforward.cpp)

target_link_libraries(llm_nodes PRIVATE backend bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions" "pthread")

