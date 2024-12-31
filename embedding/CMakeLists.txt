cmake_minimum_required(VERSION 3.10)

project(nodes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_CFLAGS "-O0 -w -g")

add_subdirectory(pybind11)
add_subdirectory(bmengine)
add_subdirectory(backend)

pybind11_add_module(llm_nodes layer_embedding.cpp)

target_link_libraries(llm_nodes PRIVATE bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions" "pthread")

