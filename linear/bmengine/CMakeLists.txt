cmake_minimum_required(VERSION 3.10)

enable_language(C)
enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(bmengine)
