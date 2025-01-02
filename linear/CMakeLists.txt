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

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
message(STATUS PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
message(STATUS PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
		print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

add_subdirectory(bmengine)
add_subdirectory(backend)

pybind11_add_module(llm_nodes layer_linear.cpp)

target_link_libraries(llm_nodes PRIVATE backend bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions" "pthread")

