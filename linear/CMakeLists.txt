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
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)
set(CMAKE_BUILD_RPATH ${ORIGIN})
set(CMAKE_INSTALL_RPATH ${ORIGIN})
set(USE_STATIC_NCCL True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_CFLAGS "-O0 -w -g")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
message(STATUS PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
message(STATUS PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE PYTHON_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_CMAKE_PREFIX_PATH}/Torch")
message(STATUS "PYTORCH_CMAKE_PREFIX_PATH ${PYTHON_CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS ATEN_INCLUDE_DIR ${TORCH_INSTALL_PREFIX}/include)
message(STATUS ATEN_INCLUDE_DIR ${TORCH_PYTHON_LIBRARY})
set(ATEN_INCLUDE_DIR ${TORCH_INSTALL_PREFIX}/include)


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
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions" 
    "pthread"
    "${TORCH_LIBRARIES}"
    "${TORCH_PYTHON_LIBRARY}"
)

