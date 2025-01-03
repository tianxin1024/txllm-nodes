cmake_minimum_required(VERSION 3.10)

project(backend)

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

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
		print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE SRC_LINEAR "*.cpp")

pybind11_add_module(llm_nodes ${SRC_LINEAR})

target_include_directories(llm_nodes PRIVATE ${CMAKE_SOURCE_DIR}/bmengine/bmengine/include)
target_include_directories(llm_nodes PRIVATE ${CMAKE_SOURCE_DIR}/backend)
target_include_directories(llm_nodes PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(llm_nodes PRIVATE bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions" 
    "pthread"
    "${TORCH_LIBRARIES}"
    "${TORCH_PYTHON_LIBRARY}"
)

