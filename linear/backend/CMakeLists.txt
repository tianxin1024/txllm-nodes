cmake_minimum_required(VERSION 3.10)
project(backend)

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

file(GLOB_RECURSE BACKEND_SOURCES
    "*.cpp"
    "*.cu"
)

file(GLOB_RECURSE FILES_BACKEND_HEADER RELATIVE "." "*.h")
message(FILES_BACKEND ${BACKEND_SOURCES})

add_library(backend SHARED ${BACKEND_SOURCES})

target_include_directories(backend
    PUBLIC FILES_BACKEND_HEADER
    PUBLIC ${Python_INCLUDE_DIRS}
    PUBLIC ${ATEN_INCLUDE_DIR})

target_link_libraries(backend
    PUBLIC bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
)
