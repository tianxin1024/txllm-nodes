cmake_minimum_required(VERSION 3.10)
project(backend)

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
    PUBLIC ${TORCH_INCLUDE_DIRS}
    PUBLIC ${ATEN_INCLUDE_DIR})

target_link_libraries(backend
    PUBLIC bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
    "stdc++fs"
    "${TORCH_LIBRARIES}"
)
