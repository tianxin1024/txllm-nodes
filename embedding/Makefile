NUM_JOBS = 8
CXX      = g++
PRO      = embedding

CMAKE_CMD = mkdir -p build && cd build && cmake ..

FLAGS = 
DEBUG_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Debug
RELEASE_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Release

all : 
	$(CMAKE_CMD) $(DEBUG_FLAGS) && make -s -j$(NUM_JOBS)

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CRAL-C to abort]" response && rm -rf build/*

.PHONY: all run clean
