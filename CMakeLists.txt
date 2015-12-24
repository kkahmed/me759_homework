# Set the required version of CMake
cmake_minimum_required(VERSION 2.8)
# Set your project title
project(ME759_Homework)

#The following will but all binary files in a "bin" directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE "Release")

#Add subdirectories for each homework
#add_subdirectory(homework_01)
#add_subdirectory(homework_02)
#add_subdirectory(homework_03)
#add_subdirectory(homework_04)
#add_subdirectory(homework_05/ArrayReduction)
#add_subdirectory(homework_05/Tiled_matrixmul)
#add_subdirectory(homework_06/Convolution_2D)
#add_subdirectory(homework_06/Profiling1Dstencil)
#add_subdirectory(homework_07)
#add_subdirectory(homework_08)
#add_subdirectory(homework_09)
#add_subdirectory(homework_10)
#add_subdirectory(homework_11)
add_subdirectory(final)

# Include macros from the ParallelUtils library
#include(ParallelUtils.cmake)
#enable_cuda_support()

# Set CUDA flags
# SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_52,-code=sm_52")
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_52,code=sm_52)


