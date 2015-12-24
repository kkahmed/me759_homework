# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Include macros from the SBEL utils library
include(SBELUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to # 
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################


#enable_mpi_support()
enable_openmp_support()
enable_cuda_support()

SET(CUDA_NVCC_FLAGS -use_fast_math; --ptxas-options=-v; -gencode arch=compute_52,code=sm_52)

cuda_add_executable(vector_reduction vector_reduction.cu)
cuda_add_executable(thrust_reduction vector_reduction_thrust.cu)
cuda_add_executable(scan_sbel scan_largearray.cu scan_gold.cpp)
cuda_add_executable(thrust_scan scan_thrust.cu scan_gold.cpp)
cuda_add_executable(problem3 problem3.cu)
