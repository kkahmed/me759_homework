/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* This software and the information contained herein is PROPRIETARY and 
* CONFIDENTIAL to NVIDIA and is being provided under the terms and 
* conditions of a Non-Disclosure Agreement.  Any reproduction or 
* disclosure to any third party without the express written consent of 
* NVIDIA is prohibited.     
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

// includes, project
//#include <cutil.h>

// includes, kernels
#include "vector_reduction_kernel.cu"
#include "vector_reduction_gold.cpp"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name);
double computeOnDevice(double* h_data, long long array_mem_size);

extern "C" void computeGold( double* reference, double* idata, const long long len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char* argv[]) 
{
	if (argc > 2) {
		runTest( argc, argv);
	} else {
		printf("Not enough arguments \n");
		return 1;
	}
	
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char* argv[]) 
{
    long long num_elements;
	int max;
	
	cudaError_t error;
	
	num_elements = strtoll(argv[1],NULL,10);
	if(num_elements < 0) num_elements = (0 - num_elements);
	max = atoi(argv[2]);
	if(max < 0) max = (0 - max);
	
    const long long array_mem_size = sizeof(double) * num_elements;

    // allocate host memory to store the input data
    double* h_data; 
	error = cudaMallocHost(&h_data, array_mem_size);
	if(error != cudaSuccess)
	{
		printf("cudaMallocHost returned error code %d, line(%d) \n", error, __LINE__);
		printf("Array must be too large \n");
		exit(EXIT_FAILURE);
	}
        
    // initialize the input data on the host to be float values
    // between -M and M
	for( long i = 0; i < num_elements; ++i) 
	{
		h_data[i] = 2.0*max*(rand()/(double)RAND_MAX) - max;
	}
	
	//Start cpu timing here
	cudaEvent_t startCPU, stopCPU;
	cudaEventCreate(&startCPU);
	cudaEventCreate(&stopCPU);
	cudaEventRecord(startCPU, 0);
	
    // compute reference solution
    double reference = 0.0;  
    computeGold(&reference , h_data, num_elements);
	
	//Stop cpu timing here
	cudaEventRecord(stopCPU, 0);
	cudaEventSynchronize(stopCPU);
	float cpuTime;
	cudaEventElapsedTime(&cpuTime, startCPU, stopCPU);
	cudaEventDestroy(startCPU);
	cudaEventDestroy(stopCPU);
	
	printf("CPU time: %f ms. \n", cpuTime);
    
    // **===-------- Modify the body of this function -----------===**
    double result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // Run accuracy test
    //float epsilon = 0.0001f;
    //unsigned int result_regtest = (abs(result - reference) <= epsilon);
    //printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    cudaFree( h_data);
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimensions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
double computeOnDevice(double* h_data, long long num_elements)
{
	//Allocate memory on the device
	double *d_data;
	cudaError_t errord;
	errord = cudaMalloc((void**) &d_data, sizeof(double)*num_elements);
	if(errord != cudaSuccess)
	{
		printf("cudaMalloc returned error code %d, line(%d) \n", errord, __LINE__);
		printf("Array must be too large \n");
		exit(EXIT_FAILURE);
	}
	
	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	//Copy onto the device
	cudaMemcpy(d_data, h_data, sizeof(double)*num_elements, cudaMemcpyHostToDevice);

	//Set up pointers
	thrust::device_ptr<double> dev_ptr(d_data);

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);

	//Use kernel to compute the reduction
	double sum = thrust::reduce(dev_ptr, dev_ptr + num_elements, (double)0.0, thrust::plus<double>());

	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

	//Copy back to the device
	cudaMemcpy(h_data, d_data, sizeof(double)*num_elements, cudaMemcpyDeviceToHost);

	//Stop inclusive timing here
	cudaEventRecord(stopIn, 0);
	cudaEventSynchronize(stopIn);
	float inTime;
	cudaEventElapsedTime(&inTime, startIn, stopIn);
	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

	//Output timing
	printf("Inclusive time: %f ms. \n", inTime);
	printf("Exclusive time: %f ms. \n", exTime);

  // Return final result
  return sum;

}
     
