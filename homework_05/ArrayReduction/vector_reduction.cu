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

// includes, project
//#include <cutil.h>

// includes, kernels
#include "vector_reduction_kernel.cu"
#include "vector_reduction_gold.cpp"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        case 1:  // One Argument
            errorM = ReadFile(h_data, argv[1]);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be float values
            // between 0 and 10
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = 10*(rand()/(float)RAND_MAX);
            }
        break;  
    }
	
	//Start cpu timing here
	cudaEvent_t startCPU, stopCPU;
	cudaEventCreate(&startCPU);
	cudaEventCreate(&stopCPU);
	cudaEventRecord(startCPU, 0);
	
    // compute reference solution
    float reference = 0.0f;  
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
    float result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // Run accuracy test
    float epsilon = 0.0001f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    //printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}


int ReadFile(float* M, char* file_name)
{
	/* //Old ReadFile
	unsigned int elements_read = NUM_ELEMENTS;
	if (cutReadFilef(file_name, &M, &elements_read, true))
        return 1;
    else
        return 0;*/

	/* Needed to create a new ReadFile.
	 * Disclaimer: Thank you to nirvedhmeshram on the forum for suggesting an alternative.
	 * With his post I decided to take my fileIO code from hw04p02 and adapt it here.
 	 */
	FILE *fp;
	fp = fopen(file_name, "r");
	int i;
	for (i=0; i<NUM_ELEMENTS; i++)
	{
		fscanf(fp, "%f", &M[i]);
	}
	fclose(fp);
	return (i == NUM_ELEMENTS);
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimensions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{
	//Allocate memory on the device
	float *d_data;
	cudaMalloc((void**) &d_data, sizeof(float)*num_elements);

	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	//Copy onto the device
	cudaMemcpy(d_data, h_data, sizeof(float)*num_elements, cudaMemcpyHostToDevice);

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);

	//Use kernel to compute the reduction
	int blocksx, blocksy, blocks;
	int threads = 512;
	int nestElements = num_elements;
	blocksx = (nestElements+511)/threads;
	blocks = blocksx;
	blocksy = 1;
	if (blocksx > 65536) {
		blocksy = (blocksx+65535)/65536;
		blocksx = 65536;
	}
	dim3 dimGrid(blocksx,blocksy);
	while(nestElements > 1)
	{
		/* Quick naive implementation to deal with when NUM_ELEMENTS
		 * is greater than 1024. Just cuts in 2 each iteration after 1st. 
		 * Assumes starting value is a multiple of 1024.
		 * Would obviously need to be different if we can't make assumptions
		 * about the starting array size being a multiple of 1024.
		 * But for those, seems to behave logN in time
		 */
		reduction <<<dimGrid,threads>>> (d_data, nestElements);
		nestElements = blocks;
		blocksx = (nestElements+511)/threads;
		blocks = blocksx;
		blocksy = 1;
		if (blocksx > 65536) {
			blocksy = (blocksx+65535)/65536;
			blocksx = 65536;
		}
		dim3 dimGrid(blocksx, blocksy, 1);
	}

	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

	//Copy back to the device
	cudaMemcpy(h_data, d_data, sizeof(float)*num_elements, cudaMemcpyDeviceToHost);

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
  return h_data[0];

}
     
