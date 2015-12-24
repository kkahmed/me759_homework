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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// You can use any other block size you wish.
#define BLOCK_SIZE 512
#define BLOCK_DUB 1024
#define DEFAULT_NUM_ELEMENTS 1024
#define MAX_RAND 2

typedef float REAL;

__global__ void prescan(REAL *odata, REAL *idata, int num)
{
	volatile __shared__ REAL temp[BLOCK_DUB];
	
	int ti = threadIdx.x;
	int ofs = 1;
	
	temp[ti] = idata[ti];
	temp[ti+blockDim.x] = idata[ti+blockDim.x];
	
	for (int i = num>>1; i>0; i>>=1)
	{
		__syncthreads();
		
		if (ti<i)
		{
			int ai = ofs*(2*ti+1)-1;
			int bi = ofs*(2*ti+2)-1;
			temp[bi] += temp[ai];
		}
		ofs <<= 1;
	}
	
	if (ti ==0) {
		temp[num-1] = 0;
	}
	
	for (int j = 1; j<num; j<<=1)
	{
		ofs >>= 1;
		__syncthreads();
		
		if (ti < j)
		{
			int ai = ofs*(2*ti+1)-1;
			int bi = ofs*(2*ti+2)-1;
			
			REAL temp2 = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += temp2;
		}
	}
	
	__syncthreads();
	
	odata[ti] = temp[ti];
	odata[ti+blockDim.x] = temp[ti+blockDim.x];
}

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls.
void prescanArray(REAL *outArray, REAL *inArray, int numElements)
{
	/*//Use kernel to compute the reduction
	int blocksx, blocksy, blocks;
	int threads = BLOCK_SIZE;
	int nestElements = num_elements;
	blocksx = (nestElements+511)/threads;
	blocks = blocksx;
	blocksy = 1;
	if (blocksx > 32768) {
		blocksy = (blocksx+32767)/32768;
		blocksx = 32768;
	}
	dim3 dimGrid(blocksx,blocksy);
	while(nestElements > 1)
	{
		// Recursive implementation to compute the reduction
		 
		reduction <<<dimGrid,threads>>> (d_data, nestElements);
		nestElements = blocks;
		blocksx = (nestElements+511)/threads;
		blocks = blocksx;
		blocksy = 1;
		if (blocksx > 32768) {
			blocksy = (blocksx+32767)/32768;
			blocksx = 32768;
		}
		dim3 dimGrid(blocksx, blocksy);
	}*/

	prescan <<<1,BLOCK_SIZE>>>(outArray, inArray, numElements);

}
// **===-----------------------------------------------------------===**



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" 
unsigned int compare( const REAL* reference, const REAL* data, 
                     const unsigned int len);
extern "C" 
void computeGold( REAL* reference, REAL* idata, const unsigned int len);

unsigned int cutComparef( REAL *reference, REAL *h_data, int num_elements, REAL err);

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
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    float device_time;
    float host_time;

    int num_elements = 0; // Must support large, non-power-of-2 arrays

    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( REAL) * num_elements;
    REAL* h_data = (REAL*) malloc( mem_size);

    switch(argc-1)
    {      
        case 0:
            num_elements = DEFAULT_NUM_ELEMENTS;
            // allocate host memory to store the input data
            mem_size = sizeof( REAL) * num_elements;
            h_data = (REAL*) malloc( mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND);
            }
            break;
        default:
            num_elements = atoi(argv[1]);
            
            // allocate host memory to store the input data
            mem_size = sizeof( REAL) * num_elements;
            h_data = (REAL*) malloc( mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND);
            }
        break;  
    }    

    cudaEvent_t time_start;
    cudaEvent_t time_end;

    cudaEventCreate(&time_start);
    cudaEventCreate(&time_end);
      
    // compute reference solution
    REAL* reference = (REAL*) malloc( mem_size);  
	// cutStartTimer(timer);
    cudaEventRecord(time_start, 0);
    computeGold( reference, h_data, num_elements);
    cudaEventRecord(time_end, 0);
    cudaEventSynchronize(time_end);
    cudaEventElapsedTime(&host_time, time_start, time_end);
	// cutStopTimer(timer);
    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Processing %d elements...\n", num_elements);

    printf("Host CPU Processing time: %f (ms)\n", host_time);


    // allocate device memory input and output arrays
    REAL* d_idata = NULL;
    REAL* d_odata = NULL;

    cudaMalloc( (void**) &d_idata, mem_size);
    cudaMalloc( (void**) &d_odata, mem_size);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
    // initialize all the other device arrays to be safe
    cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice);

    // **===-------- Allocate data structure here -----------===**
    // preallocBlockSums(num_elements);
    // **===-----------------------------------------------------------===**

    // Run just once to remove startup overhead for more accurate performance 
    // measurement
    prescanArray(d_odata, d_idata, 16);

    // Run the prescan
    // CUT_SAFE_CALL(cutCreateTimer(&timer));
    // cutStartTimer(timer);

    cudaEventRecord(time_start, 0);
    
    // **===-------- Modify the body of this function -----------===**
    prescanArray(d_odata, d_idata, num_elements);
    // **===-----------------------------------------------------------===**
    cudaThreadSynchronize();

    cudaEventRecord(time_end, 0);
    cudaEventSynchronize(time_end);

    cudaEventElapsedTime(&device_time, time_start, time_end);

    cudaEventDestroy(time_start);
    cudaEventDestroy(time_end);

    // cutStopTimer(timer);
    printf("CUDA Processing time: %g (ms)\n", device_time);
    // device_time = cutGetTimerValue(timer);
    // printf("Speedup: %fX\n", host_time/device_time);

    // **===-------- Deallocate data structure here -----------===**
    // deallocBlockSums();
    // **===-----------------------------------------------------------===**


    // copy result from device to host
    cudaMemcpy( h_data, d_odata, sizeof(REAL) * num_elements, 
                               cudaMemcpyDeviceToHost);

    // Check if the result is equivalent to the expected soluion
    unsigned int result_regtest = cutComparef( reference, h_data, num_elements, 1e-7);
    printf( "Test %s\n", (0 == result_regtest) ? "FAILED" : "PASSED");

    // cleanup memory
    free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
}

unsigned int cutComparef( REAL *reference, REAL *h_data, int num_elements, REAL err) {
    int i;
    int diff_count = 0;
    for (i = 0; i < num_elements; i++) {
        REAL diff = fabs(reference[i] - h_data[i]);
        REAL denominator = 1.f;
        if (denominator < fabs(reference[i])) {
            denominator = fabs(reference[i]);
        }
        if (!(diff / denominator < err)) {
            diff_count ++;
        }
    }
    if (diff_count > 0) {
        printf("Number of difference: %d\n", diff_count);
        return 0;
    } else {
        return 1;
    }
}
