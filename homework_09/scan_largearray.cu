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
#define BLOCK_SIZE 256
#define DEFAULT_NUM_ELEMENTS 16777216
#define MAX_RAND 2


int LOG_BLOCK_SIZE;

void getLogBlockSize(int block_size) {
	for(LOG_BLOCK_SIZE = 1;LOG_BLOCK_SIZE < 31; LOG_BLOCK_SIZE++) {
		if((1<<LOG_BLOCK_SIZE) >= block_size)
			return;
	}
	fprintf(stderr, "The size requested might be too large!\n");
	exit(-1);
}


__global__ void kernel_reduction(float *inArray, int numElements, int stride, int numRest) {
	int tid = threadIdx.x;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int idx = tid + blockDim.x * bidx + blockDim.x * gridDim.x * bidy;

	__shared__ float idata[(BLOCK_SIZE << 1)+256];
	int copyIdx = stride * ((idx << 1) + 1) - 1;
	int copyToIdx = tid<<1;
	copyToIdx += (copyToIdx>>4);

	idata[copyToIdx] = inArray[copyIdx];
	idata[copyToIdx+1] = idata[copyToIdx] + inArray[copyIdx + stride];

	__syncthreads();

	int localStride = 2;

	for(numRest>>=1;numRest > 1; numRest >>= 1, localStride <<= 1) {
		if((tid<<1) < numRest) {
			int idxOne = (localStride << 1) * (tid + 1) - 1;
			int idxTwo = idxOne - localStride;
			idxOne += (idxOne >> 4);
			idxTwo += (idxTwo >> 4);
			idata[idxOne] += idata[idxTwo];
		}
		__syncthreads();
	}

	inArray[copyIdx] = idata[copyToIdx];
	inArray[copyIdx+stride] = idata[copyToIdx+1];
}

__global__ void kernel_downtraverse(float *inArray, int numElements, int startStride, int LOG_BLOCK_SIZE) {
	int tid = threadIdx.x;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int idx = tid + blockDim.x * bidx + blockDim.x * gridDim.x * bidy;
	int finalStride = (startStride >> LOG_BLOCK_SIZE);
	if(finalStride <= 0)
		finalStride = 1;
	if((startStride << 1) == numElements) {
		__shared__ float idata[(BLOCK_SIZE<<1)+256];
		int copyIdx = finalStride * ((idx << 1) + 1) - 1;
		int copyToIdx = (tid<<1);
		copyToIdx += (copyToIdx>>4);
		if(copyIdx < numElements){
			idata[copyToIdx] = inArray[copyIdx];
			idata[copyToIdx + 1] = inArray[copyIdx+finalStride];
		}
		__syncthreads();

		int localStride = blockDim.x;
		while(localStride >= 1) {
			int idxOne = (localStride << 1) * (tid + 1) - 1;
			if(idxOne < (blockDim.x<<1)) {
				int idxTwo = idxOne - localStride;
				idxOne += (idxOne>>4);
				idxTwo += (idxTwo>>4);
				float tmp = idata[idxOne] + idata[idxTwo];
				idata[idxTwo] = idata[idxOne];
				idata[idxOne] = tmp;
			}
			localStride >>= 1;
			__syncthreads();
		}

		if(copyIdx < numElements) {
			inArray[copyIdx] = idata[copyToIdx];
			inArray[copyIdx+finalStride] = idata[copyToIdx+1];
		}
	}
	else {
		int stride = startStride;
		int idxOne = (stride << 1) * (idx + 1) - 1;
		if(idxOne < numElements) {
			int idxTwo = idxOne - stride;
			float tmp = inArray[idxOne] + inArray[idxTwo];
			inArray[idxTwo] = inArray[idxOne];
			inArray[idxOne] = tmp;
		}
	}
}
// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls.
void prescanArray(float *inArray, int numElements)
{
	unsigned numRests = numElements;
	int stride = 1;

	while(numRests > 1) {
		unsigned threads = numRests / 2;
		unsigned gridX = 1, gridY = 1;
		if(threads > BLOCK_SIZE) {
			gridX = threads / BLOCK_SIZE;
			threads = BLOCK_SIZE;
			if(gridX > 32768) {
				gridY = gridX / 32768;
				gridX = 32768;
			}
		}
		dim3 grids(gridX, gridY);
		kernel_reduction<<<grids,threads>>>(inArray, numElements, stride, numRests > (2*BLOCK_SIZE)? (2*BLOCK_SIZE) : numRests);
		stride <<= (LOG_BLOCK_SIZE + 1);
		numRests >>= (LOG_BLOCK_SIZE + 1);
	}

	/*
	cudaMemcpy(tmpArray, inArray, 10*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i<10;i++)
		printf("%f\n", tmpArray[i]);
*/
	float tmpNum = 0.0f;
	cudaMemcpy(inArray + numElements - 1, &tmpNum, sizeof(float), cudaMemcpyHostToDevice);

	
	unsigned threads = BLOCK_SIZE;
	unsigned gridX = 1, gridY = 1;
	if(threads >= (numElements>>1)) {
		threads = (numElements>>1);
		dim3 grids(gridX, gridY);
		kernel_downtraverse<<<grids, threads>>>(inArray, numElements, threads, LOG_BLOCK_SIZE);
	}
	else {
		dim3 grids(gridX, gridY);
		kernel_downtraverse<<<grids, threads>>>(inArray, numElements, numElements>>1, LOG_BLOCK_SIZE);
		int stride = numElements >> (LOG_BLOCK_SIZE + 2);
		while(stride>0) {
			gridX <<= 1;
			if(gridX > 32768) {
				gridX >>= 1;
				gridY <<= 1;
			}
			dim3 grids2(gridX, gridY);
			kernel_downtraverse<<<grids2, threads>>>(inArray, numElements, stride, LOG_BLOCK_SIZE);
			stride>>=1;
		}
	}
}
// **===-----------------------------------------------------------===**



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" 
unsigned int compare( const float* reference, const float* data, 
                     const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);
unsigned getSmallestPower2(unsigned);
unsigned int cutComparef( float *reference, float *h_data, int num_elements, float err);

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
// Get the power of 2 which is the least of the all powers that are not smaller
// than the given number
////////////////////////////////////////////////////////////////////////////////
int getSmallestPower2(int num) {
	int result = 1;
	while(result < num && result > 0)
		result <<= 1;
	if(result <= 0 || num <= 0) {
		fprintf(stderr, "The size requested might be two large!\n");
		exit(-1);
	}
	return result;
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
	int compare_size = 0;

    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( float) * num_elements;
    float* h_data = NULL;

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Randomly generate input data and write the result to
    //   file name specified by first argument
    // * Two arguments: Read the first argument which indicates the size of the array,
    //   randomly generate input data and write the input data
    //   to the second argument. (for generating random input data)
    // * Three arguments: Read the first file which indicate the size of the array,
    //   then input data from the file name specified by 2nd argument and write the
    //   SCAN output to file name specified by the 3rd argument.
    switch(argc-1)
    {      
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            // Use DEFAULT_NUM_ELEMENTS num_elements
			if(argc <= 1)
				compare_size = num_elements = DEFAULT_NUM_ELEMENTS;
			else
				compare_size = num_elements = atoi(argv[1]);

			int tmp_size = num_elements;
			num_elements = getSmallestPower2(num_elements);
            
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            //h_data = (float*) malloc( mem_size);
			cudaMallocHost(&h_data, mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = 0.0f;
            }
            for( unsigned int i = 0; i < tmp_size; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND)*2 - 1;
            }
        break;  
    }    
	getLogBlockSize(BLOCK_SIZE);

    cudaEvent_t time_start;
    cudaEvent_t time_end;

    cudaEventCreate(&time_start);
    cudaEventCreate(&time_end);
    
    // compute reference solution
    float* reference = (float*) malloc( mem_size);  
    cudaEventRecord(time_start, 0);
    computeGold( reference, h_data, num_elements);
    cudaEventRecord(time_end, 0);
    cudaEventSynchronize(time_end);
    cudaEventElapsedTime(&host_time, time_start, time_end);
    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Processing %d elements...\n", num_elements);
    printf("Host CPU Processing time: %f (ms)\n", host_time);

    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;

    cudaMalloc( (void**) &d_idata, mem_size);
    cudaMalloc( (void**) &d_odata, mem_size);
    

    // **===-------- Allocate data structure here -----------===**
    // preallocBlockSums(num_elements);
    // **===-----------------------------------------------------------===**

    // Run just once to remove startup overhead for more accurate performance 
    // measurement
    prescanArray(d_idata, 16);

    // Run the prescan
    
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
    cudaEventRecord(time_start, 0);
    // **===-------- Modify the body of this function -----------===**
    prescanArray(d_idata, num_elements);
    // **===-----------------------------------------------------------===**
    cudaThreadSynchronize();

    cudaEventRecord(time_end, 0);
    cudaEventSynchronize(time_end);
    cudaEventElapsedTime(&device_time, time_start, time_end);

    // copy result from device to host
    cudaMemcpy( h_data, d_idata, sizeof(float) * compare_size, 
                               cudaMemcpyDeviceToHost);



    printf("CUDA Processing time: %f (ms)\n", device_time);
    printf("Speedup: %fX\n", host_time/device_time);

    // **===-------- Deallocate data structure here -----------===**
    // deallocBlockSums();
    // **===-----------------------------------------------------------===**

    // Check if the result is equivalent to the expected soluion
    unsigned int result_regtest = cutComparef( reference, h_data, compare_size, 1e-6);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
	

    // cleanup memory
	cudaFreeHost(h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
	printf("------------------------------------------------------\n\n");
}

unsigned int cutComparef( float *reference, float *h_data, int num_elements, float err) {
    int i;
    int diff_count = 0;
    for (i = 0; i < num_elements; i++) {
        float diff = fabs(reference[i] - h_data[i]);
        float denominator = 1.f;
        if (denominator < fabs(reference[i])) {
            denominator = fabs(reference[i]);
        }
        if (i % 1000000 == 0) {
            //printf("Diff at %d: %g %g\n", i, diff, diff / denominator);
        }
        if (!(diff / denominator < err)) {
            //printf("Diff at %d: %g %g\n", i, diff, diff / denominator);
            getchar();
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
