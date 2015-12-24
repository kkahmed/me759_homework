#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_profiler_api.h>

#define TRIALS 20

__global__ void kernel(int integer) 
{
	//Doesn't do anything
}

//Makes random array of floats
void initializeArray(float* arr, int nElements)
{
    srand(time(NULL));
    for( int i=0; i<nElements; i++)
        arr[i] = (float)(rand());
}

int main() 
{
	printf("Allocating memory and initializing arrays \n");

	//Allocate host memory
	float *hbytes = (float *)malloc(1073741824); //2^30
	initializeArray(hbytes, 268435456); //2^28

	//Allocate pinned host memory
	float *pbytes;	cudaMallocHost(&pbytes, 1073741824);
	initializeArray(pbytes, 268435456);

	//Allocate device memory
	float *dbytes;	cudaMalloc(&dbytes, 1073741824); 

	//Declare timing
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	float inTime, inTimeTotal;

	// These for loops do the required copy operations 
	// Results averaged over number of trials

	printf("\nCopy to device using host memory \n");
	printf("For 2^0 to 2^30, in order: \n");

	for (int i=0; i<31; i++)
	{
		inTimeTotal = 0.0;

		for (int j=0; j<TRIALS; j++)
		{
			cudaEventRecord(startIn, 0);

			cudaMemcpy(dbytes,hbytes,(1<<i),cudaMemcpyHostToDevice);

			cudaEventRecord(stopIn, 0);
			cudaEventSynchronize(stopIn);
			cudaEventElapsedTime(&inTime, startIn, stopIn);

			inTimeTotal = inTimeTotal + inTime;
		}

		inTimeTotal = inTimeTotal/TRIALS;
		printf("%f \n", inTimeTotal);
	}

	printf("\nCopy to device using pinned memory \n");
	printf("For 2^0 to 2^30, in order: \n");

	for (int i=0; i<31; i++)
	{
		inTimeTotal = 0.0;

		for (int j=0; j<TRIALS; j++)
		{
			cudaEventRecord(startIn, 0);

			cudaMemcpy(dbytes,pbytes,(1<<i),cudaMemcpyHostToDevice);

			cudaEventRecord(stopIn, 0);
			cudaEventSynchronize(stopIn);
			cudaEventElapsedTime(&inTime, startIn, stopIn);

			inTimeTotal = inTimeTotal + inTime;
		}

		inTimeTotal = inTimeTotal/TRIALS;
		printf("%f \n", inTimeTotal);
	}

	printf("\nCopy from device using host memory \n");
	printf("For 2^0 to 2^30, in order: \n");

	for (int i=0; i<31; i++)
	{
		inTimeTotal = 0.0;

		for (int j=0; j<TRIALS; j++)
		{
			cudaEventRecord(startIn, 0);

			cudaMemcpy(hbytes,dbytes,(1<<i),cudaMemcpyDeviceToHost);

			cudaEventRecord(stopIn, 0);
			cudaEventSynchronize(stopIn);
			cudaEventElapsedTime(&inTime, startIn, stopIn);

			inTimeTotal = inTimeTotal + inTime;
		}

		inTimeTotal = inTimeTotal/TRIALS;
		printf("%f \n", inTimeTotal);
	}

	printf("\nCopy from device using pinned memory \n");
	printf("For 2^0 to 2^30, in order: \n");

	for (int i=0; i<31; i++)
	{
		inTimeTotal = 0.0;

		for (int j=0; j<TRIALS; j++)
		{
			cudaEventRecord(startIn, 0);

			cudaMemcpy(pbytes,dbytes,(1<<i),cudaMemcpyDeviceToHost);

			cudaEventRecord(stopIn, 0);
			cudaEventSynchronize(stopIn);
			cudaEventElapsedTime(&inTime, startIn, stopIn);

			inTimeTotal = inTimeTotal + inTime;
		}

		inTimeTotal = inTimeTotal/TRIALS;
		printf("%f \n", inTimeTotal);
	}

	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

	free(hbytes);
	cudaFree(dbytes);
	cudaFree(pbytes);

	return 0;

}
