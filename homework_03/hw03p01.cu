/*
 * hw03p01.cu
 *
 *  Created on: Oct 02, 2015
 *      Author: Kazi
 *  Usage:
 * 	Basic CUDA program that does some math on a gpu and copies 
 * 	the data back over to the host. Make sure to compile with the
 * 	right parameters for the device as this code does not check
 * 	devices to determine capability or anything.
 */

#include <stdio.h>
#include <stdlib.h>

/* 
 * A simple kernel on the gpu that sets the current entry of an array 
 * equal to the sum of the threadId and the blockId. 
 */
__global__ void sumTIandBI(int* data, int size)
{
	int ti = threadIdx.x;
	int bi = blockIdx.x;
	int entry = ti + bi*blockDim.x;
	
	//Perform the trivial operation
	if (entry < size)
	{
		data[entry] = ti + bi;
	}

	//Just have the thread print from device so we can compare to the host
	printf("%d\n", data[entry]);
}

/*
 * Entry point for the program. Runs the simple kernel sumTIandBI on a gpu.
 * This allocates an array on the device to hold some information and then
 * it copies it back to a corresponding array on the host.
 */
int main(int argc, char* argv[])
{
	const int totSize = 16;
	const int threads = 8;
	const int blocks = 2;

	//Allocate memory on the host
	int *hostArray = (int *)malloc(sizeof(int)*totSize);

	//Allocate memory on the GPU
	int *gpuArray;
	cudaMalloc((void**) &gpuArray, sizeof(int)*totSize);

	//Call the gpu kernel, 2 blocks of 8 threads
	printf("The results on the device: \n");
	sumTIandBI <<<blocks,threads>>> (gpuArray, totSize);

	//Write back to the host
	cudaMemcpy(hostArray, gpuArray, sizeof(int)*totSize, cudaMemcpyDeviceToHost);

	//Output
	int i;
	printf("The results on the host: \n");
	for(i=0; i<totSize; i++)
	{
		printf("%d\n", hostArray[i]);
	}

	//Cleanup
	if(gpuArray) cudaFree(gpuArray);
	if(hostArray) free(hostArray);

	return 0;
}
