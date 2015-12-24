/*
 * hw04p02.cu
 *
 *  Created on: Oct 04, 2015
 *      Author: Kazi
 *  Usage:
 * 	It performs integer multiplication of a 16x32 matrix with a 32x1 vector
 *	on a GPU. Does not take any arguments. Just generates predefined matrices
 *	and reports the time taken to do the multiplication.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <iostream>


/* 
 * Given an array A of size (h x w) and a vector b of size (w), it takes the product
 * Ab and writes it to a vector c of size (h).
 */
__global__ void multArray(int* A, int* b, int* c, int w, int size)
{
	int ti = threadIdx.x;

	int dsum = 0; //The sum for the dot product
	int k;
	//Perform the multiplication
	if (ti < size)
	{
		for(k=0; k<w; k++)
		{
			//Take the dot of a row of A with b
			dsum = dsum + A[ti*w+k]*b[k];
		}
		c[ti] = dsum;
	}
}

/*
 * Entry point for the program. Currently specifies matrix and vector size.
 * Allocates memory on the host and device and then creates matrices on the
 * host. Copies them over to the device to multiply them. Copies the result
 * back over to the host.
 */
int main(int argc, char* argv[])
{
	//Set the size of the arrays, threads, blocks
	int height = 16;
	int width = 32;
	int threads = height;
	int blocks = 1;

	//Allocate memory on the host
	int *hA = (int*)malloc(width*height*sizeof(int));
	int *hb = (int*)malloc(width*sizeof(int));
	int *hc = (int*)malloc(height*sizeof(int));

	//Allocate memory on the device
	int *dA, *db, *dc;
	cudaMalloc((void**) &dA, sizeof(int)*width*height);
	cudaMalloc((void**) &db, sizeof(int)*width);
	cudaMalloc((void**) &dc, sizeof(int)*height);

	//Generate the matrices on the host
	int i;
	int j;
	for(i=0; i<height; i++)
	{
		hc[i] = 0; //Let the storage array be all zeros
		for(j=0; j<width; j++)
		{
			hA[i*width + j] = i+j; //Set the matrix A
			if (i == 0) hb[j] = j; //Set the vector b
		}
	}

	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	//Copy hA,hb, hc onto dA,db, dc
	cudaMemcpy(dA, hA, sizeof(int)*width*height, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, sizeof(int)*width, cudaMemcpyHostToDevice);
	cudaMemcpy(dc, hc, sizeof(int)*width, cudaMemcpyHostToDevice);

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);

	//Use kernel to multiply A and b
	multArray <<<blocks,threads>>> (dA, db, dc, width, width*height);

	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

	//Copy dc back into hc
	cudaMemcpy(hc, dc, sizeof(int)*height, cudaMemcpyDeviceToHost);

	//Stop inclusive timing here
	cudaEventRecord(stopIn, 0);
	cudaEventSynchronize(stopIn);
	float inTime;
	cudaEventElapsedTime(&inTime, startIn, stopIn);
	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

	//For testing - to see what the result vector looks like
	for(j=0; j<height; j++)
	{
		//printf("%d\n", hc[j]);
		std::cout << j << ": " << hc[j] << std::endl;
	}

	//Output timing
	printf("Inclusive time: %f ms. \n", inTime);
	printf("Exclusive time: %f ms. \n", exTime);

	//Get device properties
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device name: %s \n", deviceProp.name);
	printf("Clock rate: %d \n", deviceProp.clockRate);
	printf("Multiprocessors: %d \n", deviceProp.multiProcessorCount);
	printf("L2 Cache: %d \n", deviceProp.l2CacheSize);
	printf("Max threads per MP: %d \n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Warp size: %d \n", deviceProp.warpSize);
	printf("Max threads per block: %d \n", deviceProp.maxThreadsPerBlock);
	printf("Max registers per block: %d \n", deviceProp.regsPerBlock);
	printf("Max blocks per MP: 32 \n"); //From table
	printf("Max warps per MP: 64 \n"); //From table
	printf("Shared memory per block (B): %d \n", deviceProp.sharedMemPerBlock);
	printf("Compute capability: %d.%d.\n", deviceProp.major, deviceProp.minor);

	//Write to file
	FILE *fp;
	fp = fopen("./problem2.out","w");
	fprintf(fp, "Results of hw04p02.cu: \n");
	for (i=0; i<height; i++)
	{
		fprintf(fp, "%d\n", hc[i]);
	}
	fclose(fp);

	//Cleanup
	if(dA) cudaFree(dA);
	if(db) cudaFree(db);
	if(dc) cudaFree(dc);
	if(hA) free(hA);
	if(hb) free(hb);
	if(hc) free(hc);

	return 0;
}
