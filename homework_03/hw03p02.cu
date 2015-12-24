/*
 * hw03p02.cu
 *
 *  Created on: Oct 02, 2015
 *      Author: Kazi
 *  Usage:
 * 	
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>


/* 
 */
__global__ void sumArray(double* a, double* b, double* c, int size)
{
	int entry = threadIdx.x + blockIdx.x*blockDim.x;
	
	//Perform the addition
	if (entry < size)
	{
		c[entry] = a[entry] + b[entry];
	}
}

/*
 */
int main(int argc, char* argv[])
{
	//Set the size of the arrays, threads, blocks
	int length = (1 << 20); //Input exponent of 2 on the right
	int threads = 1024;
	int blocks = length/threads;
	printf("The length is: %d\n", length);

	//Allocate memory on the host
	double *hA = (double*)malloc(length*sizeof(double));
	double *hB = (double*)malloc(length*sizeof(double));
	double *hC = (double*)malloc(length*sizeof(double));
	double *refC = (double*)malloc(length*sizeof(double));

	//Allocate memory on the GPU
	double *dA, *dB, *dC;
	cudaMalloc((void**) &dA, sizeof(double)*length);
	cudaMalloc((void**) &dB, sizeof(double)*length);
	cudaMalloc((void**) &dC, sizeof(double)*length);

	//Generate the random arrays
	int i;
	double temp;
	for(i=0; i<length; i++)
	{
		temp = ((double)rand())/(RAND_MAX/20)-10;
		hA[i] = temp;
		temp = ((double)rand())/(RAND_MAX/20)-10;
		hB[i] = temp;
	}

	//Add those arrays together on the cpu, for a reference
	int j;
	for(j=0; j<length; j++)
	{
		//*(refC+j) = *(hA+j) + *(hB+j);
		refC[j] = hA[j] + hB[j];
		//printf("%lf\n", refC[j]);
	}

	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	//Copy hA,hB onto dA,dB
	cudaMemcpy(dA, hA, sizeof(double)*length, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double)*length, cudaMemcpyHostToDevice);

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);

	//Use kernel to sum the two arrays
	sumArray <<<blocks,threads>>> (dA, dB, dC, length);

	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

	//Copy dC back into hC
	cudaMemcpy(hC, dC, sizeof(double)*length, cudaMemcpyDeviceToHost);

	//Stop inclusive timing here
	cudaEventRecord(stopIn, 0);
	cudaEventSynchronize(stopIn);
	float inTime;
	cudaEventElapsedTime(&inTime, startIn, stopIn);
	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

	//For testing
	int k;
	for(k=0; k<length; k++)
	{
		//printf("%lf\n", hC[k]);
	}

	printf("Inclusive time: %f ms. \n", inTime);
	printf("Exclusive time: %f ms. \n", exTime);

	//Cleanup
	if(dA) cudaFree(dA);
	if(dB) cudaFree(dB);
	if(hA) free(hA);
	if(hB) free(hB);
	if(hC) free(hC);
	if(refC) free(refC);

	return 0;
}
