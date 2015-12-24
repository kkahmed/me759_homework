#include<iostream>
#include<stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <cusp/detail/lu.h>
#include <cusp/lapack/lapack.h>

#define N 16

void initializeArray(double* arr, int nElements)
{
    const int myMin = -5;
    const int myMax = 5;
    srand(11235);

    for( int i=0; i<nElements; i++)
	{
		for( int j=0; j<nElements; j++)
		{
        		arr[(i*nElements) + j] = (double)(rand()/((double)RAND_MAX) * (myMax-myMin) + myMin);
		}		
	}
}

__global__ void LUdecomp(double *A, double *LU) {
   
	 int i = blockIdx.x*blockDim.x + threadIdx.x;
	LU[i] = A[i];

}

int main() {
  int size = N * sizeof(double); 

  double *matA;
  cudaMallocHost(&matA, size*N);

  double *matLU;
  cudaMallocHost(&matLU, size*N);

  initializeArray(matA, N);

	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	double *d_matA;  cudaMalloc(&d_matA, size*N);
	double *d_matLU; cudaMalloc(&d_matLU, size*N);

	cudaMemcpy(d_matA, matA, N*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matLU, matLU, N*size, cudaMemcpyHostToDevice);

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);

  LUdecomp<<<1,256>>>(d_matA, d_matLU);

	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

	cudaMemcpy(matLU, d_matLU, size*N, cudaMemcpyDeviceToHost);

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

	// For verification, just to output small matrices to test
	FILE *fpA, *fpL;
	fpA = fopen("./bin/matAd.inp","w");
	fpL = fopen("./bin/matLUd.inp","w");
    for( int i=0; i<N; i++)
	{
		for( int j=0; j<N; j++)
		{
			fprintf(fpA, "%f ", matA[(i*N) + j]);
			fprintf(fpL, "%f ", matLU[(i*N) + j]);
		}		
		fprintf(fpA, "\n");
		fprintf(fpL, "\n");
	}

  //free resources
  cudaFree(matLU); cudaFree(matA);
  cudaFree(d_matLU);  cudaFree(d_matA);
  return 0;
}	
