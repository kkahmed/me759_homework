#include<iostream>
#include<stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <cusp/io/matrix_market.h>
#include <cusp/precond/diagonal.h>

#define N 16


int main() {

	cusp::array2d<double, cusp::host_memory> matA(N,N);
	cusp::array1d<double, cusp::host_memory> vecb(N);

	//Initialize the array
    const int myMin = -5;
    const int myMax = 5;
    srand(11235);
    for( int i=0; i<N; i++)
	{
		for( int j=0; j<N; j++)
		{
        		matA(i,j) = (double)(rand()/((double)RAND_MAX) * (myMax-myMin) + myMin);
		}		
	}
	for( int j=0; j<N; j++)
	{
   		vecb[j] = (double)(rand()/((double)RAND_MAX) * (myMax-myMin) + myMin);
	}	

	//Write matA to MM format
	cusp::io::write_matrix_market_file(matA, "./bin/A.mtx");
    // load A from disk into a csr_matrix
    cusp::csr_matrix<int, double, cusp::device_memory> d_matB;
    cusp::io::read_matrix_market_file(d_matB, "./bin/A.mtx");

	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

	//cusp::array2d<double, cusp::device_memory> d_matA(matA);
    cusp::array1d<double, cusp::device_memory>   d_vecb(vecb);
	cusp::array1d<double, cusp::device_memory> x(N, 0);

        // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6)
        cusp::verbose_monitor<double> monitor(d_vecb, 10000, 1e-6);
        // setup preconditioner
        //cusp::precond::diagonal<double, cusp::device_memory> M(d_matB);
    cusp::krylov::cg(d_matB, x, d_vecb, monitor);

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);


	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

	//Stop inclusive timing here
	cudaEventRecord(stopIn, 0);
	cudaEventSynchronize(stopIn);
	float inTime;
	cudaEventElapsedTime(&inTime, startIn, stopIn);
	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

	// For verification, just to output small matrices to test
	//cusp::print(matA);

	//Output timing
	printf("Inclusive time: %f ms. \n", inTime);
	printf("Exclusive time: %f ms. \n", exTime);

  return 0;
}	
