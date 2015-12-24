#include<iostream>
#include<stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_profiler_api.h>

#define N 100000000
#define RADIUS 3

int checkResults(int startElem, int endElem, float* cudaRes, float* res)
{
    int nDiffs=0;
    const float smallVal = 0.000001f;
    for(int i=startElem; i<endElem; i++)
        if(fabs(cudaRes[i]-res[i])>smallVal)
            nDiffs++;
    return nDiffs;
}

void initializeWeights(float* weights, int rad)
{
    // for now hardcoded for RADIUS=3
    weights[0] = 0.50f;
    weights[1] = 0.75f;
    weights[2] = 1.25f;
    weights[3] = 2.00f;
    weights[4] = 1.25f;
    weights[5] = 0.75f;
    weights[6] = 0.50f;
}

void initializeArray(float* arr, int nElements)
{
    const int myMinNumber = -5;
    const int myMaxNumber = 5;
    srand(time(NULL));
    for( int i=0; i<nElements; i++)
        arr[i] = (float)(rand() % (myMaxNumber - myMinNumber + 1) + myMinNumber);
}

void applyStencil1D_SEQ(int sIdx, int eIdx, const float *weights, float *in, float *out) {
  
  for (int i = sIdx; i < eIdx; i++) {   
    out[i] = 0;
    //loop over all elements in the stencil
    for (int j = -RADIUS; j <= RADIUS; j++) {
      out[i] += weights[j + RADIUS] * in[i + j]; 
    }
    out[i] = out[i] / (2 * RADIUS + 1);
  }
}

__global__ void applyStencil1D(int sIdx, int eIdx, const float *weights, float *in, float *out) {
    int i = sIdx + blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float sw[7]; //Shared memory for the weights
	__shared__ float ins[519]; //Shared memory for the list 
	//hard coded to 512+7 to avoid error about blockDim being unknown

	if(threadIdx.x < 7) {
		sw[threadIdx.x] = weights[threadIdx.x]; //First 7 threads load weights
		ins[threadIdx.x] = in[i-3]; //First 7 threads load 7 numbers
	}
	ins[threadIdx.x+6] = in[i+3]; //All threads help load 512 more numbers
	__syncthreads();

    if( i < eIdx ) {
        float result = 0.f;
	//Do the math using the data that is in shared memory
        result += sw[0]*ins[threadIdx.x];
        result += sw[1]*ins[threadIdx.x+1];
        result += sw[2]*ins[threadIdx.x+2];
        result += sw[3]*ins[threadIdx.x+3];
        result += sw[4]*ins[threadIdx.x+4];
        result += sw[5]*ins[threadIdx.x+5];
        result += sw[6]*ins[threadIdx.x+6];
        result /=7.f;
        out[i] = result;
    }
}

int main() {
  int size = N * sizeof(float); 
  int wsize = (2 * RADIUS + 1) * sizeof(float); 
  //allocate resources
  float *weights;
  cudaMallocHost(&weights, wsize);
  float *in;
  cudaMallocHost(&in, size);
  float *out;
  cudaMallocHost(&out, size);
  float *cuda_out;
  cudaMallocHost(&cuda_out, size); 
  initializeWeights(weights, RADIUS);
  initializeArray(in, N);

	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);
  float *d_weights;  cudaMalloc(&d_weights, wsize);
  float *d_in;       cudaMalloc(&d_in, size);
  float *d_out;      cudaMalloc(&d_out, size);
  
  cudaMemcpy(d_weights,weights,wsize,cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);
  applyStencil1D<<<(N+511)/512, 512>>>(RADIUS, N-RADIUS, d_weights, d_in, d_out);
	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);
  cudaMemcpy(cuda_out, d_out, size, cudaMemcpyDeviceToHost);
	//Stop inclusive timing here
	cudaEventRecord(stopIn, 0);
	cudaEventSynchronize(stopIn);
	float inTime;
	cudaEventElapsedTime(&inTime, startIn, stopIn);
	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

	//Start cpu timing here
	cudaEvent_t startCPU, stopCPU;
	cudaEventCreate(&startCPU);
	cudaEventCreate(&stopCPU);
	cudaEventRecord(startCPU, 0);
  applyStencil1D_SEQ(RADIUS, N-RADIUS, weights, in, out);
	//Stop cpu timing here
	cudaEventRecord(stopCPU, 0);
	cudaEventSynchronize(stopCPU);
	float cpuTime;
	cudaEventElapsedTime(&cpuTime, startCPU, stopCPU);
	cudaEventDestroy(startCPU);
	cudaEventDestroy(stopCPU);

	//Output timing
	printf("Inclusive time: %f ms. \n", inTime);
	printf("Exclusive time: %f ms. \n", exTime);
	printf("CPU time: %f ms. \n", cpuTime);

  int nDiffs = checkResults(RADIUS, N-RADIUS, cuda_out, out);
  nDiffs==0? std::cout<<"Looks good.\n": std::cout<<"Doesn't look good: " << nDiffs << "differences\n";

  //free resources
  cudaFree(weights); cudaFree(in); cudaFree(out); cudaFree(cuda_out);
  cudaFree(d_weights);  cudaFree(d_in);  cudaFree(d_out);
  return 0;
}	
