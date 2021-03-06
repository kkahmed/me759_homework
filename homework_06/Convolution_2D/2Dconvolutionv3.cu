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

/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
// includes, project
#include "2Dconvolution.h"
#include "2Dconvolution_gold.cpp"

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void ConstCopyToDeviceMatrix(float* Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
bool ReadParams(int* params, int size, char* file_name);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);

__constant__ float cM[25];

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{
	//Set up the appropriate indices
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tidxy = tidx + tidy*blockDim.x;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int nx = tidx + bidx*blockDim.x;
	int ny = tidy + bidy*blockDim.y;

	//Hard coded to 16*16 + 2*16*4 + 16 to be large enough based on 16x16 blocks
	__shared__ float ns[400];

	//Set up the indices for the shared version of submatrix of N
	int bsx0 = (bidx < 1)? 0 : (bidx*blockDim.x-2);
	int xs0 = (bidx < 1)? 0 : 2;
	int bsx1 = (bidx = (gridDim.x-1))? (N.width-1) : ((bidx+1)*blockDim.x+2);
	int bsy0 = (bidy < 1)? 0 : (bidy*blockDim.y-2);
	int ys0 = (bidy < 1)? 0 : 2;
	int bsy1 = (bidy = (gridDim.y-1))? (N.height-1) : ((bidy+1)*blockDim.y+2);
	int bxdim = bsx1-bsx0+1;
	int bydim = bsy1-bsy0+1;
	int bsize = bxdim*bydim;

	//Create the shared memory array
	//I cannot explain this indexing with words, would need visual aid
	//It is some janky stuff I duct taped together
	if ((tidxy)<bsize)
	{
		int row = (tidxy)/bxdim;
		int ty = bsy0 + row;
		int tx = tidxy - row*bxdim + bsx0;
		ns[tidxy] = N.elements[tx + ty*N.width];
	}
	if (bsize>256 && (tidxy+256)<bsize) //16x16 hard coded here, could be changed
	{
		int row2 = (tidxy+256)/bxdim;
		int ty2 = bsy0 + row2;
		int tx2 = tidxy+256 - row2*bxdim + bsx0;
		ns[tidxy+256] = N.elements[tx2 + ty2*N.width];
	}
	__syncthreads();

   if (nx<(N.width) && ny<(N.height))
   {

      int x0 = (nx < 2)? (2-nx) : 0;
      int x1 = (nx > (N.width-3))? (N.width-nx+2) : 5;
      int y0 = (ny < 2)? (2-ny) : 0;
      int y1 = (ny > (N.height-3))? (N.height-ny+2) : 5;

      float sum = 0;

      for(int i=x0; i<x1; i++)
      {
         for(int j=y0; j<y1; j++)
         {
            //sum = sum + M.elements[i+j*M.width]*N.elements[(i+nx-2) + (ny+j-2)*N.width];
            sum = sum + M.elements[i+j*M.width]*ns[(tidx+i-xs0) + (tidy+j-ys0)*bxdim];
         }
      }
      P.elements[nx + ny*N.width] = sum;
   }

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	
	srand(2013);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
		//N  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		N =  AllocateMatrix(N_X, N_Y, 1); 
		//N_X and N_Y specified in header. Can be any positive int >= 16
		P  = AllocateMatrix(N.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*)malloc(2 * sizeof(int));
		unsigned int data_read = 2;
      	if(ReadParams(params, data_read, argv[1])){
         	printf("Error reading parameter file\n");
         	return 1;
      	}

		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		N  = AllocateMatrix(params[0], params[1], 0);		
		P  = AllocateMatrix(params[0], params[1], 0);
		(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[3]);
	}

	// M * N on the device
    ConvolutionOnDevice(M, N, P);
   
	//Start cpu timing here
	cudaEvent_t startCPU, stopCPU;
	cudaEventCreate(&startCPU);
	cudaEventCreate(&stopCPU);
	cudaEventRecord(startCPU, 0);

    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);

	//Stop cpu timing here
	cudaEventRecord(stopCPU, 0);
	cudaEventSynchronize(stopCPU);
	float cpuTime;
	cudaEventElapsedTime(&cpuTime, startCPU, stopCPU);
	cudaEventDestroy(startCPU);
	cudaEventDestroy(stopCPU);

	//Output timing
	printf("CPU time: %f ms. \n", cpuTime);

        
    // in this case check if the result is equivalent to the expected soluion

    bool res = CompareResults(reference.elements, P.elements, P.width * P.height, 0.01f);;
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	//Start inclusive timing here
	cudaEvent_t startIn, stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);

    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

	//Start exclusive timing here
	cudaEvent_t startEx, stopEx;
	cudaEventCreate(&startEx);
	cudaEventCreate(&stopEx);
	cudaEventRecord(startEx, 0);

    // Setup the execution configuration
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((N.width+dimBlock.x-1)/dimBlock.x,(N.height+dimBlock.y-1)/dimBlock.y);
 
    // Launch the device computation threads!
	ConvolutionKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd);

	//Stop exclusive timing here
	cudaEventRecord(stopEx, 0);
	cudaEventSynchronize(stopEx);
	float exTime;
	cudaEventElapsedTime(&exTime, startEx, stopEx);
	cudaEventDestroy(startEx);
	cudaEventDestroy(stopEx);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

	//Stop inclusive timing here
	cudaEventRecord(stopIn, 0);
	cudaEventSynchronize(stopIn);
	float inTime;
	cudaEventElapsedTime(&inTime, startIn, stopIn);
	cudaEventDestroy(startIn);
	cudaEventDestroy(stopIn);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);

	//Output timing
	printf("Inclusive time: %f ms. \n", inTime);
	printf("Exclusive time: %f ms. \n", exTime);

}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a host matrix to a constant device matrix
void ConstCopyToDeviceMatrix(float* Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    cudaMemcpyToSymbol(Mdevice, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps)
{
   for(unsigned int i = 0; i < elements; i++){
      float error = A[i]-B[i];
      if(error>eps){
         return false;
      } 
   }
   return true;
}

bool ReadParams(int* params, int size, char* file_name){
   ifstream ifile(file_name);
   int i=0;
   for(int i=0; i<size; i++){
      if(ifile.fail()==false){
         ifile>>params[i];
      }
   }
   return (i==size)? 1:0;

}


// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
   unsigned int data_read = M->height * M->width;
   std::ifstream ifile(file_name);

   for(unsigned int i = 0; i < data_read; i++){
      ifile>>M->elements[i];
   }
   ifile.close();
   return data_read;

}



// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
   std::ofstream ofile(file_name);
   for(unsigned int i = 0; i < M.width*M.height; i++){
      ofile<<M.elements[i];
   }
   ofile.close();
}

