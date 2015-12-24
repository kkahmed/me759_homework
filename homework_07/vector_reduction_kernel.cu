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

#ifndef _VECTOR_REDUCTION_KERNEL_H_
#define _VECTOR_REDUCTION_KERNEL_H_

#define NUM_ELEMENTS 50000000

__device__ void warpReduce(volatile double* s_data, int ti){
	s_data[ti] += s_data[ti+32];
	s_data[ti] += s_data[ti+16];
	s_data[ti] += s_data[ti+8];
	s_data[ti] += s_data[ti+4];
	s_data[ti] += s_data[ti+2];
	s_data[ti] += s_data[ti+1];	
}

// **===----------------- Modify this function ---------------------===**
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(double *g_data, int n)
{
	int ti = threadIdx.x;
	int bi = blockIdx.x + blockIdx.y*gridDim.x;
	int index = bi*blockDim.x + ti;

	//Each block is going to do calculation in shared memory
	__shared__ double s_data[512];
	if (index < n) {
		s_data[ti] = g_data[index];
	}
	else {
		s_data[ti] = 0;
	}

	//Make sure we're synched after data loaded
	__syncthreads();

	/*
	 * Disclaimer: I looked at literature online to help figure out how to do this
	 * Nvidia has some helpful examples, there are many algorithms to do this and
	 * I based this on one of them.
	 */ 

	int j;
	for (j=blockDim.x/2; j>32; j>>=1) //Bit shift to divide by two
	{
		/* This for loop collapes the calculation iteratively.
		 * Each run does a sum, so the number of elements is halved,
		 * hence the parameters of the for loop.
		 */ 
		if(ti<j)
		{
			//Set element in 1st half equal to sum with element in 2nd
			s_data[ti] += s_data[ti+j];
		}
		__syncthreads(); //Want all sums to be done before moving forward
	}
	
	if(ti<32) warpReduce(s_data,ti);
	__syncthreads();

	if(ti==0) g_data[bi] = s_data[0]; 
	//Write (0th, 511th, 1023rd, etc) result to (0th, 1st, 2nd, etc) entry in global list.


}

#endif // #ifndef _VECTOR_REDUCTION_KERNEL_H_
