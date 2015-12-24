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
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#include "tiledMatMult.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel – thread specification
/*
 * Multiplies matrices assuming they are multiples of block size.
 * Note: this was based on the kernel provided in the lecture notes.
 */
__global__ void MatrixMulKernel(const Matrix M, const Matrix N, Matrix P)
{
	/* Illustrate (3x2)x(2x3) = (3x3)
	 *       000 }
	 *       000 } N.height
	 *
	 *   00  000
	 *   00  000
	 *   00  000
	 *   |________M.width
	 */

	//Get the block and thread IDs for convenience
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//Set up stepping horizontally across sublocks of M
	int Mtop = M.width*(BLOCK_SIZE*by);
	int Mend = Mtop + M.width - 1;
	int Mstep = BLOCK_SIZE;

	//Similarly for sublocks of N, vertically
	int Ntop = BLOCK_SIZE*bx;
	int Nstep = BLOCK_SIZE*N.width;

	//The accumulator for elements of P
	float sum = 0;

	//Load a sublock into shared memory
	__shared__ float Msub[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float Nsub[BLOCK_SIZE*BLOCK_SIZE];

	int n = Ntop;
	//For loop goes through each sublock of M
	for (int m=Mtop; m<=Mend; m+=Mstep)
	{
		//Each thread takes an element into shared memory
		Msub[ty*BLOCK_SIZE+tx] = M.elements[m + tx + M.width*ty];
		Nsub[ty*BLOCK_SIZE+tx] = N.elements[n + tx + N.width*ty];

		__syncthreads();

		for (int j=0; j<BLOCK_SIZE; j++)
		{
			//Contribute to the total sum 
			sum += Msub[ty*BLOCK_SIZE+j]*Nsub[j*BLOCK_SIZE+tx];
		}

		__syncthreads();

		n+=Nstep; //Simultaneously goes down sublocks of N
	}
	int c = bx*BLOCK_SIZE + (by*BLOCK_SIZE)*N.width; //Gets you to topleft of sublock
	P.elements[c + tx + ty*N.width] = sum; //Gets you to target element within sublock
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
