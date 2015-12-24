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

/* Matrix addition: P = alpha*M + beta*N.
 * Device code.
 */

#ifndef _MATRIXADD_KERNEL_H_
#define _MATRIXADD_KERNEL_H_

#include <stdio.h>
#include "matrixadd.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix addition kernel thread specification                                                      
__global__ void MatrixAddKernel(const float* Melems, const float alpha, const float* Nelems, const float beta, float* Pelems, int size)
{
	/* This value of entry is specific to how I decided to nodalize the matrix
	 * It looks kind of like this format: (blockx, blocky, blockz)[thread]
	 * (0,0,0)[0 .. 1024] (0,1,0)[0 .. 1024] (0,2,0)[0 .. 1024] (0,3,0)[0 .. 1024]
	 * (1,0,0)[0 .. 1024] (1,1,0)[0 .. 1024] (1,2,0)[0 .. 1024] (1,3,0)[0 .. 1024]
	 * ...
	 * (4096,0,0)[0 .. 1024] (4096,1,0)[0 .. 1024] (4096,2,0)[0 .. 1024] (4096,3,0)[0 .. 1024]
	 
	//int entry = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.x*gridDim.y*blockDim.x;
	int entry = threadIdx.x + blockIdx.y*blockDim.x + blockIdx.x*gridDim.y*blockDim.x;
	if(entry < size){
		Pelems[entry] = alpha*Melems[entry] + beta*Nelems[entry];
	}*/
	/* This value of entry is specific to how I decided to nodalize the matrix
	 * It looks kind of like this format: (blockx, blocky, blockz)[thread]
	 * (0,0,0)[0 .. 64] (1,0,0)[0 .. 64] .. (62,0,0)[0 .. 64] (63,0,0)[0 .. 64]
	 * (0,1,0)[0 .. 64] (1,1,0)[0 .. 64] .. (62,1,0)[0 .. 64] (63,1,0)[0 .. 64]
	 * ...
	 * (0,4095,0)[0 .. 64] (1,4095,0)[0 .. 64] .. (62,4095,0)[0 .. 64] (63,4095,0)[0 .. 64]
	 */
	int entry = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
	int i;
	if(entry < size){
		for (i=0; i<100000; i++){
			Pelems[entry] = alpha*Melems[entry] + beta*Nelems[entry];
		}
	}
}

#endif // #ifndef _MATRIXADD_KERNEL_H_
