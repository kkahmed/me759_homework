#include <cusp/precond/diagonal.h>
#include <cusp/krylov/cg.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include <iostream>
#include <stdio.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef float ValueType;
typedef int 	IndexType;

int main(void)
{
	int i;
	int j;
	const int y = 101;
	const int x = 1001;
	//const int nodes = (y)*(x);
	//const int nums = (y*(x-2))*3 + y*2;

	float noTime;
	float diagTime;
	float aggTime;

    // create an empty sparse matrix structure (HYB format)
    cusp::csr_matrix<int, ValueType, MemorySpace> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "./bin/Amm.inp");

    // Note: A has poorly scaled rows & columns

    // solve without preconditioning
    {
        std::cout << "\nSolving with no preconditioner" << std::endl;
    
	//Start timing here
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);

        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, cusp::host_memory> b(A.num_rows, 0);
		cusp::array1d<ValueType, cusp::host_memory> xs(A.num_rows, 0);
		for (i=0;i<y;i++)
		{
			j = i*x;
			b[j] = 4000;
			xs[j] = 4000;
		}
        cusp::array1d<ValueType, MemorySpace> c(b);
        cusp::array1d<ValueType, MemorySpace> xd(xs);

        // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6)
        cusp::verbose_monitor<ValueType> monitor(b, 10000, 1e-6);
        
        // solve
        cusp::krylov::cg(A, xd, c, monitor);

	//Stop timing here
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&noTime, start1, stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
    }

    // solve with diagonal preconditioner
    {
        std::cout << "\nSolving with diagonal preconditioner (M = D^-1)" << std::endl;

    
	//Start timing here
	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
        
        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, cusp::host_memory> b(A.num_rows, 0);
		cusp::array1d<ValueType, cusp::host_memory> xs(A.num_rows, 0);
		for (i=0;i<y;i++)
		{
			j = i*x;
			b[j] = 4000;
			xs[j] = 4000;
		}
        cusp::array1d<ValueType, MemorySpace> c(b);
        cusp::array1d<ValueType, MemorySpace> xd(xs);

        // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6)
        cusp::verbose_monitor<ValueType> monitor(c, 10000, 1e-6);

        // setup preconditioner
        cusp::precond::diagonal<ValueType, MemorySpace> M(A);

        // solve
        cusp::krylov::cg(A, xd, c, monitor, M);

	//Stop cpu timing here
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&diagTime, start2, stop2);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);

		//cusp::array1d<ValueType, cusp::host_memory> xh(xd);
		/*for (i=0;i<x;i+=100)
		{
			std::cout << xd[i] << std::endl;
		}*/
    }

    // solve with aggregate preconditioner
   /* {
        std::cout << "\nSolving with smoothed aggregation preconditioner..." << std::endl;

    
	//Start timing here
	cudaEvent_t start3, stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3, 0);
        
        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, cusp::host_memory> b(A.num_rows, 0);
		cusp::array1d<ValueType, cusp::host_memory> xs(A.num_rows, 0);
		for (i=0;i<y;i++)
		{
			j = i*x;
			b[j] = 4000;
			xs[j] = 4000;
		}
        cusp::array1d<ValueType, MemorySpace> c(b);
        cusp::array1d<ValueType, MemorySpace> xd(xs);

        // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6)
        cusp::verbose_monitor<ValueType> monitor(c, 10000, 1e-6);

        // setup preconditioner
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);

        // solve
        cusp::krylov::cg(A, xd, c, monitor, M);

	//Stop cpu timing here
	cudaEventRecord(stop3, 0);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&aggTime, start3, stop3);
	cudaEventDestroy(start3);
	cudaEventDestroy(stop3);

		cusp::array1d<ValueType, cusp::host_memory> xh(xd);
		for (i=0;i<x;i+=100)
		{
			std::cout << xd[i] << std::endl;
		}


    }*/

	printf("No preconditioning time: %f ms. \n", noTime);
	printf("Diagonal preconditioning time: %f ms. \n", diagTime);
	//printf("Aggregate preconditioning time: %f ms. \n", aggTime);

    return 0;
}


