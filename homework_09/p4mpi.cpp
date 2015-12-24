/*
 * Kazi Ahmed 
 * HW 11 - p4
 * ME759 Fall 2015
 * 
 * Notes:
 * Didn't get cmake to work
 * Compiled with mpic++ -std=c++11 -Wall -O3
 * In slurm.sh, specify --nodes, and --ntasks-per-nodes
 * Then specify mpirun -np [x] ./p3mpi
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <chrono>

#include "vector_reduction_gold.cpp"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" void computeGold( double* reference, double* idata, const long long len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	if (argc > 2) {
		runTest( argc, argv);
	} else {
		printf("Not enough arguments \n");
		return 1;
	}
	
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char* argv[]) 
{
    long long num_elements;
	int max;
	
	num_elements = strtoll(argv[1],NULL,10);
	if(num_elements < 0) num_elements = (0 - num_elements);
	max = atoi(argv[2]);
	if(max < 0) max = (0 - max);
	
    const long long array_mem_size = sizeof(double) * num_elements;

    // allocate host memory to store the input data
    double* h_data = (double *)malloc(array_mem_size);

    // initialize the input data on the host to be float values
    // between -M and M
	for( long i = 0; i < num_elements; ++i) 
	{
		h_data[i] = 2.0*max*(rand()/(double)RAND_MAX) - max;
	}

    // compute reference solution
    double reference = 0.0;  
    computeGold(&reference , h_data, num_elements);

	//Start the parallel sum
	double parsum = 0.0;
	int rank;
	int p;
	long long int j;
	double sum;

	//Start mpireduce timing here
	auto r1 = std::chrono::high_resolution_clock::now();
	std::chrono::milliseconds rms;
	
	//Set up MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);

	MPI_Bcast(&num_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Each MPI process does some work
	sum = 0.0;
	for (j = rank; j<num_elements; j += p){
		sum += h_data[j];
	}

	//Sum the contributions from each MPI process
	MPI_Reduce(&sum, &parsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	MPI_Finalize();

	//Output result
	if (rank == 0) 
	{
		auto r2 = std::chrono::high_resolution_clock::now();

		rms = std::chrono::duration_cast<std::chrono::milliseconds>(r2-r1);
		printf("Processes: %d \n", p);
		printf("Sequential result: %.10f \n", reference);
		//std::cout << "Sequential timing: " << sms.count() << " ms \n";
		printf("Parallel result: %.10f \n", parsum);
		std::cout << "Parallel timing: " << rms.count() << " ms \n"; 
	}

    free(h_data);
}
