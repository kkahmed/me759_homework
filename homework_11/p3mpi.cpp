/*
 * Kazi Ahmed 
 * HW 11 - p3
 * ME759 Fall 2015
 * 
 * Notes:
 * Didn't get cmake to work
 * Compiled with mpicc++ -std=c++11 -Wall
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

// Compute the function
double evalf(double base, double x)
{
	double ans = pow(base, sin(x))*cos(x/(double)40.0);
	return ans;
}

int main( int argc, char** argv) 
{
	//e
	double base = 2.71828182845904523536;

	//If these are changed, make them consistent with eachother
	int n = 1000000;
	const double h = 0.0001;
	double xs[1000001];
	//double xp[1000001];
	double mult[4];

	double seqsum;
	double parsum;
	//Assume the initialization doesn't take long


	//Set the sequential intervals here
	//Should this be timed? 
	for (int i=0; i<1000001; i++){
		xs[i] = i*h;
	}
	mult[0] = 17.0; mult[1] = 59.0; mult[2] = 43.0; mult[3] = 49.0;

	//Start sequentional timing here
	auto s1 = std::chrono::high_resolution_clock::now();
	std::chrono::milliseconds sms;

	//Sum the unique terms
	seqsum = 0.0;
	seqsum += evalf(base,xs[0])*17.0;
	seqsum += evalf(base,xs[1])*59.0;
	seqsum += evalf(base,xs[2])*43.0;
	seqsum += evalf(base,xs[3])*49.0;
	seqsum += evalf(base,xs[n-3])*49.0;
	seqsum += evalf(base,xs[n-2])*43.0;
	seqsum += evalf(base,xs[n-1])*59.0;
	seqsum += evalf(base,xs[n])*17.0;
	seqsum = (h/(double)48.0)*seqsum;

	//Sum all the other terms
	for (int i=4; i<(n-3); i++){
		seqsum = seqsum + h*evalf(base,xs[i]);
	}
	auto s2 = std::chrono::high_resolution_clock::now();
	
	//Start parallel timing here
	auto p1 = std::chrono::high_resolution_clock::now();
	std::chrono::milliseconds pms;

	//Start the parallel sum
	parsum = 0.0;
	int rank;
	int p;
	int j;
	double sum;
	
	//Set up MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Each MPI process does some work
	sum = 0.0;
	for (j = rank + 4; j<(n-3); j += p){
		sum += evalf(base,xs[j]);
		if (j < 8){
			//The few extra terms are taken care of here
			sum += evalf(base,xs[j-4])*(mult[j-4]/(double)48.0);
			sum += evalf(base,xs[n-j+4])*(mult[j-4]/(double)48.0);
		}
	}
	sum = h*sum;

	//Sum the contributions from each MPI process
	MPI_Reduce(&sum, &parsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	MPI_Finalize();

	//Output result
	if (rank ==0) {
	auto p2 = std::chrono::high_resolution_clock::now();
	sms = std::chrono::duration_cast<std::chrono::milliseconds>(s2-s1);
	pms = std::chrono::duration_cast<std::chrono::milliseconds>(p2-p1);
	printf("Processes: %d \n", p);
	printf("Sequential result: %.15f \n", seqsum);
	std::cout << "Sequential timing: " << sms.count() << " ms \n";
	printf("Parallel result: %.15f \n", parsum);
	std::cout << "Parallel timing: " << pms.count() << " ms \n"; }

    return EXIT_SUCCESS;
}
