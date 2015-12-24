/*
 * Kazi Ahmed 
 * HW 10
 * ME759 Fall 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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
	const int n = 1000000;
	const double h = 0.0001;
	double xs[1000001];
	double xp[1000001];

	double seqsum;
	double parsum;
	//Assume the initialization doesn't take long


	//Set the sequential intervals here
	//Should this be timed? 
	for (int i=0; i<1000001; i++){
		xs[i] = i*h;
	}

	//Start sequentional timing here
	double sTime1 = omp_get_wtime();

	//Sum the unique terms
	seqsum = 0.0;
	seqsum += evalf(base,xs[0])*(17.0);
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
	double sTime2 = omp_get_wtime();
	
	//Start parallel timing here
	double pTime1 = omp_get_wtime();

	//Eh, just do this part in serial
	parsum = 0.0;
	parsum += evalf(base,xs[0])*(17.0);
	parsum += evalf(base,xs[1])*59.0;
	parsum += evalf(base,xs[2])*43.0;
	parsum += evalf(base,xs[3])*49.0;
	parsum += evalf(base,xs[n-3])*49.0;
	parsum += evalf(base,xs[n-2])*43.0;
	parsum += evalf(base,xs[n-1])*59.0;
	parsum += evalf(base,xs[n])*17.0;
	parsum = (h/(double)48.0)*parsum;

	int j;
	//Sum all the other terms
	#pragma omp parallel for reduction(+:parsum)
	for (int j=4; j<(n-3); j++){
		parsum += h*evalf(base,xs[j]);
	}

	double pTime2 = omp_get_wtime();

	//Output result
	printf("Sequential result: %.15f \n", seqsum);
	printf("Sequential timing: %.15f \n", (sTime2-sTime1));
	printf("Parallel result: %.15f \n", parsum);
	printf("Parallel timing: %.15f \n", (pTime2-pTime1));

    return EXIT_SUCCESS;
}
