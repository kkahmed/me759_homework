#include<iostream>
#include<stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 4096

/*
 * Barebones crout algorithm. Not much too this.
 * Adapted from "Crout matrix decomposition" from Wikipedia
 */
void crout(double const *A, double *L, double *U, int n) {
	int i, j, k;
	double sum = 0;

	for (i = 0; i < n; i++) {
		U[(i*n) + i] = 1;
	}

	for (j = 0; j < n; j++) {
		for (i = j; i < n; i++) {
			sum = 0;
			for (k = 0; k < j; k++) {
				sum = sum + L[(i*n) + k] * U[(k*n) + j];	
			}
			L[(i*n) + j] = A[(i*n) + j] - sum;
		}

		for (i = j; i < n; i++) {
			sum = 0;
			for(k = 0; k < j; k++) {
				sum = sum + L[(j*n) + k] * U[(k*n) + i];
			}
			if (L[(j*n) + j] == 0) {
				printf("Divide by zero error \n");
				exit(EXIT_FAILURE);
			}
			U[(j*n) + i] = (A[(j*n) + i] - sum) / L[(j*n) + j];
		}
	}
}

void initializeArray(double* arr, int nElements)
{
    const int myMin = -5;
    const int myMax = 5;
    srand(11235);

    for( int i=0; i<nElements; i++)
	{
		for( int j=0; j<nElements; j++)
		{
        		arr[(i*nElements) + j] = (double)(rand()/((double)RAND_MAX) * (myMax-myMin) + myMin);
		}		
	}
}

void initializeb(double* arr, int nElements)
{
    const int myMin = -5;
    const int myMax = 5;
    srand(11235);

    for( int i=0; i<nElements; i++)
	{
        	arr[i] = (double)(rand()/((double)RAND_MAX) * (myMax-myMin) + myMin);	
	}
}

//Given LU decomposition and an equation for LUx=b, solves it
void linsolve(double const *L, double const *U, double *b, double *x, int n)
{
	double *vecy = (double *)malloc(sizeof(double)*n);
	double sum;

	vecy[0] = b[0]/L[0];
	for (int i=1; i<n; i++)
	{
		sum = 0;
		for (int j=0; j<i; j++)
		{
			sum = L[(i*n+j)]*vecy[j] + sum;
		}
		vecy[i] = (1/L[(i*n+i)])*(b[i]-sum);
	}

	x[n-1] = vecy[n-1]/U[(n*n-1)];
	for (int i=n-2; i>=0; i--)
	{
		sum = 0;
		for (int j=i+1; j<=n-1; j++)
		{
			sum += U[(i*n+j)]*x[j];
		}

		x[i] = (1/U[(i*n+i)])*(vecy[i]-sum);
	}

	free(vecy);
}


/*
 * Initialize a random dense array and use crout's method to find the LU decomposition
 */
int main() {
  int size = N * sizeof(double); 

  double *matL = (double *)malloc(size*N); 
  double *matU = (double *)malloc(size*N); 
  double *matA = (double *)malloc(size*N); 

	double *vecb = (double *)malloc(size); 
	double *vecx = (double *)malloc(size); 

	//Set up random matrices
 	initializeArray(matA, N);
	initializeb(vecb,N);

	//Start timing
	clock_t time;
	time = clock();

	//Do the solve
	crout(matA, matL, matU, N);
	linsolve(matL,matU,vecb,vecx,N);

	time = clock() - time;

	printf("The calculation took %f seconds. \n", ((double)time)/CLOCKS_PER_SEC);

	
	// For verification, just to output small matrices to test
	/*FILE *fpA, *fpL, *fpU, *fpb, *fpx;
	fpA = fopen("./bin/matA.inp","w");
	fpL = fopen("./bin/matL.inp","w");
	fpU = fopen("./bin/matU.inp","w");
	fpb = fopen("./bin/vecb.inp","w");
	fpx = fopen("./bin/vecx.out","w");
    for( int i=0; i<N; i++)
	{
		for( int j=0; j<N; j++)
		{
			fprintf(fpA, "%f ", matA[(i*N) + j]);
			fprintf(fpL, "%f ", matL[(i*N) + j]);
			fprintf(fpU, "%f ", matU[(i*N) + j]);
		}		
		fprintf(fpA, "\n");
		fprintf(fpL, "\n");
		fprintf(fpU, "\n");
		fprintf(fpb, "%f \n", vecb[i]);
		fprintf(fpx, "%f \n", vecx[i]);
	}*/

	free(matA);
	free(matL);
	free(matU);
	free(vecb);
	free(vecx);
} 
