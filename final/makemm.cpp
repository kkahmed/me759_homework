/*
* Write a diffusion matrix in market format
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[])
{
	const int y = 11;
	const int x = 101;
	const int nodes = (y)*(x);
	const int nums = (y*(x-2))*3 + y*2;

	int i;
	int j;

	FILE *fp;
	fp = fopen("./bin/Amm.inp","w");
	fprintf(fp, "%%%%MatrixMarket matrix coordinate real general \n");
	fprintf(fp, "%%\n");
	fprintf(fp, "%d %d %d \n", nodes, nodes, nums);
	for (i=1; i<y+1; i++)
	{
		fprintf(fp, "%d %d %f \n", (x*(i-1)+1), (x*(i-1)+1), 1.0);
		for (j=1; j<x-1; j++)
		{
			fprintf(fp, "%d %d %f \n", x*(i-1)+1+j, x*(i-1)+j, -0.5);
			fprintf(fp, "%d %d %f \n", x*(i-1)+1+j, x*(i-1)+1+j, 1.0);	
			fprintf(fp, "%d %d %f \n", x*(i-1)+1+j, x*(i-1)+2+j, -0.5);		
		}
		fprintf(fp, "%d %d %f \n", (x*(i)), (x*(i)), 1.0);	
	}
	fclose(fp);

}
