/*
 * hw02p04.c
 *
 *  Created on: Sep 15, 2015
 *      Author: Kazi
 *  Usage:
 * 	Must provide the code with a command line argument which is
 * 	the file location of a list of integers, one per line.
 * 	Then it will read those to an array and exclusive scan the array.
 * 	The program displays time taken.
 * 	Make sure to give the full path of the file when running.
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>


/* 
 * Performs an exclusive scan on the provided array.
 * It does this scan in place. 
 * Temporary variables keep track of values as terms in the array are reassigned.
 */
void scan(int *arr, int end) {
	int j;
	int temp;
	int temp2;
	temp = *arr;
	*arr = 0;
	//Step through the array
	for (j=1; j<=end; j++) {
		//Make additions here
		temp2 = *(arr+j);
		*(arr+j) = temp + *(arr+j-1);
		temp = temp2;
	}
}

/* 
 * Reads the specified file and tries to make an int array with it.
 * Steps through the array and performs the scan operation.
 * Outputs ints read, last entry, and time taken.
 */
int main(int argc, char *argv[]) {

	//Check for input
	if (argc == 1) {
		printf("You did not enter an input file \n");
		return 1;
	}

	//Try to open the file
	FILE *pFile;
	pFile = fopen(argv[1],"r");
	if (!pFile) {
		printf("File not found or inputted wrong \n");
		return 1;
	}

	//Get number of lines in file
	int numLines;
	numLines = 0;
	char buf[10];
	while (fgets(buf, 10, pFile) != NULL) {
		numLines++;
	}

	/* 
	 * Read each line and build int array
	 * Tried to be memory efficient doing it this way
	 * instead of using int readArray[1000000] every time
	 */
	pFile = fopen(argv[1],"r");
	int readArray[numLines]; //Array to be sorted
	int step;
	step = 0;
	while (fgets(buf, 10, pFile) != NULL) {
		readArray[step] = (int)strtol(buf, (char **)NULL, 10);
		step++;
	}

	//Make array pointer so array can be manipulated
	int *p;
	p = readArray;

	//Perform the scan and time it
	clock_t time;
	time = clock();
	scan(p, numLines-1);
	time = clock() - time;

	//For testing output some lines
	/*int i;
	for (i=0; i<numLines ;i++){
		printf("%d\n", readArray[i]);
	}*/

	//And do other output
	printf("The number of integers read: %d\n", numLines);
	printf("The last entry of the scanned array: %d\n", readArray[numLines-1]);
	printf("Assuming a clocks per sec value of: %d\n", CLOCKS_PER_SEC);
	printf("The calculation took %f seconds. \n", ((double)time)/CLOCKS_PER_SEC);

	fclose(pFile);
	return 0;

}
