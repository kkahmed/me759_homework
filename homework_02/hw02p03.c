/*
 * hw02p03.c
 *
 *  Created on: Sep 15, 2015
 *      Author: Kazi
 *  Usage:
 * 	Must provide the code with a command line argument which is
 * 	the file location of a list of integers, one per line.
 * 	Then it will read those to an array and sort the array.
 * 	The program displays time taken and compares to C's qsort.
 * 	Make sure to give the full path of the file when running.
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

/* 
 * Integer comparison for using C's sort, copy pasted from online
 * http://www.anyexample.com/programming/c/qsort__sorting_array_of_strings__integers_and_structs.xml
 * Not used for my own methods, just for testing C's functions
 */
int int_cmp(const void *a, const void *b) 
{ 
    const int *ia = (const int *)a; // casting pointer types 
    const int *ib = (const int *)b;
    return *ia  - *ib; 
	/* integer comparison: returns negative if b > a 
	and positive if a > b */ 
} 

/*
 * Recursive quickSort implementation. Determines a pivot number,
 * moves everything below that number earlier in the array, and
 * above that number later in the array, then calls itself for 
 * each half created. Full disclaimer: as explained in the readme,
 * I based it (and the split function) on pseudocode I found
 * in my notes from CS367 with Professor Skrentny.
 * 
 * Arguments:
 * 	int *arr: pointer to array that should be sorted
 *	int start: starting index within array
 * 	int end: ending index within array
 */
void quick2Sort(int *arr, int start, int end) {

	int swapLine;
	if (start < end) {
		swapLine = split(arr, start, end);
		//Recursive calls
		quick2Sort(arr, swapLine+1, end);
		quick2Sort(arr, start, swapLine-1);
	}
}

/* 
 * This function does the actual work for quick2Sort.
 * It takes item pointed to at the end location as the pivot,
 * then walks through the array and makes swaps in the original
 * array itself to put every item on the proper side of pivot.
 * 
 * Arguments:
 * 	int *arr: pointer to array that should be sorted
 *	int start: starting index within array
 * 	int end: ending index within array
 */
int split(int *arr, int start, int end) {
	int pivot = *(arr+end);
	int i = start;
	int j;
	int temp;
	//Step through the array
	for (j=start; j<end; j++) {
		//Make swaps here
		if (*(arr+j) < pivot) {
			temp = *(arr+j);
			*(arr+j) = *(arr+i);
			*(arr+i) = temp;
			i++;
		}
	}
	//Finally put the pivot in the right place
	temp = *(arr+end);
	*(arr+end) = *(arr+i);
	*(arr+i) = temp;
	return i;
}

/* 
 * Reads the specified file and tries to make an int array with it.
 * Uses a very efficient quick sort and compares to C's qsort, 
 * which ends up being a little slower but is much smarter.
 * Outputs ints read, min, max, and time to sort.
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
	int compArray[numLines]; //Copy to have C sort it
	int step;
	step = 0;
	while (fgets(buf, 10, pFile) != NULL) {
		readArray[step] = (int)strtol(buf, (char **)NULL, 10);
		compArray[step] = (int)strtol(buf, (char **)NULL, 10);
		step++;
	}

	//Make array pointer so array can be manipulated
	int *p;
	p = readArray;

	//Perform the sort and time it
	clock_t time;
	time = clock();
	quick2Sort(p, 0, numLines-1);
	time = clock() - time;

	//Now have C sort the array
	clock_t time2;
	time2 = clock();
	qsort(compArray, numLines, sizeof(int), int_cmp);
	time2 = clock() - time2;

	//For testing output some lines
	/*int i;
	for (i=0; i<20 ;i++){
		printf("%d\n", readArray[i]);
	}*/

	//And do other output
	printf("The number of integers read: %d\n", numLines);
	printf("The smallest integer read: %d\n", readArray[0]);
	printf("The largest integer read: %d\n", readArray[numLines-1]);
	printf("Assuming a clocks per sec value of: %d\n", CLOCKS_PER_SEC);
	printf("The calculation took %f seconds. \n", ((double)time)/CLOCKS_PER_SEC);
	printf("C sorted it in %f seconds. \n", ((double)time2)/CLOCKS_PER_SEC);

	fclose(pFile);
	return 0;

}





