/*
 * hw02p02.c
 *
 *  Created on: Sep 15, 2015
 *      Author: Kazi
 *
 *  Usage:
 *  	User must provide string inputs. May or may not include quotes.
 *  	Takes everything provided and counts non-space characters.
 *  	Outputs the number of characters in the string(s).
 */

#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {

	int iCount;		//Will store number of characters
	iCount = 0;		//Final result value to be incremented

	int j;
	//So 0 is returned if nothing provided
	if (argc > 1) {
		//Step through all the arguments and send to counter
		for (j=1; argv[j]; j++) {
			iCount = iCount + counter(argv[j]); //Add results to iCount
		}
	}

	//Output the result
	printf("The number of characters is: %d\n", iCount);
	return 0;
}

/*
 * Counts the characters in a string starting from the beginning.
 * Does not count spaces. Continues until end of string.
 */
int counter(char cIn[]) {

	int i;
	int iCharCount = 0;

	//Step through the string and increment the count until finished
	for (i=0; cIn[i]; i++) {
		//Ignore spaces
		if (cIn[i] != ' ') {
			iCharCount++;
		}
	}

	return iCharCount;
}

