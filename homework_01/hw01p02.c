/*
 * hw01p02.c
 *
 *  Created on: Sep 8, 2015
 *      Author: Kazi
 *
 *  Usage:
 *  	Asks for user ID as an input. Accepts long long.
 *  	Defaults to Kazi's user ID if entered incorrectly.
 *  	Output includes the first 3 digits.
 */

#include <stdio.h>
#include <inttypes.h>
#include <string.h>

int main() {

	long long llId;			//Will store the ID
	char cIDBuffer[4];		//Will store the digits to output
	char cOutBuffer[30];	//Full string to output
	char enter;				//Extra character for 'return'

	//Ask user for the input
	printf("Please enter your student ID then hit enter: ");
	fflush(stdout);

	/*
	 * Check if a number was entered.
	 * If not compatible with long long, assume Kazi's ID
	 * Note that if the input string starts with a number at all,
	 * even just 1 digit (e.g. 3asdf) it will use '3' in the output.
	 */
	if (scanf("%lld%c", &llId, &enter) != 2) {
		printf("Invalid input, assuming Kazi's ID \n");
		llId = 9066665317LL; //Assume Kazi's ID as default
	}

	//Build string of first 3 digits of the ID
	snprintf(cIDBuffer, 4, "%lld", llId);

	//Build the complete output string
	strcpy(cOutBuffer, "Hello! I am student ");
	strcat(cOutBuffer, cIDBuffer);
	strcat(cOutBuffer, ".");

	//Output
	printf(cOutBuffer);

	return 0;
}


