/*
 * Kazi Ahmed 
 * Problem 3 - HW9
 * ME759 Fall 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>


int main( int argc, char** argv) 
{
	//day 	0  0  1  2  5  5  6  6  7  8  9  9  9  10 11
	thrust::host_vector<int> hday(15,0);
	//site 	2  3  0  1  1  2  0  1  2  1  3  4  0  1  2
	thrust::host_vector<int> hsit(15,0);
	//msrmnt	9  5  6  3  3  8  2  6  5  10 9  11 8  4  1
	thrust::host_vector<int> hmsr(15,0);
	 hday[ 0]= 0;  hday[ 1]= 0;  hday[ 2]= 1;
	 hday[ 3]= 2;  hday[ 4]= 5;  hday[ 5]= 5;
	 hday[ 6]= 6;  hday[ 7]= 6;  hday[ 8]= 7;
	 hday[ 9]= 8;  hday[10]= 9;  hday[11]= 9;
	 hday[12]= 9;  hday[13]=10;  hday[14]=11;
	 hsit[ 0]= 2;  hsit[ 1]= 3;  hsit[ 2]= 0;
	 hsit[ 3]= 1;  hsit[ 4]= 1;  hsit[ 5]= 2;
	 hsit[ 6]= 0;  hsit[ 7]= 1;  hsit[ 8]= 2;
	 hsit[ 9]= 1;  hsit[10]= 3;  hsit[11]= 4;
	 hsit[12]= 0;  hsit[13]= 1;  hsit[14]= 2;
	 hmsr[ 0]= 9;  hmsr[ 1]= 5;  hmsr[ 2]= 6;
	 hmsr[ 3]= 3;  hmsr[ 4]= 3;  hmsr[ 5]= 8;
	 hmsr[ 6]= 2;  hmsr[ 7]= 6;  hmsr[ 8]= 5;
	 hmsr[ 9]=10;  hmsr[10]= 9;  hmsr[11]=11;
	 hmsr[12]= 8;  hmsr[13]= 4;  hmsr[14]= 1;

	//Now make the device vectors
	thrust::device_vector<int> dday(15,0);
	thrust::device_vector<int> dsit(15,0);
	thrust::device_vector<int> dmsr(15,0);
	thrust::copy(hday.begin(),hday.end(),dday.begin());
	thrust::copy(hsit.begin(),hsit.end(),dsit.begin());
	thrust::copy(hmsr.begin(),hmsr.end(),dmsr.begin());

	using namespace thrust::placeholders;

	printf("See comments for explanation of logic \n");


	/* Number of days for which rainfall exceeded 5 at any one location
	 *
	 * According to the prof, this should be # of days at a given location
	 * that rainfall exceeded 5, then that number summed over all locations.
	 * Therefore the answer should be 8.
	 * We asked this in class and so I made the code only to calculate based
	 * on that interpretation. 
	 * 
	 * Alternatively if days should not be counted twice, answer is 6.
	 * This would require additional code, didn't think about it.
	 */ 
	int n5;
	n5 = thrust::count_if(dmsr.begin(), dmsr.end(), _1>5);
	printf("Sum over locations of days > 5 rain: %d \n",n5);

	
	/* Total rainfall at each site
	 * 
	 * Expected answer 16, 26, 23, 14, 11
	 * 
	 * Sort by site, then reduce by key. Use copied versions of the data since
	 * this doesn't zip and so won't keep the dates in tact as well.
	 */
	thrust::device_vector<int> sit(15,0);
	thrust::device_vector<int> msr(15,0);
	thrust::copy(dsit.begin(),dsit.end(),sit.begin());
	thrust::copy(dmsr.begin(),dmsr.end(),msr.begin());

	thrust::sort_by_key(sit.begin(), sit.end(), msr.begin());
	thrust::reduce_by_key(sit.begin(),sit.end(),msr.begin(),dsit.begin(),dmsr.begin());

	thrust::copy(dsit.begin(),dsit.end(),hsit.begin());
	thrust::copy(dmsr.begin(),dmsr.end(),hmsr.begin());

	for (int i=0; i<5; i++)
	{
		printf("Site %d rainfall is %d \n", hsit[i], hmsr[i]);
	}

    return EXIT_SUCCESS;
}