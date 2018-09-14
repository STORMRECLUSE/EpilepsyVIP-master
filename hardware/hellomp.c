#include <omp.h>
#include <stdio.h>
/*include omp.h for parallel directives*/
/*include stdio.h for printf*/

//hellomp.c Multiprocessing Hello World
//Copyright 2015, Morganne Lerch, All rights reserved.

int main(){
    //directive parallelizes code in next block
    //across all processors
    #pragma omp parallel
    {
    	//runs on each processor, gets ID number of thread
   	int ID = omp_get_thread_num();
    	//prints hello world from each processor with ID
    	printf("hello world %d \n",ID);
    }
}
