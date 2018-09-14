#include <omp.h>
#include <stdio.h>
//parfor2.c Multiprocessing 'For' Loop
//Copyright 2015, Morganne Lerch, All rights reserved.
int main()
{
    int x=0;
    int y;
    #pragma omp parallel for num_threads(2) reduction(+:x)
    for (y=0; y<8; y++)
        {
            int ID = omp_get_thread_num();
            x+=2;
            printf("x = %d, y = %d, thread number = %d \n", x, y, ID);
        }

}
