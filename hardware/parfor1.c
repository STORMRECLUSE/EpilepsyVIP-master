#include <omp.h>
#include <stdio.h>
//parfor2.c Multiprocessing 'For' Loop
//Copyright 2015, Morganne Lerch, All rights reserved.
int main()
{
    int y;
    #pragma omp parallel for private(y)
    for (y=0; y<10; y++)
        {
            int ID = omp_get_thread_num();
            printf("%d \n",ID);
        }
}
