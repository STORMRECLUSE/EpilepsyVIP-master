#ifndef EPILEPSYVIP2_ENERGY_H
#define EPILEPSYVIP2_ENERGY_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sliding_structs.h"

void Energy_Stats(double* data, double *en_stat,int start_index, int WINDOWSIZE, int length);
void Energy_Stats_vect (sliding_vector<double> data, double *en_stat,int start_index, int WINDOWSIZE);

#endif