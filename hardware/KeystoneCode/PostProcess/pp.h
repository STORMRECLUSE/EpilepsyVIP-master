#ifndef EPILEPSYVIP2_PP_H
#define EPILEPSYVIP2_PP_H

//#include <stdio.h>
#include <iostream>

#include "sliding_structs.h"

int post_processing(sliding_vector <int> &novelty_sequence, sliding_vector <double> &outlier_sequence, int adaptive_rate, double outlier_threshold);

#endif