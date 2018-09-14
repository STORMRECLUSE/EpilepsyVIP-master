#ifndef EPILEPSYVIP2_FILTER_H
#define EPILEPSYVIP2_FILTER_H
//#include <stdlib.h>
//#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
//#include <vector>
#include <math.h>
#include <cstdlib>
//#include <stdlib.h>
#include <complex>

#include "sliding_structs.h"

//#include <fftw3.h>

using namespace std;

//#include "tmwtypes.h"
//#include "mat.h"
//#include "matrix.h"

#define N 10 //The number of images which construct a time series for each pixel
#ifndef PI
#define PI 3.14159
#endif
double *ComputeLP( int FilterOrder );

double *ComputeHP( int FilterOrder );

double *TrinomialMultiply( int FilterOrder, double *b, double *c );

double *ComputeNumCoeffs(int FilterOrder);

double *ComputeDenCoeffs( int FilterOrder, double Lcutoff, double Ucutoff );

void filter(int ord, double *a, double *b, int np, double **x, double **y, int row);

void filter_vect(int ord, double *a, double *b, int start_index,int end_index, const sliding_vector <double> &x, sliding_vector <double> &y);
//int filter_freq(fftw_complex* Xin, fftw_complex* Aco, fftw_complex* Bco, int Np, double* y);

double sf_bwbp( int n, double f1f, double f2f );

double sf_bwbs( int n, double f1f, double f2f );

double *ccof_bwbs( int n, double f1f, double f2f );

double *dcof_bwbs( int n, double f1f, double f2f );


#endif
