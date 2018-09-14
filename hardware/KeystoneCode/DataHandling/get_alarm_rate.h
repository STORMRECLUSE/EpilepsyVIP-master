#ifndef EPILEPSYVIP2_GET_ALARM_RATE_H
#define EPILEPSYVIP2_GET_ALARM_RATE_H

#include "keystone_svm.h"
#include "major_votes.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "sliding_structs.h"
#include "filter.h"
#include "energy.h"
#include <omp.h>
#include "pp.h"


int read_model_params(const char* params_filename, double* outlier_threshold, double*weights, int CHANSIZE);

void initialize_pipeline(int FiltOrder, double* freq_bands, int WinStep, const char* filename, int n_input);

void destroy_pipeline(double *NumC, double* DenC, double **en_stat, struct svm_model **seizmodels,
                      sliding_array <int> &novelty_sequence, sliding_array <double>  &outlier_sequence,
                      int *alarm_sequence, double *weights, double* outlier_threshold,
                      sliding_array<double> &filtered_data, sliding_array<double> &raw_data, int CHANSIZE);

double get_window_decision(int WINDOWSIZE, int CHANSIZE, int FilterOrder, double*NumC, double*DenC,
                        sliding_array<double> &filtered_data,sliding_array<double> &raw_data,
                        double **en_stat, struct svm_model **seizmodels,sliding_array<int>&novelty_sequence,
                        sliding_array<double>&outlier_sequence, int adaptive_rate,
                        double* outlier_thresholds,double* maj_weights, int* alarm_sequence, int num_filter);

int get_alarm_rate(int WINDOWSIZE, int FilterOrder, double *NumC, double*DenC,
                       sliding_vector <double> &filtered_data, sliding_vector <double> &raw_data,
                       double *en_stat,
        //int windowStep, int numWin, int start_index,
                       struct svm_model *seizmodel, sliding_vector <int> &novelty_sequence,
                       sliding_vector <double> &outlier_sequence, int adaptive_rate,
                       double outlier_threshold, int WindowStep);
void update_windows(int WindowStep,sliding_array <int> &novelty_sequence,
                    sliding_array<double> &outlier_sequence,sliding_array<double >&filtered_data,
                    sliding_array<double> &raw_data);

#endif
