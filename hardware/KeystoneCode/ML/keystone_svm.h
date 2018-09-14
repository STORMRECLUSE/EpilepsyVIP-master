#ifndef EPILEPSYVIP2_KEYSTONE_SVM_H
#define EPILEPSYVIP2_KEYSTONE_SVM_H

//#ifndef LIBSVM_VERSION
#include "svm.h"
//#endif
#include <iostream>
#include <stdlib.h>
#include <string.h>

//#ifndef EPILEPSYVIP2_SLIDING_STRUCTS_H
#include "sliding_structs.h"
//#endif

void novelty_create(struct svm_model *seizmodel, double* test, sliding_vector <int> &novelty_sequence);
struct svm_model** svm_load_models(const char* filename_base,int num_chans);
#endif