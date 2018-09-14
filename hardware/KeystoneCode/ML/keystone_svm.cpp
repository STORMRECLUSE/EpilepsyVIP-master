#include "keystone_svm.h"


/**
   predict takes in a model and a test vector and spits out a novelty sequence
   
   Step 1: create a model
   Step 2: create a test vector nodes
   
**/
using namespace std;

//seizmodel = svm_load_model("TS041.model");

void novelty_create(struct svm_model *seizmodel, double* test, sliding_vector <int> &novelty_sequence){
  struct svm_node* test_vector = new svm_node[4];
  int i;

  //test_vector = (struct svm_node *) realloc(test_vector,3*sizeof(struct svm_node));
  
  for(i=0;i<3;i++){
    test_vector[i].index = i+1;
    test_vector[i].value = test[i];
  }
  test_vector[3].index = -1;
  
  novelty_sequence[0] = svm_predict(seizmodel, test_vector);
  delete [] test_vector;
}

struct svm_model** svm_load_models(const char* filename_base,int num_chans){
  struct svm_model** svm_models = new struct svm_model*[num_chans];
  char filename_with_int[40];
  //strcpy(filename_copy,filename_base);
  for(int i = 0; i < num_chans;i++){
    sprintf(filename_with_int,filename_base,i);
    svm_models[i] = svm_load_model(filename_with_int);
  }
  return svm_models;

}
