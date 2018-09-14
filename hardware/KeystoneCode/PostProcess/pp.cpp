
#include "pp.h"
#include <iostream>

int post_processing(sliding_vector<int> &novelty_sequence, sliding_vector<double> &outlier_sequence,
                    int adaptive_rate, double outlier_threshold){
  /**
     inputs:
        1. novelty_sequence: prediction result from one-class SVM
	2. adaptive_rate: last x sequence values to consider
	3. outlier_threshold: threshold applied to outlier_sequence

     outputs:
        1. outlier_sequence: smoothed rate from novelty_sequence
  **/


    outlier_sequence[0] = (double)(outlier_sequence[-1]*adaptive_rate -
            novelty_sequence[-adaptive_rate] + novelty_sequence[0])/
                            (double)(adaptive_rate);
    //std::cout<< novelty_sequence[-adaptive_rate]  << " ";

  return 2*(outlier_sequence[0] < outlier_threshold) - 1;
}

