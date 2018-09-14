#include "major_votes.h"


int major_votes(double*  weights,int* alarm_rules, int numChannels){
  

  int length = numChannels;
  double vote=0;
 
  for(int i=0;i<length;i++){                     // wTa
    vote += weights[i]*alarm_rules[i];
  }

  return (vote > 0) ? 1:-1;
}

  



