#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "energy.h"


void Energy_Stats(double* data, double *en_stat,int start_index, int WINDOWSIZE, int length)
{
  
  int m;
  
  double CL=0;
  double E=0;
  double TE=0;
  

  if (length < start_index + WINDOWSIZE){
    printf("Inconsistent dimensions. See energy_stats function.\n");
    return;
  }

  for(m=start_index + 1;m<start_index + WINDOWSIZE;m++){  //mean curve length
    CL = CL + fabs(data[m]-data[m-1]);  
  }

  CL = log(1.0/WINDOWSIZE*CL);


  for(m=start_index;m<start_index + WINDOWSIZE;m++){   //mean energy
    E = E + pow(data[m],2);     
  }
  
 E = log(1.0/WINDOWSIZE*E);
  
 for(m=start_index + 2;m<start_index + WINDOWSIZE;m++){   //Teager Energy
    TE = TE + pow(data[m-1],2) - data[m]*data[m-2];
      }
  
  TE = log(1.0/WINDOWSIZE*TE);
  
  en_stat[0] = CL;
  en_stat[1] = E;
  en_stat[2] = TE;

  return ;
}


void Energy_Stats_vect(sliding_vector<double> data, double *en_stat,int start_index, int WINDOWSIZE)
{

    int m;

    double CL=0;
    double E=0;
    double TE=0;


    if (data._length < start_index + WINDOWSIZE){
        printf("Inconsistent dimensions. See energy_stats function.\n");
        return;
    }

    for(m=start_index + 1;m<start_index + WINDOWSIZE;m++){  //mean curve length
        CL = CL + fabs(data[m]-data[m-1]);
    }

    CL = log(1.0/WINDOWSIZE*CL);


    for(m=start_index;m<start_index + WINDOWSIZE;m++){   //mean energy
        E = E + pow(data[m],2);
    }

    E = log(1.0/WINDOWSIZE*E);

    for(m=start_index + 2;m<start_index + WINDOWSIZE;m++){   //Teager Energy
        TE = TE + pow(data[m-1],2) - data[m]*data[m-2];
    }

    TE = log(1.0/WINDOWSIZE*TE);

    en_stat[0] = CL;
    en_stat[1] = E;
    en_stat[2] = TE;

    return ;
}