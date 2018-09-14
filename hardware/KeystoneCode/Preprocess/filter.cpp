#include "filter.h"
#define N 10 //The number of images which construct a time series for each pixel
#define PI 3.14159

double *ComputeLP( int FilterOrder )
{
    double *NumCoeffs;
    int m;
    int i;

    NumCoeffs = (double *)calloc( FilterOrder+1, sizeof(double) );
    if( NumCoeffs == NULL ) return( NULL );

    NumCoeffs[0] = 1;
    NumCoeffs[1] = FilterOrder;
    m = FilterOrder/2;
    for( i=2; i <= m; ++i)
    {
        NumCoeffs[i] =(double) (FilterOrder-i+1)*NumCoeffs[i-1]/i;
        NumCoeffs[FilterOrder-i]= NumCoeffs[i];
    }
    NumCoeffs[FilterOrder-1] = FilterOrder;
    NumCoeffs[FilterOrder] = 1;

    return NumCoeffs;
}

double *ComputeHP( int FilterOrder )
{
    double *NumCoeffs;
    int i;

    NumCoeffs = ComputeLP(FilterOrder);
    if(NumCoeffs == NULL ) return( NULL );

    for( i = 0; i <= FilterOrder; ++i)
        if( i % 2 )
            NumCoeffs[i] = -NumCoeffs[i];

    return NumCoeffs;
}

double *TrinomialMultiply( int FilterOrder, double *b, double *c )
{
    int i, j;
    double *RetVal;

    RetVal = new double[ 4 * FilterOrder]();
    if( RetVal == NULL ) return( NULL );

    RetVal[2] = c[0];
    RetVal[3] = c[1];
    RetVal[0] = b[0];
    RetVal[1] = b[1];

    for( i = 1; i < FilterOrder; ++i )
    {
        RetVal[2*(2*i+1)]   += c[2*i] * RetVal[2*(2*i-1)]   - c[2*i+1] * RetVal[2*(2*i-1)+1];
        RetVal[2*(2*i+1)+1] += c[2*i] * RetVal[2*(2*i-1)+1] + c[2*i+1] * RetVal[2*(2*i-1)];

        for( j = 2*i; j > 1; --j )
        {
            RetVal[2*j]   += b[2*i] * RetVal[2*(j-1)]   - b[2*i+1] * RetVal[2*(j-1)+1] +
                c[2*i] * RetVal[2*(j-2)]   - c[2*i+1] * RetVal[2*(j-2)+1];
            RetVal[2*j+1] += b[2*i] * RetVal[2*(j-1)+1] + b[2*i+1] * RetVal[2*(j-1)] +
                c[2*i] * RetVal[2*(j-2)+1] + c[2*i+1] * RetVal[2*(j-2)];
        }

        RetVal[2] += b[2*i] * RetVal[0] - b[2*i+1] * RetVal[1] + c[2*i];
        RetVal[3] += b[2*i] * RetVal[1] + b[2*i+1] * RetVal[0] + c[2*i+1];
        RetVal[0] += b[2*i];
        RetVal[1] += b[2*i+1];
    }

    return RetVal;
};

double *ComputeNumCoeffs(int FilterOrder)
{
    double *TCoeffs;
    double *NumCoeffs;
    int i;

    NumCoeffs = (double *)calloc( 2*FilterOrder+1, sizeof(double) );
    if( NumCoeffs == NULL ) return( NULL );

    TCoeffs = ComputeHP(FilterOrder);
    if( TCoeffs == NULL ) return( NULL );

    for( i = 0; i < FilterOrder; ++i)
    {
        NumCoeffs[2*i] = TCoeffs[i];
        NumCoeffs[2*i+1] = 0.0;
    }
    NumCoeffs[2*FilterOrder] = TCoeffs[FilterOrder];

    free(TCoeffs);

    return NumCoeffs;
}
double *ComputeDenCoeffs( int FilterOrder, double Lcutoff, double Ucutoff ){
    int k;            // loop variables
    double theta;     // PI * (Ucutoff - Lcutoff) / 2.0
    double cp;        // cosine of phi
    double st;        // sine of theta
    double ct;        // cosine of theta
    double s2t;       // sine of 2*theta
    double c2t;       // cosine 0f 2*theta
    double *RCoeffs;     // z^-2 coefficients
    double *TCoeffs;     // z^-1 coefficients
    double *DenomCoeffs;     // dk coefficients
    double PoleAngle;      // pole angle
    double SinPoleAngle;     // sine of pole angle
    double CosPoleAngle;     // cosine of pole angle
    double a;         // workspace variables

    cp = cos(PI * (Ucutoff + Lcutoff) / 2.0);
    theta = PI * (Ucutoff - Lcutoff) / 2.0;
    st = sin(theta);
    ct = cos(theta);
    s2t = 2.0*st*ct;        // sine of 2*theta
    c2t = 2.0*ct*ct - 1.0;  // cosine of 2*theta

    RCoeffs = (double *)calloc( 2 * FilterOrder, sizeof(double) );
    TCoeffs = (double *)calloc( 2 * FilterOrder, sizeof(double) );

    for( k = 0; k < FilterOrder; ++k )
    {
        PoleAngle = PI * (double)(2*k+1)/(double)(2*FilterOrder);
        SinPoleAngle = sin(PoleAngle);
        CosPoleAngle = cos(PoleAngle);
        a = 1.0 + s2t*SinPoleAngle;
        RCoeffs[2*k] = c2t/a;
        RCoeffs[2*k+1] = s2t*CosPoleAngle/a;
        TCoeffs[2*k] = -2.0*cp*(ct+st*SinPoleAngle)/a;
        TCoeffs[2*k+1] = -2.0*cp*st*CosPoleAngle/a;
    }

    DenomCoeffs = TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs );
    free(TCoeffs);
    free(RCoeffs);

    DenomCoeffs[1] = DenomCoeffs[0];
    DenomCoeffs[0] = 1.0;
    for( k = 3; k <= 2*FilterOrder; ++k )
        DenomCoeffs[k] = DenomCoeffs[2*k-2];


    return DenomCoeffs;
};

void filter(int ord, double *a, double *b, int np, double **x, double **y, int row)
{
  int i,j;
  y[row][0]=b[0]*x[row][0];
  for (i=1;i<2*ord+1;i++)   //changed to 2*order
    {
      y[row][i]=0.0;
      for (j=0;j<i+1;j++)
	y[row][i]=y[row][i]+b[j]*x[row][i-j];
      for (j=0;j<i;j++)
	y[row][i]=y[row][i]-a[j+1]*y[row][i-j-1]; 
    }
  for (i=2*ord+1;i<np+1;i++)
    {
      y[row][i]=0.0;
      for (j=0;j<2*ord+1;j++)
	y[row][i]=y[row][i]+b[j]*x[row][i-j];
      for (j=0;j<2*ord;j++)
	y[row][i]=y[row][i]-a[j+1]*y[row][i-j-1];
    }
};

void filter_vect(int ord, double *a, double *b, int start_index, int end_index, const sliding_vector <double> &x, sliding_vector <double> &y){
    //int i,j;

    for (int i=start_index;i<end_index+1;i++)
    {
        y[i]=0.0;
        for (int j=0;j<2*ord+1;j++)
            y[i]+= b[j]*x[i-j];
        for (int j=0;j<2*ord;j++)
            y[i]-=a[j+1]*y[i-j-1];
    }
}

/*
int filter_freq(fftw_complex* Xin, fftw_complex* Aco, fftw_complex* Bco, int Np, double* y) {         
  
  int i;                                                                  
  fftw_complex* filt_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*Np);   // No flag?/use a different flag than estimate?? 
  fftw_plan q = fftw_plan_dft_c2r_1d(Np, filt_fft, y, FFTW_ESTIMATE);    
  
  
  // Using convolution                                                    
  for (i = 0; i < Np; i++) {                                               
    filt_fft[i][0] = Xin[i][0]*Bco[i][0]*Aco[i][0]-Aco[i][0]*Bco[i][1]*Xin[i][1]+Aco[i][1]*Bco[i][0]*Xin[i][1]+Aco[i][1]*Bco[i][1]*Xin[i][0];
    filt_fft[i][0] = filt_fft[i][0]/(Aco[i][0]*Aco[i][0] + Aco[i][1]*Aco[i][1]);
    filt_fft[i][1] = Aco[i][0]*Bco[i][0]*Xin[i][1]+Aco[i][0]*Bco[i][1]*Xin[i][0]-Aco[i][1]*Bco[i][0]*Xin[i][0]+Aco[i][1]*Bco[i][1]*Xin[i][1];
    filt_fft[i][1] = filt_fft[i][1]/(Aco[i][0]*Aco[i][0] + Aco[i][1]*Aco[i][1]);

    //printf("%d\n", i);                                              
    //printf("%11.7f %11.7f %11.7f %11.7f %11.7f %11.7f\n", Bco[i][0]/Aco[i][0], Bco/Aco };
    // inverse fft                                                          
    fftw_execute(q);                                                        
    fftw_destroy_plan(q);                                                   
    fftw_free(filt_fft); 
    
  }
};
*/
double sf_bwbp( int n, double f1f, double f2f )
{
    int k;            // loop variables
    double ctt;       // cotangent of theta
    double sfr, sfi;  // real and imaginary parts of the scaling factor
    double parg;      // pole angle
    double sparg;     // sine of pole angle
    double cparg;     // cosine of pole angle
    double a, b, c;   // workspace variables

    ctt = 1.0 / tan(M_PI * (f2f - f1f) / 2.0);
    sfr = 1.0;
    sfi = 0.0;

    for( k = 0; k < n; ++k )
    {
        parg = M_PI * (double)(2*k+1)/(double)(2*n);
        sparg = ctt + sin(parg);
        cparg = cos(parg);
        a = (sfr + sfi)*(sparg - cparg);
        b = sfr * sparg;
        c = -sfi * cparg;
        sfr = b - c;
        sfi = a - b - c;
    }

    return( 1.0 / sfr );
}
//
//double sf_bwbs( int n, double f1f, double f2f )
//{
//    int k;            // loop variables
//    double tt;        // tangent of theta
//    double sfr, sfi;  // real and imaginary parts of the scaling factor
//    double parg;      // pole angle
//    double sparg;     // sine of pole angle
//    double cparg;     // cosine of pole angle
//    double a, b, c;   // workspace variables
//
//    tt = tan(M_PI * (f2f - f1f) / 2.0);
//    sfr = 1.0;
//    sfi = 0.0;
//
//    for( k = 0; k < n; ++k )
//    {
//        parg = M_PI * (double)(2*k+1)/(double)(2*n);
//        sparg = tt + sin(parg);
//        cparg = cos(parg);
//        a = (sfr + sfi)*(sparg - cparg);
//        b = sfr * sparg;
//        c = -sfi * cparg;
//        sfr = b - c;
//        sfi = a - b - c;
//    }
//
//    return( 1.0 / sfr );
//}

//double *ccof_bwbs( int n, double f1f, double f2f )
//{
//    double alpha;
//    double *ccof;
//    int i, j;
//
//    alpha = -2.0 * cos(M_PI * (f2f + f1f) / 2.0) / cos(M_PI * (f2f - f1f) / 2.0);
//
//    ccof = (double *)calloc( 2*n+1, sizeof(double) );
//
//    ccof[0] = 1.0;
//
//    ccof[2] = 1.0;
//    ccof[1] = alpha;
//
//    for( i = 1; i < n; ++i )
//    {
//        ccof[2*i+2] += ccof[2*i];
//        for( j = 2*i; j > 1; --j )
//            ccof[j+1] += alpha * ccof[j] + ccof[j-1];
//
//        ccof[2] += alpha * ccof[1] + 1.0;
//        ccof[1] += alpha;
//    }
//
//    return( ccof );
//}
//
//double *dcof_bwbs( int n, double f1f, double f2f )
//{
//    int k;            // loop variables
//    double theta;     // M_PI * (f2f - f1f) / 2.0
//    double cp;        // cosine of phi
//    double st;        // sine of theta
//    double ct;        // cosine of theta
//    double s2t;       // sine of 2*theta
//    double c2t;       // cosine 0f 2*theta
//    double *rcof;     // z^-2 coefficients
//    double *tcof;     // z^-1 coefficients
//    double *dcof;     // dk coefficients
//    double parg;      // pole angle
//    double sparg;     // sine of pole angle
//    double cparg;     // cosine of pole angle
//    double a;         // workspace variables
//
//    cp = cos(M_PI * (f2f + f1f) / 2.0);
//    theta = M_PI * (f2f - f1f) / 2.0;
//    st = sin(theta);
//    ct = cos(theta);
//    s2t = 2.0*st*ct;        // sine of 2*theta
//    c2t = 2.0*ct*ct - 1.0;  // cosine 0f 2*theta
//
//    rcof = (double *)calloc( 2 * n, sizeof(double) );
//    tcof = (double *)calloc( 2 * n, sizeof(double) );
//
//    for( k = 0; k < n; ++k )
//    {
//        parg = M_PI * (double)(2*k+1)/(double)(2*n);
//        sparg = sin(parg);
//        cparg = cos(parg);
//        a = 1.0 + s2t*sparg;
//        rcof[2*k] = c2t/a;
//        rcof[2*k+1] = -s2t*cparg/a;
//        tcof[2*k] = -2.0*cp*(ct+st*sparg)/a;
//        tcof[2*k+1] = 2.0*cp*st*cparg/a;
//    }
//
//    dcof = trinomial_mult( n, tcof, rcof );
//    free( tcof );
//    free( rcof );
//
//    dcof[1] = dcof[0];
//    dcof[0] = 1.0;
//    for( k = 3; k <= 2*n; ++k )
//        dcof[k] = dcof[2*k-2];
//    return( dcof );
//}