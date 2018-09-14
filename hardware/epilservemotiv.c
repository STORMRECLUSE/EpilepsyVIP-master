#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <complex>
using namespace std;

#include "tmwtypes.h"                                                           
#include "mat.h"                                                                
#include "matrix.h"



#define N 10 //The number of images which construct a time series for each pixel
#define PI 3.14159

#define BUFFERSIZE 32
#define WINDOWSIZE 1024
#define CHANSIZE 15

//epilserv.c EVM Server for Data Transfer
//Copyright 2015, Erik Biegert, All rights reserved.

/* Function ERROR
	Throws some errors.
	Check perror man page
	'man perror' in terminal */
void error(const char *msg)
{
	perror(msg);
	exit(1);
}

/* Function CHECKYES
	Asks user a y/n question based on input
	string msg. Returns 1 or 0 */
int checkyes(char msg[]) 
{
	char ans;
	while(1) {
		printf("%s (y/n) ", msg);
		scanf("%c", &ans);
		if (ans == 'y')
			return 1;
		if (ans == 'n')
			return 0;
	}
}


/* Function PRINTFILE
	Reads the file that was saved 
	at the end of function f
	This function is unused, but is a working example 
	of how to read an array from a file */
int printfile(char *filename)
{
	int i;
	FILE *fp = fopen(filename, "r");
	double win[WINDOWSIZE];
	fread(win, sizeof(double), WINDOWSIZE, fp);
	for (i = 0; i < WINDOWSIZE; i++) {
		printf("%lf ", win[i]);
	}
	printf("\n");
	fclose(fp);
	return 1;
}

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
        if( i % 2 ) NumCoeffs[i] = -NumCoeffs[i];

    return NumCoeffs;
}

double *TrinomialMultiply( int FilterOrder, double *b, double *c )
{
    int i, j;
    double *RetVal;

    RetVal = (double *)calloc( 4 * FilterOrder, sizeof(double) );
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
}

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
double *ComputeDenCoeffs( int FilterOrder, double Lcutoff, double Ucutoff )
{
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
}

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
}

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
}
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


/* Function STARTSERV
	Starts the server with input 'portno' waiting for a client to connect*/
int startserv(int portno){
	/*-------Server Hosting Stuff-------*/	
	//Open a socket
	int sockfd, newsockfd;

	struct sockaddr_in serv_addr, cli_addr;
	socklen_t clilen;

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd 	< 0)
		error("ERROR opening socket\n");

	//Sets values in buffer to 0
	//for more information, 'man bzero' in terminal
	bzero((char *) &serv_addr, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET; //sets the >A<ddress >F<amily
	serv_addr.sin_addr.s_addr = INADDR_ANY; //grabs the IP address of host
	serv_addr.sin_port = htons(portno); //sets port number, must convert using htons

	//bind the socket
	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
		error("ERROR binding socket!\n");
	
	printf("Server established! Waiting for client to connect...\n");
	//listen for connections
	listen(sockfd, 5);

	//accepting a connection
	clilen = sizeof(cli_addr);
	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (newsockfd < 0)
		error("ERROR accepting client\n");
	printf("Client Connected!\n");
	return newsockfd;
}





/* Function MAIN
	Server setup and data receiving
	usage: 1 argument (portno) */
int main(int argc, char *argv[])
{
	//initializations
	int sockfd, newsockfd, portno, n, i, j, q, r, k, flag;
	struct sockaddr_in serv_addr, cli_addr;
	double buffer[BUFFERSIZE], *in;
	double bufferout[BUFFERSIZE];
	double y[WINDOWSIZE];
	double *win;
	double* filtnum = ComputeNumCoeffs(10);
	double* filtden = ComputeDenCoeffs(10,100/1000,500/1000);
	double* filtout;

	
	fftw_complex *out; //for use of fftw
	fftw_plan p; //for use of fftw

	//allocate memory for data of CHANSIZE rows and WINDOWSIZE*2 columns
	double **data = (double **)malloc(CHANSIZE * sizeof(double *));
	for ( i = 0; i < CHANSIZE; i++ ) {
		data[i] =(double *)malloc(WINDOWSIZE * sizeof(double)*2);
	}
	  //win[row][column] brings you to the index
	  //each row is a channel, columns of width WINDOWSIZE
	
	//No input error catcher
	if (argc < 2) {
		printf("Usage: portnumber\n");
	}

	//Grabs port number from input argument
	portno = atoi(argv[1]);

	//start the server
	newsockfd = startserv(portno);

/*------------------------------------------------------------------------------*/


	//setup FFTW
	
	in = (double*) fftw_malloc(sizeof(double) * WINDOWSIZE);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * WINDOWSIZE);
	p =fftw_plan_dft_r2c_1d(WINDOWSIZE,in,out,FFTW_MEASURE);
		
	int counter = 0;	
	int rowcount = 0;
	int timecount = 0;
	int fftcount = 0;
	flag = 0;
	k = 0;
	r = 0;

/*------------------------------------------------------------------------------*/

	//IGNORE FIRST 5 SECONDS OF DATA (since it is mostly zeros)
	//ALLOW THE WINDOW TO FILL UP

	/**
	while(1) {
		read(newsockfd, buffer, BUFFERSIZE * sizeof(double));
		for (i = 0; i < BUFFERSIZE; i++){
			//write incoming buffer to data
			data[rowcount][timecount] = buffer[i];
			counter++;
			//printf("%d %d %f\n",rowcount,timecount,data[rowcount][timecount]);
			memset(bufferout, 0, sizeof(bufferout));
			rowcount++; //go to next channel
			if (rowcount == CHANSIZE) { //at max channel, go back to channel 0
				timecount++;//go to next column
				rowcount = 0;
				if (!(timecount % 1000)) //DEBUG
				printf("%d\n",timecount); //DEBUG
				if (timecount == WINDOWSIZE) {
					printf("BOOM\n"); //DEBUG
					win = &data[0][0];
					flag = 1;
					break;

				}
			}

		}
		write(newsockfd, bufferout, BUFFERSIZE*sizeof(double));	
		if (flag == 1)
			break;
	} **/
/*------------------------------------------------------------------------------*/

	//while loop of server reading data that is being streamed to buffer
	//While loop for receiving data. Fills buffer of size BUFFERSIZE and then 
	while (1) {
		flag = 0;
		//read the socket! new data from newsockfd goes to bufer
		n = read(newsockfd, buffer, BUFFERSIZE * sizeof(double));
			if (n < 0)
				error("ERROR reading socket\n");
			//check if received all 0s
			for (i = 0; i < BUFFERSIZE; i++) {
				if (buffer[i] != 0) {
					flag = 1; 
					break;

				}
			}
			//flag stays 0 if all zeros are read on socket
			if (flag == 0) {
				printf("Data Transfer Complete! Writing data back now...\n");
				break;
			}
		for (i = 0; i < BUFFERSIZE; i++){
			printf("%lf ",buffer[i]);
			/*counter++; //debugging
			if (!(counter % 10000)) 
				//printf("%d\n",counter);*/
			data[rowcount][timecount] = buffer[i]; //write incoming data to buffer
			win = &data[rowcount][timecount-WINDOWSIZE]; //move the window pointer to next channel
			//printf("%d %d %f %f\n",rowcount,timecount-WINDOWSIZE,*win,data[rowcount][timecount-WINDOWSIZE]);
			//if (fftcount == WINDOWSIZE/25) { //take fft every fraction of the windowsize
/*			memcpy((void*)in, (void *)win, WINDOWSIZE*sizeof(double));
			fftw_execute(p); //EXECUTES FFT
			//fix FFT with conjugates
			for (j = (WINDOWSIZE/2)+1; j<WINDOWSIZE; j++) {
				out[j][0] = out[WINDOWSIZE-j][0]; 
				out[j][1] = -1.0*out[WINDOWSIZE-j][1];
			}
*/			fftcount++;
			//}
			
	
			//filter_freq(out,Aco,Bco,WINDOWSIZE,y);
			/*		
			while (q == 1){
				//Fills up the buffer of size BUFFERSIZE
				for (k = 0; k < BUFFERSIZE; k++){
					bufferout[k] = *(win+r);
					r++;
				}

				//If we reach the end of the matrix, stop sending data
				if(r >= WINDOWSIZE-1)
					q = 0;
		
				//when the buffer is full, send the buffer
				write(newsockfd,bufferout,BUFFERSIZE * sizeof(double));

			}*/

			if (timecount == WINDOWSIZE*2-1) {
				memcpy((void*)data[rowcount],(void*)win,WINDOWSIZE*sizeof(double));
			}
			rowcount++; //go to next channel
			//printf("Debug: %d %f\n", rowcount, buffer[i]); //DEBUG
			if (rowcount == CHANSIZE) { //at max channel, go back to channel 0
				timecount++;
				if (timecount == WINDOWSIZE*2)
					timecount = WINDOWSIZE;
				rowcount = 0;
				if (!(timecount % 1000)) //DEBUG
					printf("%d\n",timecount); //DEBUG
			}
		}
		memset(bufferout, 0, sizeof(bufferout));
		write(newsockfd, bufferout, BUFFERSIZE*sizeof(double));	
	}

	/*for ( i = 0; i < 32*3; i++){
		printf("%f %d ",data[0][i],i);
	}*/


/************    TIME DOMAIN ******************/
 double FrequencyBands[2] = {0.1,0.4};
 int FiltOrd = 5;

 	double *DenC = 0;
 	double *NumC = 0;
 	//double y_filtered[WINDOWSIZE]; 
 	double **y_filtered = (double **)malloc(CHANSIZE * sizeof(double *));
 	for ( i = 0; i < CHANSIZE; i++ ) {
 		y_filtered[i] =(double *)malloc(WINDOWSIZE * sizeof(double));
 	}

 DenC = ComputeDenCoeffs(FiltOrd, FrequencyBands[0], FrequencyBands[1]);

 NumC = ComputeNumCoeffs(FiltOrd);

double scalf;
scalf = sf_bwbp(FiltOrd, FrequencyBands[0], FrequencyBands[1]);
 for(int k = 0; k<2*FiltOrd+1; k++)         // why k <11?  2*Order + 1
   {
     NumC[k] = scalf*NumC[k];
     //printf("NumC is: %lf\n", NumC[k]);
     //printf("Denc is %lf\n", DenC[k]);
   }

for (i = 0; i < CHANSIZE; i++){
 	filter(FiltOrd, DenC, NumC, WINDOWSIZE, data, y_filtered, i);
}

/*------------------------------------------------------------------------------*/

//AT THIS POINT WE HAVE RECEIVED ALL DATA, NOW WE SEND BACK THE LAST WINDOW

q = 1;
rowcount = 0;
r = timecount-WINDOWSIZE;
win = &data[rowcount][r];
while (q == 1){
	//Fills up the buffer of size BUFFERSIZE
	for (k = 0; k < BUFFERSIZE; k++){
		//bufferout[k] = data[rowcount][r];
		bufferout[k] = y_filtered[rowcount][r];

		//counter++;
		r++;
		if(r == WINDOWSIZE) {
			rowcount++;
			r = 0;
//printf("%d ", rowcount);
//printf("%f ", bufferout[k]);
//printf("%f ", data[0][r]);			
			if (rowcount == CHANSIZE) {
				printf("Data Send Complete!\n");
				q = 0;
				break;
			}
		}
	}
	//when the buffer is full, send the buffer
	write(newsockfd,bufferout,BUFFERSIZE * sizeof(double));
}

for (i = 0; i < BUFFERSIZE; i++){
	bufferout[i] = 666;
}
write(newsockfd, bufferout, BUFFERSIZE*sizeof(double));
	//ask user if they want to save the final window to a file
/*
	if (checkyes("Save current window to file?") == 1) {
		char outpath[20];
		printf("Please type output file name:\n");
		scanf("%s", outpath);
		 //for (i = 0; i < WINDOWSIZE; i++){ 
		//	printf("%lf ", win[i]);}
		/printf("\n");  //this just prints the window
		FILE *outfile = fopen(outpath, "w");
		fwrite(win, sizeof(double), WINDOWSIZE, outfile);
		fclose(outfile);
		printf("The file has been written! To access this file, please check function PRINTFILE in the source code for this program!\n");
	}
*/




	//close connection
	//close(newsockfd);
	//close(sockfd);
	//fclose(outfile);
	free(data);
	//free();
	fftw_free(in);
	fftw_free(out);
	return 0;
}
