#include "../Preprocess/filter.h"
//#include <fftw3.h>

#define N 10 //The number of images which construct a time series for each pixel
#define PI 3.14159

#define BUFFERSIZE 32
#define WINDOWSIZE 4096
#define CHANSIZE 154

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

void ReadXBytes(int socket, unsigned int x, void* buffer)
{
    int bytesRead = 0;
    int result;
    while (bytesRead < x)
    {
        result = read(socket, buffer + bytesRead, x - bytesRead);
        if (result < 1 )
        {
            // Throw your error.
        }

        bytesRead += result;
    }
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
	double bufferout[BUFFERSIZE], bufferin[BUFFERSIZE];
	double y[WINDOWSIZE];
	double *win;
	double* filtnum = ComputeNumCoeffs(10);
	double* filtden = ComputeDenCoeffs(10,100/1000,500/1000);
	double* filtout;

	/*
	fftw_complex *out; //for use of fftw
	fftw_plan p; //for use of fftw
	*/

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

	/*
	//setup FFTW
	in = (double*) fftw_malloc(sizeof(double) * WINDOWSIZE);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * WINDOWSIZE);
	p =fftw_plan_dft_r2c_1d(WINDOWSIZE,in,out,FFTW_MEASURE);
	*/
	int counter = 0;	
	int rowcount = 0;
	int timecount = 0;
	int fftcount = 0;
	int sendcount = 0;
	flag = 0;
	k = 0;
	r = 0;

/*------------------------------------------------------------------------------*/

	//IGNORE FIRST 5 SECONDS OF DATA (since it is mostly zeros)
	//ALLOW THE WINDOW TO FILL UP

	while(1) {
		ReadXBytes(newsockfd,BUFFERSIZE*sizeof(double),buffer);
		for (i = 0; i < BUFFERSIZE; i++){
			//write incoming buffer to data
			data[rowcount][timecount] = buffer[i];
			counter++;
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
		if (flag == 1)
			break;
	}
/*------------------------------------------------------------------------------*/

	//while loop of server reading data that is being streamed to buffer
	//While loop for receiving data. Fills buffer of size BUFFERSIZE and then 
	while (1) {
		flag = 0;
		ReadXBytes(newsockfd,BUFFERSIZE*sizeof(double),buffer);
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
			data[rowcount][timecount] = buffer[i]; //write incoming data to buffer
			win = &data[rowcount][timecount-WINDOWSIZE]; //move the window pointer to next channel
			//if (fftcount == WINDOWSIZE/25) { //take fft every fraction of the windowsize
/*				memcpy((void*)in, (void *)win, WINDOWSIZE*sizeof(double));
				fftw_execute(p); //EXECUTES FFT
				//fix FFT with conjugates
				for (j = (WINDOWSIZE/2)+1; j<WINDOWSIZE; j++) {
					out[j][0] = out[WINDOWSIZE-j][0]; 
					out[j][1] = -1.0*out[WINDOWSIZE-j][1];
				}
*/			fftcount++;
			//}

			//move window back
			if (timecount == WINDOWSIZE*2-1) {
				memcpy((void*)data[rowcount],(void*)win,WINDOWSIZE*sizeof(double));
			}
			rowcount++; //go to next channel
			if (rowcount == CHANSIZE) { //at max channel, go back to channel 0
				timecount++;
				if (timecount == WINDOWSIZE*2)
					timecount = WINDOWSIZE;
				rowcount = 0;
				if (!(timecount % 1000)) //DEBUG
					printf("%d\n",timecount); //DEBUG
			}
		}
	}

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
	timecount = 0;;//timecount-WINDOWSIZE;
	win = &data[rowcount][timecount];
	while (q == 1){
		//Fills up the buffer of size BUFFERSIZE
		bufferout[0] = rowcount; //first index of buffer is the start row
		bufferout[1] = timecount; //next index of buffer is the start column
		for (k = 0; k < BUFFERSIZE-2; k++){
			//bufferout[k+2] = data[rowcount][timecount];
			bufferout[k+2] = y_filtered[rowcount][timecount];

			//counter++;
			timecount++;
			if(timecount == WINDOWSIZE) {
				rowcount++;
				timecount = 0;		
				if (rowcount == CHANSIZE) {
					printf("Data Send Complete!\n");
					q = 0;
					break;
				}
			}
		}
		//when the buffer is full, send the buffer
		write(newsockfd,bufferout,BUFFERSIZE * sizeof(double));
		sendcount++;
	}

	for (i = 0; i < BUFFERSIZE; i++){
		bufferout[i] = 666;
	}
	write(newsockfd, bufferout, BUFFERSIZE*sizeof(double));




	//close connection
	//close(newsockfd);
	//close(sockfd);
	free(data);
	//free();
	/*
	fftw_free(in);
	fftw_free(out);
	*/
	return 0;
}
