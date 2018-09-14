#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include "tmwtypes.h"
#include "mat.h"
#include "matrix.h"

#define BUFFERSIZE 32
#define WINDOWSIZE 2048
#define CHANSIZE 154

//epilclient.c Reads and Send Doubles over TCP/IP
//Copyright 2015, Erik Biegert, All rights reserved.

/* HEADER FILES: all are standard C libraries except 
tmwtypes.h
mat.h
matrix.h
These are matlab C headers and are located with the MATLAB install
/usr/local/MATLAB/yourversion/extern/include

You also need the library dependencies. Basically compile with the following command:
gcc -o epilclient epilclient.c -Wl,-rpath /usr/local/MATLAB/R2013a_Student/bin/glnxa64/ -I/usr/local/MATLAB/R2013a_Student/extern/include/ -L/usr/local/MATLAB/R2013a_Student/bin/glnxa64 -lmat -lmx
*/

/* Function ERROR
	Throws some errors.
	Check perror man page
	'man perror' in terminal */
void error(const char *msg)
{
	perror(msg);
	exit(0);
}

void ReadXBytes(int socket, unsigned int x, void* buffer)
{
    int bytesRead = 0;
    int result;
    while (bytesRead < x)
    {
        result = read(socket, ((char*) buffer )+ bytesRead, x - bytesRead);
        if (result < 1 )
        {
            // Throw your error.
        }

        bytesRead += result;
    }
}
/* Function TXCLIENTDATA
 * Sends a sendtime's amount worth of data
 * Uses datacounter to remember where it is in the large matlab file being read
 * Matrixsize is total # of elements in the data being sent
 * txrows is the number of rows (channels) in the data being sent
 * Returns 1 if it hasn't reached the end of the data
 * Returns 0 if it has reached the end of data and has TX'd a buffer of 666's
*/
int txclientdata(int sockfd, double* bufferout, double *data, int sendtime, int &datacounter,
					int matrixsize, int txrows) {
	int k,i,q = 1;
	int rowcount = 0;
	int timecount = 0;;//timecount-WINDOWSIZE;
	for(i = 0; i < BUFFERSIZE; i++){
		bufferout[i] = 0;
	}
	bufferout[0] = sendtime;
	write(sockfd, bufferout, BUFFERSIZE * sizeof(double));
	while (q == 1) { //here we start sending back the filtered data, dataout
		//Fills up the buffer of size BUFFERSIZE
		memset(bufferout, 0, sizeof(bufferout));
		for (k = 0; k < BUFFERSIZE; k++) {
			bufferout[k] = *(data + datacounter);
			datacounter++;
			rowcount++;
			if (rowcount == txrows){
				timecount++;
				rowcount = 0;
			}
			if (timecount == sendtime) {
				q = 0;
				break;
			}
			//printf("%lf ", buffer[i]); //WARNING: this prints ALL the values that you are sending and will probably give you a segmentation fault.
		}
		//If we reach the end of the matrix, stop sending data
		if (datacounter > matrixsize - 1) {
			q = 0;
		}
		//when the buffer is full, send the buffer
		int n = write(sockfd, bufferout, BUFFERSIZE * sizeof(double));
	}

	if (datacounter >= matrixsize - 1) {
		for (i = 0; i < BUFFERSIZE; i++) {
			bufferout[i] = 666;
		}
		write(sockfd, bufferout, BUFFERSIZE * sizeof(double));

		return 0;
	}
	return 1;
}
/* Function RXCLIENTDATA
 * Receives final decision, energystats[3], and a windowsize amount of data.
 * The windowsize is determined by rxwindowsize and rxchansize as specified by epilserv when sending
 * The first buffer received is a header buffer with relevant information
 * Afterwards it receives data and writes it to file F and out.
*/
void rxclientdata(int sockfd, double* buffer, FILE *decision_f,FILE* energy_stats_f,
				  FILE* filt_data_f, double finaldecision, double* energystats, double** out, int print_last_m) {
	int i, k, n;
	double bufferout[BUFFERSIZE];
	ReadXBytes(sockfd, BUFFERSIZE * sizeof(double), buffer); //very first buffer received is the energystats and finaldecision
	finaldecision = buffer[0];
	int rxwindowsize = buffer[4]; //rxwindow/chansize allow us to receive an arbitrary sized data set from server
	int rxchansize = buffer[5];
	for (k = 0; k < 3; k++) {
		energystats[k] = buffer[k + 1];
	}
	fprintf(decision_f,"%f\n", finaldecision);
	//printf("final decison = %f\n", finaldecision);
	//fprintf(energy_stats_f,"");
	for (int i = 0;i <3;i++){
		fprintf(energy_stats_f,"%f, ",energystats[i]);
		//printf("energy_stats = %f\n", energystats[i]);
	}
	fprintf(energy_stats_f,"\n" );
//	char c;
//	do
//		c = fgetc(in);
//	while (c != '\n');


	//fprintf(filt_data_f,'Filtered Window: \n');
	while (1) { //Receive data
		int flag = 0;
		//printf("Receiving\n");
		ReadXBytes(sockfd, BUFFERSIZE * sizeof(double), buffer);
		//n = read(sockfd, buffer, BUFFERSIZE * sizeof(double));
		//printf("%f, %f\n", buffer[0],buffer[1]);
		int rowcount = buffer[0];
		int timecount = buffer[1];

		//printf("%d ", n);
		if (n < 0)
			error("ERROR reading socket\n");
		//check if received all 666s
		//if (n == BUFFERSIZE*sizeof(double)) {
		for (i = 0; i < 10; i++) {
			if (buffer[i] != 666) {
				flag = 1;
				break;
			}
		}
		//flag stays 0 if all zeros are read on socket
		if (flag == 0) {
			break;
		}
		for (i = 0; i < BUFFERSIZE - 2; i++) {
			if (rowcount == rxchansize){
				break;
			}
			out[rowcount][timecount] = buffer[i + 2];
			//printf("%f ", buffer[i+2]);
			//fprintf(filt_data_f, "%f, ", out[rowcount][timecount]);
			timecount++;
			if (timecount == rxwindowsize) {
				rowcount++;
				timecount = 0;
				//fprintf(filt_data_f, "\n");
			}
		}
	}

	for(int l=WINDOWSIZE-print_last_m;l<rxwindowsize;l++){
		for(int chan =0; chan<rxchansize;chan++){
			fprintf(filt_data_f,"%f, ",out[chan][l]);
		}
		fprintf(filt_data_f,"\n");
	}


//write(sockfd, bufferout, BUFFERSIZE * sizeof(double));
};

void write_file_headers(int CHAN_SIZE,FILE* decision_f,FILE* energy_f,FILE*filt_data_f){
	fprintf(decision_f,"Decision\n");
	fprintf(energy_f,"Energy_1, Energy_2, Energy_3\n");
	for (int chan = 0; chan<CHAN_SIZE; chan++){
		fprintf(filt_data_f,"CHAN_%d, ",chan);
	}
	fprintf(filt_data_f,"\n");
}

/* Function MAIN
	Connects to server
	Reads doubles from textfile
	Sends values read over buffer
	Usage: hostname port matfile */
int main(int argc, char *argv[])
{
	

	//some initializations
	int sockfd, portno, n, i, q, k, j, matrixsize,flag,counter,rowcount,timecount,sendcount=0,rxwindowsize,rxchansize;
	double finaldecision;
	struct sockaddr_in serv_addr;
	struct hostent *server;
	double buffer[BUFFERSIZE], bufferin[BUFFERSIZE], bufferout[BUFFERSIZE];


	//initialize 'out' variable
	double **out = (double **)malloc(CHANSIZE * sizeof(double *));
	for ( i = 0; i < CHANSIZE; i++ ) {
		out[i] =(double *)malloc(WINDOWSIZE * sizeof(double)*2);
	}
	
	if (argc < 4) {	//make sure input is correct
		printf("Usage: %s hostname port textfile\n", argv[0]);
	}
	portno = atoi(argv[2]); //grab port #
	sockfd = socket(AF_INET, SOCK_STREAM, 0); //create socket
	if (sockfd < 0) 
		error("ERROR opening socket\n");
	server = gethostbyname(argv[1]);  //grab serverid
	if (server == NULL) {
		fprintf(stderr,"ERROR, no such host\n");
		exit(0);
	}
	bzero((char *) &serv_addr, sizeof(serv_addr)); //zero out buffer
	serv_addr.sin_family = AF_INET; //set inet settings
	bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
	serv_addr.sin_port = htons(portno);
	
	//connected
	if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
		error("ERROR connecting\n");
	
	//clear the buffer
	bzero(buffer,BUFFERSIZE);

	bufferout[1] = WINDOWSIZE;
	bufferout[2] = CHANSIZE;
	write(sockfd, bufferout, BUFFERSIZE * sizeof(double));

	bzero(buffer,BUFFERSIZE);

	//find filename
	char *path = argv[3];
	//gets tricky here, we are using matlab C commands now

	// this is the equivalent of 'fopen'
	MATFile *datafile = matOpen(path, "r");
	if (datafile == NULL)
		perror("ERROR opening file!\n");

	//matlab structure type is mxArray
	mxArray *pa;
	double *data;
	//set 'pa' to the matrix containing the data
	pa = matGetVariable(datafile, "record_RMpt5"); 

	//'data' is a pointer that points to the first value in the matrix
	data = mxGetPr(pa);
	matrixsize = mxGetNumberOfElements(pa); //number of total elements in the matrix
	memset(buffer, 0, sizeof(buffer)); //pad with 0s
	int txrows = mxGetM(pa);
	int txcol = mxGetN(pa);
	i = 0;
	j = 0;
	q = 1;
	int g = 0,datacounter=0;
	flag = 0;
	rowcount = 0;
	timecount = 0;
	double energystats[3];
	FILE *decision_f = fopen("decision.txt", "w");
	FILE *energy_stats_f = fopen("energy.txt","w");
	FILE *filt_data_f = fopen("filt.csv","w");
	write_file_headers(CHANSIZE,decision_f,energy_stats_f,filt_data_f);


	//fputc(filt_data_f,"\n");
	//prepare to write to a file

	if (decision_f == NULL | energy_stats_f == NULL| filt_data_f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}


	//THIS IS A SUPER IMPORTANT VALUE ***************************************************************************************************************************************************************
	//This determines the offset time! Epilserv will read it from the first buffer sent each time
	int offsettime = 512;
	q = 1;

	//Transmit a WINDOWSIZE worth of data in order to fill the first window
	printf("Sending First Window");
	q = txclientdata(sockfd, buffer, data, WINDOWSIZE, datacounter, matrixsize, txrows);
	rxclientdata(sockfd, buffer, decision_f,energy_stats_f,filt_data_f, finaldecision, energystats, out,WINDOWSIZE);
	printf("First Window Received\n");
	while (1) { //n is zero when we finish sending all data
		printf("Start Sending\n");
		q = txclientdata(sockfd, buffer, data, offsettime, datacounter, matrixsize, txrows);
		if (q != 1){
			break;
		}
		printf("Start Receiving\n");
		rxclientdata(sockfd, buffer, decision_f,energy_stats_f,filt_data_f, finaldecision, energystats, out,offsettime);


	}
	//close connection and file
	fclose(decision_f);fclose(energy_stats_f);fclose(filt_data_f);
	close(sockfd);
	matClose(datafile); 
	return 0;
}
