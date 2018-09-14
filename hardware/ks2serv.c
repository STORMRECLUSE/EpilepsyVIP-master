#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define BUFFERSIZE 32
#define WINDOWSIZE 5000

//ks2serv.c EVM Server for Data Transfer
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
	at the end of function MAIN
	This function is unused, but is a working example 
	of how to read an array from a file */
int printfile(char *filename)
{
	int i;
	FILE *fp = fopen(filename, "r");
	double win[windowsize];
	fread(win, sizeof(double), windowsize, fp);
	for (i = 0; i < windowsize; i++) {
		printf("%lf ", win[i]);
	}
	printf("\n");
	fclose(fp);
	return 1;
}

/* Function MAIN
	Server setup and data receiving
	usage: 1 argument (portno) */
int main(int argc, char *argv[])
{
	//initializations
	int sockfd, newsockfd, portno, clilen, socklen_t, n, i,j, q, flag;
	struct sockaddr_in serv_addr, cli_addr;
	double buffer[BUFFERSIZE], win[WINDOWSIZE];
	memset(win, 0, sizeof(win));
	
	//No input error catcher
	if (argc < 2) {
		printf("Usage: portnumber\n");
	}
	
	//Open a socket
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0)
		error("ERROR opening socket\n");

	//Sets values in buffer to 0
	//for more information, 'man bzero' in terminal
	bzero((char *) &serv_addr, sizeof(serv_addr));

	//Grabs port number from input argument
	portno = atoi(argv[1]);

	serv_addr.sin_family = AF_INET; //sets the >A<ddress >F<amily
	serv_addr.sin_addr.s_addr = INADDR_ANY; //grabs the IP address of host
	serv_addr.sin_port = htons(portno); //sets port number, must convert using htons

	//bind the socket
	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
		error("ERROR binding socket!\n");
	
	//listen for connections
	listen(sockfd, 5);

	//accepting a connection
	clilen = sizeof(cli_addr);
	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (newsockfd < 0)
		error("ERROR accepting client\n");
	printf("Client Connected!\n");
	


	//while loop of server reading data that is being streamed to buffer
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
				printf("Data Transfer Complete!\n");
				break;
			}

		//print what data is received
		for (i = 0; i < BUFFERSIZE; i++){
			printf("%lf ",buffer[i]);
			
			//update window matrix
			for (j = 0; j < WINDOWSIZE; j++) {
				win[j] = win[j+1];
			}
			win[WINDOWSIZE] = buffer[i];
		}
		printf("\n");
	}

	//ask user if they want to save the final window to a file
	if (checkyes("Save current window to file?") == 1) {
		char outpath[20];
		printf("Please type output file name:\n");
		scanf("%s", outpath);
		/* for (i = 0; i < WINDOWSIZE; i++){ 
			printf("%lf ", win[i]);}
		printf("\n"); */ //this just prints the window
		FILE *outfile = fopen(outpath, "w");
		fwrite(win, sizeof(double), WINDOWSIZE, outfile);
		fclose(outfile);
		printf("The file has been written! To access this file, please check function PRINTFILE in the source code for this program!\n");
	}

	//close connection
	close(newsockfd);
	close(sockfd);
	return 0;
}
