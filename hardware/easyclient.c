#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

#define BUFFERSIZE 32

//easyclient.c Reads and Send Doubles over TCP/IP
//Copyright 2015, Erik Biegert, All rights reserved.

/* Function ERROR
	Throws some errors.
	Check perror man page
	'man perror' in terminal */
void error(const char *msg)
{
	perror(msg);
	exit(0);
}

/* Function MAIN
	Connects to server
	Reads doubles from textfile
	Sends values read over buffer
	Usage: hostname port textfile */
int main(int argc, char *argv[])
{
	//some initializations
	int sockfd, portno, n, i, q;
	struct sockaddr_in serv_addr;
	struct hostent *server;
	double buffer[BUFFERSIZE];
	
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
	
	//connect
	if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
		error("ERROR connecting\n");
	
	//clear the buffer
	bzero(buffer,BUFFERSIZE);
	
	//find filename
	char *path = argv[3];
	FILE *data = fopen(path, "r");
	if (data == NULL)
		perror("ERROR opening file!\n");

	//read & send data
	q = 1;
	while (q == 1){
		//each while loop sends 1 buffer
		//q is set to 0 when the file is done reading
		memset(buffer, 0, sizeof(buffer)); //pad with 0s

		//fills buffer with data read from txt file
		for (i = 0; i < BUFFERSIZE; i++){
			q = fscanf(data, "%lf", &buffer[i]);
			printf("%lf ", buffer[i]);
		}
		printf("\n");

		//write command sends data on socket
		write(sockfd, buffer, BUFFERSIZE * sizeof(double));
	}

	//Send transmit complete
	//All zero buffer tells server no more data is coming
	memset(buffer, 0, sizeof(buffer));
	write(sockfd, buffer, BUFFERSIZE*sizeof(double));
	printf("Data Transmit Complete!\n");
	
	
	//close connection and file
	close(sockfd);
	fclose(data); 
	return 0;
}
