#include <stdio.h>

//filecreate.c Quick Fake Data Generator
//Copyright 2015, Erik Biegert, All rights reserved.

int main(int argc, char *argv[])
{
	if (argc !=3) {
		printf("Usage: filename value\n");
		return 1;
	}
	FILE *fp = fopen(argv[1], "w");
	int value = atoi(argv[2]);
	int i = 1; //counting
	for (i = 1; i <= value; i++) {
		double d = i;
		d = d/10;
		fprintf(fp, "%lf ", d);
	}
	printf("File written!\n");
	close(fp);
	return 0;
}
