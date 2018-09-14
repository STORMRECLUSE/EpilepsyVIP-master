#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

//ks2blnk.c LED Blinking Program
//Copyright 2015, Erik Biegert, All rights reserved.

int export(char *pin) // sets up GPIO drivers for LEDs
{ //USAGE: export(GPIOpin#)
	//open export file
	printf("Export GPIO%s\n", pin);
	FILE *exp = fopen("/sys/class/gpio/export","w");
	if(exp == NULL){
		printf("Error opening gpio export file!\n");
		return 1;
	}
	//write to export file
	fprintf(exp, "%s", pin);
	//close export file
	int close = fclose(exp);
	if(close != 0){
		printf("Error closing gpio export file!\n");
		return 1;
	}
	return 0;
}

int unexport(char *pin) //cleans up GPIO drivers for LEDs
{ //USAGE: unexport(GPIOpin#)
	printf("Unexport GPIO%s\n", pin);
	FILE *unexp = fopen("/sys/class/gpio/unexport","w");
	if(unexp == NULL){
		printf("Error opening gpio unexport file!\n");
		return 1;
	}
	fprintf(unexp, "%s", pin); //writes the file
	int close = fclose(unexp);
	if(close != 0){
		printf("Error closing gpio unexport file!\n");
		return 1;
	}
	return 0;
}

void direct(){ //sets direction files (to 'out')
	//create path to 'direction'
	char *dirpath = "/sys/class/gpio/gpio12/direction";
	printf("Set Direction 'out', dirpath = %s\n", dirpath);
	FILE *dir = fopen(dirpath, "w"); //open direction file
	fprintf(dir, "out"); //write "out" to direction file
	int close = fclose(dir); //close the direction file

	char *dirpath2 = "/sys/class/gpio/gpio13/direction";
	printf("Set Direction 'out', dirpath = %s\n", dirpath2);
	FILE *dir2 = fopen(dirpath2, "w"); //open direction file
	fprintf(dir2, "out"); //write "out" to direction file
	int close2 = fclose(dir2); //close direction file

	char *dirpath3 = "/sys/class/gpio/gpio14/direction";
	printf("Set Direction 'out', dirpath = %s\n", dirpath3);
	FILE *dir3 = fopen(dirpath3, "w"); //open direction file
	fprintf(dir3, "out"); //write "out" to direction file
	int close3 = fclose(dir3); //close direction file

	char *dirpath4 = "/sys/class/gpio/gpio15/direction";
	printf("Set Direction 'out', dirpath = %s\n", dirpath4);
	FILE *dir4 = fopen(dirpath4, "w"); //open direction file
	fprintf(dir4, "out"); //write "out" to direction file
	int close4 = fclose(dir4); //close direction file
}

void red(char *pwr) //turns red LED on or off
{
	FILE *val = fopen("/sys/class/gpio/gpio13/value", "w");
	fprintf(val, "%s", pwr); //1 or 0, ON or OFF respectively
	int close = fclose(val);
	if (strcmp(pwr,"1") ==0){ //print led on or off to terminal
		printf("Red led ON\n");	
	}
	else {
		printf("Red led OFF\n");	
	}
}

void grn(char *pwr) //turns green LED on or off
{
	FILE *val = fopen("/sys/class/gpio/gpio12/value", "w");
	fprintf(val, "%s", pwr); //1 or 0, ON or OFF respectively
	int close = fclose(val);
	if (strcmp(pwr,"1") ==0){ //print led on or off to terminal
		printf("Green led ON\n");	
	}
	else {
		printf("Green led OFF\n");	
	}
}

void lblu(char *pwr) //turns left blue LED on or off
{
	FILE *val = fopen("/sys/class/gpio/gpio15/value", "w");
	fprintf(val, "%s", pwr); //1 or 0, ON or OFF respectively
	int close = fclose(val);
	if (strcmp(pwr,"1") ==0){ //print led on or off to terminal
		printf("Left Blue led ON\n");	
	}
	else {
		printf("Left Blue led OFF\n");	
	}
}

void rblu(char *pwr) //turns right blue LED on or off
{
	FILE *val = fopen("/sys/class/gpio/gpio14/value", "w");
	fprintf(val, "%s", pwr); //1 or 0, ON or OFF respectively
	int close = fclose(val);
	if (strcmp(pwr,"1") ==0){ //print led on or off to terminal
		printf("Right Blue led ON\n");	
	}
	else {
		printf("Right Blue led OFF\n");	
	}
}

int main(int argc, char *argv[]) //this function ties it all together
{
	if (argc != 2){
		printf("USAGE: ks2blnk <# of blinks>\n");
		return 1;
	}

	char *gpio1 = "12";
	char *gpio2 = "13";
	char *gpio3 = "14";
	char *gpio4 = "15";
	int blinks = atoi(argv[1]);

	export(gpio1);
	export(gpio2);
	export(gpio3);
	export(gpio4);

	direct();

	int i = 1;
	while (i <= blinks) {
	/* Change this while loop in order to change the blinking pattern*/
		printf("Blink number: %d\n", i);
		red("1"); //red on
		lblu("1"); //leftblu on
		sleep(1); //wait 1 second
		red("0"); //red off
		lblu("0");//leftblu off
		grn("1"); //green on
		rblu("1"); //rightblu on
		sleep(1); //wait 1 second
		grn("0"); //green off
		rblu("0"); //rightblu off
		i++;
	}
	
	unexport(gpio1);
	unexport(gpio2);
	unexport(gpio3);
	unexport(gpio4);

	return 0;
}
