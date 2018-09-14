#include <stdio.h>

//hello2.c Expanded Helloworld Introduction
//Copyright 2015, Erik Biegert, All rights reserved.

int main() 
{
    char name[30];
    
    printf("=====Erik's Simple Program=====\n");
    printf("Please type your name and press enter: \n");

    scanf("%s", &name);

    if (strncmp(name,"Erik",4) == 0) 
        {
        printf("Hello %s, all glorious master. We bow to your presence.\n", name);
        }
    else 
        {
        printf("Hello %s, it's nice to meet you!\n", name);
        }
    printf("=====END PROGRAM=====\n");
    return 0;
}
