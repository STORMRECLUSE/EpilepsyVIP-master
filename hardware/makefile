all: epilserv

epilserv: energy.o epilserv.o filter.o

	g++ -I. -ggdb -O2 -I/usr/include epilserv.o energy.o filter.o /usr/lib/libfftw_acc.a -lOpenCL -fopenmp -locl_util -lrt -lm -L/lib -L/usr/lib -Wl,-rpath-link,/lib -Wl,-rpath-link,/usr/lib -o epilserv

epilserv.o: epilserv.c energy.h filter.h
	g++ -I. -c epilserv.c -g

filter.o: filter.c filter.h
	g++ -I. -c filter.c -g

energy.o: energy.c energy.h
	g++ -I. -c energy.c  -g

clean: 
	rm energy.o epilserv.o filter.o	