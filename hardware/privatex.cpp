#include <omp.h>
#include <iostream>
using namespace std;

int main()
{
     std::string a = "m", b = "y";
     
     #pragma omp parallel private(a) shared(b) num_threads(2){
         a += "p";
         b += "?";
         std::cout << "Inside of parallel region 1: \n a is (" << a << "), b is (" << b << ")\n";
         //a should be p because private(a) created an empty local variable of the same name
         //each thread has its own copy of variable a, so only one p is seen
         //b should be y?? because ? was appended by each of the two threads
     }
     std::cout << "Outside of parallel region 1: \n a is (" << a << "), b is (" << b << ")\n";
     //a should be m because the variable was not changed outside of the parallel region
     //b should be y?? because changes to a shared variable are global

     #pragma omp parallel firstprivate(a) shared(b) num_threads(2){
         a += "p";
         b += "?";
         std::cout << "Inside of parallel region 2: \n a is (" << a << "), b is (" << b << ")\n";
         //a should be mp because firstprivate(a) created a copy of the original value under the same name
         //only one p is seen because each thread has its own copy
         //b should be y???? because ? was appended by each of the two threads
         //it adds on from the last parallel region since b is a shared variable in both cases
     }
     std::cout << "Outside of parallel region 2: \n a is (" << a << "), b is (" << b << ")\n";
     //a should be m because the variable was not changed outside of the parallel region
     //b should be y???? because changes to a shared variable are global
}
