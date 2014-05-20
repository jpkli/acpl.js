#include <stdlib.h>
#include <stdio.h>
#include "testthrust.hpp"
//#include "testcuda.h"
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/sort.h>
using namespace std;

//extern "C" 
//void test_thrust(void);

int main(void){
   test_thrust();

/*
   float* h_idata = (float*) malloc(1);

   h_idata[0] = 0.0f;


   test();
*/
   return 0;
}
