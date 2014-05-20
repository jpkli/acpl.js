#include <stdlib.h>
#include <stdio.h>
#include "acp.hpp"
#include <math.h>       
#include <sys/time.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "testcuda.h"
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/sort.h>
using namespace std;

//extern "C" 
//void test_thrust(void);
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main( int argc, char** argv){
    int N = 1024 * 1024 * 16;
    clock_t cpu0, cpu1;
    double wall0,wall1;

    if(argc > 1) {
         N = atoi(argv[1]);
    }
    float *h_A = (float*) malloc( sizeof(float) * N );
    float *h_B = (float*) malloc( sizeof(float) * N );


    for( int i = 0; i < N; ++i)
    {
      h_A[i] = (float) static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
      h_B[i] = (float) static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
    }

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    acp::sort(h_A, N);
    //acp::transform(h_A, h_B, N, "+");
    //  Stop timers

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //cout << "Transform - Elapsed Time = " << elapsedTime << endl;
    cout << "," << elapsedTime << endl;
/*
    for( int i = 0; i < N; ++i)
    {
      h_A[i] = (float) static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
    }    


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    acp::reduce(h_A, N, "+");
    //  Stop timers

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    
    elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Reduce - Elapsed Time (ms) = " << elapsedTime << endl;

    
    for( int i = 0; i < N; ++i)
    {
      h_A[i] = (float) static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    acp::sort(h_A, N);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //cout << "Sort - Elapsed Time (ms) = " << elapsedTime << endl;
    cout << "," << elapsedTime << endl;

/*
    for( int i = 0; i < N; ++i)
    {
      h_A[i] = (float) static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
      h_B[i] = (float) static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
    }
    wall0 = get_wall_time();
    cpu0 = clock();


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    acp::scan(h_A, N);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Scan - Elapsed Time (ms) = " << elapsedTime << endl;
*/
   free(h_A);
   free(h_B);
   return 0;
}