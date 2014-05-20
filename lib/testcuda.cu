#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda.h>
#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)

// includes, kernels
//#include <test1_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel template for flops test
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result=1;
    float r0=12345.6F, r1=23456.7F, r2=34567.8F, r3=45678.9F, r4=56789.0F, r5=98765.4F, r6=87654.3F, r7=76543.2F;
    float val1 = g_idata[idx];
    //float val2 = g_idata[1];

    for(int i=0; i<1024; i++){
        r0 = r7 * val1 + r2;
        r1 = r6 * val1 + r4;
        r2 = r5 * val1 + r6;
        r3 = r4 * val1 + r0;
        r4 = r3 * val1 + r7;
        r5 = r2 * val1 + r3;
        r6 = r1 * val1 + r5;
        r7 = r0 * val1 + r1;
        result = (r0*r1+r2) * (r3*r4+r5) + (r6*r7+result);
    }
    //result = val2 + (result * val1);

    g_idata[idx] = result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void testCUDA()
{
    unsigned int num_blocks, threads_per_block;
    if(argc < 3) {
        printf("\nusage: %s <num_blocks> <num_thread> \nUsing default setting ... \n\n", argv[0]);
        num_blocks = 12;
        threads_per_block = 512;
    } else {
        num_blocks = atoi(argv[1]);
        threads_per_block = atoi(argv[2]);
    }

    unsigned int num_threads = (int)num_blocks * (int)threads_per_block;
    unsigned int mem_size = sizeof(float) * num_threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t error;
    
    // allocate host memory
    float* h_idata = (float*) malloc(mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i)
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_data;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data, mem_size));
    CUDA_SAFE_CALL( cudaMemcpy( d_data, h_idata, mem_size, cudaMemcpyHostToDevice) );

    // setup execution parameters
    // adjust thread block sizes here
    dim3  grid(num_blocks, 1, 1);
    dim3  threads(threads_per_block, 1, 1);

    // execute the kernel
    cudaEventRecord(start, 0);
    testKernel<<< grid, threads>>>(d_data);
    error = cudaGetLastError();
    if (error != cudaSuccess) 
    {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }    

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    // check if kernel execution generated and error
    // CUT_CHECK_ERROR("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy(h_odata, d_data, sizeof( float) * num_threads, cudaMemcpyDeviceToHost) );

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Elapsed Time: %f ms\n", elapsedTime );
    printf( "Number of Threads: %d (%d x %d)\n", num_threads, num_blocks, threads_per_block);
    long double gflops = num_threads / (elapsedTime) / 1e6 * 24 * 1024;
    printf( "GFLOPS: %Lf \n", gflops);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cleanup memory
    free( h_idata);
    free( h_odata);
    CUDA_SAFE_CALL(cudaFree(d_data));
}


