#include "acp.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>


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

__global__ void
cudaVectorAdd(float *A, float *B, int numElements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < numElements) {
        A[tid] = A[tid] + B[tid];
    }
}

namespace acp {

    void testCUDA(float* h_A, float* h_B, int N) {
        float *d_A = NULL, *d_B = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_A, N) );
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_B, N) );
        
        CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, N, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, N, cudaMemcpyHostToDevice) );

        int threads_per_block = 512;
        int num_blocks = N / threads_per_block + 1;

        dim3  grid(num_blocks, 1, 1);
        dim3  threads(threads_per_block, 1, 1);

        cudaVectorAdd<<<grid, threads>>>(d_A, d_B, N);
        
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) 
        {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        CUDA_SAFE_CALL( cudaMemcpy(h_A, d_A, N, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL(cudaFree(d_A));
        CUDA_SAFE_CALL(cudaFree(d_B));
    }

    void testThrust(void) {
        int N = 1<<24;
    
        thrust::host_vector<int> hv( N );
        thrust::generate(hv.begin(), hv.end(), rand);

        thrust::device_vector<int> dv = hv;
        thrust::sort(dv.begin(), dv.end());
        thrust::copy(dv.begin(), dv.end(), hv.begin());

        //std::cout << hv[0] << " - " << hv[1024] << std::endl;
    }

    void sort(float* hV, int N) {
        thrust::device_vector<float> dV(hV, hV+N);
        thrust::sort(dV.begin(), dV.end());
        thrust::copy(dV.begin(), dV.end(), hV);
    }


    void transform(float* hA, float* hB, int N, const char* op) {

        thrust::device_vector<float> dA(hA, hA+N);
        thrust::device_vector<float> dB(hB, hB+N);
        thrust::device_vector<float> dC(N);

        if( strcmp(op, "-") == 0 ){
            thrust::transform(dA.begin(), dA.end(), dB.begin(), dC.begin(), thrust::minus<float>());
        } else if( strcmp(op, "*") == 0 ) {
            thrust::transform(dA.begin(), dA.end(), dB.begin(), dC.begin(), thrust::multiplies<float>());
        } else if( strcmp(op, "/") == 0 ) {
            thrust::transform(dA.begin(), dA.end(), dB.begin(), dC.begin(), thrust::divides<float>());
        } else {
            thrust::transform(dA.begin(), dA.end(), dB.begin(), dC.begin(), thrust::plus<float>());
        }

        thrust::copy(dC.begin(), dC.end(), hA);
    
    }

    void scan(float* hA, int N) {

        thrust::device_vector<float> dA(hA, hA+N);
        thrust::exclusive_scan(dA.begin(), dA.end(), dA.begin());
        thrust::copy(dA.begin(), dA.end(), hA);
    
    }    

    float reduce(float* hA, int N, const char* op) {

        thrust::device_vector<float> dA(hA, hA+N);
        float result;

        if( strcmp(op, "-") == 0 ){
            result = thrust::reduce(dA.begin(), dA.end(), 0, thrust::minus<float>());
        } else if( strcmp(op, "*") == 0 ) {
            result = thrust::reduce(dA.begin()+1, dA.end(), hA[0], thrust::multiplies<float>());
        } else if( strcmp(op, "/") == 0 ) {
            result = thrust::reduce(dA.begin()+1, dA.end(), hA[0], thrust::divides<float>());
        } else if( strcmp(op, "max") == 0 ) {
            result = thrust::reduce(dA.begin(), dA.end(), hA[0], thrust::maximum<float>());
        } else if( strcmp(op, "min") == 0 ) {
            result = thrust::reduce(dA.begin(), dA.end(), hA[0], thrust::minimum<float>());
        } else {
            result = thrust::reduce(dA.begin(), dA.end(), 0, thrust::plus<float>());
        }

        return result;
    
    }

}