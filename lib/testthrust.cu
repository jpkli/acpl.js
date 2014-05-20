#include "testthrust.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void test_thrust(void) {
    int N = 1<<24;
    thrust::host_vector<int> hv( N );
    thrust::generate(hv.begin(), hv.end(), rand);

    thrust::device_vector<int> dv = hv;

    thrust::sort(dv.begin(), dv.end());

    thrust::copy(dv.begin(), dv.end(), hv.begin());

/*
	for(int i=0; i<N; i++) {
		std::cout << hv[i] << " " ;
	}
*/
	std::cout << hv[0] << " - " << hv[1024] << std::endl;

}


