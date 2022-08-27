#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <math.h>
#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "cuda.h"

#define N 1000

inline
cudaError_t checkCuda(cudaError_t result) 
{
#if defined(DEBUG) || defined(_DEBUG)
	if(result != cudaSuccess){
		fprintf(stderr,"CUDA RUNTIME ERROR: %s\n",cudaGetErrorString(result));
	assert(result==cudaSuccess);
	}
#endif
	return result;
}

__global__ void vector_add(unsigned long int *out, unsigned long int *a, unsigned long int *b, int n){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid < n) 
		out[tid]=a[tid]+b[tid];
}

__global__ void vector_multiply(unsigned long int *out, unsigned long int *a, unsigned long int *b, int n){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid < n) 
		out[tid]=a[tid]*b[tid];
}

__global__ void vector_shl(unsigned long int *out, unsigned long int *a, unsigned long int *b, int n){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid < n) 
		out[tid]=a[tid]<<b[tid];
}

__global__ void vector_shr(unsigned long int *out, unsigned long int *a, unsigned long int *b, int n){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid < n) 
		out[tid]=a[tid]>>b[tid];
}

void printResults(unsigned long int *data_){
	for (int i=0;i<N;i++){
		printf("%lu\n",data_[i]);
	}
}

int main(){
	int count;
	checkCuda(cudaGetDeviceCount(&count));
	printf("Device count: %d\n",count);
	// INFO & ALLOCATE LOOP
	for (int i=0; i<count;i++){
		size_t freeDevMem,totalDevMem;
		checkCuda(cudaSetDevice(i));
		cudaDeviceProp dev;
		checkCuda(cudaGetDeviceProperties(&dev,i));
		checkCuda(cudaMemGetInfo(&freeDevMem,&totalDevMem));
		printf("Device: %s\n", dev.name);
		printf("\tFree device memory: %lu, Total device memory: %lu\n", freeDevMem,totalDevMem);
		printf("\tMax Threads per block: %d\n",dev.maxThreadsPerBlock);
		printf("\tMax Threads Dim: %d,%d,%d,%d\n", dev.maxThreadsDim[0],dev.maxThreadsDim[1],dev.maxThreadsDim[2],dev.maxThreadsDim[3]);
		printf("\tMax Grid Size: %d,%d,%d,%d\n", dev.maxGridSize[0],dev.maxGridSize[1],dev.maxGridSize[2],dev.maxGridSize[3]);
		printf("\tMap Host data: %s\n", dev.canMapHostMemory ? "True":"False");

		unsigned long int *h_a, *h_b, *h_out, *d_a, *d_b, *d_out;
		h_a = (unsigned long int*)malloc(N*sizeof(unsigned long int));
		h_b = (unsigned long int*)malloc(N*sizeof(unsigned long int));
		h_out = (unsigned long int*)malloc(N*sizeof(unsigned long int));
		//memset(h_out,0,sizeof(unsigned long int)*N);

		for (unsigned long int i=0;i<N;i++){
			h_a[i] = i;
			h_b[i] = 4;
		}

		checkCuda(cudaMallocHost((void **)&d_a,N*sizeof(unsigned long int)));
		checkCuda(cudaMallocHost((void **)&d_b,N*sizeof(unsigned long int)));
		checkCuda(cudaMallocHost((void **)&d_out,N*sizeof(unsigned long int)));

		checkCuda(cudaMemcpy(d_a, h_a, N*sizeof(unsigned long int),cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_b, h_b, N*sizeof(unsigned long int),cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_out, h_out, N*sizeof(unsigned long int),cudaMemcpyHostToDevice));

		int gridSize=100;
		int blockSize=100;
		//vector_add<<<gridSize,blockSize>>>(d_out,d_a,d_b,N);
		vector_multiply<<<gridSize,blockSize>>>(d_out,d_a,d_b,N);
		//vector_shl<<<gridSize,blockSize>>>(d_out,d_a,d_b,N);
		//vector_shr<<<gridSize,blockSize>>>(d_out,d_a,d_b,N);

		checkCuda(cudaMemcpy(h_out,d_out,N*sizeof(unsigned long int),cudaMemcpyDeviceToHost));
		//cudaMemcpy(h_a,d_a,N*sizeof(unsigned long int),cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_b,d_b,N*sizeof(unsigned long int),cudaMemcpyDeviceToHost);
		//printResults(h_a);
		//printResults(h_b);
		printResults(h_out);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_out);
		cudaFree(h_a);
		cudaFree(h_b);
		cudaFree(h_out);
	}

	return 0;
/*
	// RUN KERNEL LOOP
	int gridSize=1;
	int blockSize=1;
	for (int i=0; i<count;i++){
		cudaSetDevice(i);
		vector_add<<<gridSize,blockSize>>>(d_out,d_a,d_b,N);
		printf("GPU FUNCTION...\n");
	}


	// ALLOCATE DATA BACK TO HOST
	for (int i=0;i<count;i++){
		printf("Device: %d",i);
		cudaSetDevice(i);
		cudaMemcpy(h_out,d_out,N*sizeof(unsigned long int),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_a,d_a,N*sizeof(unsigned long int),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_b,d_b,N*sizeof(unsigned long int),cudaMemcpyDeviceToHost);
		for (int j=0; i<N;j++){
			printf("%lu\n",h_out[j]);
		}
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_out);
	}


	// FREE DEVICE & HOST MEMORY 
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_out);
*/
	
}
