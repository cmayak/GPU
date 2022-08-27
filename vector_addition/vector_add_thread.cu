#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <math.h>

#define N 536870912

using namespace std;
__global__ void vector_add(float *out, float *a, float *b, int n){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid < n) 
		out[tid]=a[tid]+b[tid];
}

int main(){
	// TRANSFER DATA FROM HOST TO DEVICE FOR FUNCTION TEST
	float *a, *b, *out;
	float *d_a,*d_b,*d_out;

	// Initilize HOST memory
	a = (float*)malloc(sizeof(float)*N);
	b = (float*)malloc(sizeof(float)*N);
	out = (float*)malloc(sizeof(float)*N);
	memset(a,0,sizeof(float)*N);
	memset(b,0,sizeof(float)*N);
	memset(out,0,sizeof(float)*N);

	// Assign HOST memory values
	for (int j=0; j<N; j++){
		a[j] = 1.0f; b[j] = 2.0f;
	}

	// ALLOCATE DEVICE MEMORY SPACE
	cudaMalloc((void **)&d_a,sizeof(float)*N);
	cudaMalloc((void **)&d_b,sizeof(float)*N);
	cudaMalloc((void **)&d_out,sizeof(float)*N);

	cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(d_out,out,sizeof(float)*N,cudaMemcpyHostToDevice);

	// Execute DEVICE function
	int blockSize = 32;
	int gridSize = ceil((float) N/blockSize);
	vector_add<<<gridSize,blockSize>>>(d_out,d_a,d_b,N);

	// TRANSFER DATA FROM DEVICE TO HOST
	//cudaMemcpy(a,d_a,sizeof(float)*N,cudaMemcpyDeviceToHost);
	//cudaMemcpy(b,d_b,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(out,d_out,sizeof(float)*N,cudaMemcpyDeviceToHost);

	// Print Results after DEVICE FUNCTION to HOST DATA
	//for(int i=0; i<N; i++){
	//	printf("a[%d]: %f + b[%d] %f = %f\n", i,a[i],i,b[i], out[i]);
	//}
	// FREE DEVICE & HOST MEMORY 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	free(a);
	free(b);
	free(out);
}
