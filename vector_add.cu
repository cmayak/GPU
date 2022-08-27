#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <iterator>
#include <algorithm>

#define N 100000

__global__ void vector_add(float *out, float *a, float *b, int n){
	for (int i=0; i<n; i++){
		out[i]=a[i]+b[i];
	}
}

int main(){
	float *a, *b, *out;
	float *d_a,*d_b,*d_out;

	// Initilize HOST memory
	a = (float*)malloc(sizeof(float)*N);
	b = (float*)malloc(sizeof(float)*N);
	out = (float*)malloc(sizeof(float)*N);

	// Assign HOST memory values
	for (int i=0; i<N; i++){
		a[i] = 1.0f; b[i] = 2.0f;
	}

	// ALLOCATE DEVICE MEMORY SPACE
	cudaMalloc((void **)&d_a,sizeof(float)*N);
	cudaMalloc((void **)&d_b,sizeof(float)*N);
	cudaMalloc((void **)&d_out,sizeof(float)*N);

	// TRANSFER DATA FROM HOST TO DEVICE FOR FUNCTION TEST
	cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(d_out,out,sizeof(float)*N,cudaMemcpyHostToDevice);

	// Execute DEVICE function
	vector_add<<<1,256>>>(d_out,d_a,d_b,N);

	// TRANSFER DATA FROM DEVICE TO HOST
	cudaMemcpy(a,d_a,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(b,d_b,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(out,d_out,sizeof(float)*N,cudaMemcpyDeviceToHost);

	// Print Results after DEVICE FUNCTION to HOST DATA
	/*
	for(int i=0; i<N; i++){
		printf("a[%d]: %f + b[%d] %f = %f\n", i,a[i],i,b[i], out[i]);
	}
	*/

	// FREE DEVICE & HOST MEMORY 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	free(a);
	free(b);
	free(out);
}
