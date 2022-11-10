#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include <string.h>
#include <iomanip>
#include <sstream>
#include <map>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
using namespace std;

typedef uint32_t WORD;
typedef unsigned char BYTE;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	    if (code != cudaSuccess) 
		        {
				        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
					        if (abort) { exit(code); getchar(); }
						    }
}

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define N32 4294967296
map<char,int> ascii_hex {{'0',0}, {'1',1}, {'2',2}, {'3',3}, {'4',4}, {'5',5}, {'6',6}, {'7',7}, {'8',8}, {'9',9}, {'A',10}, {'B',11}, {'C',12}, {'D',13}, {'E',14}, {'F',15},};
map<int,char> ascii_int {{0,'0'}, {1,'1'}, {2,'2'}, {3,'3'}, {4,'4'}, {5,'5'}, {6,'6'}, {7,'7'}, {8,'8'}, {9,'9'}, {10,'A'}, {11,'B'}, {12,'C'}, {13,'D'}, {14,'E'}, {15,'F'},};

int primes[64] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311};


void k2_8(unsigned long int *k2,int *primes_){                                                          
	for (int i=0;i<8;i++){                                                                              
		k2[i] = (sqrt(primes_[i])-floor(sqrt(primes_[i])))*pow(2,32);
	}                                                                                                   
}                                                                                                       

void k3_64(unsigned long int *k3,int *primes_){                                                         
	for (int i=0;i<64;i++){                                                                             
		k3[i] = (cbrt(primes_[i])-floor(cbrt(primes_[i])))*pow(2,32);
	}                                                                                                   
}


unsigned long int hex_int(string in){               
	unsigned long int x;                            
	stringstream ss;                                
	ss<<hex<<in;                                    
	ss>>x;                                          
	return x;                                       
}                                                   

string int_hex(unsigned long int in){               
	stringstream ss;                                
	ss<<setw(8)<<setfill('0')<<hex<<in;             
	return ss.str();                                
}      

char*** pad_chara(char *in){
	stringstream ss;
	ss<<in<<'8'<<setfill('0')<<setw(95)<<hex<<(strlen(in)*4);
	const string pad = ss.str();
	const char* padd = pad.c_str();
	cout << strlen(padd) << endl;
	char ***chara = (char***)malloc(sizeof(char**)*(strlen(padd)/128));
	for (int i=0;i<(strlen(padd)/128);i++){
		chara[i] = (char**)malloc(sizeof(char*)*64);
		for (int j=0;j<64;j++){
			chara[i][j] = (char*)malloc(sizeof(char)*8);
			for(int k=0;k<8;k++){
				chara[i][j][k] = padd[i*128+j*8+k];
			}
		}
	}
	return chara;
}

unsigned long int char_8_int(char* in){
	unsigned long int out=0;
	for (int i=7,j=0;i>0,j<8;i--,j++){
		out+= ascii_hex[in[j]]*pow(16,i);
	}
	return out;
}

char* int_8_char(unsigned long int in){
	char *out = (char*)malloc(sizeof(char)*8);
	for(int i=0;i<8;i++){
		out[i] = ascii_int[in%16];
		in /= 16;
	}
	reverse(out,out+strlen(out));
	return out;
}

//	return ((rotr(i2,17)^rotr(i2,19)^shr(i2,10))+i7+(rotr(i15,7)^rotr(i15,18)^shr(i15,3))+i16)%N32;

template<typename T>
__device__ T d_rotr(T x, unsigned int n){
	return (x>>n)|(x<<(32-n));
}

template<typename T>
__device__ T d_shr(T x,unsigned int n){
	return x>>n;
}

template<typename T>
__device__ T d_wt(T i_2,T i_7,T i_15,T i_16){
	return ((d_rotr(i_2,17)^d_rotr(i_2,19)^d_shr(i_2,10))+i_7+(d_rotr(i_15,7)^d_rotr(i_15,18)^d_shr(i_15,3))+i_16)%N32;
}

template<typename T>
__device__ T d_t1d(T d,T e, T f, T g, T h, T k3, T m){
	return (d + (h + EP1(e) + CH(e,f,g) + k3 + m)%N32)%N32;
}

template<typename T>
__device__ T d_t12(T a, T b, T c, T e, T f, T g, T h, T k3, T m){
	return (((EP1(e) + CH(e,f,g) + h + k3 + m)%N32) + (EP0(a) + MAJ(a,b,c))%N32)%N32;
}

template<typename T>
__device__ T* d_h0(T *h0, T k3, T m){
	unsigned long int t12 = d_t12(h0[0],h0[1],h0[2],h0[4],h0[5],h0[6],h0[7],k3,m);
	unsigned long int t1d = d_t1d(h0[3],h0[4],h0[5],h0[6],h0[7],k3,m);
	h0[7] = h0[6];
	h0[6] = h0[5];
	h0[5] = h0[4];
	h0[4] = t1d;
	h0[3] = h0[2];
	h0[2] = h0[1];
	h0[1] = h0[0];
	h0[0] = t12;
	return h0;
}

template<typename T>
__device__ T d_pow(T n, T x){
	unsigned long long int out = 1;
	for(int i=0;i<n;i++){
		out *= x;
	}
	return out;
}

__device__ unsigned long int* target(unsigned char *&s, unsigned long int *&o){
	unsigned char in[9];
	unsigned char tar[65];
	unsigned int x=0;
	unsigned int exp=0;
	o = new unsigned long int[8];
	for(int h=0,i=144;h<8,i<152;h++,i++){
		if(149<i && i<152){
			x = (s[i] >= 'A') ? (s[i] - 'A' + 10) : (s[i] - '0');
			exp += d_pow((unsigned int)(151-i),(unsigned int)16)*x;
		}
		in[h] = s[i];
	}
	exp = exp*2 - 6;
	for(int i=0;i<65;i++){
		if(i<(64-exp))
			tar[i] = '0';
		if((64-exp-1)<i && i<(64-exp+6)){
			tar[i] = in[64-exp-i+5];
		}
		else
			tar[i] = '0';
	}

	unsigned long int y=0;
	unsigned int i=65;
	for(unsigned int j=0;j<i;j++){
		if(j%8 == 0 && j>0){
			o[(j/8)-1] = y;
			y = 0;
		}
		x = (tar[j] >= 'A') ? (tar[j] - 'A' + 10) : (tar[j] - '0');
		y += d_pow(7-j%8,(unsigned int)16)*x;
	}
	o[7] = y;
}

__device__ unsigned long int* d_padd_s(unsigned char *&s,unsigned long int *&ino, unsigned long int &nblocks){
	unsigned int i=0;
	unsigned int x = 0;
	unsigned long int y = 0;
	unsigned long int *z;
	while(s[i] != '\0'){
		i++;
	}
	z = new unsigned long int[i/8];
	for(unsigned int j=0;j<i;j++){
		if(j%8 == 0 && j>0){
			z[(j/8)-1] = y;
			y = 0;
		}
		x = (s[j] >= 'A') ? (s[j] - 'A' + 10) : (s[j] - '0');
		y += d_pow(7-j%8,(unsigned int)16)*x;
	}
	z[7] = y;
	unsigned int n = i/8;
	nblocks = (n/16) + ((n%16) !=0);
	unsigned long int *nno = new unsigned long int[nblocks*16];
	for(int i=0;i<(nblocks*16);i++){
		if(i<n)
			nno[i] = z[i];
		if(i==n)
			nno[i] = d_pow((unsigned int)7,(unsigned int)16)*8;
		if(i>n && i<(nblocks*16)-1)
			nno[i] = 0;
		if(i==(nblocks*16)-1)
			nno[i] = n*32;
	}
	return nno;
	delete ino;
	delete z;
}


__device__ void hash_block(unsigned char *&s, unsigned long int *&h0i, unsigned long int *&ik2, unsigned long int* iblock, unsigned long int* ik3, unsigned long int nonce, unsigned long int nblocks){

	iblock = d_padd_s(s,iblock,nblocks);

	ik2 = new unsigned long int[8];
	for(int i=0;i<8;i++)
		ik2[i] = h0i[i];

	unsigned long int h0[8];
	unsigned long int block[64];
	for(int i=0;i<nblocks;i++){
		for(int j=0;j<8;j++){
			h0[j] = ik2[j];
		}
		for(int k=0;k<64;k++){
			if(k<16){
				block[k] = iblock[(i*16)+k];
				if(i==1 && k==3)
					block[k] = nonce;
			}
			else
				block[k] = d_wt(block[k-2], block[k-7], block[k-15], block[k-16]);
			ik2 = d_h0(ik2,ik3[k],block[k]);
		}
		for(int l=0;l<8;l++){
			ik2[l] = (ik2[l] + h0[l])%N32;
		}
	}
	s = new unsigned char[65];
	unsigned long temp[8]; 
	for(int i=0;i<8;i++)
		temp[i]= ik2[i];
	for(int i=0;i<8;i++){
		for(int j=7;j>-1;j--){
			s[i*8+j] = (char)((temp[i]%16)<10) ? (temp[i]%16)+48 : (char)(temp[i]%16)-10+65;
			temp[i]/=16;
		}
	}

}


__device__ void check_tar(unsigned long int *&it, unsigned long int *&h0, bool &v){
	unsigned int i=0;
	while(!v){
		if(h0[i] < it[i]){
			printf("True\n");
			v = true;
		}
		i++;
	}
}

__device__ void swap_endian(unsigned char *&in){
	unsigned char temp[65];
	for(int i=0;i<64;i=i+2){
		temp[64-i] = in[i];
		temp[63-i] = in[i+1];
	}
	temp[65] = '\0';
	in = temp;

	unsigned long int y=0;
	unsigned long int x=0;
	unsigned long int *z = new unsigned long int[8];
	for(unsigned int j=0;j<64;j++){
		if(j%8 == 0 && j>0){
			z[(j/8)-1] = y;
			printf("%lu\t",y);
			y = 0;
		}
		x = (in[j] >= 'A') ? (in[j] - 'A' + 10) : (in[j] - '0');
		y += d_pow(7-j%8,(unsigned int)16)*x;
	}
	z[7] = y;
	printf("%lu\t",z[7]);
}

__global__ void hash_256(unsigned char *ins, unsigned char *ot, unsigned long int *k2, unsigned long int *k3, unsigned long int offset, bool valid){
	int blockid = (gridDim.x * blockIdx.y) + blockIdx.x;
	int threadid = (blockid * blockDim.x) + threadIdx.x;
	__shared__ unsigned long int *o;
	__shared__ unsigned long int* tar;
	unsigned long int nonce = threadid+offset;
	unsigned long int nblocks;
	unsigned long int *h0i;
	__syncthreads();
	target(ins,tar);
	hash_block(ins,k2,h0i,o,k3,nonce,nblocks); //1
	hash_block(ins,k2,h0i,o,k3,nonce,nblocks); //2
	//for(int i=0;i<8;i++)
	//	printf("%lu\t",h0i[i]);
	check_tar(tar,h0i,valid);
	swap_endian(ins);
	printf("\nOUT:%s\n",ins);
}

void device_info(cudaDeviceProp &dp, int id){
	cudaSetDevice(id);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp,id);
	printf("Device Global Mememory %lu\n", devProp.totalGlobalMem);
	printf("Device shared memory per block %lu\n", devProp.sharedMemPerBlock);
	printf("Device shared memory per multiprocessor %lu\n", devProp.sharedMemPerMultiprocessor);
	printf("Device Max threads per block %d\n", devProp.maxThreadsPerBlock);
	printf("Device Multiprocessor Count %d\n", devProp.multiProcessorCount);
	printf("Device %d\n",id);
}

int main(int argc, char **argv){
	unsigned char ar[161] = "0100000000000000000000000000000000000000000000000000000000000000000000003BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A29AB5F49FFFF001D00000000";
//	char goal[258] = "0100000000000000000000000000000000000000000000000000000000000000000000003BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A29AB5F49FFFF001D1DAC2B7C";
	unsigned char t[257];
	unsigned long int k2[8];
	unsigned long int k3[64];
	k2_8(k2,primes);
	k3_64(k3,primes);

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp,0);
	int count;
	cudaGetDeviceCount(&count);
	unsigned long int *d_k2, *d_k3;
	unsigned char *d_s, *ot;
	//unsigned long int offset = 1835008;
	unsigned long int offset = 497822588;
	unsigned long int threads;
	unsigned long int blocks = 1;
	bool valid = false;
	bool *d_valid;

	for (int i=0;i<count;i++){
		device_info(devProp,i);
		//threads = devProp.maxThreadsPerBlock;
		threads = 1;

		cudaMallocHost((void**)&d_s,sizeof(unsigned char)*257);
		cudaMallocHost((void**)&ot,sizeof(unsigned char)*65);
		cudaMallocHost((void**)&d_k2,sizeof(unsigned long int)*8);
		cudaMallocHost((void**)&d_k3,sizeof(unsigned long int)*64);
		cudaMallocHost((void**)&d_valid,sizeof(bool));
		cudaMemcpy(d_s,&ar,sizeof(unsigned char)*257,cudaMemcpyHostToDevice);
		cudaMemcpy(d_k2,&k2,sizeof(unsigned long int)*8,cudaMemcpyHostToDevice);
		cudaMemcpy(d_k3,&k3,sizeof(unsigned long int)*64,cudaMemcpyHostToDevice);
		cudaMemcpy(d_valid,&valid,sizeof(bool),cudaMemcpyHostToDevice);
		//hash_256<<<1,2>>>(d_s,ot,d_k2,d_k3,i*2);
		//cudaPeekAtLastError();
		//cudaDeviceSynchronize();
		////gpuErrchk(cudaMemcpy(&t,ot,sizeof(unsigned char)*65,cudaMemcpyDeviceToHost));
		//cudaMemcpy(t,&ot,sizeof(unsigned char)*65,cudaMemcpyDeviceToHost);
		//printf("OUT:%s\n",t);
		//cudaFree(d_k2);
		//cudaFree(d_k3);
	}

	unsigned long long int zuper = 18446744073709;
	unsigned long long int z = 0;

	while(z < zuper){
		for(int i=0;i<count;i++){
			//system("clear");
			auto start = std::chrono::high_resolution_clock::now();
			//for(unsigned long int j=(N32/count)*i;j<(N32/count)*(i+1);j=j+(blocks*threads)){
			for(unsigned long int j=497822588;j<497822589;j++){
				hash_256<<<blocks,threads>>>(d_s,ot,d_k2,d_k3,j,d_valid);
				cudaMemcpy(&valid,d_valid,sizeof(bool),cudaMemcpyDeviceToHost);
				printf("%lu:%d\n",j,valid);
				if(valid)
					break;
			}
			if(valid){
				printf("Valid\n");
				exit(1);
			}
			auto finish = std::chrono::high_resolution_clock::now();
			auto dif_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
			auto dif_s = std::chrono::duration_cast<std::chrono::seconds>(finish-start).count();
			zuper = zuper + (offset*count*threads*blocks);
			//printf("%llu:%llu:%lu:%lu\n",zuper,zuper/(dif_ns),dif_s,dif_ns);
		}
		if(z<zuper){
			exit(1);
		}
	}

	for(int i=0;i<count;i++){
		cudaMemcpy(&t,ot,sizeof(unsigned char)*65,cudaMemcpyDeviceToHost);
		printf("OUT:%s\n",t);
	}

	for(int i=0;i<count;i++){
		cudaFree(d_k2);
		cudaFree(d_k3);
	}
}
