#include "cuda.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include <iostream>

using namespace std;


void dev_info(){
	int deviceCount;
	int dev,driverVersion=0,runtimeVersion=0;
	int canAccess;
	cudaGetDeviceCount(&deviceCount);

	for(dev=0;dev<deviceCount;dev++){
//		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,dev);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("\n");

		printf("NAME: %s", deviceProp.name);
		printf("Cuda Driver Version: %d.%d\n", driverVersion/1000,(driverVersion%100)/10);
		printf("Cuda Runtime Version: %d.%d\n", runtimeVersion/1000,(runtimeVersion%100)/10);
		printf("Cuda Capability Major/Minor: %d.%d\n",deviceProp.major,deviceProp.minor);
		printf("Cuda total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
		printf("Cuda Multiprocessor count: %03d, Cuda Cores/MP: %d, Cuda Cores: %d\n", deviceProp.multiProcessorCount,(deviceProp.major*deviceProp.minor),((deviceProp.major*deviceProp.minor)*deviceProp.multiProcessorCount));
		printf("GPU MAX Clock rate: %dHz\n", deviceProp.clockRate);
		printf("GPU Memory Clock rate: %dHz\n", deviceProp.memoryClockRate);
		printf("GPU Bus memory width: %d\n", deviceProp.memoryBusWidth);

		printf("GPU L2 cache size: %d\n", deviceProp.l2CacheSize);
		printf("Max Texture Dimension Size(x,y,z): (%d,%d,%d)\n", deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
		printf("Total amount of constant memory: %luB\n", deviceProp.totalConstMem);
		printf("Total amount of shared memory per block: %luB\n", deviceProp.sharedMemPerBlock);
		printf("Total amount of shared memory per multiprocessor: %luB\n", deviceProp.sharedMemPerMultiprocessor);
		printf("Total number of registers available per block %d\n", deviceProp.regsPerBlock);
		printf("Warp size: %d\n", deviceProp.warpSize);
		printf("Max number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Max number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("Max dimension Size of thread block(x,y,z): (%d,%d,%d)\n", deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
		printf("Max dimension Size of grid size: (%d,%d,%d)\n", deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
		printf("Integrated GPU sharing Host Memory: %s\n", deviceProp.integrated ? "YES" : "NO");
		printf("Support for host page-locked memory mapping %s\n", deviceProp.canMapHostMemory ? "YES" : "NO");
		printf("Device Supports ECC: %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
		printf("Cuda Device Driver mode (TCC or WDDM): %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
		printf("Device supports Unified addressing(UVA): %s\n", deviceProp.unifiedAddressing ? "YES" : "NO");
		printf("Device supports Managed Memory: %s\n", deviceProp.managedMemory ? "YES" : "NO");
		printf("Device supports Compute Preemption: %s\n", deviceProp.computePreemptionSupported ? "YES" : "NO");
		printf("Cooperative Kernel Launch: %s\n", deviceProp.cooperativeLaunch ? "YES":"NO");
		printf("MultiDevice Co-op Kernel Launch: %s\n", deviceProp.cooperativeMultiDeviceLaunch ? "YES" : "NO");
		printf("Device PCI DOMAIN ID / BUS ID / Location ID: %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
		printf("Device compute Mode: < %d >\n", deviceProp.computeMode);
		cudaDeviceEnablePeerAccess(dev==0 ? 1:0,0);
		cudaDeviceCanAccessPeer(&canAccess,dev,dev==0 ? 1:0);
		printf("Device %d Can access peer dev %d: %d", dev, dev==0 ? 1:0,canAccess);
	}
	cudaDeviceReset();

}

#define DEFAULT_SIZE (256*(1e6))
#define DEFAULT_INCREMENT (4*(1e6))
#define MEMCOPY_ITERATIONS 100
#define FLUSH_SIZE (256*1024*1024)
#define CACHE_CLEAR_SIZE (16*(1e6))

#define SHMOO_MEMSIZE_MAX (64*(1e6))
#define SHMOO_MEMSIZE_START (1e3)
#define SHMOO_INCREMENT_1KB (1e3)
#define SHMOO_INCREMENT_2KB (2*1e3)
#define SHMOO_INCREMENT_10KB (10*1e3)
#define SHMOO_INCREMENT_100KB (100*1e3)
#define SHMOO_INCREMENT_1MB (1e6)
#define SHMOO_INCREMENT_2MB (2*1e6)
#define SHMOO_INCREMENT_4MB (4*1e6)
#define SHMOO_LIMIT_20KB (20*1e3)
#define SHMOO_LIMIT_50KB (50*1e3)
#define SHMOO_LIMIT_100KB (100*1e3)
#define SHMOO_LIMIT_1MB (1e6)
#define SHMOO_LIMIT_16MB (16*1e6)
#define SHMOO_LIMIT_32MB (32*1e6)



enum testMode { QUICK_MODE, RANGE_MODE, SHMOO_MODE };
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };

static bool bDontUseGPUTiming;
int *pArgc = NULL;
char **pArgv = NULL;
char *flush_buf;
static const char *sSDKsample = "CUDA Bandwidth Test";

const char *sMemoryCopyKind[] = {"Device to Host", "Host to Device", "Device to Device", NULL};
const char *sMemoryMode[] = {"PINNED", "PAGEABLE", NULL};

// Declerations
int runTest(const int argc, const char **argv);
void testBandwidth(unsigned int start, unsigned int end, unsigned int increment, testMode mode, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int incr, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);

float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testDeviceToDeviceTransfer(unsigned int memSize);

void printResultsReadable(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc);
void printResultsCSV(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc); 
void printHelp(void);
// Declerations

void printHelp(void) {
  printf("Usage:  bandwidthTest [OPTION]...\n");
  printf(
      "Test the bandwidth for device to host, host to device, and device to "
      "device transfers\n");
  printf("\n");
  printf(
      "Example:  measure the bandwidth of device to host pinned memory copies "
      "in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n");
  printf(
      "./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 "
      "--increment=1024 --dtoh\n");

  printf("\n");
  printf("Options:\n");
  printf("--help\tDisplay this help menu\n");
  printf("--csv\tPrint results as a CSV\n");
  printf("--device=[deviceno]\tSpecify the device device to be used\n");
  printf("  all - compute cumulative bandwidth on all the devices\n");
  printf("  0,1,2,...,n - Specify any particular device to be used\n");
  printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
  printf("  pageable - pageable memory\n");
  printf("  pinned   - non-pageable system memory\n");
  printf("--mode=[MODE]\tSpecify the mode to use\n");
  printf("  quick - performs a quick measurement\n");
  printf("  range - measures a user-specified range of values\n");
  printf("  shmoo - performs an intense shmoo of a large range of values\n");

  printf("--htod\tMeasure host to device transfers\n");
  printf("--dtoh\tMeasure device to host transfers\n");
  printf("--dtod\tMeasure device to device transfers\n");
#if CUDART_VERSION >= 2020
  printf("--wc\tAllocate pinned memory as write-combined\n");
#endif
  printf("--cputiming\tForce CPU-based timing always\n");

  printf("Range mode options\n");
  printf("--start=[SIZE]\tStarting transfer size in bytes\n");
  printf("--end=[SIZE]\tEnding transfer size in bytes\n");
  printf("--increment=[SIZE]\tIncrement size in bytes\n");
}

void printResultsCSV(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc){
	std::string sConfig;

	if (kind == DEVICE_TO_DEVICE){
		sConfig+="D2D";
	} else {
		if (kind == DEVICE_TO_HOST) {
			sConfig+="D2H";
		} else if (kind == HOST_TO_DEVICE){
			sConfig+="H2D";
		}
	if(memMode == PAGEABLE){
		sConfig+="-Paged";
	} else if (memMode == PINNED){
		sConfig+="-Pinned";
		if (wc){
			sConfig+="-WriteCombined";
		}
	}
	}

	unsigned int i;
	double dSeconds = 0.0;

	for (i=0; i<count; i++){
		dSeconds = (double)memSizes[i] / (bandwidths[i] * (double)(1e9));
		printf("bandwidthTest-%s, Bandwidth = %.1f GB/s, Time = %.5f s, Size = %u bytes, NumDevsUsed = %d\n", sConfig.c_str(), bandwidths[i], dSeconds, memSizes[i], iNumDevs);
	}
}

void printResultsReadable(unsigned int *memSizes, double *bandwidths,
                          unsigned int count, memcpyKind kind,
                          memoryMode memMode, int iNumDevs, bool wc) {
  printf(" %s Bandwidth, %i Device(s)\n", sMemoryCopyKind[kind], iNumDevs);
  printf(" %s Memory Transfers\n", sMemoryMode[memMode]);

  if (wc) {
    printf(" Write-Combined Memory Writes are Enabled");
  }

  printf("   Transfer Size (Bytes)\tBandwidth(GB/s)\n");
  unsigned int i;

  for (i = 0; i < (count - 1); i++) {
    printf("   %u\t\t\t%s%.1f\n", memSizes[i],
           (memSizes[i] < 10000) ? "\t" : "", bandwidths[i]);
  }

  printf("   %u\t\t\t%s%.1f\n\n", memSizes[i],
         (memSizes[i] < 10000) ? "\t" : "", bandwidths[i]);
}

int runTest(const int argc, const char **argv){
	int start = DEFAULT_SIZE;
	int end = DEFAULT_SIZE;
	int startDevice = 0;
	int endDevice = 1;
	int increment = DEFAULT_INCREMENT;
	testMode mode = QUICK_MODE;
	bool htod = false;
	bool dtoh = false;
	bool dtod = false;
	bool wc = false;
	char *modeStr;
	char *device = NULL;
	printMode printmode = USER_READABLE;
	char *memModeStr = NULL;
	memoryMode memMode = PINNED;

	if (checkCmdLineFlag(argc,argv,"help")){
		printHelp();
		return 0;
	}

	if(checkCmdLineFlag(argc,argv,"csv")){
		printmode = CSV;
	}

	if (getCmdLineArgumentString(argc,argv,"memory",&memModeStr)){
		if (strcmp(memModeStr,"pinned") == 0){
			memMode = PAGEABLE;
		} else if (strcmp(memModeStr,"pinned") == 0){
			memMode = PINNED;
		} else {
			printf("Invalid memory mode - valid modes are pageable or pinned\n");
			printf("See --help for more info\n");
			return -1000;
		}
	} else {
		memMode = PINNED;
	}

	if (getCmdLineArgumentString(argc,argv,"device",&device)){
		int deviceCount;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
		if(error_id != cudaSuccess){
			printf("cudaGetDeviceCount returned %d\n-> %s\n",(int)error_id,cudaGetErrorString(error_id));
			exit(EXIT_FAILURE);
		}
		if(deviceCount ==0){
			printf("!!!NO DEVICES FOUND!!!\n");
			return -2000;
		}

		if (strcmp(device,"all") == 0){
			printf("\n!!!Cumulative Bandwidth to be computed from all devices!!!!\n");
			startDevice=0;
			endDevice = deviceCount;
		} else {
			startDevice=endDevice=atoi(device);

			if(startDevice >= deviceCount || startDevice < 0){
				printf("\n!!!Invalid GPU number %d given hence default gpu %d will be used!!!\n",startDevice,0);
				startDevice=endDevice=0;
			}
		}
	}
	printf("\nRunning on... \n\n");
	for (int currentDevice=startDevice;currentDevice<=endDevice;currentDevice++){
		cudaDeviceProp deviceProp;
		cudaError_t error_id = cudaGetDeviceProperties(&deviceProp,currentDevice);

		if (error_id == cudaSuccess){
			printf("Device %d: %s\n", currentDevice, deviceProp.name);
			if (deviceProp.computeMode == cudaComputeModeProhibited){
				fprintf(stderr,"Error: device is running <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
				checkCudaErrors(cudaSetDevice(currentDevice));
				exit(EXIT_FAILURE);
			}
		} else {
			printf("cudaGetDeviceProperties returned %d\n -> %s\n", (int)error_id,cudaGetErrorString(error_id));
			checkCudaErrors(cudaSetDevice(currentDevice));
			exit(EXIT_FAILURE);
		}
	}

	if (getCmdLineArgumentString(argc,argv,"mode",&modeStr)){
		if (strcmp(modeStr,"quick") == 0){
			printf("Quick Mode \n\n");
			mode = QUICK_MODE;
		} else if(strcmp(modeStr,"shmoo") == 0) {
			printf("Shmoo Mode \n\n");
			mode = SHMOO_MODE;
		} else if(strcmp(modeStr,"range") == 0) {
			printf("Range Mode \n\n");
			mode = RANGE_MODE;
		} else {
			printf("INVAILD mode - valid mode are quick, range, or shmoo\n");
			printf("See --help for more info\n");
			return -3000;
		}
	} else {
		printf("Quick Mode\n\n");
		mode = QUICK_MODE;
	}

	if (checkCmdLineFlag(argc,argv,"htod")){
		htod = true;
	}
	if (checkCmdLineFlag(argc,argv,"dtoh")){
		dtoh = true;
	}
	if (checkCmdLineFlag(argc,argv,"dtod")){
		dtod = true;
	}
#if CUDART_VERSION >= 2020
	if (checkCmdLineFlag(argc,argv,"wc")){
		wc = true;
	}
#endif
	if(!htod && !dtoh && !dtod){
		htod = true;
		dtoh = true;
		dtod = true;
	}

	if (RANGE_MODE == mode){
		if (checkCmdLineFlag(argc, (const char **)argv, "start")){
			start = getCmdLineArgumentInt(argc,argv,"start");
			if (start<=0){
				printf("Illegal argument - start must be greater than zero\n");
				return -4000;
			}
		} else {
			printf("Must specify a start size in range mode\n");
			printf("See --help for more info\n");
			return -5000;
		}

		if(checkCmdLineFlag(argc,(const char **)argv,"end")){
			end = getCmdLineArgumentInt(argc,argv,"end");

			if (end<=0){
				printf("Illegal argument - start must be greater than zero\n");
				return -6000;
			}
			
			if (start > end){
				printf("Illegal argument - start must be greater than zero\n");
				return -7000;
			} 
		} else {
			printf("Must specify an end size in range mode.\n");
			printf("See --help for more info\n");
			return -8000;
		}

		if(checkCmdLineFlag(argc,argv,"increment")){
			increment = getCmdLineArgumentInt(argc,argv,"increment");
			if(increment <=0){
				printf("Illegal argument - start must be greater than zero\n");
				return -9000;
			}
		} else {
			printf("Must specify an end size in range mode.\n");
			printf("See --help for more info\n");
			return -10000;
		}
	}
			
	if (htod){
		testBandwidth(start,end,increment,mode,HOST_TO_DEVICE,printmode,memMode,startDevice,endDevice,wc);
	}

	if (dtoh){
		testBandwidth(start,end,increment,mode,DEVICE_TO_HOST,printmode,memMode,startDevice,endDevice,wc);
	}
	if (dtod){
		testBandwidth(start,end,increment,mode,DEVICE_TO_DEVICE,printmode,memMode,startDevice,endDevice,wc);
	}

	for (int nDevice = startDevice; nDevice<=endDevice;nDevice++){
		cudaSetDevice(nDevice);
	}
	return 0;
}





void testBandwidth(unsigned int start, unsigned int end, unsigned int increment, testMode mode, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc){
	printf("\nBandwidth test\n");
	switch (mode){
		case QUICK_MODE:
			testBandwidthQuick(DEFAULT_SIZE, kind, printmode, memMode, startDevice, endDevice, wc);
			break;
		case RANGE_MODE:
			testBandwidthRange(start, end, increment, kind, printmode, memMode, startDevice, endDevice, wc);
			break;
		case SHMOO_MODE:
			testBandwidthShmoo(kind, printmode, memMode, startDevice, endDevice, wc);
			break;
		default:
			break;
	}
}

void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc){
	testBandwidthRange(size,size, DEFAULT_INCREMENT, kind, printmode, memMode, startDevice, endDevice, wc);
}

void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc){
	unsigned int count = 1+((end-start)/increment);
	unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
	double *bandwidths = (double *)malloc(count * sizeof(double));

	for (unsigned int i=0; i<count; i++){
		bandwidths[i] = 0.0;
	}

	for (unsigned int i=0; i<count; i++){
	       memSizes[i] = start + i * increment;

		switch(kind){
			case DEVICE_TO_HOST:
				bandwidths[i] += testDeviceToHostTransfer(memSizes[i], memMode, wc);
				break;
			case HOST_TO_DEVICE:
			       bandwidths[i] += testHostToDeviceTransfer(memSizes[i], memMode, wc);
			       break;
			case DEVICE_TO_DEVICE:
			       bandwidths[i] += testDeviceToDeviceTransfer(memSizes[i]);
			       break;
		}
	}

	if (printmode == CSV){
		printResultsCSV(memSizes, bandwidths, count, kind, memMode, (1+ endDevice - startDevice), wc);
	}
	else{
		printResultsReadable(memSizes, bandwidths, count, kind, memMode, (1+endDevice-startDevice),wc);
	}
	free(memSizes);
	free(bandwidths);
}

void testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc){
	unsigned int count = 1 + (SHMOO_LIMIT_20KB / SHMOO_INCREMENT_1KB) + 
	      	((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB) +
		((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB) +
		((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB) +
		((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB) + 
		((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB) + 
		((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

	unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
	double *bandwidths = (double *)malloc(count * sizeof(double));

	for (unsigned int i=0; i<count; i++){
		bandwidths[i] = 0.0;
	}

	for (int currentDevice = startDevice; currentDevice<=endDevice; currentDevice++){
		cudaSetDevice(currentDevice);
		int iteration = 0;
		unsigned int memSize = 0;
		while (memSize <= SHMOO_MEMSIZE_MAX){
			if(memSize< SHMOO_LIMIT_20KB){
				memSize+=SHMOO_INCREMENT_1KB;
			} else if (memSize < SHMOO_LIMIT_50KB){
				memSize+=SHMOO_INCREMENT_2KB;
			} else if (memSize < SHMOO_LIMIT_100KB){
				memSize+=SHMOO_INCREMENT_10KB;
			} else if (memSize < SHMOO_LIMIT_1MB){
				memSize+=SHMOO_INCREMENT_100KB;
			} else if (memSize < SHMOO_LIMIT_16MB){
				memSize+=SHMOO_INCREMENT_1MB;
			} else if (memSize < SHMOO_LIMIT_32MB){
				memSize+=SHMOO_INCREMENT_2MB;
			} else {
				memSize+=SHMOO_INCREMENT_4MB;
			}
			
			memSizes[iteration] = memSize;

			switch(kind){
				case DEVICE_TO_HOST:
					bandwidths[iteration] += testDeviceToHostTransfer(memSizes[iteration],memMode,wc);
					break;
				case HOST_TO_DEVICE:
					bandwidths[iteration] += testHostToDeviceTransfer(memSizes[iteration],memMode,wc);
					break;
				case DEVICE_TO_DEVICE:
					bandwidths[iteration] += testDeviceToDeviceTransfer(memSizes[iteration]);
					break;
			}
			iteration++;
			printf(".");
			fflush(0);
		}
	}

	printf("\n");
	if (CSV == printmode){
		printResultsCSV(memSizes,bandwidths,count,kind,memMode,(1+endDevice-startDevice),wc);
	} else {
		printResultsReadable(memSizes,bandwidths,count,kind,memMode,(1+endDevice-startDevice),wc);
	}

	free(memSizes);
	free(bandwidths);
}




float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc){
	StopWatchInterface *timer = NULL;
	float elapsedTimeInMs=0.0f;
	float bandwidthInGBs = 0.0f;
	unsigned char *h_idata = NULL;
	unsigned char *h_odata = NULL;
	cudaEvent_t start, stop;

	sdkCreateTimer(&timer);
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	if(PINNED == memMode){
#if CUDART_VERSION >=2020
		checkCudaErrors(cudaHostAlloc((void **)&h_idata,memSize,(wc) ? cudaHostAllocWriteCombined : 0));
		checkCudaErrors(cudaHostAlloc((void **)&h_odata,memSize,(wc) ? cudaHostAllocWriteCombined : 0));
#else
		checkCudaErrors(cudaMallocHost((void **)&h_idata,memSize));
		checkCudaErrors(cudaMallocHost((void **)&h_odata,memSize));
#endif
	} else {
		h_idata = (unsigned char *)malloc(memSize);
		h_odata = (unsigned char *)malloc(memSize);
		
		if (h_idata == 0 || h_odata == 0){
			fprintf(stderr, "Not enough memory avaiable on Host to run test!\n");
			exit(EXIT_FAILURE);
		}
	}

	for (unsigned int i=0; i<memSize/sizeof(unsigned char);i++){
		h_idata[i] = (unsigned char)(i&0xff);
	}
	unsigned char *d_idata;
	checkCudaErrors(cudaMalloc((void **)&d_idata, memSize));

	checkCudaErrors(cudaMemcpy(d_idata,h_idata,memSize, cudaMemcpyHostToDevice));
	if (PINNED == memMode){
		if (bDontUseGPUTiming) sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start,0));
		for (unsigned int i=0; i<MEMCOPY_ITERATIONS;i++){
			checkCudaErrors(cudaMemcpyAsync(h_odata,d_idata,memSize,cudaMemcpyDeviceToHost,0));
		}
		checkCudaErrors(cudaEventRecord(stop,0));
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs,start,stop));
		if(bDontUseGPUTiming){
			sdkStopTimer(&timer);
			elapsedTimeInMs = sdkGetTimerValue(&timer);
			sdkResetTimer(&timer);
		}
	} else {
		elapsedTimeInMs = 0;
		for(unsigned int i=0; i<MEMCOPY_ITERATIONS;i++){
			sdkStartTimer(&timer);
			elapsedTimeInMs += sdkGetTimerValue(&timer);
			sdkResetTimer(&timer);
			memset(flush_buf,i,FLUSH_SIZE);
		}
	}

	double time_s = elapsedTimeInMs/1e3;
	bandwidthInGBs = (memSize *(float)MEMCOPY_ITERATIONS)/(double)1e9;
	bandwidthInGBs = bandwidthInGBs/time_s;
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));
	sdkDeleteTimer(&timer);

	if(PINNED == memMode){
		checkCudaErrors(cudaFreeHost(h_idata));
		checkCudaErrors(cudaFreeHost(h_odata));
	}
	else {
		free(h_idata);
		free(h_odata);
	}

	checkCudaErrors(cudaFree(d_idata));

	return bandwidthInGBs;
}

float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc){
	StopWatchInterface *timer = NULL;
	float elapsedTimeInMs = 0.0f;
	float bandwidthInGBs = 0.0f;
	cudaEvent_t start, stop;
	sdkCreateTimer(&timer);
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	unsigned char *h_odata = NULL;

	if (PINNED == memMode){
#if CUDART_VERSION >= 2020
		checkCudaErrors(cudaHostAlloc((void **)&h_odata,memSize,(wc) ? cudaHostAllocWriteCombined : 0));
#else
		checkCudaErrors(cudaMallocHost((void **)&h_odata,memSize));
#endif
	} else {
		h_odata = (unsigned char *)malloc(memSize);
		if(h_odata == 0){
			fprintf(stderr, "Not enough memory available on host to run test!\n");
			exit(EXIT_FAILURE);
		}
	}

	unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
	unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

	if (h_cacheClear1 == 0 || h_cacheClear2 == 0){
		fprintf(stderr, "Not enough memory available on host to run test!\n");
		exit(EXIT_FAILURE);
	}

	for (unsigned int i=0; i<CACHE_CLEAR_SIZE/sizeof(unsigned char);i++){
		h_cacheClear1[i] = (unsigned char)(i & 0xff);
		h_cacheClear2[i] = (unsigned char)(0xff -(i & 0xff));
	}

	unsigned char * d_idata;
	checkCudaErrors(cudaMalloc((void **)&d_idata,memSize));

	if (PINNED == memMode){
		if (bDontUseGPUTiming) sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start, 0));
		for (unsigned int i=0; i<MEMCOPY_ITERATIONS;i++){
			checkCudaErrors(cudaMemcpyAsync(d_idata,h_odata, memSize,cudaMemcpyHostToDevice,0));
		}
		checkCudaErrors(cudaEventRecord(stop,0));
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs,start,stop));
		if (bDontUseGPUTiming){
			sdkStopTimer(&timer);
			elapsedTimeInMs = sdkGetTimerValue(&timer);
			sdkResetTimer(&timer);
		}
	} else {
		elapsedTimeInMs = 0;
		for (unsigned int i=0; i<MEMCOPY_ITERATIONS; i++){
			sdkStartTimer(&timer);
			checkCudaErrors(cudaMemcpy(d_idata,h_odata,memSize,cudaMemcpyHostToDevice));
			sdkStopTimer(&timer);
			elapsedTimeInMs += sdkGetTimerValue(&timer);
			sdkResetTimer(&timer);
			memset(flush_buf,i,FLUSH_SIZE);
		}
	}

	double time_s = elapsedTimeInMs /1e3;
	bandwidthInGBs = (memSize*(float)MEMCOPY_ITERATIONS)/(double)1e9;
	bandwidthInGBs = bandwidthInGBs/time_s;

	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));
	sdkDeleteTimer(&timer);

	if(PINNED == memMode){
		checkCudaErrors(cudaFreeHost(h_odata));
	} else { 
		free(h_odata);
	}

	free(h_cacheClear1);
	free(h_cacheClear2);
	checkCudaErrors(cudaFree(d_idata));

	return bandwidthInGBs;
}

float testDeviceToDeviceTransfer(unsigned int memSize){
	StopWatchInterface *timer = NULL;
	float elapsedTimeInMs = 0.0f;
	float bandwidthInGBs = 0.0f;
	cudaEvent_t start,stop;

	sdkCreateTimer(&timer);
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	unsigned char *h_idata = (unsigned char *)malloc(memSize);

	if(h_idata == 0){
		fprintf(stderr,"Not enough memory available on host to run test\n");
		exit(EXIT_FAILURE);
	}

	for (unsigned int i=0; i<memSize/sizeof(unsigned char);i++){
		h_idata[i] = (unsigned char)(i & 0xff);
	}

	unsigned char *d_idata;
	checkCudaErrors(cudaMalloc((void **)&d_idata,memSize));
	unsigned char *d_odata;
	checkCudaErrors(cudaMalloc((void **)&d_odata,memSize));

	checkCudaErrors(cudaMemcpy(d_idata,h_idata,memSize,cudaMemcpyHostToDevice));

	sdkStartTimer(&timer);
	checkCudaErrors(cudaEventRecord(start,0));

	for (unsigned int i=0; i<MEMCOPY_ITERATIONS;i++){
		checkCudaErrors(cudaMemcpy(d_odata,d_idata,memSize,cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaDeviceSynchronize());
	
	sdkStopTimer(&timer);
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs,start,stop));

	if(bDontUseGPUTiming){
		elapsedTimeInMs = sdkGetTimerValue(&timer);
	}

	double time_s = elapsedTimeInMs/1e3;
	bandwidthInGBs = (2.0f * memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
	bandwidthInGBs = bandwidthInGBs/time_s;

	sdkDeleteTimer(&timer);
	free(h_idata);
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));

	return bandwidthInGBs;
}




int main(int argc,char **argv){
	pArgc = &argc;
	pArgv = argv;

	flush_buf = (char *)malloc(FLUSH_SIZE);

	printf("[%s] - Starting...\n",sSDKsample);
	dev_info();
	int iRetVal = runTest(argc,(const char **)argv);

	if (iRetVal < 0){
		checkCudaErrors(cudaSetDevice(0));
	}

	printf("%s\n", (iRetVal==0) ? "Result = Pass" : "Result = FAIL");
	printf("\nNOTE: The CUDA samples are not meant for performance measurements. Results may vary when GPU Boost is enabled\n");
	free(flush_buf);
	exit((iRetVal == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}
