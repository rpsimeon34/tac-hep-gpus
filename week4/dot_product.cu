#include <stdio.h>
#include <time.h>


#define BLOCK_SIZE 32

const int DSIZE = 256;
const int a = 1;
const int b = 1;
const int block_size = 32;
const int grid_size = DSIZE/block_size;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)

// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *A, const int *B, int *C, int N) {

	// FIXME
	// Use atomicAdd	
}


int main() {
	
	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 
	h_A = new int[DSIZE];
	h_B = new int[DSIZE];
	h_C = new int;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;
		h_B[i] = b;
	}

	*h_C = 0;


	// Allocate device memory 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));
	
	// Check memory allocation for errors
    cudaCheckErrors("Error while allocating device memory");

	// Copy the matrices on GPU
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
	
	// Check memory copy for errors
    cudaCheckErrors("Error while copying vectors to device");

	// Define block/grid dimentions and launch kernel
    dot_product<<<grid_size, block_size>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("Error while running kernel");
	
	// Copy results back to host
	
    // Check copy for errors

	// Verify result

	// Free allocated memory
	
	return 0;

}
