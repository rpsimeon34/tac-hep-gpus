#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0);

const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *A, float *B, int v_size) {

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;
    // Swap the vector elements - make sure you are not out of range
    if (idx < v_size) {
        float temp_C = A[idx];
        A[idx] = B[idx];
        B[idx] = temp_C;
    }

}


int main() {


    float *h_A, *h_B, *d_A, *d_B;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    printf("A[0] is %f before\n",h_A[0]);
    printf("B[0] is %f before\n",h_B[0]);


    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaCheckErrors("This is an error! Woohoo!");

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copying host to device error! Noooo");

    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, DSIZE);
    cudaCheckErrors("Error running kernel, boooo");

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copying back to host error, awww");

    // Print and check some elements to make sure swapping was successfull
    printf("A[0] is %f after\n",h_A[0]);
    printf("B[0] is %f after\n",h_B[0]);

    // Free the memory 
    free(h_A);
    free(h_B);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
