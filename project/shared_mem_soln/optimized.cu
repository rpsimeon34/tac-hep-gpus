// Version of stencil and matrix multiplication with explicit memory copies
// and default CUDA stream

#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 512
#define RADIUS 3
#define BLOCK_SIZE 32
#define A_val 1
#define B_val 2

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
   } while (0)


__global__ void stencil_2d(int *in, int *out) {

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex_x = threadIdx.x + RADIUS;
    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
    int lindex_y = threadIdx.y + RADIUS;

    // Read input elements into shared memory
    int size = N + 2*RADIUS;
    temp[lindex_x][lindex_y] = in[gindex_y+gindex_x*N];

    if (threadIdx.x < RADIUS) {
        temp[lindex_x - RADIUS][lindex_y] = in[gindex_y+(gindex_x - RADIUS)*N];
        temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_y+(gindex_x + BLOCK_SIZE)*N];
    }

    if (threadIdx.y < RADIUS ) {
        temp[lindex_x][lindex_y - RADIUS] = in[gindex_y-RADIUS+N*gindex_x];
        temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_y+BLOCK_SIZE+N*gindex_x];
    }

    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        result += temp[lindex_x+offset][lindex_y];
        result += temp[lindex_x][lindex_y+offset];
    }
    // Avoid double-counting the center
    result -= temp[lindex_x][lindex_y];

    //FIXME
    // Store the result
    out[gindex_y+size*gindex_x] = result;
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const int *A, const int *B, int *C, int size) {

    // create thread x index
    // create thread y index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        int temp = 0;
        for (int i = 0; i < size; i++){
            temp += A[idy*size+i]*B[i*size+idx];
        }
        C[idy*size+idx] = temp;                    
    }

}


void fill_ints(int *x, int n, int val) {
   // Store the result
   // https://en.cppreference.com/w/cpp/algorithm/fill_n
   fill_n(x, n, val);
}


int main(void) {

    int *in_A, *out_A, *in_B, *out_B, *h_C; // host copies
    int *d_in_A, *d_out_A, *d_in_B, *d_out_B, *d_C; // device copies

    // Alloc space for host copies and setup values
    int size = (N + 2*RADIUS)*(N + 2*RADIUS) * sizeof(int);
    int DSIZE = N + 2*RADIUS;
    in_A = (int *)malloc(size); fill_ints(in_A, (N + 2*RADIUS)*(N + 2*RADIUS),A_val);
    out_A = (int *)malloc(size); fill_ints(out_A, (N + 2*RADIUS)*(N + 2*RADIUS),A_val);
    in_B = (int *)malloc(size); fill_ints(in_B, (N + 2*RADIUS)*(N + 2*RADIUS),B_val);
    out_B = (int *)malloc(size); fill_ints(out_B, (N + 2*RADIUS)*(N + 2*RADIUS),B_val);
    h_C = (int *)malloc(size); fill_ints(h_C, (N + 2*RADIUS)*(N + 2*RADIUS),0);

    // Alloc space for device copies
    cudaMalloc((void **)&d_in_A, size);
    cudaMalloc((void **)&d_out_A, size);
    cudaMalloc((void **)&d_in_B, size);
    cudaMalloc((void **)&d_out_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaCheckErrors("Error while allocating device memory");

    // Copy to device
    cudaMemcpy(d_in_A, in_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_A, out_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_B, in_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_B, out_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Error while copying from host to device");

    // Launch stencil_2d() kernel on GPU
    int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 grid(gridSize, gridSize);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // Launch the kernel 
    // Properly set memory address for first element on which the stencil will be applied
    stencil_2d<<<grid,block>>>(d_in_A + RADIUS*(N + 2*RADIUS) + RADIUS , d_out_A + RADIUS*(N + 2*RADIUS) + RADIUS);
    stencil_2d<<<grid,block>>>(d_in_B + RADIUS*(N + 2*RADIUS) + RADIUS , d_out_B + RADIUS*(N + 2*RADIUS) + RADIUS);
    cudaCheckErrors("Error while launching stencil kernel");

    // Launch mat_mult kernel on GPU
    int m_gridSize = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 m_grid(m_gridSize, m_gridSize);
    dim3 m_block(BLOCK_SIZE, BLOCK_SIZE);
    // Launch the kernel
    matrix_mul_gpu<<<m_grid, m_block>>>(d_out_A, d_out_B, d_C, DSIZE);
    cudaCheckErrors("Error while launching multiplication kernel");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Error while copying from device to host");

    // Error Checking
    int exp_edge = A_val*B_val*((RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    int exp_center = A_val*B_val*((RADIUS*4+1)*(RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    for (int i = 0; i < N + 2 * RADIUS; ++i) {
        for (int j = 0; j < N + 2 * RADIUS; ++j) {

            if ((i < RADIUS || i >= N + RADIUS) && (j < RADIUS || j >= N+RADIUS)) {
                if (h_C[j+i*(N + 2 * RADIUS)] != A_val*B_val*DSIZE) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_C[j+i*(N + 2 * RADIUS)], A_val*B_val*DSIZE);
                    return -1;
                }
            }
            else if ((j < RADIUS || j >= N + RADIUS) && (i >= RADIUS && i< N+RADIUS)){
                if (h_C[j+i*(N + 2 * RADIUS)] != exp_edge) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_C[j+i*(N + 2 * RADIUS)], exp_edge);
                    return -1;
                }
            }        
            else if ((i < RADIUS || i >= N + RADIUS) && (j >= RADIUS && j< N+RADIUS)){
                if (h_C[j+i*(N + 2 * RADIUS)] != exp_edge) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_C[j+i*(N + 2 * RADIUS)], exp_edge);
                    return -1;
                }
            }
            else {
                if (h_C[j+i*(N + 2 * RADIUS)] != exp_center) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_C[j+i*(N + 2 * RADIUS)], exp_center);
                    return -1;
                }
            }
        }
    }

    // Cleanup
    free(in_A);
    free(out_A);
    free(in_B);
    free(out_B);
    free(h_C);
    cudaFree(d_in_A);
    cudaFree(d_out_A);
    cudaFree(d_in_B);
    cudaFree(d_out_B);
    cudaFree(d_C);
    printf("Success!\n");

    return 0;
}
