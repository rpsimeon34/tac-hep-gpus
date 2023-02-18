#include <stdio.h>


const int DSIZE_X = 256;
const int DSIZE_Y = 256;

__global__ void add_matrix(float* A, float*B, float* C, int NX, int NY)
{
    // Express in terms of threads and blocks
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    // Add the two matrices - make sure you are not out of range
    if (idx <  NX && idy < NY ) {
        C[idy*NX + idx] = A[idy*NX + idx] + B[idy*NX + idx];
    }

}

int main()
{

    // Create and allocate memory for host and device pointers 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE_X*DSIZE_Y];
    h_B = new float[DSIZE_X*DSIZE_Y];
    h_C = new float[DSIZE_X*DSIZE_Y];

    // Fill in the matrices
    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            h_A[i*DSIZE_Y+j] = rand()/(float)RAND_MAX;
            h_B[i*DSIZE_Y+j] = rand()/(float)RAND_MAX;
            h_C[i*DSIZE_Y+j] = 0.0;
        }
    }
    printf("A[0,0] is %f before\n",h_A[0]);
    printf("B[0,0] is %f before\n",h_B[0]);

    // Copy from host to device
    cudaMalloc(&d_A, DSIZE_X*DSIZE_Y*sizeof(float));
    cudaMalloc(&d_B, DSIZE_X*DSIZE_Y*sizeof(float));
    cudaMalloc(&d_C, DSIZE_X*DSIZE_Y*sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    dim3 blockSize(32,32); 
    dim3 gridSize((DSIZE_X+31)/32,(DSIZE_Y+31/32)); 
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    // Copy back to host
    cudaMemcpy(h_C, d_C, DSIZE_X*DSIZE_Y*sizeof(float),cudaMemcpyDeviceToHost);

    // Print and check some elements to make the addition was succesfull
    printf("A[0,0] is now %f\n",h_A[0]);
    printf("B[0,0] is now %f\n",h_B[0]);
    printf("C[0,0] is now %f\n",h_C[0]);

    printf("A[157,20] is %f\n",h_A[157*DSIZE_Y+20]);
    printf("B[157,20] is %f\n",h_B[157*DSIZE_Y+20]);
    printf("C[157,20] is %f\n",h_C[157*DSIZE_Y+20]);

    // Free the memory     
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
