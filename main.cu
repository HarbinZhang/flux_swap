#include <stdio.h>
#include <math.h>

#define NUM_THREADS 10000
#define ARRAY_SIZE  5

#define BLOCK_WIDTH 1000

__global__ void init(double *g)
{
	// which thread is this?
	// int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int i = threadIdx.x;
	int j = threadIdx.y;

	// g[i*ARRAY_SIZE + j] = i*ARRAY_SIZE + j;
	printf("Hello from sin %f, cos %f, thready %d\n", sinf(i), cosf(i-j), i*i+j);
	g[i*ARRAY_SIZE + j] = sinf(i*i + j)*sinf(i*i + j) + cosf(i - j);
	
	__syncthreads();
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
}


int main(int argc, char ** argv) {
    // declare and allocate host memory
    double h_array[ARRAY_SIZE][ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(double);
 
    // declare, allocate, and zero out GPU memory
    double * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    dim3 dimGrid(2, 2);
    dim3 dimBlock(ARRAY_SIZE, ARRAY_SIZE);
    init<<<1, dimBlock>>>(d_array);
    // init<<<1, ARRAY_SIZE*ARRAY_SIZE>>>(d_array);
    cudaDeviceSynchronize();


    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("{ ");
    for (int i = 0; i < ARRAY_SIZE; i++)  {
    	for (int j = 0; j < ARRAY_SIZE; j++)
    		{ printf("%f ", h_array[i][j]); }
    	printf("\n");
    }
   
    printf("}\n");


    cudaFree(d_array);


	return 0;
}
