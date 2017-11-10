#include <stdio.h>

#define NUM_THREADS 10000
#define ARRAY_SIZE  5

#define BLOCK_WIDTH 1000

__global__ void init(int *g)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	g[i] = i;
}

void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)  {
    	for (int j = 0; j < size; j++)
    		{ printf("%d ", array[i][j]); }
    	printf("\n");
    }
    
    printf("}\n");
}

int main(int argc, char ** argv) {
    // declare and allocate host memory
    int h_array[ARRAY_SIZE][ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
 
    // declare, allocate, and zero out GPU memory
    int * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    dim3 dimGrid(2, 2);
    dim3 dimBlock(ARRAY_SIZE, ARRAY_SIZE);
    init<<<1, dimBlock>>>(d_array);

    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);

    cudaFree(d_array);


	return 0;
}
