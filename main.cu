#include <stdio.h>
#include <math.h>

#define NUM_THREADS 10000
#define ARRAY_SIZE  1000

#define BLOCK_WIDTH 1000

__global__ void init(double *g)
{
	// which thread is this?
	// int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int i = blockIdx.x;
	int j = threadIdx.x;

	// g[i*ARRAY_SIZE + j] = i*ARRAY_SIZE + j;
	// printf("Hello from sin %f, cos %f, thready %d\n", sinf(i), cosf(i-j), i*i+j);
	// g[i*ARRAY_SIZE + j] = sinf(i*i + j)*sinf(i*i + j) + cosf(i - j);
	g[i*ARRAY_SIZE + j] = j;
	
	__syncthreads();
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
}


__device__ int partition(double* input, int p, int r)
{
    int pivot = input[r];
    
    while ( p < r )
    {
        while ( input[p] < pivot )
            p++;
        
        while ( input[r] > pivot )
            r--;
        
        if ( input[p] == input[r] )
            p++;
        else if ( p < r ) {
            double tmp = input[p];
            input[p] = input[r];
            input[r] = tmp;
        }
    }
    
    return r;
}


__device__ double quick_select(double* input, int p, int r, int k)
{
    if ( p == r ) return input[p];
    int j = partition(input, p, r);
    int length = j - p + 1;
    if ( length == k ) return input[j];
    else if ( k < length ) return quick_select(input, p, j - 1, k);
    else  return quick_select(input, j + 1, r, k - length);
}


__global__ void running(double *g)
{

	// buffer
	double arr[5];
	int i = blockIdx.x;
	int j = threadIdx.x;
	int index = i * ARRAY_SIZE + j;
	if(i == 0 || i == ARRAY_SIZE - 1  || j == 0 || j == ARRAY_SIZE - 1){

	}else{
		arr[0] = g[index];
		arr[1] = g[index + 1];
		arr[2] = g[index - 1];
		arr[3] = g[index + ARRAY_SIZE];
		arr[4] = g[index - ARRAY_SIZE];

		int temp = quick_select(arr, 0, 4, 0);

		g[index] = temp;
	}


	__syncthreads();
	// get mediean
}




__global__ void getResult(double *g, double *r){
	int i = blockIdx.x;
	int j = threadIdx.x;
	int index = i * ARRAY_SIZE + j;

	// for (int s = )

}



__global__ void handle(double *g)
{
	// which thread is this?
	// int i = blockIdx.x * blockDim.x + threadIdx.x; 

	init<<<1000, 1000>>>(g);

	running<<<1000, 1000>>>(g);



	
	__syncthreads();
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
}




int main(int argc, char ** argv) {
    // declare and allocate host memory
    double h_array[ARRAY_SIZE][ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(double);
 
    clock_t cpu_startTime, cpu_endTime;
    double cpu_ElapseTime=0;
	

    // declare, allocate, and zero out GPU memory
    double * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    // dim3 dimGrid(2, 2);
    // dim3 dimBlock(ARRAY_SIZE, ARRAY_SIZE);
    // init<<<1, dimBlock>>>(d_array);
    // // init<<<1, ARRAY_SIZE*ARRAY_SIZE>>>(d_array);
    // cudaDeviceSynchronize();

	cpu_startTime = clock();


	handle<<<1, 1>>>(d_array);
	cudaDeviceSynchronize();



    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	
	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);
	printf("Time using in CPU is : %f\n", cpu_ElapseTime);


    printf("{ ");
    for (int i = 0; i < 5; i++)  {
    	for (int j = 0; j < 5; j++)
    		{ printf("%f ", h_array[i][j]); }
    	printf("\n");
    }
   
    printf("}\n");


    cudaFree(d_array);


	return 0;
}
