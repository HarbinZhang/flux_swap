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


__device__ int partition(double *items, int left, int right, int pivotIndex)
{
        int pivot = items[pivotIndex];
        int partitionIndex = left;

       	double temp = items[right];
		items[right] = items[partitionIndex];
		items[partitionIndex] = temp;
        for(int i=left; i < right; i++) {
                if(items[i]<pivot) {
                		double temp = items[i];
                		items[i] = items[partitionIndex];
                		items[partitionIndex] =	temp;
                        partitionIndex++;
                }
        }
		double temp = items[right];
		items[right] = items[partitionIndex];
		items[partitionIndex] = temp;
        return partitionIndex;
}


__device__ int quickSelect(double *items, int first, int last, int k) {
    int pivot = partition(items, first, last);
    if (k < pivot-first+1) { //boundary was wrong
        return quickSelect(items, first, pivot, k);
    } else if (k > pivot-first+1) {//boundary was wrong
        return quickSelect(items, pivot+1, last, k-pivot);
    } else {
        return items[pivot];//index was wrong
    }
}



__global__ void running(double *g)
{

	// buffer
	double arr[5];
	int i = blockIdx.x;
	int j = threadIdx.x;

	int index = i * ARRAY_SIZE + j;
	if(i == 0 || i == ARRAY_SIZE || j == 0 || j == ARRAY_SIZE){

	}else{
		arr[0] = g[index];
		arr[1] = g[index + 1];
		arr[2] = g[index - 1];
		arr[3] = g[index + ARRAY_SIZE];
		arr[4] = g[index - ARRAY_SIZE];

		int temp = quickSelect(arr, 0, 4, 2);

		g[index] = temp;
	}


	__syncthreads();
	// get mediean
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
