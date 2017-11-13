#include <stdio.h>
#include <math.h>
#include <ctime>

#define ARRAY_SIZE 1000
#define X 1
#define Y X
#define N ARRAY_SIZE*X

__global__ void init(double *g)
{
	// which thread is this?
	// int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int i = blockIdx.x;
	int j = threadIdx.x;

	int m = i + blockIdx.z * ARRAY_SIZE;
	int n = j + blockIdx.y * ARRAY_SIZE;


	// g[i*ARRAY_SIZE + j] = i*ARRAY_SIZE + j;
	// printf("Hello from sin %d, cos %d, thready %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
	// printf("hi blockDim %d \t %d \t %d \n", blockDim.x, blockDim.y, blockDim.z);
	// printf("hi threadIdx %d \t%d \t%d \n", threadIdx.x, threadIdx.y, threadIdx.z);
	// printf("hi m: %d   n: %d  index: %d  value:%f\n", m, n, m * ARRAY_SIZE * X + n, sinf(m*m + n)*sinf(m*m + n) + cosf(m - n));
	g[m * ARRAY_SIZE * X + n] = sinf(m*m + n)*sinf(m*m + n) + cosf(m - n);
	// g[m * ARRAY_SIZE * X + n] = n*1.0;
	__syncthreads();
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
}


__device__ int partition(double* input, int p, int r)
{
    double pivot = input[r];
    
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
	int m = i + blockIdx.z * ARRAY_SIZE;
	int n = j + blockIdx.y * ARRAY_SIZE;
	int index = m * ARRAY_SIZE * X + n;

	// if(i == 0 || i == ARRAY_SIZE - 1  || j == 0 || j == ARRAY_SIZE - 1){
	// if( (y == 0 && i == 0) || ( y == Y-1 && i == ARRAY_SIZE - 1) ||
	// 	(x == 0 && j == 0) || ( x == X-1 && j == ARRAY_SIZE - 1)){
	if(m == 0 || m == X*ARRAY_SIZE - 1 || n == 0 || n == Y*ARRAY_SIZE - 1){

	}else{
		arr[0] = g[index];
		arr[1] = g[index + 1];
		arr[2] = g[index - 1];
		arr[3] = g[index + ARRAY_SIZE * X];
		arr[4] = g[index - ARRAY_SIZE * X];

		double temp = quick_select(arr, 0, 4, 2);

		g[index] = temp;
	}


	__syncthreads();
	// get mediean
}




__global__ void getResult(double *g, double *r){
	int i = blockIdx.x;
	int j = threadIdx.x;
	int m = i + blockIdx.z * ARRAY_SIZE;
	int n = j + blockIdx.y * ARRAY_SIZE;
	int index = m * ARRAY_SIZE * X + n;


	int mid = 500 * X - 1;
	if(m == mid && n == mid){
		printf("mid: %f\n", g[mid * ARRAY_SIZE + mid]);
		r[1] = g[mid * ARRAY_SIZE + mid];
	}

	if(m == 17 && n == 31){
		printf("17, 31 : %f\n", g[17*ARRAY_SIZE + 31]);
		r[2] = g[17 * ARRAY_SIZE + 31];
	}

	__syncthreads();

	for (int s = ARRAY_SIZE/2; s > 0; s >>= 1 ){
		if(j < s){
			g[index] += g[index + s];
		}
		__syncthreads();
	}

	for (int s = ARRAY_SIZE/2; s > 0; s >>= 1){
		if(i < s && j == 0){
			g[index] += g[index + s*ARRAY_SIZE];
		}
		__syncthreads();
	}

	if(m == 0 && n == 0){
		printf("sum: %f\n", g[0]);
		r[0] = g[0];
	}


}



__global__ void handle(double *g, double *r)
{
	// which thread is this?
	// int i = blockIdx.x * blockDim.x + threadIdx.x; 

	
	for(int i = 0; i < 10; i++){
		// running<<<dim3(ARRAY_SIZE, X, Y), ARRAY_SIZE>>>(g);
		__syncthreads();
	}	

	running<<<dim3(ARRAY_SIZE, X, Y), ARRAY_SIZE>>>(g);

	getResult<<<dim3(ARRAY_SIZE, X, Y), ARRAY_SIZE>>>(g, r);

	
	__syncthreads();
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
}




int main(int argc, char ** argv) {
    // declare and allocate host memory
    double h_array[N*N];
    const int ARRAY_BYTES = N*N*sizeof(double);
 
    clock_t cpu_startTime, cpu_endTime;
    double cpu_ElapseTime=0;
	

    printf("The N is : %d\n",N);

    // declare, allocate, and zero out GPU memory
    double * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    double * r;
    cudaMalloc((void **) &r, 3 * sizeof(double));
    

    for(int i = 0; i < X*ARRAY_SIZE; i++){
    	for(int j = 0; j < Y*ARRAY_SIZE; j++){
    		h_array[i*N + j] = sin(i*i + j) * sin(i*i + j) + cos(i - j);
    	}
    }


    cudaMemcpy(d_array, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // dim3 dimGrid(2, 2);
    // dim3 dimBlock(ARRAY_SIZE, ARRAY_SIZE);
    // init<<<1, dimBlock>>>(d_array);
    // // init<<<1, ARRAY_SIZE*ARRAY_SIZE>>>(d_array);

    // init<<<dim3(ARRAY_SIZE,X,Y), ARRAY_SIZE>>>(d_array);
    // cudaDeviceSynchronize();

	cpu_startTime = clock();


	handle<<<1, 1>>>(d_array, r);
	cudaDeviceSynchronize();



    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	
	cpu_endTime = clock();
	cpu_ElapseTime = (cpu_endTime - cpu_startTime);
	printf("Time using in CPU is : %f\n", cpu_ElapseTime);


    // printf("{ ");
    // for (int i = 0; i < 10; i++)  {
    //	for(int j = 0; j < 10; j++)
    //		{ printf("%f ", h_array[i*ARRAY_SIZE +j]); }
    //	printf("\n");
    // }
   
    printf("}\n");


    cudaFree(d_array);


	return 0;
}
