#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>

#define ARRAY_SIZE 1000
#define X 4
#define Y X
#define N ARRAY_SIZE*X

__global__ void init(double *g)
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	int m = i + blockIdx.z * ARRAY_SIZE;
	int n = j + blockIdx.y * ARRAY_SIZE;

	g[m * ARRAY_SIZE * X + n] = sinf(m*m + n)*sinf(m*m + n) + cosf(m - n);
	__syncthreads();
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

__device__ double bubble_sort(double *input, int p, int r, int k){
	for(int i = 0; i < 5; i++){
		for(int j = i+1; j < 5; j ++){
			if(input[i] < input[j]){
				double temp = input[i];
				input[i] = input[j];
				input[j] = temp;
			}
		}
	}
	return input[k];
}

__global__ void running(double *g, double *mid_array)
{
	// buffer
	double arr[5];
	int i = blockIdx.x;
	int j = threadIdx.x;
	int m = i + blockIdx.z * ARRAY_SIZE;
	int n = j + blockIdx.y * ARRAY_SIZE;
	int index = m * ARRAY_SIZE * X + n;

	arr[2] = g[index];

	__syncthreads();

	if(m != 0 && m != N - 1 && n != 0 && n != N - 1){
		arr[1] = g[index + 1];
		arr[0] = g[index - 1];
		arr[3] = g[index + ARRAY_SIZE * X];
		arr[4] = g[index - ARRAY_SIZE * X];

		// arr[2] = quick_select(arr, 0, 4, 2);
		for(int i = 0; i < 5; i++){
			for(int j = 0; j < 5 - i; j++){
				if(arr[j+1] < arr[j]){
					double temp = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = temp;
				}
			}
		}
	}
	__syncthreads();
	mid_array[index] = arr[2];
	__syncthreads();
}


__global__ void getRowSum(double *g, double *r, double *getSumArray){
	int i = blockIdx.x;
	int j = threadIdx.x;
	int m = i + blockIdx.z * ARRAY_SIZE;
	int n = j + blockIdx.y * ARRAY_SIZE;
	int index = m * ARRAY_SIZE * X + n;
	__shared__ double sdata[ARRAY_SIZE];

	sdata[j] = g[index];
	__syncthreads();

	int mid = ARRAY_SIZE/2 * X;
	if(m == mid && n == mid){
		r[1] = sdata[j];
		printf("mid: %f\n", r[1]);
	}

	if(m == 17 && n == 31){
		r[0] = sdata[j];
		printf("17 31: %f\n", r[0]);
	}

	__syncthreads();

	for (int s = 1024/2; s > 0; s >>= 1 ){
		if(j < s && j + s < ARRAY_SIZE){
			sdata[j] += sdata[j + s];
		}
		__syncthreads();
	}

	if(j == 0){
		getSumArray[i + ARRAY_SIZE * (Y * blockIdx.z + blockIdx.y)] = sdata[0];
	}
	__syncthreads();
}


__global__ void getSum(double *getSumArray, double *csum){
	int i = threadIdx.x;
	int index = i + ARRAY_SIZE * (Y*blockIdx.z + blockIdx.y);	

	__shared__ double sdata[ARRAY_SIZE];

	sdata[i] = getSumArray[index];

	__syncthreads();

	for (int s = 512; s > 0; s >>= 1 ){
		if(i < s && i + s < ARRAY_SIZE){
				sdata[i] += sdata[i + s];
		}
		__syncthreads();
	}

	if(i == 0){
		csum[2 + blockIdx.y * X + blockIdx.z] = sdata[0];	
		printf("sum: %f\n", sdata[0]);
	}
	__syncthreads();
}


__global__ void getRes(double *r){
	__shared__ double sdata[X*Y];
	int i = threadIdx.x;
	sdata[i] = r[2+i];
	__syncthreads();

	for(int s = X*Y/2; s > 0; s >>=1){
		if(i < s){
			sdata[i] += sdata[i + s];
		}
		__syncthreads();
	}
	if(i == 0){
		r[2] = sdata[0];

	}
	__syncthreads();
}


__global__ void handle(double *g, double *mid_array)
{
	for(int i = 0; i < 10; i++){
		running<<<dim3(ARRAY_SIZE, X, Y), ARRAY_SIZE>>>(g, mid_array);
		double *temp = g;
		g = mid_array;
		mid_array = temp;
	}	
}




int main(int argc, char ** argv) {
    // declare and allocate host memory
    double h_array[N*N];
    const int ARRAY_BYTES = N*N*sizeof(double);

    printf("The N is : %d\n",N);

    // declare, allocate, and zero out GPU memory
    double * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    double * mid_array;
    cudaMalloc((void **) &mid_array, ARRAY_BYTES);
    cudaMemset((void *) mid_array, 0, ARRAY_BYTES); 

    double * r;
    cudaMalloc((void **) &r, (2+X*Y) * sizeof(double));

    double * getSumArray;
    cudaMalloc((void **) &getSumArray, X * Y * ARRAY_SIZE * sizeof(double));
    
    for(int i = 0; i < X*ARRAY_SIZE; i++){
    	for(int j = 0; j < Y*ARRAY_SIZE; j++){
    		h_array[i*N + j] = sin(i*i + j) * sin(i*i + j) + cos(i - j);
    	}
    }

    cudaMemcpy(d_array, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);



	auto start = std::chrono::system_clock::now();


	handle<<<1, 1>>>(d_array, mid_array);
	cudaDeviceSynchronize();

	getRowSum<<<dim3(ARRAY_SIZE, X, Y), ARRAY_SIZE>>>(d_array, r, getSumArray);
	cudaDeviceSynchronize();

	getSum<<<dim3(1,X,Y), ARRAY_SIZE>>>(getSumArray, r);
	cudaDeviceSynchronize();

	getRes<<<1, X*Y>>>(r);
	cudaDeviceSynchronize();

	double res[3];
    cudaMemcpy(res, r, 3*sizeof(double), cudaMemcpyDeviceToHost);


	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	printf("Time using in CPU is : %f\n", elapsed_seconds);


    printf("Sum From CPU: %f \n", res[2]);
    printf("A[17][31] :  %f \n", res[0]);
    printf("A[mid][mid]: %f \n", res[1]);


    cudaFree(d_array);
    cudaFree(r);
    cudaFree(cres);
    cudaFree(mid_array);


	return 0;
}
