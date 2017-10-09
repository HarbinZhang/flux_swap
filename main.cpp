#include <mpi.h>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include "f.h"

using namespace std;

// helper funcions
vector<vector<long long> > initMatrix(int n, int p, int rank);
void printMatrix(vector<vector<long long> > matrix);
static int bandwidth;

int main(int argc, char **argv){
	if(argc != 2){
		cout << "Error, please specify the size of the matrix"<< argc << endl;
		exit(1);
	}

	int rank;
	int p;		// size of processors
	int n = atoi(argv[1]);



	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	vector<vector<long long> > matrix = initMatrix(n, p, rank);

	MPI_Barrier(MPI_COMM_WORLD);
	double startTime = MPI_Wtime();

	if(rank == 0){cout<<"<<<<<<<<<<<<<<<<<<<<<< start <<p "<<p<<" n: "<<n<<endl;}


	// send row value to rank - 1


	
	int row_start = 0;
	if (rank == 0){
		row_start = 1;
	}
	int row_end = bandwidth;
	if (rank == p - 1){
		row_end = bandwidth - 1;
	}

	for(int k = 0; k < 10; k++){
		if(rank != 0){
			MPI_Send(&matrix[0].front(), n, MPI_LONG_LONG, rank - 1, 1, MPI_COMM_WORLD);
		}
		if(rank != p - 1){
			MPI_Recv(&matrix[bandwidth].front(), n, MPI_LONG_LONG, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}




		// iteration
		for(int i = row_start; i < row_end; i++){
			for(int j = 1; j < n - 1; j++){
				matrix[i][j] = f(matrix[i][j], matrix[i+1][j], matrix[i][j+1], matrix[i+1][j+1]);
			}
		}
	}

	if(rank == 0){cout<<"<<<<<<<<<<<<<<<<<<<<<< iteration done "<<endl;}

	// printMatrix(matrix);


	// sum
	long long sum = 0;
	for(int i = 0; i < bandwidth; i++){
		for(int j = 0; j < n; j++){
			sum += matrix[i][j];
		}
	}

	int temp = 0;
	if(rank == 3){
		cout<<"p3 sum is : "<<sum<<" "<<bandwidth<<endl;
	}else if(rank == 0 && p==1){
		for(int i = 2*floor(n/4); i < 3*floor(n/4); i++){
			for(int j = 0; j < n; j++){
				temp += matrix[i][j];

			}

		}
	}



	// send back
	if(rank == 0){
		long long recv_sum;
		for(int i = 1; i < p; i++){
			MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum += recv_sum;
		}
	}else{
		MPI_Send(&sum, 1, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD);
	}
	


	
	long long mid = matrix[bandwidth/2][n/2];
	int mid_rank = p / 2;

	if(rank == 0 && p != 1){
		MPI_Recv(&mid, 1, MPI_LONG_LONG, mid_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}else if(rank == mid_rank && p != 1){
		int mid_x = n/2 - mid_rank*bandwidth;
		cout<<"mid"<<mid_x<<" "<<mid_rank<<endl;
		MPI_Send(&matrix[mid_x][n/2], 1, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);


	

	double endTime, totalTime;
	if(rank == 0){
		double endTime = MPI_Wtime();
		double totalTime = endTime - startTime;
		cout << "The total time is: " << totalTime<<endl;
		cout << "sum is: "<<sum<<endl;
		cout << "mid is: "<<mid<<endl;
	}




	MPI_Finalize();
	return 0;

}

vector<vector<long long> >initMatrix(int n, int p, int rank){

	

	bandwidth = floor(n/p);
	if(rank == p - 1){
		bandwidth = n - (p-1) * floor(n/p);
		if(rank==3){cout<<bandwidth<<" "<<(p-1) * floor(n/p)<<endl;}
	}

	vector<long long> row(n, 0);
	vector<vector<long long> > res(bandwidth+1, row);

	
	// res.push_back(null_vec);


	for(int i = rank*floor(n/p); i < rank*floor(n/p) + bandwidth; i++){
		for(int j = 0; j < n; j++){
			long long val = i + j * n;
			res[i - rank*floor(n/p)][j] = val;
		}

	}



	return res;
}

void printMatrix(vector<vector<long long> > matrix){
	cout << "<<<<<<<<<<<<<<<<<<<<	Matrix print 	<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
	for(int i = 0; i < matrix.size(); i++){
		for (int j = 0; j < matrix[i].size(); j++){
			cout<<matrix[i][j]<<"\t";
		}
		cout<<endl;
	}
	cout<<endl;
}
