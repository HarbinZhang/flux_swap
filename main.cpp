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



	// send row value to rank - 1
	if(rank != 0){
		MPI_Send(&matrix[0].front(), n, MPI_LONG_LONG, rank - 1, 1, MPI_COMM_WORLD);
	}
	if(rank != p - 1){
		MPI_Recv(&matrix[bandwidth].front(), n, MPI_LONG_LONG, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// if(rank == 0){cout<<"send row done "<< matrix[75][1] <<endl;}
	// else{
	// 	cout<<"hi "<<matrix[0][1]<<endl;
	// }

	// iteration
	int row_start = 0;
	if (rank == 0){
		row_start = 1;
	}
	int row_end = bandwidth;
	if (rank == p - 1){
		row_end = bandwidth - 1;
	}

	for(int k = 0; k < 10; k++){
		for(int i = row_start; i < row_end; i++){
			for(int j = 1; j < n - 1; j++){
				matrix[i][j] = f(matrix[i][j], matrix[i+1][j], matrix[i][j+1], matrix[i+1][j+1]);
			}
		}
	}

	int temp = 0;
	if(rank == 0){
		for(int i= 0; i<150; i++){
			temp += matrix[75][i];
		}
		cout<<"0 ceshi"<<temp<<" "<<matrix[75][75]<<endl;
	}else{
		for(int i= 0; i<150; i++){
			temp += matrix[0][i];
		}
		cout<<"1 ceshi"<<temp<<" "<<matrix[0][75]<<endl;		
	}

	
	if(rank == 0){cout<<"iteration done "<<" "<<row_end<<endl;}

	// sum
	long long sum = 0;
	for(int i = 0; i < bandwidth; i++){
		for(int j = 0; j < n; j++){
			sum += matrix[i][j];
		}
	}

	if(rank == 0){cout<<"sum done "<<sum<<endl;}

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
	
	if(rank == 0){cout<<"send back done"<<endl;}

	
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

	if(rank == 0){cout<<"send mid done"<<endl;}
	

	double endTime, totalTime;
	if(rank == 0){
		double endTime = MPI_Wtime();
		double totalTime = endTime - startTime;
		cout << "The total time is: " << totalTime<<endl;
		cout << "sum is: "<<sum<<endl;
		cout << "mid is: "<<mid<<endl;
	}

	// cout << matrix.size() << endl;
	//printMatrix(matrix);

	MPI_Finalize();
	return 0;

}

vector<vector<long long> >initMatrix(int n, int p, int rank){
	vector<vector<long long> > res;
	vector<long long> row;

	bandwidth = n/p;
	if(rank == p - 1){
		bandwidth = n - (p-1) * n/p;
	}

	vector<long long> null_vec(n, 0);
	// res.push_back(null_vec);

	for(int i = rank*n/p; i < rank*n/p + bandwidth; i++){
		for(int j = 0; j < n; j++){
			long long val = i + j * n;
			row.push_back(val);
		}
		res.push_back(row);
		row.clear();
	}


	res.push_back(null_vec);
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
