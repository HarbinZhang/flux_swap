#include <mpi.h>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include "f.h"
#include <fstream>
using namespace std;


vector<vector<long long> > initMatrix(int n, int p, int rank);
void printMatrix(vector<vector<long long> > matrix, int rank);
static int bandwidth_X, bandwidth_Y;
static int start_row, start_column;


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

	if(rank == 0){cout<<"<<<<<<<<<<<<<<<<<<<<<< start "<<endl;}

	int row_start_func = 0;
	if (start_row == 0){
		row_start_func = 1;
	}
	int row_end_func = bandwidth_X;
	if (start_row == floor(n/sqrt(p)) * (sqrt(p)-1)){
		row_end_func = bandwidth_X - 1;
	}
	int column_start_func = 0;
	if (start_column == 0){
		column_start_func = 1;
	}
	int column_end_func = bandwidth_Y;
	if ((rank+1) % (int)sqrt(p) == 0){
		column_end_func--;
	}
	// if(rank == 2) cout<<column_end_func<<" "<<column_start_func<<endl;

	for(int k = 0; k < 10; k++){
		// if not bottom
		if(rank < sqrt(p)*(sqrt(p)-1)){
			MPI_Recv(&matrix[bandwidth_X].front(), bandwidth_Y, MPI_LONG_LONG, rank + sqrt(p), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// if not top
		if(start_row != 0){
			MPI_Send(&matrix[0].front(), bandwidth_Y, MPI_LONG_LONG, rank - sqrt(p), 1, MPI_COMM_WORLD);
		}

		// if not right
		if((rank+1) % (int)sqrt(p) != 0){
			long long column_vec[bandwidth_X+1];
			MPI_Recv(&column_vec[0], bandwidth_X+1, MPI_LONG_LONG, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);			
			for(int i = 0; i < bandwidth_X+1; i++){
				matrix[i][bandwidth_Y] = column_vec[i];
			}
		}
		// if not left
		if(start_column != 0){
			long long column_vec[bandwidth_X+1];
			for(int i = 0; i < bandwidth_X+1; i++){
				column_vec[i] = matrix[i][0];
			}
			MPI_Send(&column_vec[0], bandwidth_X+1, MPI_LONG_LONG, rank - 1, 1, MPI_COMM_WORLD);			
		}

		// iteration
		for(int i = row_start_func; i < row_end_func; i++){
			for(int j = column_start_func; j < column_end_func; j++){
				matrix[i][j] = f(matrix[i][j], matrix[i+1][j], matrix[i][j+1], matrix[i+1][j+1]);
			}
		}


	}

	// printMatrix(matrix, rank);


	long long sum = 0;
	for(int i = 0; i < bandwidth_X; i++){
		for(int j = 0; j < bandwidth_Y; j++){
			sum += matrix[i][j];
		}
	}
	// if not right
	long long recv_sum_row = 0;
	if((rank+1) % (int)sqrt(p) != 0){
		MPI_Recv(&recv_sum_row, 1, MPI_LONG_LONG, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	// if not left
	if(start_column != 0){
		sum += recv_sum_row;
		MPI_Send(&sum, 1, MPI_LONG_LONG, rank - 1, 1, MPI_COMM_WORLD);		
	}
	// if not bottom && column == 0
	long long recv_sum_column = 0;
	if(rank < sqrt(p)*(sqrt(p)-1) && start_column == 0){
		MPI_Recv(&recv_sum_column, 1, MPI_LONG_LONG, rank + sqrt(p), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
	}
	// if not top && column == 0
	if(start_row != 0 && start_column == 0){
		sum += recv_sum_row;
		sum += recv_sum_column;
		MPI_Send(&sum, 1, MPI_LONG_LONG, rank - sqrt(p), 1, MPI_COMM_WORLD);				
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0){
		sum += recv_sum_row;
		sum += recv_sum_column; 
		cout<<"Sum is : "<<sum<<endl;

		double endTime = MPI_Wtime();
		cout << "The total time is: " << endTime - startTime<<endl;
	}
	
	if(rank == (sqrt(p) + 1) * floor(sqrt(p) / 2)){
		int index = n/2 % (int)(n/sqrt(p));
		cout<<"The mid is : "<<matrix[index][index]<<" index is "<<index<<endl;
		// printMatrix(matrix,rank);
	}


	MPI_Finalize();
	return 0;
}



vector<vector<long long> >initMatrix(int n, int p, int rank){

	

	bandwidth_X = floor(n/sqrt(p));
	bandwidth_Y = floor(n/sqrt(p));
	start_row = bandwidth_X * floor(rank / sqrt(p));
	start_column = bandwidth_Y * (rank - floor(rank / sqrt(p))*sqrt(p));
	// p is bottom
	if(rank >= sqrt(p)*(sqrt(p)-1)){
		bandwidth_X = n - (sqrt(p)-1)*floor(n/sqrt(p));
	}
	// p is right
	if((rank+1) % (int)sqrt(p) == 0){
		bandwidth_Y = n - (sqrt(p)-1)*floor(n/sqrt(p));
	}

	// if(rank == 3) cout<<start_row<<" "<<start_column<<endl;
	vector<long long> row(bandwidth_Y+1, 0);
	vector<vector<long long> > res(bandwidth_X+1, row);

	for(int i = start_row; i < start_row + bandwidth_X; i++){
		for(int j = start_column; j < start_column + bandwidth_Y; j++){
			long long val = i + j * n;
			res[i - start_row][j - start_column] = val;
		}
	}


	return res;
}

void printMatrix(vector<vector<long long> > matrix, int rank){
	cout << "<<<<<<<<<<<<<<<<<<<<	Matrix print :"<<rank<<" 	<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
	for(int i = 0; i < matrix.size(); i++){
		for (int j = 0; j < matrix[i].size(); j++){
			cout<<matrix[i][j]<<"\t";
		}
		cout<<endl;
	}
	cout<<endl;
}