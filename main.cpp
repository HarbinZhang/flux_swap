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

	for(int k = 1; k < 10; k++){
		for(int i = 1; i < n - 1; i++){
			for(int j = 1; j < n - 1; j++){
				matrix[i][j] = f(matrix[i][j], matrix[i][j+1], matrix[i+1][j], matrix[i+1][j+1]);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	double endTime, totalTime;
	if(rank == 0){
		double endTime = MPI_Wtime();
		double totalTime = endTime - startTime;
		cout << "The total time is: " << totalTime<<endl;
	}

	printMatrix(matrix);

}

vector<vector<long long> >initMatrix(int n, int p, int rank){
	vector<vector<long long> > res;
	vector<long long> row;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			long long val = i + j * n;
			row.push_back(val);
		}
		res.push_back(row);
		row.clear();
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
