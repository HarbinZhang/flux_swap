#include <omp.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stack>
#include <utility>
#include "g.h"
#include <queue>
// #include "Myqueue.h"
using namespace std;


static const int s = 12;
static const double fai = 1e-2;


int main(){

	queue<pair<double, double> > q;
	double res = 0;
	double M = 0;
	q.push(make_pair(1.0, 5.0));

	omp_lock_t lock;
	omp_lock_t lock_q;
	omp_lock_t lock_threads;
	omp_init_lock(&lock);
	omp_init_lock(&lock_q);
	omp_init_lock(&lock_threads);

	int Nthreads = omp_get_thread_num();
	int num_available_threads = Nthreads;


	double start = omp_get_wtime();
	#pragma omp parallel
	{
		double local_M = 0;
		stack<pair <double, double> > stack;


		omp_set_lock(&lock_q);
		// while(!q.empty() && num_available_threads){
		while(!q.empty()){
			num_available_threads--;
			// omp_unset_lock(&lock);
			pair<double, double> temp = q.front();
			q.pop();
			omp_unset_lock(&lock_q);

			// if(temp.second == 0){continue;}

			// omp_set_lock(&lock);
			// cout<<"temp: "<<temp.first<<" "<<temp.second<<endl;
			// omp_unset_lock(&lock);			

			double a = temp.first;
			double b = temp.second;

			double ga = g(a);
			double gb = g(b);

			omp_set_lock(&lock);
			M = max(M, ga);
			M = max(M, gb);
			// cout<<"new M: "<<M<<" a: "<<a<<" b: "<<b<<endl;
			omp_unset_lock(&lock);

			double newg = (g(a) + g(b) + s*(b-a)) / 2;

			omp_set_lock(&lock);
			// cout<<"newg: "<<newg<<" M: "<<M<<endl;
			if(newg < M + fai){
				// res = max(M, res);
				omp_unset_lock(&lock);
			}else{
				// omp_set_lock(&lock);
				// cout<<"a+b: "<<a+b<<endl;
				omp_unset_lock(&lock);

				omp_set_lock(&lock_q);
				q.push(make_pair (a, (a+b)/2));
				q.push(make_pair ((a+b)/2, b));
				omp_unset_lock(&lock_q);
			}
		

		}
		omp_unset_lock(&lock_q);
		omp_set_lock(&lock);
		cout<<"M: "<<M<<" with id= "<<omp_get_thread_num()<<endl;
		omp_unset_lock(&lock);
	
	}


	double end = omp_get_wtime();
	cout<<"M: "<<M<<" duration: "<<end - start<<endl;
	omp_destroy_lock(&lock);
	omp_destroy_lock(&lock_q);

	return 0;

}




