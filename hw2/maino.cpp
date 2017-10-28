#include <omp.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stack>
#include <utility>
#include "g.h"
#include <queue>
using namespace std;


static const int s = 12;
static const double fai = 1e-2;


int main(){

	queue<pair<double, double> > q;
	double res = 0;
	double M = 0;
	q.push(make_pair(1.0, 5.0));

	omp_lock_t lock_M;
	omp_lock_t lock_q;
	omp_lock_t lock_thread;
	omp_lock_t lock_qchanged;
	omp_init_lock(&lock_M);
	omp_init_lock(&lock_q);
	omp_init_lock(&lock_thread);
	omp_init_lock(&lock_qchanged);

	double start = omp_get_wtime();
	int Nthread = omp_get_num_threads();
	int curt_available_nthread = Nthread;
	#pragma omp parallel
	{
		stack<pair <double, double> > stack;
		omp_set_lock(&lock_thread);
		omp_set_lock(&lock_q);
		while(!q.empty() && curt_available_nthread < Nthread){
			if(q.empty()){
				// curt_available_nthread++;
				omp_unset_lock(&lock_thread);
				omp_unset_lock(&lock_q);
				while(1){

					if(q.empty() && curt_available_nthread == Nthread){
						break;
					}
				}
			}else{
				curt_available_nthread--;
				omp_unset_lock(&lock_thread);
				pair<double, double> temp = q.front();
				q.pop();
				omp_unset_lock(&lock_q);
				
				stack.push(temp);
				while(!stack.empty()){
					pair<double, double> temp = stack.top();
					stack.pop();

					double a = temp.first;
					double b = temp.second;

					double ga = g(a);
					double gb = g(b);

					omp_set_lock(&lock_M);
					M = max(M, ga);
					M = max(M, gb);
					// cout<<"new M: "<<M<<" a: "<<a<<" b: "<<b<<endl;
					omp_unset_lock(&lock_M);					

					double newg = (g(a) + g(b) + s*(b-a)) / 2;

					omp_set_lock(&lock_M);
					// cout<<"newg: "<<newg<<" M: "<<M<<endl;
					if(newg < M + fai){
					// res = max(M, res);
					omp_unset_lock(&lock_M);
					}else{
						// omp_set_lock(&lock);
						// cout<<"a+b: "<<a+b<<endl;
						omp_unset_lock(&lock_M);

						
						stack.push(make_pair (a, (a+b)/2));

						omp_set_lock(&lock_q);
						q.push(make_pair ((a+b)/2, b));
						omp_unset_lock(&lock_q);
					}
				}
				omp_set_lock(&lock_thread);
				curt_available_nthread++;
				omp_unset_lock(&lock_thread);
			}
		}
		omp_unset_lock(&lock_thread);
		omp_unset_lock(&lock_q);
	}

	double end = omp_get_wtime();
	cout<<"M: "<<M<<" duration: "<<end - start<<endl;
	omp_destroy_lock(&lock_M);
	omp_destroy_lock(&lock_q);
	omp_destroy_lock(&lock_thread);
	omp_destroy_lock(&lock_qchanged);

	return 0;
}



