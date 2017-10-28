#include <omp.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stack>
#include <utility>
#include "g.h"
#include "Myqueue.h"
using namespace std;


static const int s = 12;
static const double fai = 1e-6;


int main(){

	Myqueue q;
	double res = 0;
	double M = 0;
	q.push(make_pair(1.0, 100.0));

	omp_lock_t lock;
	omp_lock_t lock_q;
	omp_init_lock(&lock);
	omp_init_lock(&lock_q);

	double start = omp_get_wtime();
	#pragma omp parallel
	{
		double local_M = 0;
		stack<pair <double, double> > stack;
		// omp_set_lock(&lock);
		while(q.size() > 0){
			// omp_unset_lock(&lock);
			pair<double, double> temp = q.pop();
			// omp_unset_lock(&lock);

			if(temp.second == 0){continue;}

			// omp_set_lock(&lock);
			// cout<<"temp: "<<temp.first<<" "<<temp.second<<endl;
			// omp_unset_lock(&lock);			

			double a = temp.first;
			double b = temp.second;

			omp_set_lock(&lock);
			M = max(M, g(a));
			M = max(M, g(b));
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

				q.push(make_pair (a, (a+b)/2));
				q.push(make_pair ((a+b)/2, b));
				// omp_unset_lock(&lock);
			}
		}
		// omp_unset_lock(&lock);
		// omp_set_lock(&lock);
		// cout<<"M: "<<M<<" with id= "<<omp_get_thread_num()<<endl;
		// omp_unset_lock(&lock);

		#pragma omp barrier			
	}


	double end = omp_get_wtime();
	cout<<"M: "<<M<<" duration: "<<end - start<<endl;

	omp_destroy_lock(&lock);
	omp_destroy_lock(&lock_q);

	return 0;

}




