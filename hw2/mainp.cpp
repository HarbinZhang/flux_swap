#include <omp.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stack>
#include <utility>
#include "g.h"
#include <queue>
#include <vector>
#include <stack>

using namespace std;

	
static const int s = 12;
static const double fai = 1e-4;

class Node
{
public:
	double first, second;
	double ga, gb;
	bool ga_valid, gb_valid;
	double M;

	Node(double first, double second, double M, double ga, double gb, bool ga_valid, bool gb_valid){
		this->first = first;
		this->second = second;
		this->ga = ga;
		this->gb = gb;
		this->ga_valid = ga_valid;
		this->gb_valid = gb_valid;
		this->M = M;
	}
	~Node();
	
};

struct LessThanByM
{
  bool operator()(const Node* lhs, const Node* rhs) const
  {
    return lhs->M < rhs->M;
  }
};


int main(){

	// stack<Node *> q;
	priority_queue<Node*, vector<Node*>, LessThanByM> q; 
	// double res = 0;
	double M = 0;
	Node* node_p = new Node(1.0, 100.0, 0, 0, 0, false, false);
	q.push(node_p);

	omp_lock_t lock;
	omp_lock_t lock_q;
	omp_lock_t lock_qchanged;

	omp_init_lock(&lock);
	omp_init_lock(&lock_q);
	omp_init_lock(&lock_qchanged);

	int num_active_threads = 0;

	double start = omp_get_wtime();


	#pragma omp parallel
	{
		// double local_M = 0;
		
		while(true){
			omp_set_lock(&lock_q);
			if(q.empty()){
				// cout<<"nthreads: "<<num_active_threads<<" M: "<<M<<endl;
				if(num_active_threads == 0){
					omp_unset_lock(&lock_q);
					break;
				}
				else{
					omp_unset_lock(&lock_q);
					continue;
				}
			}

			Node* temp = q.top();
			q.pop();				
			num_active_threads++;
			omp_unset_lock(&lock_q);			

			

			// if(temp.second == 0){continue;}

			// omp_set_lock(&lock);
			// cout<<"temp: "<<temp.first<<" "<<temp.second<<endl;
			// omp_unset_lock(&lock);			

			double a = temp->first;
			double b = temp->second;

			double ga = temp->ga_valid ? temp->ga:g(a);
			double gb = temp->gb_valid ? temp->gb:g(b);
			// double ga = g(a);
			// double gb = g(b);

			omp_set_lock(&lock);
			M = max(M, ga);
			M = max(M, gb);
			// cout<<"new M: "<<M<<" a: "<<a<<" b: "<<b<<endl;
			omp_unset_lock(&lock);

			double newg = (ga + gb + s*(b-a)) / 2;

			omp_set_lock(&lock);
			// cout<<"newg: "<<newg<<" M: "<<M<<endl;
			if(newg < M + fai){
				// res = max(M, res);
				omp_unset_lock(&lock);
				omp_set_lock(&lock_q);
				num_active_threads--;
				omp_unset_lock(&lock_q);
			}else{
				// omp_set_lock(&lock);
				// cout<<"a+b: "<<a+b<<endl;
				omp_unset_lock(&lock);

				omp_set_lock(&lock_q);
				// q.push(make_pair (a, (a+b)/2));
				// q.push(make_pair ((a+b)/2, b));
				Node* temp = new Node(a, (a+b)/2, newg, ga, 0, true, false);
				q.push(temp);
				temp = new Node((a+b)/2, b, newg, 0, gb, false, true);
				q.push(temp);
				num_active_threads--;
				omp_unset_lock(&lock_q);
			}
		}
		omp_unset_lock(&lock_q);
		// omp_set_lock(&lock);
		// cout<<"M: "<<M<<" with id= "<<omp_get_thread_num()<<endl;
		// omp_unset_lock(&lock);
		
		cout<< " #threads: "<<omp_get_num_threads()<<endl;			
		// #pragma omp barrier	

	}


	double end = omp_get_wtime();
	cout<<"M: "<<M<<" duration: "<<end - start<<endl;
	omp_destroy_lock(&lock);
	omp_destroy_lock(&lock_q);

	return 0;

}




