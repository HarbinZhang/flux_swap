#include <queue>
#include <utility>

class Myqueue{
private:
	omp_lock_t lock;
	std::queue<std::pair<double, double> > q;
public:
	Myqueue(){
		omp_init_lock(&lock);
	}
	~Myqueue(){
		omp_destroy_lock(&lock);
	}

	void push(const std::pair<double, double>& item){
		omp_set_lock(&lock);
		this->q.push(item);
		omp_unset_lock(&lock);
	}

	bool try_push(const std::pair<double, double>& item){
		bool res = omp_test_lock(&lock);
		if(res){
			this->q.push(item);
			omp_unset_lock(&lock);
		}
		return res;
	}

	std::pair<double, double> pop(){
		omp_set_lock(&lock);

		std::pair<double, double> res = this->q.front();
		this->q.pop();
		omp_unset_lock(&lock);
		return res;

	}

	int size(){
		omp_set_lock(&lock);
		int res = this->q.size();
		omp_unset_lock(&lock);
		return res;
	}

};