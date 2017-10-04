#ifndef F_H
#define F_H

inline int fast_mod_exp(int x, int e, int m) {
  int ret = 1;
  while (e > 0) {
	  if (e & 1) {
		 ret = (ret * x) % m;
	  }
	  x = (x * x) % m;
	  e = e >> 1;
  }
  return ret;
}

long long f(long long a, long long b, long long c, long long d) {
  int e = static_cast<int>((b + c + d) % 20016); //Euler's theorem
  int x = static_cast<int>(a % 20017);
  for (int i = 0; i < 100; i++) {
    int next_e = x;
    x = fast_mod_exp(x,e,20017);
    e = next_e;
  }
  return static_cast<long long>(x);
}

#endif