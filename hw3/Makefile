git:
	git add .
	git commit -m "auto"
	git push

sta:
	qstat -u haibinzh
main:
	nvcc -arch=sm_35 -o main.out main.cu -rdc=true -lcudadevrt

main2:
	nvcc -arch=sm_35 -o main.out main2.cu -rdc=true -lcudadevrt -std=c++11

main4:
	nvcc -arch=sm_35 -o main.out main4.cu -rdc=true -lcudadevrt

simple:
	nvcc -arch=sm_35 -o simple.out simplePrintf.cu -rdc=true -lcudadevrt


atomic:
	nvcc -arch=sm_35 -o atomic.out atomic.cu -rdc=true -lcudadevrt
