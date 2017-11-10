git:
	git add .
	git commit -m "auto"
	git push


main:
	nvcc -arch=sm_35 -o main.out main.cu -rdc=true -lcudadevrt


simple:
	nvcc -arch=sm_35 -o simple.out simplePrintf.cu -rdc=true -lcudadevrt


atomic:
	nvcc -arch=sm_35 -o atomic.out atomic.cu -rdc=true -lcudadevrt
