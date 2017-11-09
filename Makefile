git:
	git add .
	git commit -m "auto"
	git push


main:
	nvcc -arch=sm_35 -o main main.cu -rdc=true -lcudadevrt
	./main


simple:
	nvcc -arch=sm_35 -o simple simplePrintf.cu -rdc=true -lcudadevrt
	./simple


atomic:
	nvcc -arch=sm_35 -o atomic atomic.cu -rdc=true -lcudadevrt
	./atomic
