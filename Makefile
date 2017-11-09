git:
	git add .
	git commit -m "auto"
	git push


main:
	nvcc -arch=sm_35 -o main.o main.cu -rdc=true -lcudadevrt
	./main.o