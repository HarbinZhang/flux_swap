CC = g++
CFLAGS = -g -Wall -Wextra -Werror -std=c++11 -pthread
LIBS = -L. -lcpu -ldl

EXECS=hello
MPICC?=mpicc
MPICPP?=mpic++

.phony: all, clean 


module:
	module load gcc/5.4.0
	module load openmpi/1.10.2/gcc/5.4.0

test:
	g++ main.cpp -g -Wall -ldl -pthread -std=c++11 -o main.out
	./main.out $(N)

hello: hello.c
	${MPICC} -o hello hello.c

mpi_test: main.cpp
	${MPICPP} -o main main.cpp
	mpirun -n 2 ./main 4


git:
	git add .
	git commit -m "auto"
	git push
