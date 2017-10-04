CC = g++
CFLAGS = -g -Wall -Wextra -Werror -std=c++11 -pthread
LIBS = -L. -lcpu -ldl

EXECS=hello
MPICC?=mpicc
MPICPP?=mpic++

.phony: all, clean 


test:
	g++ main.cpp -g -Wall -ldl -pthread -std=c++11 -o main.out
	./main.out $(N)

hello: hello.c
	${MPICC} -o hello hello.c

mpi_test: main.cpp
	${MPICPP} -o main main.cpp

	./main $(N)

git:
	git add .
	git commit -m "auto"
	git push