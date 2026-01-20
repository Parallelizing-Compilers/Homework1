CC = gcc
CXX = g++
CXXFLAGS += -std=c++20 -fopenmp
CFLAGS += -fopenmp
LDLIBS +=

all: dgemm-naive dgemm-optimized

clean:
	rm -rf dgemm-naive dgemm-optimized
	rm -rf *.o *.dSYM *.trace

dgemm-naive.o: dgemm-naive.c
	$(CC) -c $< -o $@

dgemm-optimized.o: dgemm-optimized.c
	$(CC) $(CFLAGS) -c $< -o $@

benchmark.o: benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dgemm-naive: benchmark.o dgemm-naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

dgemm-optimized: benchmark.o dgemm-optimized.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)