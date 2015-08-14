# GNU
CC = gcc
CXX = g++
LD = g++

CXXFLAGS = -std=c++0x -Wall -O2 -fopenmp -DREPORT_VERBOSE
LDFLAGS = $(CXXFLAGS)

DIRS = . ./common ./utility ./external

HDR = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.h))
SRC = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.cpp))
OBJ = $(patsubst %.cpp,%.o,$(SRC))

INC = -I./common -I./utility -I./external
LIBS = -ldl -lfftw3 -lfftw3f -lfftw3f_omp

all: main.x

main.x: $(HDR) $(OBJ) 	
	$(LD) $(LDFLAGS) $(INC) $(OBJ) $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

.PHONY: clean
clean: 
	rm -f *.x *~ $(OBJ)
