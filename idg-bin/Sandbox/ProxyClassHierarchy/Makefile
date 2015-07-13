CC = gcc
CXX = g++
LD = g++

DEBUG = 1
PERFORMANCE = 0

CXXFLAGS = -std=c++0x -Wall -O2 -fopenmp -DDEBUG=$(DEBUG) -DPERF=$(PERFORMANCE)
LDFLAGS = $(CXXFLAGS)

DIRS = . ./utility ./external

HDR = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.h))
SRC = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.cpp))
OBJ = $(patsubst %.cpp,%.o,$(SRC))

INC = -I./utility -I./external
LIBS = -ldl -lfftw3 -lfftw3f

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
