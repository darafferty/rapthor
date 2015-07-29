# GNU
# CC = gcc
# CXX = g++
# LD = g++

# Intel
CC = icc
CXX = icpc
LD = icpc

DEBUG = 0
REPORT_VERBOSE = 1
REPORT_TOTAL = 1

CXXFLAGS = -std=c++0x -Wall -O2 -xAVX -fopenmp -DDEBUG=$(DEBUG) -DREPORT_VERBOSE=$(REPORT_VERBOSE) -DREPORT_TOTAL=$(REPORT_TOTAL)
LDFLAGS = $(CXXFLAGS)

DIRS = . ./common ./utility ./external

HDR = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.h))
SRC = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.cpp))
OBJ = $(patsubst %.cpp,%.o,$(SRC))

INC = -I./common -I./utility -I./external
# LIBS = -ldl -lfftw3 -lfftw3f
LIBS = -ldl -mkl -lmkl_vml_avx -lmkl_avx

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
