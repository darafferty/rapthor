# GNU
# CC = gcc
# CXX = g++
# LD = g++

# Intel
CC = icc
CXX = icpc
LD = icpc

# CXXFLAGS = -std=c++0x -Wall -O2 -xAVX -fopenmp -DDEBUG -DREPORT_VERBOSE
# CXXFLAGS = -std=c++0x -Wall -O2 -xAVX -fopenmp -DREPORT_TOTAL
CXXFLAGS = -std=c++0x -Wall -O2 -xAVX -fopenmp -DREPORT_VERBOSE
# CXXFLAGS = -std=c++0x -Wall -O2 -xAVX -fopenmp 
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
