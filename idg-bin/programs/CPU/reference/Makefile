# Directories
ROOT_DIR = ../../..
COMMON_DIR = ${ROOT_DIR}/common
UTILITY_DIR = ${ROOT_DIR}/utility
EXTERNAL_DIR = ${ROOT_DIR}/external
DIRS = . ${COMMON_DIR} ${UTILITY_DIR} ${EXTERNAL_DIR}

# GNU
CC = gcc
CXX = g++
LD = g++

# Flags
CXXFLAGS = -std=c++0x -Wall -O2 -fopenmp -DREPORT_VERBOSE
LDFLAGS = $(CXXFLAGS)
INC = -I${COMMON_DIR} -I${UTILITY_DIR} -I${EXTERNAL_DIR}
LIBS = -ldl -lfftw3 -lfftw3f -lfftw3f_omp

# Expand variables
HDR = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.h))
SRC = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.cpp))
OBJ = $(patsubst %.cpp,%.o,$(SRC))

# Rules
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
