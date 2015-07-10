CC = gcc
CXX = g++
LD = g++

CXXFLAGS = -std=c++0x -Wall -O2 -fopenmp
LDFLAGS = $(CXXFLAGS)

HDR = $(wildcard *.h)
SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp,%.o,$(SRC))

INC = 
LIBS = -ldl -lfftw3 -lfftw3f

all: main.x

main.x: $(HDR) $(OBJ) 	
	$(LD) $(LDFLAGS) $(OBJ) $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $<

%.o: %.cc 
	$(CXX) $(CXXFLAGS) $(INC) -c $<

.PHONY: clean
clean: 
	rm -f *.x *~ *.o
