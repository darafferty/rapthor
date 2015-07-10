CC = gcc
CXX = g++
LD = g++

CXXFLAGS = -std=c++0x -Wall -O2 -fopenmp
LDFLAGS = $(CXXFLAGS)

HDR = ../Parameters.h ../CompilerEnvironment.h 
SRC = ../Parameters.cpp ../CompilerEnvironment.cpp
OBJ = $(patsubst %.cpp,%.o,$(SRC))

INC = -I..

all: test-CompileTimeConstants.x test-CompilerEnvironment.x

test-CompileTimeConstants.x: test-CompileTimeConstants.o $(HDR) $(OBJ) 	
	$(LD) $(LDFLAGS) test-CompileTimeConstants.o $(OBJ) $(LIBS) -o $@

test-CompilerEnvironment.x: test-CompilerEnvironment.o $(HDR) $(OBJ) 	
	$(LD) $(LDFLAGS) test-CompilerEnvironment.o $(OBJ) $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $<

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $<

.PHONY: clean
clean: 
	rm -f *.x *~ *.o
