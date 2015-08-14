#include <iostream>
#include "CompilerEnvironment.h"

using namespace std;

int main(int argc, char *argv[])
{
  idg::CompilerEnvironment cc;

  // this is what has been set by the default constructor
  cc.print();

  // set values explicitly
  cc.set_c_compiler("/user/bin/gcc");
  cc.set_c_flags("-Wall -O0 -pg -DDEBUG -fopenmp");
  cc.set_cpp_compiler("/user/bin/g++");
  cc.set_cpp_flags("-O2 -fopenmp");

  // new values, now using operator<< for output
  cout << cc;

  return 0;
}
