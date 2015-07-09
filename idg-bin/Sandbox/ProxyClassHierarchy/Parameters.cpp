
#include <iostream>
#include "Parameters.h"

using namespace std;

namespace idg {

  // helper functions
  ostream& operator<<(ostream& os, const Parameters& c) 
  {
    c.print(os);
    return os;
  }


} // namespace idg
