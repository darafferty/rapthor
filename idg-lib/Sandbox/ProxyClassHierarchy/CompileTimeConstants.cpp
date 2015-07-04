
#include <iostream>
#include "CompileTimeConstants.h"

using namespace std;

namespace idg {

  // helper functions
  ostream& operator<<(ostream& os, const CompileTimeConstants& ctc) 
  {
    ctc.print(os);
    return os;
  }


} // namespace idg
