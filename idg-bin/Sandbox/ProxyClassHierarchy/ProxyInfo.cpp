#include <iostream>
#include "ProxyInfo.h"

using namespace std;

namespace idg {




    // helper functions
    ostream& operator<<(ostream& os, const ProxyInfo& pi) {
      pi.print(os);
      return os;
    }

} // namepace idg
