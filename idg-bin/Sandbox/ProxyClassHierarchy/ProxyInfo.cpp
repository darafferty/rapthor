#include <iostream>
#include <vector>
#include <string>
#include "ProxyInfo.h"

using namespace std;

namespace idg {

  ProxyInfo::ProxyInfo() // set default values
	: path_to_src("./src"),
	  path_to_lib("./lib")
  { }


  void ProxyInfo::add_lib(std::string libname) 
  {
    libs[libname] = vector<string>();
  }

  void ProxyInfo::add_src_file_to_lib(std::string libname, std::string filename) 
  {
    if (libs.find(libname) != libs.end()) {
	// libname exist as a key, add a file name to the value vector
	libs.at(libname).push_back(filename);
      }
  }

  void ProxyInfo::print(std::ostream& os) const {
    os << "PROXY INFO" << endl;
    os << "Path to source files: " << path_to_src << std::endl;
    os << "Path to library: " << path_to_lib << std::endl;
    
    for (auto& s : libs) {
      os << s.first << ": ";
      for (auto& e : s.second)
	os << e << " ";
      os << endl;
    }
  }
  
  // helper functions
  ostream& operator<<(ostream& os, const ProxyInfo& pi) {
    pi.print(os);
    return os;
  }

} // namepace idg
