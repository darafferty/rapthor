#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "idg-config.h"
#include "ProxyInfo.h"

using namespace std;

inline bool exists (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

namespace idg {

  ProxyInfo::ProxyInfo() // set default values
	: path_to_src("./src"),
	  path_to_lib("./lib")
  { delete_libs=true; }


  void ProxyInfo::add_lib(std::string libname) 
  {
    libs[libname] = vector<string>();
  }

  void ProxyInfo::add_src_file_to_lib(std::string libname, std::string filename, bool optional) 
  {
    if (libs.find(libname) != libs.end()) {
      // libname exist as a key, add a file name to the value vector
      if (!optional || exists(path_to_src + "/" + filename)) {
          libs.at(libname).push_back(filename);
      }
    }
  }


  vector<string> ProxyInfo::get_lib_names() const 
  {
    vector<string> v;
    for (auto iter = libs.begin(); iter!=libs.end(); iter++)
      v.push_back(iter->first);
    return v;
  }

  
  vector<string> ProxyInfo::get_source_files(string libname) const 
  {
    return libs.at(libname);
  }
  

  void ProxyInfo::print(std::ostream& os) const 
  {
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
  ostream& operator<<(ostream& os, const ProxyInfo& pi) 
  {
    pi.print(os);
    return os;
  }

} // namepace idg
