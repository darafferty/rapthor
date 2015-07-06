/** 
 *  \class ProxyInfo
 *
 *  \brief Collection of additonal information for the Proxy
 *
 *  Have a more detailed description here
 */

#ifndef IDG_PROXYINFO_H_
#define IDG_PROXYINFO_H_

#include <iostream>
#include <string>

namespace idg {
      
    class ProxyInfo {
    public:
      ProxyInfo() // set default values
	: path_to_src("./src"),
	  path_to_lib("./lib")
	{ }

      // copy constructor, assigment: default okay
      
      ~ProxyInfo() = default;

      // set and get methods
      std::string get_path_to_src() const { return path_to_src; }
      std::string get_path_to_lib() const { return path_to_lib; }      

      // auxiliary functions
      void print() const { print(std::cout); }
      void print(std::ostream& os) const {
	os << "Path to source files: " << path_to_src << std::endl;
	os << "Path to library: " << path_to_lib << std::endl;
      }

    private:
      std::string path_to_src;
      std::string path_to_lib;      
    
    };

    // helper functions
    std::ostream& operator<<(std::ostream& os, const ProxyInfo& pi);

} // namepace idg

#endif
