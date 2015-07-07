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
	path_to_lib("./lib"),
	src_gridder("KernelGridder.cpp"),
	src_degridder("KernelDegridder.cpp"),
	src_fft("KernelFFT.cpp"),
	src_adder("KernelAdder.cpp"),
	src_splitter("KernelSplitter.cpp"),
	so_gridder("Gridder.so"),
	so_degridder("Degridder.so"),
	so_fft("FFT.so"),
	so_adder("Adder.so"),
	so_splitter("Splitter.so")
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
      std::string src_gridder; 
      std::string src_degridder; 
      std::string src_fft; 
      std::string src_adder; 
      std::string src_splitter; 
      std::string so_gridder; 
      std::string so_degridder; 
      std::string so_fft; 
      std::string so_adder; 
      std::string so_splitter; 
    };

    // helper functions
    std::ostream& operator<<(std::ostream& os, const ProxyInfo& pi);

} // namepace idg

#endif
