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
#include <map>
#include <vector>

#include "idg-config.h"

namespace idg {

    class ProxyInfo {
        public:
        // Default constructor
        ProxyInfo();

        // copy constructor, assigment: default okay

        ~ProxyInfo() = default;

        // get methods
        std::string get_path_to_src() const { return path_to_src; }
        std::string get_path_to_lib() const { return path_to_lib; }

        std::vector<std::string> get_lib_names() const;
        std::vector<std::string> get_source_files(std::string libname) const;

        bool delete_shared_objects() { return delete_libs; }
        bool keep_shared_objects() { return !delete_libs; }

        // set methods
        void set_path_to_src(std::string s) { path_to_src = s; }
        void set_path_to_lib(std::string s) { path_to_lib = s; }

        void add_lib(std::string libname);
        void add_src_file_to_lib(std::string libname, std::string filename, bool optional = false);

        void set_delete_shared_objects(bool b) { delete_libs=b; }
        void set_keep_shared_objects(bool b) { delete_libs=!b; }

        // auxiliary functions
        void print() const { print(std::cout); }
        void print(std::ostream& os) const;

    private:
        std::string path_to_src; // path where src code that is put into libs lives
        std::string path_to_lib; // path where to create the libs
        std::map< std::string, std::vector<std::string> > libs; // maps library name to list of files
        bool delete_libs;
    }; // end class ProxyInfo

    // helper functions
    std::ostream& operator<<(std::ostream& os, const ProxyInfo& pi);

} // end namepace idg

#endif
