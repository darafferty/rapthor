/**
 *  \class CPU
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_H_
#define IDG_CPU_H_

#include <dlfcn.h>
#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "Proxy.h"
#include "Kernels.h"
#include "LikwidPowerSensor.h"

namespace idg {
    namespace proxy {

        class CPU : public Proxy {
            public:
                /// Constructors
                CPU(Parameters params,
                    Compiler compiler = default_compiler(),
                    Compilerflags flags = default_compiler_flags(),
                    ProxyInfo info = default_info());

                /// Copy constructor
                //CPU(const CPU& v) = delete;

                /// Destructor
                virtual ~CPU();

                /// Assignment
                CPU& operator=(const CPU& rhs) = delete;

                // Get default values
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();

                // Get parameters of proxy
                const Parameters& get_parameters() const { return mParams; }
                const ProxyInfo& get_info() const { return mInfo; }

            // High level routines
            public:
                virtual void grid_visibilities(GRIDDING_PARAMETERS);
                virtual void degrid_visibilities(DEGRIDDING_PARAMETERS);
                virtual void transform(DomainAtoDomainB direction, void* grid);

            // Low level routines
            public:
                /** \brief Grid the visibilities onto uniform subgrids
                           (visibilities -> subgrids). */
                virtual void grid_onto_subgrids(GRIDDER_PARAMETERS) override;

                /** \brief Add subgrids to a grid
                           (subgrids -> grid). */
                virtual void add_subgrids_to_grid(ADDER_PARAMETERS) override;

                /** \brief Exctract subgrids from a grid
                           (grid -> subgrids). */
                virtual void split_grid_into_subgrids(SPLITTER_PARAMETERS) override;

                /** \brief Degrid the visibilities from uniform subgrids
                           (subgrids -> visibilities). */
                virtual void degrid_from_subgrids(DEGRIDDER_PARAMETERS) override;

            protected:
                static std::string make_tempdir();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

                void compile(Compiler compiler, Compilerflags flags);
                void parameter_sanity_check();
                void load_shared_objects();
                void find_kernel_functions();

                // data
                ProxyInfo mInfo; // info about shared object files

                // store the ptr to Module, which each loads an .so-file
                std::vector<runtime::Module*> modules;
                std::map<std::string,int> which_module;
        }; // class CPU

    } // namespace proxy
} // namespace idg

#endif
