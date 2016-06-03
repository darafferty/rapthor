/**
 *  \class Kepler
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_KEPLER_H_
#define IDG_CUDA_KEPLER_H_

#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Kepler : public CUDA {
                public:
                    /// Constructor
                    Kepler(
                        Parameters params,
                        unsigned deviceNumber = 0,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    /// Destructor
                    ~Kepler() = default;

                protected:
                    virtual dim3 get_block_gridder() const override;
                    virtual dim3 get_block_degridder() const override;
                    virtual dim3 get_block_adder() const override;
                    virtual dim3 get_block_splitter() const override;
                    virtual dim3 get_block_scaler() const override;
                    virtual int get_gridder_batch_size() const override;
                    virtual int get_degridder_batch_size() const override;
                    virtual std::string append(Compilerflags flags) const override;
            }; // class Kepler

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
