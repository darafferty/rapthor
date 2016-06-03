/**
 *  \class Maxwell
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_MAXWELL_H_
#define IDG_CUDA_MAXWELL_H_

#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Maxwell : public CUDA {
                public:
                    /// Constructor
                    Maxwell(
                        Parameters params,
                        unsigned deviceNumber = 0,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    /// Destructor
                    ~Maxwell() = default;

                protected:
                    virtual dim3 get_block_gridder() const override;
                    virtual dim3 get_block_degridder() const override;
                    virtual dim3 get_block_adder() const override;
                    virtual dim3 get_block_splitter() const override;
                    virtual dim3 get_block_scaler() const override;
                    virtual int get_gridder_batch_size() const override;
                    virtual int get_degridder_batch_size() const override;
                    virtual std::string append(Compilerflags flags) const override;
            }; // class Maxwell

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
