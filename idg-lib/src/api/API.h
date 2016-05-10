/*
 * API.h
 * Implementation of IDG's high level interfaces
 */

#ifndef IDG_INTERFACE_H_
#define IDG_INTERFACE_H_

#include <complex>

namespace idg {


    enum class Type {
        CPU_REFERENCE,
        CPU_OPTIMIZED,
        CUDA_KEPLER,
        CUDA_MAXWELL
    };


    class WSCLEAN_API
    {
    public:
        virtual ~WSCLEAN_API() {};

        virtual void set_frequencies(const double* frequencyList,
                                     size_t channelCount) = 0;

        virtual void set_stations(const size_t nr_stations) = 0;

        /* virtual void set_kernel(size_t kernelSize, */
        /*                         const double* spheroidal) = 0; */
        // Better this?
        /* virtual void set_spheroidal(const double* spheroidal, size_t size) override; */

        virtual void start_w_layer(double layerWInLambda) = 0;
        virtual void finish_w_layer() = 0;

        /* virtual void start_aterm(const std::complex<double>* aterm) = 0; */
        /* virtual void finish_aterm() = 0; */

        virtual void set_grid(std::complex<double>* grid,
                              size_t height, size_t width) = 0;

        virtual void bake() = 0;

        // Gridding
        virtual void grid_visibilities(
            const std::complex<float>* visibilities, // size CH x PL
            const double* uvwInMeters,
            size_t antenna1,
            size_t antenna2,
            size_t timeIndex) = 0;

        /* // Deridding */
        /* virtual void queue_visibility_sampling( */
        /*     //std::complex<float>* visibility, // size CH x PL */
        /*     const double* uvwInMeters, */
        /*     size_t antenna1, */
        /*     size_t antenna2, */
        /*     size_t timeIndex, */
        /*     size_t rowId, */
        /*     bool& isBufferFull) = 0; */

        virtual void execute() = 0;

        /* virtual void finish_sampled_visibilities() = 0; */

        /* virtual size_t get_sampling_buffer_size() const = 0; */

        /* virtual void get_sampling_buffer(size_t index, */
        /*                                  std::complex<double>* data, */
        /*                                  size_t& rowID) const = 0; */
    };

} // end namespace idg

#endif
