/*
 * GridderBuffer.h
 * Access to IDG's high level gridder routines
 */


#include "GridderBufferImpl.h"
#include "BufferSetImpl.h"

#include <mutex>
#include <csignal>

#include <omp.h>

using namespace std;

namespace idg {
namespace api {

    GridderBufferImpl::GridderBufferImpl(
        BufferSetImpl* bufferset,
        proxy::Proxy* proxy,
        size_t bufferTimesteps)
        : BufferImpl(bufferset, proxy, bufferTimesteps),
          m_bufferUVW2(0,0),
          m_bufferStationPairs2(0),
          m_buffer_weights(0,0,0,0),
          m_buffer_weights2(0,0,0,0),
          m_default_aterm_correction(bufferset->m_default_aterm_correction),
          m_avg_aterm_correction(bufferset->m_avg_aterm_correction),
          m_average_beam(bufferset->m_average_beam),
          m_do_gridding(bufferset->m_do_gridding),
          m_do_compute_avg_beam(bufferset->m_do_compute_avg_beam),
          m_apply_aterm(bufferset->m_apply_aterm)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        m_aterm_offsets2 = m_default_aterm_offsets;
    }


    GridderBufferImpl::~GridderBufferImpl()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        if (m_flush_thread.joinable()) m_flush_thread.join();
    }


    void GridderBufferImpl::grid_visibilities(
        size_t timeIndex,
        size_t antenna1,
        size_t antenna2,
        const double* uvwInMeters,
        complex<float>* visibilities,
        const float* weights)
    {
        // exclude auto-correlations
        if (antenna1 == antenna2) return;

        int    local_time = timeIndex - m_timeStartThisBatch;
        size_t local_bl   = baseline_index(antenna1, antenna2);

        if (local_time < 0) {
            m_timeStartThisBatch = 0;
            m_timeStartNextBatch = m_bufferTimesteps;
            local_time = timeIndex;
        }

        if (local_time >= m_bufferTimesteps) {
            // Empty buffer before filling it up again
            flush();

            while (local_time >= m_bufferTimesteps) {
                m_timeStartThisBatch += m_bufferTimesteps;
                local_time = timeIndex - m_timeStartThisBatch;
            }
            m_timeStartNextBatch = m_timeStartThisBatch + m_bufferTimesteps;
        }

        // Keep track of all time indices pushed into the buffer
        m_timeindices.insert(timeIndex);

        // Copy data into buffers
        m_bufferUVW(local_bl, local_time) = {
            static_cast<float>(uvwInMeters[0]),
            static_cast<float>(uvwInMeters[1]),
            static_cast<float>(uvwInMeters[2])
        };

        m_bufferStationPairs(local_bl) = {
            static_cast<int>(antenna1),
            static_cast<int>(antenna2)
        };

        std::copy(visibilities,
                  visibilities + m_nr_channels * m_nrPolarizations,
                  (complex<float>*) &m_bufferVisibilities(local_bl, local_time, 0));
        std::copy(weights, weights + m_nr_channels*4, &m_buffer_weights(local_bl, local_time, 0, 0));
    }

    void GridderBufferImpl::compute_avg_beam()
    {
        m_bufferset->m_avg_beam_watch->Start();

        const unsigned int subgrid_size    = m_subgridsize;
        const unsigned int nr_correlations = 4;
        const unsigned int nr_aterms       = m_aterm_offsets2.size() - 1;
        const unsigned int nr_antennas     = m_nrStations;
        const unsigned int nr_baselines    = m_bufferStationPairs2.get_x_dim();
        const unsigned int nr_timesteps    = m_bufferUVW2.get_x_dim();
        const unsigned int nr_channels     = get_frequencies_size();

        // Define multidimensional types
        typedef std::complex<float> AverageBeam[subgrid_size*subgrid_size][nr_correlations][nr_correlations];
        typedef std::complex<float> ATerms[nr_aterms][nr_antennas][subgrid_size][subgrid_size][nr_correlations];
        typedef unsigned int ATermOffsets[nr_aterms+1];
        typedef unsigned int StationPairs[nr_baselines][2];
        typedef float UVW[nr_baselines][nr_timesteps][3];
        typedef float Weights[nr_baselines][nr_timesteps][nr_channels][nr_correlations];
        typedef float SumOfWeights[nr_baselines][nr_aterms][nr_correlations];

        // Cast class members to multidimensional types used in this method
        ATerms       *aterms        = (ATerms *) m_aterms2.data();
        AverageBeam  *average_beam  = (AverageBeam *) m_average_beam.data();
        ATermOffsets *aterm_offsets = (ATermOffsets *) m_aterm_offsets2.data();
        StationPairs *station_pairs = (StationPairs *) m_bufferStationPairs2.data();
        UVW          *uvw           = (UVW *) m_bufferUVW2.data();
        Weights      *weights       = (Weights *) m_buffer_weights2.data();

        // Initialize sum of weights
        std::vector<float> sum_of_weights_buffer(nr_baselines*nr_aterms*nr_correlations, 0.0);
        SumOfWeights &sum_of_weights = *((SumOfWeights *) sum_of_weights_buffer.data());

        // Compute sum of weights
        #pragma omp parallel for
        for (int n = 0; n < nr_aterms; n++) {
            int time_start = (*aterm_offsets)[n];
            int time_end = (*aterm_offsets)[n+1];

            // loop over baselines
            for (int bl = 0; bl < nr_baselines; bl++) {

                for (int t = time_start; t < time_end; t++)
                {
                    if (std::isinf((*uvw)[bl][t][0])) continue;

                    for (int ch = 0; ch < nr_channels; ch++)
                    {
                        for (int pol = 0; pol < nr_correlations; pol++)
                        {
                            sum_of_weights[bl][n][pol] += (*weights)[bl][t][ch][pol];
                        }
                    }
                }
            }
        }

        // Compute average beam for all pixels
        #pragma omp parallel for
        for (int i = 0; i < (subgrid_size * subgrid_size); i++) {
            std::complex<double> sum[nr_correlations][nr_correlations];

            // Loop over aterms
            for (int n = 0; n < nr_aterms; n++) {
                // Loop over baselines
                for (int bl = 0; bl < nr_baselines; bl++) {
                    unsigned int antenna1 = (*station_pairs)[bl][0];
                    unsigned int antenna2 = (*station_pairs)[bl][1];

                    // Check whether stationPair is initialized
                    if (antenna1 >= nr_antennas || antenna2 >= nr_antennas) {
                        continue;
                    }

                    std::complex<float> aXX1 = (*aterms)[n][antenna1][0][i][0];
                    std::complex<float> aXY1 = (*aterms)[n][antenna1][0][i][1];
                    std::complex<float> aYX1 = (*aterms)[n][antenna1][0][i][2];
                    std::complex<float> aYY1 = (*aterms)[n][antenna1][0][i][3];

                    std::complex<float> aXX2 = std::conj((*aterms)[n][antenna2][0][i][0]);
                    std::complex<float> aXY2 = std::conj((*aterms)[n][antenna2][0][i][1]);
                    std::complex<float> aYX2 = std::conj((*aterms)[n][antenna2][0][i][2]);
                    std::complex<float> aYY2 = std::conj((*aterms)[n][antenna2][0][i][3]);

                    std::complex<float> kp[16] = {};
                    kp[0 +  0] = aXX2*aXX1;
                    kp[0 +  4] = aXX2*aXY1;
                    kp[0 +  8] = aXY2*aXX1;
                    kp[0 + 12] = aXY2*aXY1;

                    kp[1 +  0] = aXX2*aYX1;
                    kp[1 +  4] = aXX2*aYY1;
                    kp[1 +  8] = aXY2*aYX1;
                    kp[1 + 12] = aXY2*aYY1;

                    kp[2 +  0] = aYX2*aXX1;
                    kp[2 +  4] = aYX2*aXY1;
                    kp[2 +  8] = aYY2*aXX1;
                    kp[2 + 12] = aYY2*aXY1;

                    kp[3 +  0] = aYX2*aYX1;
                    kp[3 +  4] = aYX2*aYY1;
                    kp[3 +  8] = aYY2*aYX1;
                    kp[3 + 12] = aYY2*aYY1;

                    for (int ii = 0; ii < nr_correlations; ii++) {
                        for (int jj = 0; jj < nr_correlations; jj++) {
                            // Load weights for current baseline, aterm
                            float *weights = &sum_of_weights[bl][n][0];

                            // Compute real and imaginary part of update separately
                            float update_real = 0;
                            float update_imag = 0;
                            for (int p = 0; p < nr_correlations; p++) {
                                float kp1_real =  kp[4*ii+p].real();
                                float kp1_imag = -kp[4*ii+p].imag();
                                float kp2_real =  kp[4*jj+p].real();
                                float kp2_imag =  kp[4*jj+p].imag();
                                update_real += weights[p] * (kp1_real * kp2_real - kp1_imag * kp2_imag);
                                update_imag += weights[p] * (kp1_real * kp2_imag + kp1_imag * kp2_real);
                            }

                            // Add kronecker product to sum
                            sum[ii][jj] += std::complex<float>(update_real, update_imag);
                        }
                    }
                } // end for baselines
            } // end for aterms

            // Set average beam from sum of kronecker products
            for(size_t ii = 0; ii < 4; ii++) {
                for(size_t jj = 0; jj < 4; jj++) {
                    (*average_beam)[i][ii][jj] += sum[ii][jj];
                }
            }
        } // end for pixels

        m_bufferset->m_avg_beam_watch->Pause();

    } // end compute_avg_beam

    void GridderBufferImpl::flush_thread_worker()
    {

        if (m_do_compute_avg_beam)
        {
            compute_avg_beam();
        }

        if (!m_do_gridding) return;

        Array4D<std::complex<float>> *aterm_correction;
        if (m_apply_aterm)
        {
            aterm_correction = &m_avg_aterm_correction;
        }
        else
        {
            m_aterm_offsets_array = Array1D<unsigned int>(m_default_aterm_offsets.data(), m_default_aterm_offsets.size());
            m_aterms_array = Array4D<Matrix2x2<complex<float>>>(m_default_aterms.data(), m_default_aterm_offsets.size()-1, m_nrStations, m_subgridsize, m_subgridsize);
            aterm_correction = &m_default_aterm_correction;
        }

        // Set Plan options
        Plan::Options options;
        options.w_step      = m_wStepInLambda;
        options.nr_w_layers = m_nr_w_layers;
        options.plan_strict = false;

        // Create plan
        m_bufferset->m_plan_watch->Start();
        Plan plan(
            m_kernel_size,
            m_subgridsize,
            m_gridHeight,
            m_cellHeight,
            m_frequencies,
            m_bufferUVW2,
            m_bufferStationPairs2,
            m_aterm_offsets_array,
            options);
        m_bufferset->m_plan_watch->Pause();

        // Run gridding
        m_bufferset->m_gridding_watch->Start();
        m_proxy->set_grid(m_grid);
        m_proxy->gridding(
            plan,
            m_wStepInLambda,
            m_shift,
            m_cellHeight,
            m_kernel_size,
            m_subgridsize,
            m_frequencies,
            m_bufferVisibilities2,
            m_bufferUVW2,
            m_bufferStationPairs2,
            *m_grid,
            m_aterms_array,
            m_aterm_offsets_array,
            m_spheroidal);
        m_proxy->get_grid();
        m_bufferset->m_gridding_watch->Pause();
    }

    // Must be called whenever the buffer is full or no more data added
    void GridderBufferImpl::flush()
    {
        // Return if no input in buffer
        if (m_timeindices.size() == 0) return;


        // if there is still a flushthread running, wait for it to finish
        if (m_flush_thread.joinable()) m_flush_thread.join();

        std::swap(m_bufferUVW, m_bufferUVW2);
        std::swap(m_bufferStationPairs, m_bufferStationPairs2);
        std::swap(m_bufferVisibilities, m_bufferVisibilities2);
        std::swap(m_buffer_weights, m_buffer_weights2);
        std::swap(m_aterm_offsets, m_aterm_offsets2);
        m_aterm_offsets_array = Array1D<unsigned int>(m_aterm_offsets2.data(), m_aterm_offsets2.size());

        std::swap(m_aterms, m_aterms2);
        assert(m_aterms2.size()==(m_aterm_offsets_array.get_x_dim()-1)*m_nrStations*m_subgridsize*m_subgridsize);
        m_aterms_array = Array4D<Matrix2x2<complex<float>>>(m_aterms2.data(), m_aterm_offsets_array.get_x_dim()-1, m_nrStations, m_subgridsize, m_subgridsize);

        m_flush_thread = std::thread(&GridderBufferImpl::flush_thread_worker, this);

        // Prepare next batch
        m_timeStartThisBatch += m_bufferTimesteps;
        m_timeStartNextBatch += m_bufferTimesteps;
        m_timeindices.clear();
        reset_aterm();
        set_uvw_to_infinity();
    }

    // Reset the a-term for a new buffer; copy the last a-term from the
    // previous buffer;
    void GridderBufferImpl::reset_aterm()
    {
      if (m_aterm_offsets.size()!=2) {
        m_aterm_offsets = std::vector<unsigned int>(2, 0);
      }
      m_aterm_offsets[0] = 0;
      m_aterm_offsets[1] = m_bufferTimesteps;

      size_t n_old_aterms = m_aterm_offsets2.size()-1; // Nr aterms in previous chunk

      size_t atermBlockSize = m_nrStations*m_subgridsize*m_subgridsize;
      m_aterms.resize(atermBlockSize);
      std::copy(m_aterms2.data()+(n_old_aterms-1)*atermBlockSize,
                m_aterms2.data()+(n_old_aterms)*atermBlockSize,
                (Matrix2x2<complex<float>>*) m_aterms.data());
    }


    void GridderBufferImpl::finished()
    {
        flush();
        // if there is still a flushthread running, wait for it to finish
        if (m_flush_thread.joinable()) {
            m_flush_thread.join();
        }
    }


    void GridderBufferImpl::malloc_buffers()
    {
        BufferImpl::malloc_buffers();

        m_bufferUVW2 = m_proxy->allocate_array2d<UVW<float>>(m_nr_baselines, m_bufferTimesteps);
        m_bufferVisibilities2 = m_proxy->allocate_array3d<Visibility<std::complex<float>>>(m_nr_baselines, m_bufferTimesteps, m_nr_channels);
        m_bufferStationPairs2 = m_proxy->allocate_array1d<std::pair<unsigned int,unsigned int>>(m_nr_baselines);
        m_buffer_weights = m_proxy->allocate_array4d<float>(m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4);
        m_buffer_weights2 = m_proxy->allocate_array4d<float>(m_nr_baselines, m_bufferTimesteps, m_nr_channels, 4);
    }

} // namespace api
} // namespace idg
