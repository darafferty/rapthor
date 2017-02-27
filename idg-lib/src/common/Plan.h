#ifndef IDG_PLAN_H_
#define IDG_PLAN_H_

#include <vector>
#include <limits>
#include <stdexcept> // runtime_error
#include <cmath>
#include <numeric>
#include <iterator>
#include <omp.h>

#include "Types.h"


namespace idg {

        class Plan {

        public:
            // Constructors
            Plan() {};

            Plan(
                const int kernel_size,
                const int subgrid_size,
                const int grid_size,
                const float cell_size,
                const Array1D<float>& frequencies,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array1D<unsigned int>& aterms_offsets,
                const float w_step = 0.0,
                const int nr_w_layers = 1,
                const int max_nr_timesteps_per_subgrid = std::numeric_limits<int>::max());

            // Destructor
            virtual ~Plan() = default;

            void initialize(
                const int kernel_size,
                const int subgrid_size,
                const int grid_size,
                const float cell_size,
                const Array1D<float>& frequencies,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array1D<unsigned int>& aterms_offsets,
                const float w_step = 0.0,
                const int nr_w_layers = 1,
                const int max_nr_timesteps_per_subgrid = std::numeric_limits<int>::max());

            // total number of subgrids
            int get_nr_subgrids() const;

            // number of subgrids one baseline
            int get_nr_subgrids(int baseline) const;

            // number of subgrids for baselines b1 to b1+n-1
            int get_nr_subgrids(int baseline, int n) const;

            // returns index of first index of baseline
            int get_subgrid_offset(int baseline) const;

            // max number of subgrids for n baselines between bl1 and bl2+n
            int get_max_nr_subgrids(int bl1, int bl2, int n) const;

            // max number of subgrids for 1 baseline
            int get_max_nr_subgrids() const;

            // total number of timesteps
            int get_nr_timesteps() const;

            // number of timesteps one baseline
            int get_nr_timesteps(int baseline) const;

            // number of timesteps for baselines b1 to b1+n-1
            int get_nr_timesteps(int baseline, int n) const;

            // total number of visibilities
            int get_nr_visibilities() const;

            // number of visibilities one baseline
            int get_nr_visibilities(int baseline) const;

            // number of visibilities for baselines b1 to b1+n-1
            int get_nr_visibilities(int baseline, int n) const;

            // number of baselines
            int get_nr_baselines() const {
                return total_nr_timesteps_per_baseline.size();
            }

            const Metadata* get_metadata_ptr(int baseline = 0) const;

            void copy_metadata(void *ptr) const;
            
            bool get_needs_w_stacking() const {return needs_w_stacking;}

        private:
            std::vector<Metadata> metadata;
            std::vector<int> subgrid_offset;
            std::vector<int> total_nr_timesteps_per_baseline;
            std::vector<int> total_nr_visibilities_per_baseline;
            float w_step;
            bool needs_w_stacking;

        }; // class Plan

} // namespace idg

#endif
