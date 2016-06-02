#ifndef IDG_PLAN_H_
#define IDG_PLAN_H_

#include <vector>
#include <limits>
#include <stdexcept> // runtime_error
#include <cmath>
#include <numeric>
#include <iterator>

#include "Types.h"
#include "Parameters.h"


namespace idg {

        class Plan {

        public:
            Plan(
                 const Parameters& params,
                 const float *uvw,
                 const float *wavenumbers,
                 const int *baselines,
                 const int *aterm_offsets,
                 const int kernel_size,
                 const int max_nr_timesteps = std::numeric_limits<int>::max());

            void init_metadata(
                const float *_uvw,
                const float *wavenumbers,
                const int *_baselines,
                const int *aterm_offsets,
                const int kernel_size,
                const int max_nr_timesteps = std::numeric_limits<int>::max());

            // returns index of first index of baseline
            int get_subgrid_offset(int baseline) const;

            // total number of subgrids
            int get_nr_subgrids() const;

            // number of subgrids one baseline
            int get_nr_subgrids(int baseline) const;

            // number of subgrids for baselines b1 to b1+n-1
            int get_nr_subgrids(int baseline, int n) const;

            // max number of subgrids for n baselines between bl1 and bl2+n
            int get_max_nr_subgrids(int bl1, int bl2, int n);

            void print_subgrid_offset() const;

            // total number of timesteps
            int get_nr_timesteps() const;

            // number of timesteps one baseline
            int get_nr_timesteps(int baseline) const;

            // number of timesteps for baselines b1 to b1+n-1
            int get_nr_timesteps(int baseline, int n) const;

            const Metadata* get_metadata_ptr(int baseline = 0) const;
            std::vector<Metadata> copy_metadata() const;

        private:
            const Parameters mParams;
            std::vector<Metadata> metadata;
            std::vector<int> subgrid_offset;
            std::vector<int> timesteps_per_baseline;

        }; // class Plan

} // namespace idg

#endif
