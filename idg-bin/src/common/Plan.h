#ifndef IDG_PLAN_H_
#define IDG_PLAN_H_

#include <vector>
#include <limits>
#include <stdexcept> // runtime_error
#include <cmath>

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
                 const int kernel_size);

            void init_metadata(
                const float *_uvw,
                const float *wavenumbers,
                const int *_baselines,
                const int *aterm_offsets,
                const int kernel_size);

            // returns index of first index of baseline
            int get_subgrid_offset(int baseline) const;

            // total number of subgrids
            int get_nr_subgrids() const;

            // number of subgrids one baseline
            int get_nr_subgrids(int baseline) const;

            // number of subgrids for baselines b1 to b1+n-1
            int get_nr_subgrids(int baseline, int n) const;

            // split subgrid into multiple subgrids when
            // max_nr_timesteps is exceeded
            void split_subgrids(int max_nr_timesteps);

            void print_subgrid_offset() const;

            const Metadata* get_metadata_ptr(int baseline = 0) const;
            std::vector<Metadata> copy_metadata() const;

        private:
            const Parameters mParams;
            std::vector<Metadata> metadata;
            std::vector<int> subgrid_offset;

        }; // class Plan

} // namespace idg

#endif
