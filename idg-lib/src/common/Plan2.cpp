#include <cassert> // assert

#include "Plan2.h"

using namespace std;

namespace idg {

    Plan2::Plan2(
        const int kernel_size,
        const int subgrid_size,
        const int grid_size,
        const float cell_size,
        const Array1D<float>& frequencies,
        const Array2D<UVWCoordinate<float>>& uvw,
        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
        const Array1D<unsigned int>& aterms_offsets,
        const int max_nr_timesteps_per_subgrid)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        initialize(
            kernel_size, subgrid_size, grid_size, cell_size,
            frequencies, uvw, baselines, aterms_offsets,
            max_nr_timesteps_per_subgrid);
    }

    class Subgrid {
        public:
            Subgrid(
                const int kernel_size,
                const int subgrid_size,
                const int grid_size) :
                kernel_size(kernel_size),
                subgrid_size(subgrid_size),
                grid_size(grid_size)
            {
                reset();
            }

            void reset() {
                u_min =  std::numeric_limits<float>::infinity();
                u_max = -std::numeric_limits<float>::infinity();
                v_min =  std::numeric_limits<float>::infinity();
                v_max = -std::numeric_limits<float>::infinity();
                uv_width = 0;
            }

            bool add_visibility(float u_pixels, float v_pixels) {
                // Initialize candidate uv limits
                float u_min_ = fmin(u_min, u_pixels);
                float u_max_ = fmax(u_max, u_pixels);
                float v_min_ = fmin(v_min, v_pixels);
                float v_max_ = fmax(v_max, v_pixels);

                // Compute candidate uv width
                int u_width_  = u_max_ - u_min_ + 1;
                int v_width_  = v_max_ - v_min_ + 1;
                int uv_width_ = fmax(u_width_, v_width_);

                // Return false if the visibility does not fit
                if ((uv_width + kernel_size) >= subgrid_size) {
                    return false;
                } else {
                    u_min = u_min_;
                    u_max = u_max_;
                    v_min = v_min_;
                    v_max = v_max_;
                    uv_width = uv_width_;
                    return true;
                }
            }

            Coordinate get_coordinate() {
                // Compute middle point in pixels
                int u_pixels = roundf((u_max + u_min) / 2);
                int v_pixels = roundf((v_max + v_min) / 2);

                // Shift center from middle of grid to top left
                u_pixels += (grid_size/2);
                v_pixels += (grid_size/2);

                // Shift from middle of subgrid to top left
                u_pixels -= (subgrid_size/2);
                v_pixels -= (subgrid_size/2);

                return {u_pixels, v_pixels};
            }

            const int kernel_size;
            const int subgrid_size;
            const int grid_size;
            float u_min;
            float u_max;
            float v_min;
            float v_max;
            float uv_width;
    }; // end class Subgrid


    inline float meters_to_pixels(float meters, float imagesize, float frequency) {
        const double speed_of_light = 299792458.0;
        return meters * imagesize * (frequency / speed_of_light);
    }


    void Plan2::initialize(
        const int kernel_size,
        const int subgrid_size,
        const int grid_size,
        const float cell_size,
        const Array1D<float>& frequencies,
        const Array2D<UVWCoordinate<float>>& uvw,
        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
        const Array1D<unsigned int>& aterms_offsets,
        const int max_nr_timesteps_per_subgrid)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        // Check arguments
        assert(baselines.get_x_dim() == uvw.get_y_dim());

        // Initialize arguments
        nr_baselines      = uvw.get_y_dim();
        auto nr_timesteps = uvw.get_x_dim();
        auto nr_timeslots = aterms_offsets.get_x_dim() - 1;
        auto nr_channels  = frequencies.get_x_dim();
        auto image_size   = cell_size * grid_size; // TODO: remove

        // Allocate metadata
        metadata.reserve(nr_baselines * nr_timesteps / nr_timeslots);

        // Temporary metadata vector for individual baselines
        std::vector<Metadata> metadata_[nr_baselines];
        for (int i = 0; i < nr_baselines; i++) {
            metadata_[i].reserve(nr_timesteps / nr_timeslots);
        }

        // Iterate all baselines
        #pragma omp parallel for
        for (int bl = 0; bl < nr_baselines; bl++) {
            // Get thread id
            const int thread_id = omp_get_thread_num();

            // Get baseline
            Baseline baseline = (Baseline) {baselines(bl).first, baselines(bl).second};

            // Iterate all time slots
            for (int timeslot = 0; timeslot < nr_timeslots; timeslot++) {
                // Get aterm offset
                const int current_aterms_offset = aterms_offsets(timeslot);
                const int next_aterms_offset    = aterms_offsets(timeslot+1);

                // The aterm index is equal to the timeslot
                const int aterm_index = timeslot;

                // Determine number of timesteps in current aterm
                const int nr_timesteps_per_aterm = next_aterms_offset - current_aterms_offset;

                // Compute uv coordinates in pixels
                struct DataPoint {
                    int timestep;
                    int channel;
                    float u_pixels;
                    float v_pixels;
                };

                DataPoint datapoints[nr_timesteps_per_aterm][nr_channels];

                for (int t = 0; t < nr_timesteps_per_aterm; t++) {
                    for (int c = 0; c < nr_channels; c++) {
                        // U,V in meters
                        float u_meters = uvw(bl, t).u;
                        float v_meters = uvw(bl, t).v;

                        float u_pixels = meters_to_pixels(u_meters, image_size, frequencies(c));
                        float v_pixels = meters_to_pixels(v_meters, image_size, frequencies(c));

                        datapoints[t][c] = {t, c, u_pixels, v_pixels};
                    }
                } // end for time

                // Initialize subgrid
                Subgrid subgrid(kernel_size, subgrid_size, grid_size);

                int time_offset = 0;
                while (time_offset < nr_timesteps_per_aterm) {
                    // Load first visibility
                    DataPoint first_datapoint = datapoints[time_offset][0];
                    const int first_timestep = first_datapoint.timestep;

                    // Create subgrid
                    subgrid.reset();
                    int nr_timesteps_subgrid = 0;

                    // Iterate all datapoints
                    int time_limit = abs(time_offset + max_nr_timesteps_per_subgrid);
                    int time_max = time_limit > 0 ? min(time_limit, nr_timesteps_per_aterm) : nr_timesteps_per_aterm;
                    for (; time_offset < time_max; time_offset++) {
                        DataPoint visibility = datapoints[time_offset][0];
                        const float u_pixels = visibility.u_pixels;
                        const float v_pixels = visibility.v_pixels;

                        // Try to add visibility to subgrid
                        if (subgrid.add_visibility(u_pixels, v_pixels)) {
                            nr_timesteps_subgrid++;
                        } else {
                            break;
                        }
                    } // end for time

                    // Check whether current subgrid is in grid range
                    Coordinate coordinate = subgrid.get_coordinate();
                    bool uv_max_pixels = max(coordinate.x, coordinate.y);
                    bool uv_in_range = uv_max_pixels > 0 && uv_max_pixels < (grid_size - subgrid_size);

                    // Add subgrid to metadata
                    if (uv_in_range && nr_timesteps_subgrid > 0) {
                        Metadata m = {
                            bl * (int) nr_timesteps,                // baseline offset, TODO: store bl index
                            current_aterms_offset + first_timestep, // time offset, TODO: store time index
                            nr_timesteps_subgrid,                   // nr of timesteps
                            aterm_index,                            // aterm index
                            baseline,                               // baselines
                            coordinate                              // coordinate
                        };
                        //cout << "new metadata: " << endl;
                        //cout << m;
                        metadata_[bl].push_back(m);
                    }
                } // end while
            } // end for timeslot
        } // end for bl

        // Combine data structures
        for (int bl = 0; bl < nr_baselines; bl++) {
            // The subgrid offset is the number of subgrids for all prior baselines
            subgrid_offset.push_back(metadata.size());

            for (int i = 0; i < metadata_[bl].size(); i++) {
                metadata.push_back(metadata_[bl][i]);
            }

            // The number of timesteps per baseline is always nr_timesteps
            timesteps_per_baseline.push_back(nr_timesteps);
        } // end for bl

        // Set sentinel
        subgrid_offset.push_back(metadata.size());

    } // end initialize


    int Plan2::get_nr_subgrids() const {
        return metadata.size();
    }


    int Plan2::get_nr_subgrids(int bl) const {
        return get_nr_subgrids(bl,1);
    }


    int Plan2::get_nr_subgrids(int bl, int n) const {
        if (n < 1) {
            throw invalid_argument("n should be at least one.");
        }
        return get_subgrid_offset(bl+n) - get_subgrid_offset(bl);
    }

    int Plan2::get_subgrid_offset(int bl) const {
        return subgrid_offset[bl];
    }

    int Plan2::get_max_nr_subgrids(int bl1, int bl2, int n) {
        int nr_baselines = bl1 + n > bl2 ? bl2 - bl1 : n;
        int max_nr_subgrids = get_nr_subgrids(bl1, nr_baselines);
        for (int bl = bl1 + n; bl <  bl2; bl += n) {
            nr_baselines = bl + n > bl2 ? bl2 - bl : n;
            int nr_subgrids = get_nr_subgrids(bl, nr_baselines);
            if (nr_subgrids > max_nr_subgrids) {
                max_nr_subgrids = nr_subgrids;
            }
        }
        return max_nr_subgrids;
    }

    int Plan2::get_max_nr_subgrids() {
        return get_max_nr_subgrids(0, nr_baselines, 1);
    }

    int Plan2::get_nr_timesteps() const {
        return accumulate(timesteps_per_baseline.begin(), timesteps_per_baseline.end(), 0);
    }

    int Plan2::get_nr_timesteps(int baseline) const {
        return timesteps_per_baseline[baseline];
    }

    int Plan2::get_nr_timesteps(int baseline, int n) const {
        auto begin = next(timesteps_per_baseline.begin(), baseline);
        auto end   = next(begin, n);
        return accumulate(begin, end, 0);
    }

    const Metadata* Plan2::get_metadata_ptr(int bl) const {
        auto offset = get_subgrid_offset(bl);
        return &(metadata[offset]);
    }

} // namespace idg
