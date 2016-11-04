#include "Plan.h"

using namespace std;

namespace idg {

    Plan::Plan(const Parameters& parameters,
               const float *uvw,
               const float *wavenumbers,
               const int *baselines,
               const int *aterm_offsets,
               const int kernel_size,
               const int max_nr_timesteps)
        : mParams(parameters)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        init_metadata(uvw, wavenumbers, baselines,
                      aterm_offsets, kernel_size, max_nr_timesteps);
    }

    int Plan::get_subgrid_offset(int bl) const {
        return subgrid_offset[bl];
    }


    int Plan::get_nr_subgrids() const {
        return metadata.size();
    }


    int Plan::get_nr_subgrids(int bl) const {
        return get_nr_subgrids(bl,1);
    }


    int Plan::get_nr_subgrids(int bl, int n) const {
        if (n < 1) {
            throw invalid_argument("n should be at least one.");
        }
        return get_subgrid_offset(bl+n) - get_subgrid_offset(bl);
    }


    int Plan::get_max_nr_subgrids(int bl1, int bl2, int n) {
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

    int Plan::get_max_nr_subgrids() {
        return get_max_nr_subgrids(0, nr_baselines, 1);
    }

    int Plan::get_nr_timesteps() const {
        return accumulate(timesteps_per_baseline.begin(), timesteps_per_baseline.end(), 0);
    }


    int Plan::get_nr_timesteps(int baseline) const {
        return timesteps_per_baseline[baseline];
    }


    int Plan::get_nr_timesteps(int baseline, int n) const {
        auto begin = next(timesteps_per_baseline.begin(), baseline);
        auto end   = next(begin, n);
        return accumulate(begin, end, 0);
    }


    const Metadata* Plan::get_metadata_ptr(int bl) const {
        auto offset = get_subgrid_offset(bl);
        return &(metadata[offset]);
    }


    vector<Metadata> Plan::copy_metadata() const {
        return metadata;
    }


    void Plan::print_subgrid_offset() const {
        int k = 0;
        for (auto& e: subgrid_offset) {
            cout << "subgrid_offset[" << k++ << "]:" << e << endl;
        }
    }


    static int find_aterm_index(const int time,
                                const int *aterm_offsets)
    {
        int k = 0;
        while (time >= aterm_offsets[k+1]) k++;
        return k;
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
                //if (uv_width_ >= kernel_size) {
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
    };


    float uv_meters_to_pixels(float meters, float imagesize, float wavenumber) {
        return meters * imagesize * wavenumber / (2 * M_PI);
    }

    void Plan::init_metadata(
        const float *_uvw,
        const float *wavenumbers,
        const int *_baselines,
        const int *aterm_offsets,
        const int kernel_size,
        const int max_nr_timesteps)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        // Load parameters
        nr_baselines      = mParams.get_nr_baselines();
        auto nr_time      = mParams.get_nr_time();
        auto nr_timeslots = mParams.get_nr_timeslots();
        auto nr_channels  = mParams.get_nr_channels();
        auto grid_size    = mParams.get_grid_size();
        auto subgrid_size = mParams.get_subgrid_size();
        auto imagesize    = mParams.get_imagesize();

        // Allocate metadata
        metadata.reserve(nr_baselines * nr_time / nr_timeslots);

        // Temporary metadata vector for individual baselines
        std::vector<Metadata> metadata_[nr_baselines];
        for (int i = 0; i < nr_baselines; i++) {
            metadata_[i].reserve(nr_time / nr_timeslots);
        }

        // Iterate all baselines
        #pragma omp parallel for
        for (int bl = 0; bl < nr_baselines; bl++) {
            // Get thread id
            const int thread_id = omp_get_thread_num();

            // Get baseline
            Baseline baseline = ((Baseline *) (_baselines))[bl];

            // Compute baseline offset
            const int baseline_offset = bl * nr_time;

            // Iterate all time slots
            for (int timeslot = 0; timeslot < nr_timeslots; timeslot++) {
                // Get aterm offset
                const int current_aterm_offset = aterm_offsets[timeslot];
                const int next_aterm_offset    = aterm_offsets[timeslot+1];

                // The aterm index is equal to the timeslot
                const int aterm_index = timeslot;

                // Determine number of timesteps in current aterm
                const int nr_timesteps = next_aterm_offset - current_aterm_offset;

                // Get pointer to current uvw coordinates
                UVW *uvw =  ((UVW *) _uvw) + (baseline_offset + current_aterm_offset);

                // Compute uv coordinates in pixels
                struct Visibility {
                    int timestep;
                    int channel;
                    float u_pixels;
                    float v_pixels;
                };

                Visibility visibilities[nr_timesteps][nr_channels];

                for (int t = 0; t < nr_timesteps; t++) {
                    for (int c = 0; c < nr_channels; c++) {
                        // U,V in meters
                        float u_meters = uvw[t].u;
                        float v_meters = uvw[t].v;

                        float u_pixels = uv_meters_to_pixels(u_meters, imagesize, wavenumbers[c]);
                        float v_pixels = uv_meters_to_pixels(v_meters, imagesize, wavenumbers[c]);

                        visibilities[t][c] = {t, c, u_pixels, v_pixels};
                    }
                } // end for time

                // Initialize subgrid
                Subgrid subgrid(kernel_size, subgrid_size, grid_size);

                int time_offset = 0;
                while (time_offset < nr_timesteps) {
                    // Load first visibility
                    Visibility first_visibility = visibilities[time_offset][0];
                    const int first_timestep = first_visibility.timestep;

                    // Create subgrid
                    subgrid.reset();
                    int current_nr_timesteps = 0;

                    // Iterate all visibilities
                    int time_limit = abs(time_offset + max_nr_timesteps);
                    int time_max = time_limit > 0 ? min(time_limit, nr_timesteps) : nr_timesteps;
                    for (; time_offset < time_max; time_offset++) {
                        Visibility visibility = visibilities[time_offset][0];
                        const float u_pixels = visibility.u_pixels;
                        const float v_pixels = visibility.v_pixels;

                        // Try to add visibility to subgrid
                        if (subgrid.add_visibility(u_pixels, v_pixels)) {
                            current_nr_timesteps++;
                        } else {
                            break;
                        }
                    } // end for time

                    // Check whether current subgrid is in grid range
                    Coordinate coordinate = subgrid.get_coordinate();
                    bool uv_max_pixels = max(coordinate.x, coordinate.y);
                    bool uv_in_range = uv_max_pixels > 0 && uv_max_pixels < (grid_size - subgrid_size);

                    // Add subgrid to metadata
                    if (uv_in_range && current_nr_timesteps > 0) {
                        Metadata m = {
                            baseline_offset,                       // baseline offset
                            current_aterm_offset + first_timestep, // time offset
                            current_nr_timesteps,                  // nr of timesteps
                            aterm_index,                           // aterm index
                            baseline,                              // baselines
                            subgrid.get_coordinate()               // coordinate
                        };
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

            // The number of timesteps per baseline is always nr_time
            timesteps_per_baseline.push_back(nr_time);
        } // end for bl

        // Set sentinel
        subgrid_offset.push_back(metadata.size());
    } // end init_metadata
} // namespace idg
