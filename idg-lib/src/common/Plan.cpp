#include <iostream>
#include <cassert> // assert
#include <algorithm> // max_element
#include <memory.h> // memcpy

#include "Plan.h"

using namespace std;

namespace idg {

    Plan::Plan(
        const int kernel_size,
        const int subgrid_size,
        const int grid_size,
        const float cell_size,
        const Array1D<float>& frequencies,
        const Array2D<UVWCoordinate<float>>& uvw,
        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
        const Array1D<unsigned int>& aterms_offsets,
        Options options)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        initialize(
            kernel_size, subgrid_size, grid_size, cell_size,
            frequencies, uvw, baselines, aterms_offsets, options);
    }

    class Subgrid {
        public:
            Subgrid(
                const int   kernel_size,
                const int   subgrid_size,
                const int   grid_size,
                const float w_step,
                const unsigned nr_w_layers) :
                kernel_size(kernel_size),
                subgrid_size(subgrid_size),
                grid_size(grid_size),
                w_step(w_step),
                nr_w_layers(nr_w_layers)
            {
                reset();
            }

            void reset() {
                u_min =  std::numeric_limits<float>::infinity();
                u_max = -std::numeric_limits<float>::infinity();
                v_min =  std::numeric_limits<float>::infinity();
                v_max = -std::numeric_limits<float>::infinity();
                uv_width = 0;
                w_index  = 0;
                finished = false;
            }

            bool add_visibility(float u_pixels, float v_pixels, float w_lambda) {
                // Return false when finish() has been called
                if (finished) {
                    return false;
                }

                // Return false for invalid visibilities
                if (std::isinf(u_pixels) || std::isinf(v_pixels)) {
                    return false;
                }

                int w_index_ = 0;
                if (w_step) w_index_ = int(std::floor(w_lambda/w_step));

                // if this is not the first sample, it should map to the
                // same w_index as the others, if not, return false
                if (std::isfinite(u_min) && (w_index_ != w_index)) {
                    return false;
                }

                // Initialize candidate uv limits
                float u_min_ = fmin(u_min, u_pixels);
                float u_max_ = fmax(u_max, u_pixels);
                float v_min_ = fmin(v_min, v_pixels);
                float v_max_ = fmax(v_max, v_pixels);

                // Compute candidate uv width
                float u_width_  = u_max_ - u_min_;
                float v_width_  = v_max_ - v_min_;
                float uv_width_ = fmax(u_width_, v_width_);

                // Return false if the visibility does not fit
                if ((uv_width_ + kernel_size) >= subgrid_size) {
                    return false;
                } else {
                    u_min = u_min_;
                    u_max = u_max_;
                    v_min = v_min_;
                    v_max = v_max_;
                    uv_width = uv_width_;
                    w_index = w_index_;
                    return true;
                }
            }

            bool in_range() {
                Coordinate coordinate = get_coordinate();

                // Compute extremes of subgrid position in grid
                int uv_max_pixels = max(coordinate.x, coordinate.y);
                int uv_min_pixels = min(coordinate.x, coordinate.y);

                // Index in w-stack
                int w_index = coordinate.z;

                // Return whether the subgrid fits in grid and w-stack
                return  uv_min_pixels >= 1 &&
                        uv_max_pixels <= (grid_size - subgrid_size) &&
                        w_index       >= -((int) nr_w_layers) &&
                        w_index       <   ((int) nr_w_layers);
            }

            void compute_coordinate() {
                // Compute middle point in pixels
                int u_pixels = roundf((u_max + u_min) / 2);
                int v_pixels = roundf((v_max + v_min) / 2);

                // Shift center from middle of grid to top left
                u_pixels += (grid_size/2);
                v_pixels += (grid_size/2);

                // Shift from middle of subgrid to top left
                u_pixels -= (subgrid_size/2);
                v_pixels -= (subgrid_size/2);

                coordinate = {u_pixels, v_pixels, w_index};
            }

            void finish() {
                finished = true;
                compute_coordinate();
            }

            Coordinate get_coordinate() {
                if (!finished) {
                    throw std::runtime_error("finish the subgrid before retrieving its coordinate");
                }
                return coordinate;
            }

            const int kernel_size;
            const int subgrid_size;
            const int grid_size;
            float u_min;
            float u_max;
            float v_min;
            float v_max;
            float uv_width;
            int   w_index;
            float w_step;
            int nr_w_layers;
            bool finished;
            Coordinate coordinate;
    }; // end class Subgrid


    inline float meters_to_pixels(float meters, float imagesize, float frequency) {
        const double speed_of_light = 299792458.0;
        return meters * imagesize * (frequency / speed_of_light);
    }

    inline float meters_to_lambda(float meters, float frequency) {
        const double speed_of_light = 299792458.0;
        return meters * (frequency / speed_of_light);
    }

    void Plan::initialize(
        const int kernel_size,
        const int subgrid_size,
        const int grid_size,
        const float cell_size,
        const Array1D<float>& frequencies,
        const Array2D<UVWCoordinate<float>>& uvw,
        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
        const Array1D<unsigned int>& aterms_offsets,
        const Options& options)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        std::clog << "kernel_size  : " << kernel_size << std::endl;
        std::clog << "subgrid_size : " << subgrid_size << std::endl;
        std::clog << "grid_size    : " << grid_size << std::endl;
        #endif

        // Get options
        float w_step = options.w_step;
        int nr_w_layers = options.nr_w_layers;
        int max_nr_timesteps_per_subgrid = options.max_nr_timesteps_per_subgrid;
        bool plan_strict = options.plan_strict;

        // Check arguments
        assert(baselines.get_x_dim() == uvw.get_y_dim());

        // Initialize arguments
        auto nr_baselines = uvw.get_y_dim();
        auto nr_timesteps = uvw.get_x_dim();
        auto nr_timeslots = aterms_offsets.get_x_dim() - 1;
        auto nr_channels  = frequencies.get_x_dim();
        auto image_size   = cell_size * grid_size; // TODO: remove

        // Spectral-line imaging
        bool simulate_spectral_line = options.simulate_spectral_line;
        auto nr_channels_ = nr_channels;
        if (simulate_spectral_line) {
            nr_channels = 1;
        }

        // Allocate metadata
        metadata.reserve(nr_baselines * nr_timesteps / nr_timeslots);

        // Temporary metadata vector for individual baselines
        std::vector<Metadata> metadata_[nr_baselines];
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            metadata_[bl].reserve(nr_timesteps / nr_timeslots);
        }

        // Iterate all baselines
        #pragma omp parallel for
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            // Get baseline
            Baseline baseline = (Baseline) {baselines(bl).first, baselines(bl).second};

            // Iterate all time slots
            for (unsigned timeslot = 0; timeslot < nr_timeslots; timeslot++) {
                // Get aterm offset
                const unsigned current_aterms_offset = aterms_offsets(timeslot);
                const unsigned next_aterms_offset    = aterms_offsets(timeslot+1);

                // The aterm index is equal to the timeslot
                const unsigned aterm_index = timeslot;

                // Determine number of timesteps in current aterm
                const unsigned nr_timesteps_per_aterm = next_aterms_offset - current_aterms_offset;

                // Compute uv coordinates in pixels
                struct DataPoint {
                    unsigned timestep;
                    unsigned channel;
                    float u_pixels;
                    float v_pixels;
                    float w_lambda;
                };

                std::vector<DataPoint> datapoints(nr_timesteps_per_aterm * nr_channels);

                for (unsigned t = 0; t < nr_timesteps_per_aterm; t++) {
                    for (unsigned c = 0; c < nr_channels; c++) {
                        // U,V in meters
                        float u_meters = uvw(bl, current_aterms_offset + t).u;
                        float v_meters = uvw(bl, current_aterms_offset + t).v;
                        float w_meters = uvw(bl, current_aterms_offset + t).w;

                        float u_pixels = meters_to_pixels(u_meters, image_size, frequencies(c));
                        float v_pixels = meters_to_pixels(v_meters, image_size, frequencies(c));

                        float w_lambda = meters_to_lambda(w_meters, frequencies(c));

                        datapoints[t*nr_channels + c] = {t, c, u_pixels, v_pixels, w_lambda};
                    }
                } // end for time

                // Initialize subgrid
                Subgrid subgrid(kernel_size, subgrid_size, grid_size, w_step, nr_w_layers);

                unsigned time_offset = 0;
                while (time_offset < nr_timesteps_per_aterm) {
                    // Load first visibility
                    DataPoint first_datapoint = datapoints[time_offset*nr_channels];
                    const int first_timestep = first_datapoint.timestep;

                    // Create subgrid
                    subgrid.reset();
                    int nr_timesteps_subgrid = 0;

                    // Iterate all datapoints
                    for (; time_offset < nr_timesteps_per_aterm; time_offset++) {
                        // Visibility for first channel
                        DataPoint visibility0 = datapoints[time_offset*nr_channels];
                        const float u_pixels0 = visibility0.u_pixels;
                        const float v_pixels0 = visibility0.v_pixels;
                        const float w_lambda0 = visibility0.w_lambda;

                        DataPoint visibility1 = datapoints[time_offset*nr_channels + nr_channels - 1];
                        const float u_pixels1 = visibility1.u_pixels;
                        const float v_pixels1 = visibility1.v_pixels;

                        // Try to add visibilities to subgrid
                        if (subgrid.add_visibility(u_pixels0, v_pixels0, w_lambda0) &&
                            // HACK also pass w_lambda0 below
                            subgrid.add_visibility(u_pixels1, v_pixels1, w_lambda0)) {
                            nr_timesteps_subgrid++;
                            if (nr_timesteps_subgrid == max_nr_timesteps_per_subgrid) break;
                        } else {
                            break;
                        }
                    } // end for time

                    // Handle empty subgrid
                    if (nr_timesteps_subgrid == 0) {
                        DataPoint visibility = datapoints[time_offset*nr_channels];
                        const float u_pixels = visibility.u_pixels;
                        const float v_pixels = visibility.v_pixels;

                        if (std::isfinite(u_pixels) && std::isfinite(v_pixels) && plan_strict) {
                            // Coordinates are valid, but did not (all) fit onto subgrid
                            #pragma omp critical
                            throw std::runtime_error("could not place (all) visibilities on subgrid (too many channnels, kernel size too large)");
                        } else {
                            // Advance to next timeslot when visibilities for current timeslot had infinite coordinates
                            time_offset++;
                            continue;
                        }
                    }

                    // Finish subgrid
                    subgrid.finish();

                    // Add subgrid to metadata
                    if (subgrid.in_range()) {
                        Metadata m = {
                            (int) (bl * nr_timesteps),                      // baseline offset, TODO: store bl index
                            (int) (current_aterms_offset + first_timestep), // time offset, TODO: store time index
                            nr_timesteps_subgrid,                   // nr of timesteps
                            (int) aterm_index,                      // aterm index
                            baseline,                               // baselines
                            subgrid.get_coordinate()                // coordinate
                        };
                        metadata_[bl].push_back(m);

                        // Add additional subgrids for subsequent frequencies
                        if (simulate_spectral_line) {
                            for (unsigned c = 1; c < nr_channels_; c++) {
                                // Compute shifted subgrid for current frequency
                                float shift = frequencies(c) / frequencies(0);
                                Metadata m = metadata_[bl].back();
                                m.coordinate.x *= shift;
                                m.coordinate.y *= shift;
                                metadata_[bl].push_back(m);
                            }
                        }
                    }
                    else if (plan_strict)
                    {
                        #pragma omp critical
                        throw std::runtime_error("subgrid falls not within grid");
                    }
                } // end while
            } // end for timeslot
        } // end for bl

        // Combine data structures
        for (unsigned bl = 0; bl < nr_baselines; bl++) {
            // The subgrid offset is the number of subgrids for all prior baselines
            subgrid_offset.push_back(metadata.size());

            // Count total number of timesteps for baseline
            int total_nr_timesteps = 0;

            for (unsigned i = 0; i < metadata_[bl].size(); i++) {
                Metadata& m = metadata_[bl][i];

                // Append subgrid
                metadata.push_back(metadata_[bl][i]);

                // Accumulate timesteps
                total_nr_timesteps += m.nr_timesteps;
            }

            // Set total total number of timesteps for baseline
            total_nr_timesteps_per_baseline.push_back(total_nr_timesteps);

            // Either all or no channels of a timestep are gridded
            // onto a subgrid, hence total_nr_timesteps * nr_channels
            int total_nr_visibilities = total_nr_timesteps * nr_channels;
            total_nr_visibilities_per_baseline.push_back(total_nr_visibilities);
        } // end for bl

        // Set sentinel
        subgrid_offset.push_back(metadata.size());
    } // end initialize


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

    int Plan::get_subgrid_offset(int bl) const {
        return subgrid_offset[bl];
    }

    int Plan::get_max_nr_subgrids(int bl1, int bl2, int n) const {
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

    int Plan::get_max_nr_subgrids() const {
        return get_max_nr_subgrids(0, get_nr_baselines(), 1);
    }

    int Plan::get_nr_timesteps() const {
        return accumulate(
            total_nr_timesteps_per_baseline.begin(),
            total_nr_timesteps_per_baseline.end(), 0);
    }

    int Plan::get_nr_timesteps(int baseline) const {
        return total_nr_timesteps_per_baseline[baseline];
    }

    int Plan::get_nr_timesteps(int baseline, int n) const {
        auto begin = next(
            total_nr_timesteps_per_baseline.begin(),
            baseline);
        auto end   = next(begin, n);
        return accumulate(begin, end, 0);
    }

    int Plan::get_max_nr_timesteps() const {
        return *max_element(
            total_nr_timesteps_per_baseline.begin(),
            total_nr_timesteps_per_baseline.end());
    }

    int Plan::get_nr_visibilities() const {
        return accumulate(
            total_nr_visibilities_per_baseline.begin(),
            total_nr_visibilities_per_baseline.end(), 0);
    }

    int Plan::get_nr_visibilities(int baseline) const {
        return total_nr_visibilities_per_baseline[baseline];
    }

    int Plan::get_nr_visibilities(int baseline, int n) const {
        auto begin = next(
            total_nr_visibilities_per_baseline.begin(),
            baseline);
        auto end   = next(begin, n);
        return accumulate(begin, end, 0);
    }

    const Metadata* Plan::get_metadata_ptr(int bl) const {
        auto offset = get_subgrid_offset(bl);
        return &(metadata[offset]);
    }

    void Plan::copy_metadata(void *ptr) const {
        memcpy(ptr, get_metadata_ptr(), get_nr_subgrids() * sizeof(Metadata));
    }

    void Plan::initialize_job(
        const unsigned int nr_baselines,
        const unsigned int jobsize,
        const unsigned int bl,
        unsigned int *first_bl_,
        unsigned int *last_bl_,
        unsigned int *current_nr_baselines_) const
    {
        // Determine maximum number of baselines in this job
        auto current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

        // Determine first and last baseline in this job
        auto first_bl = bl;
        auto last_bl  = bl + current_nr_baselines;

        // Skip empty baselines
        while (get_nr_timesteps(first_bl, 1) == 0 && first_bl < last_bl) {
            first_bl++;
        }

        // Update parameters
        (*first_bl_) = first_bl;
        (*last_bl_)  = last_bl;
        (*current_nr_baselines_) = last_bl - first_bl;
    }


    void Plan::mask_visibilities(
        Array3D<Visibility<std::complex<float>>>& visibilities) const
    {
        // Get visibilities dimensions
        auto nr_baselines = visibilities.get_z_dim();
        auto nr_timesteps = visibilities.get_y_dim();
        auto nr_channels  = visibilities.get_x_dim();

        // The visibility mask is zero
        const Visibility<std::complex<float>> zero = {0.0f, 0.0f, 0.0f, 0.0f};

        // Sanity check
        assert((unsigned) get_nr_baselines() == nr_baselines);

        // Find offset for first subgrid
        const Metadata& m0 = metadata[0];
        int baseline_offset_1 = m0.baseline_offset;

        // Iterate all metadata elements
        int nr_subgrids = get_nr_subgrids();
        for (int i = 0; i < nr_subgrids; i++) {
            const Metadata& m_current = metadata[i];

            // Determine which visibilities are used in the plan
            unsigned current_offset       = (m_current.baseline_offset - baseline_offset_1) + m_current.time_offset;
            unsigned current_nr_timesteps = m_current.nr_timesteps;

            // Determine which visibilities to mask
            unsigned first = current_offset + current_nr_timesteps;
            unsigned last = 0;
            if (i < nr_subgrids-1) {
                const Metadata& m_next = metadata[i+1];
                int next_offset = (m_next.baseline_offset - baseline_offset_1) + m_next.time_offset;
                last = next_offset;
            } else {
                last = nr_baselines * nr_timesteps;
            }

            // Mask all selected visibilities for all channels
            for (unsigned t = first; t < last; t++) {
                for (unsigned c = 0; c < nr_channels; c++) {
                    visibilities(0, t, c) = zero;
                }
            }
        }
    }

} // namespace idg

#include "PlanC.h"
