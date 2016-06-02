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
        auto nr_baselines = mParams.get_nr_baselines();
        auto nr_time = mParams.get_nr_time();
        auto nr_timeslots = mParams.get_nr_timeslots();
        auto nr_channels = mParams.get_nr_channels();
        auto grid_size = mParams.get_grid_size();
        auto subgrid_size = mParams.get_subgrid_size();
        auto imagesize = mParams.get_imagesize();

        // Pointers to datastructures
        UVW *uvw = (UVW *) _uvw;
        Baseline *baselines = (Baseline *) _baselines;
        metadata.reserve(nr_baselines * nr_time / nr_timeslots);
        Baseline *bptr = (Baseline *) baselines;

        // Get wavenumber for first and last frequency
        float wavenumber_first = wavenumbers[0];
        float wavenumber_last  = wavenumbers[nr_channels-1];

        // Iterate all baselines
        for (int bl = 0; bl < nr_baselines; bl++) {
            Baseline baseline = baselines[bl];
            subgrid_offset.push_back(metadata.size());
            int timesteps_current_baseline = 0;

            // Iterate all timesteps
            int time = 0;
            while (time < nr_time) {
                // Find mininmum and maximum u and v for current set of
                // measurements in pixels
                float u_min =  std::numeric_limits<float>::infinity();
                float u_max = -std::numeric_limits<float>::infinity();
                float v_min =  std::numeric_limits<float>::infinity();
                float v_max = -std::numeric_limits<float>::infinity();

                int nr_timesteps = 0;
                int u_pixels_previous;
                int v_pixels_previous;
                int aterm_index_subgrid = find_aterm_index(time, aterm_offsets);
                for (int t = time; t < nr_time; t++) {
                    int baseline_offset = bl * nr_time;
                    auto aterm_index = find_aterm_index(t, aterm_offsets);
                    UVW current = uvw[baseline_offset + t];

                    // U,V in meters
                    float u_meters = current.u;
                    float v_meters = current.v;

                    float uv_max_meters  = fmax(fabs(u_meters), fabs(v_meters));
                    float uv_max_pixels  = uv_max_meters * imagesize * wavenumbers[0] /
                                           (2 * M_PI);
                    bool uv_in_range     = uv_max_pixels < grid_size/2;

                    // Iterate all channels
                    // TODO: split channels if they do not fit on a subgrid
                    for (int chan = 0; chan < nr_channels; chan += nr_channels) {
                        float wavenumber = wavenumbers[chan];
                        float scaling = imagesize * wavenumber / (2 * M_PI);

                        // U,V in pixels
                        float u_pixels = u_meters * scaling;
                        float v_pixels = v_meters * scaling;

                        if (u_pixels < u_min) u_min = u_pixels;
                        if (u_pixels > u_max) u_max = u_pixels;
                        if (v_pixels < v_min) v_min = v_pixels;
                        if (v_pixels > v_max) v_max = v_pixels;
                    } // end for chan

                    // Compute u,v width
                    int u_width = u_max - u_min + 1;
                    int v_width = v_max - v_min + 1;
                    int uv_width = u_width < v_width ? v_width : u_width;

                    // Compute middle point in pixels
                    int u_pixels = roundf((u_max + u_min) / 2);
                    int v_pixels = roundf((v_max + v_min) / 2);

                    // Shift center from middle of grid to top left
                    u_pixels += (grid_size/2);
                    v_pixels += (grid_size/2);

                    // Shift from middle of subgrid to top left
                    u_pixels -= (subgrid_size/2);
                    v_pixels -= (subgrid_size/2);

                    // Stop conditions
                    bool same_aterm = (aterm_index == aterm_index_subgrid);
                    bool last_iteration = (t == nr_time - 1);
                    bool timestep_fits = uv_in_range && (uv_width + kernel_size) < subgrid_size;

                    // Check whether current set of measurements fit in subgrid
                    if (timestep_fits && same_aterm && !last_iteration) {
                        // Continue to next measurement
                        nr_timesteps++;
                    } else {
                        // Measurement no longer fits, create new subgrid

                        // Use current u,v pixels for last measurement
                        if (timestep_fits && t == nr_time - 1) {
                            u_pixels_previous = u_pixels;
                            v_pixels_previous = v_pixels;
                            nr_timesteps++;
                        }

                        // // TODO: split also on channels dynamically
                        // if (nr_timesteps == 0) {
                        //     printf("max_nr_timesteps=%d\n", max_nr_timesteps);
                        //     throw runtime_error("Could not fit any timestep on subgrid,
                        //                         nr_channels is probably too high");
                        // }

                        // Construct coordinate
                        Coordinate coordinate = { u_pixels_previous,
                                                  v_pixels_previous };

                        if (nr_timesteps > 0) {
                            // Split into subgrids with at most max_nr_timesteps
                            for (int i = 0; i < nr_timesteps; i += max_nr_timesteps) {
                                int current_nr_timesteps = i + max_nr_timesteps < nr_timesteps ? max_nr_timesteps : nr_timesteps - i;
                                timesteps_current_baseline += current_nr_timesteps;

                                // Set metadata
                                Metadata m = { baseline_offset,
                                               time + i,
                                               current_nr_timesteps,
                                               aterm_index_subgrid,
                                               baseline,
                                               coordinate };
                                metadata.push_back(m);

                                // cout << "New subgrid: " << endl
                                //      << m << endl;
                            }
                        } else {
                            // Skip timestep
                            nr_timesteps++;
                        }

                        // Go to next subgrid
                        time += nr_timesteps;
                        break;
                    }

                    // Store current u,v pixels
                    u_pixels_previous = u_pixels;
                    v_pixels_previous = v_pixels;
                }
            } // end while
            timesteps_per_baseline.push_back(timesteps_current_baseline);
        } // end for bl

        // Set sentinel
        subgrid_offset.push_back(metadata.size());
    } // end init_metadata

} // namespace idg
