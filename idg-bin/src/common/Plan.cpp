#include "Plan.h"

using namespace std;

namespace idg {

    Plan::Plan(const Parameters& parameters,
               const float *uvw,
               const float *wavenumbers,
               const int *baselines,
               const int *aterm_offsets,
               const int kernel_size)
        : mParams(parameters)
    {
        init_metadata(uvw, wavenumbers, baselines,
                      aterm_offsets, kernel_size);
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


    const Metadata* Plan::get_metadata_ptr(int bl) const {
        auto offset = get_subgrid_offset(bl);
        return &(metadata[offset]);
    }


    vector<Metadata> Plan::copy_metadata() const {
        return metadata;
    }


    void Plan::init_metadata(
        const float *_uvw,
        const float *wavenumbers,
        const int *_baselines,
        const int *aterm_offsets,
        const int kernel_size)
    {
        // Load parameters
        auto nr_baselines = mParams.get_nr_baselines();
        auto nr_time = mParams.get_nr_time();
        auto nr_channels = mParams.get_nr_channels();
        auto grid_size = mParams.get_grid_size();
        auto subgrid_size = mParams.get_subgrid_size();
        auto imagesize = mParams.get_imagesize();

        // Pointers to datastructures
        UVW *uvw = (UVW *) _uvw;
        Baseline *baselines = (Baseline *) _baselines;
        metadata.reserve(nr_baselines); // TODO: put more accurate reservation
        Baseline *bptr = (Baseline *) baselines;

        // Get wavenumber for first and last frequency
        float wavenumber_first = wavenumbers[0];
        float wavenumber_last  = wavenumbers[nr_channels-1];

        // Iterate all baselines
        for (int bl = 0; bl < nr_baselines; bl++) {
            Baseline baseline = baselines[bl];
            subgrid_offset.push_back(metadata.size());

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
                for (int time_offset = time; time_offset < nr_time; time_offset++) {
                    int baseline_offset = bl * nr_time;
                    UVW current = uvw[baseline_offset + time_offset];

                    // U,V in meters
                    float u_meters = current.u;
                    float v_meters = current.v;

                    // Iterate all channels
                    // if (bl==0 && time==0) printf("WARNING: some channels not gridded!\n");
                    for (int chan = 0; chan < 1; chan++) {
                        // for (int chan = 0; chan < nr_channels; chan++) {
                        float wavenumber = wavenumbers[chan];
                        float scaling = imagesize * wavenumber / (2 * M_PI);

                        // U,V in pixels
                        float u_pixels = u_meters * scaling;
                        float v_pixels = v_meters * scaling;

                        if (u_pixels < u_min) u_min = u_pixels;
                        if (u_pixels > u_max) u_max = u_pixels;
                        if (v_pixels < v_min) v_min = v_pixels;
                        if (v_pixels > v_max) v_max = v_pixels;
                    }

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

                    // TODO: add a MAX_NR_TIMESTEPS to be put onto a grid?
                    // TODO: add changing A-terms
                    bool same_aterm = true;
                    bool last_iteration = (time_offset == nr_time - 1);
                    bool timestep_fits = (uv_width + kernel_size) < subgrid_size;

                    // Check whether current set of measurements fit in subgrid
                    if (timestep_fits && same_aterm && !last_iteration) {
                        // Continue to next measurement
                        nr_timesteps++;
                    } else {
                        // Measurement no longer fits, create new subgrid

                        // Use current u,v pixels for last measurement
                        if (time_offset == nr_time - 1) {
                            u_pixels_previous = u_pixels;
                            v_pixels_previous = v_pixels;
                            nr_timesteps++;
                        }

                        // TODO: split also on channels dynamically
                        if (nr_timesteps==0) {
                            throw runtime_error("Probably too many channels!");
                        }

                        // Construct coordinate
                        Coordinate coordinate = { u_pixels_previous,
                                                  v_pixels_previous };

                        // Set metadata
                        Metadata m = { time,
                                       nr_timesteps,
                                       baseline,
                                       coordinate };
                        // TODO: include aterm_index
                        metadata.push_back(m);

                        // cout << "Create new subgrid: " << m << endl;

                        // Go to next subgrid
                        time += nr_timesteps;
                        break;
                    }

                    // Store curren u,v pixels
                    u_pixels_previous = u_pixels;
                    v_pixels_previous = v_pixels;
                }
            } // end while
        } // end for bl
        // Set sentinel
        subgrid_offset.push_back(metadata.size());
    } // end init_metadata


    // auxiliary for debugging
    void Plan::print_subgrid_offset() const {
        int k = 0;
        for (auto& e: subgrid_offset) {
            cout << "subgrid_offset[" << k++ << "]:" << e << endl;
        }
    }


} // namespace idg
