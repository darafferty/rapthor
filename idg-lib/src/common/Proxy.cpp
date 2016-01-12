#include <exception> // runtime_error
#include "Proxy.h"

using namespace std;

namespace idg {
    namespace proxy{

        // TODO: remove
        vector<Metadata> Proxy::init_metadata(
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
            vector<Metadata> metadata;
            metadata.reserve(nr_baselines); // TODO: put more accurate reservation
            Baseline *bptr = (Baseline *) baselines;

            // Get wavenumber for first and last frequency
            float wavenumber_first = wavenumbers[0];
            float wavenumber_last  = wavenumbers[nr_channels-1];

            // Iterate all baselines
            for (int bl = 0; bl < nr_baselines; bl++) {
                Baseline baseline = baselines[bl];

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

                            // cout << "Create new subgrid: "
                            //     << "offset = " << offset << ", "
                            //     << "time_offset = " << time << ", "
                            //     << "nr_timesteps = " << nr_timesteps << endl;

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

            return metadata;
        } // end init_metadata

    } // namespace proxy
} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::Proxy Proxy_t;

    int Proxy_get_nr_stations(Proxy_t* p) {
        return p->get_nr_stations(); }
    int Proxy_get_nr_baselines(Proxy_t* p) {
        return p->get_nr_baselines(); }
    int Proxy_get_nr_channels(Proxy_t* p) {
        return p->get_nr_channels(); }
    int Proxy_get_nr_time(Proxy_t* p) {
        return p->get_nr_time(); }
    int Proxy_get_nr_timeslots(Proxy_t* p) {
        return p->get_nr_timeslots(); }
    int Proxy_get_nr_polarizations(Proxy_t* p) {
        return p->get_nr_polarizations(); }
    float Proxy_get_imagesize(Proxy_t* p) {
        return p->get_imagesize(); }
    int Proxy_get_grid_size(Proxy_t* p) {
        return p->get_grid_size(); }
    int Proxy_get_subgrid_size(Proxy_t* p) {
        return p->get_subgrid_size(); }
    int Proxy_get_job_size(Proxy_t* p) {
        return p->get_job_size(); }
    int Proxy_get_job_size_gridding(Proxy_t* p) {
        return p->get_job_size_gridding(); }
    int Proxy_get_job_size_degridding(Proxy_t* p) {
        return p->get_job_size_degridding(); }
    int Proxy_get_job_size_gridder(Proxy_t* p) {
        return p->get_job_size_gridder(); }
    int Proxy_get_job_size_adder(Proxy_t* p) {
        return p->get_job_size_adder(); }
    int Proxy_get_job_size_splitter(Proxy_t* p) {
        return p->get_job_size_splitter(); }
    int Proxy_get_job_size_degridder(Proxy_t* p) {
        return p->get_job_size_degridder(); }
    void Proxy_set_job_size(Proxy_t* p, int n) {
        p->set_job_size(n); }
    void Proxy_set_job_size_gridding(Proxy_t* p, int n) {
        p->set_job_size_gridding(n); }
    void Proxy_set_job_size_degridding(Proxy_t* p, int n) {
        p->set_job_size_degridding(n); }

    int Proxy_get_nr_subgrids(
        Proxy_t* p, void *uvw, void *wavenumbers,
        void *baselines, void *aterm_offsets, int kernel_size)
    {
        return p->init_metadata(
            (float *) uvw, (float *) wavenumbers,
            (int *) baselines, (int *) aterm_offsets, kernel_size).size();
    }

    void Proxy_init_metadata(
        Proxy_t* p, void *metadata, void *uvw, void *wavenumbers,
        void *baselines, void *aterm_offsets, int kernel_size)
    {
        auto _metadata = p->init_metadata(
            (float *) uvw, (float *) wavenumbers,
            (int *) baselines, (int *) aterm_offsets, kernel_size);
        memcpy(metadata, _metadata.data(), _metadata.size() * sizeof(idg::Metadata));
    }
} // end extern "C"
