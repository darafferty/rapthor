#include "Proxy.h"

using namespace std;

namespace idg {
    namespace proxy{

        vector<Metadata> Proxy::init_metadata(
            const float *_uvw,
            const float *wavenumbers,
            const int *_baselines)
        {
            // Load parameters
            auto nr_baselines = mParams.get_nr_baselines();
            auto nr_timeslots = mParams.get_nr_timeslots();
            auto nr_timesteps = mParams.get_nr_timesteps();
            auto nr_channels = mParams.get_nr_channels();
            auto grid_size = mParams.get_grid_size();
            auto subgrid_size = mParams.get_subgrid_size();
            auto imagesize = mParams.get_imagesize();

            // Compute number of subgrids
            auto nr_subgrids = nr_baselines * nr_timeslots;

            // nr_time is the total number of timesteps for a baseline
            auto nr_time = nr_timesteps * nr_timeslots;

            // Pointers to datastructures
            UVW *uvw = (UVW *) _uvw;
            Baseline *baselines = (Baseline *) _baselines;
            vector<Metadata> metadata;
            metadata.reserve(nr_subgrids);
            Baseline *bptr = (Baseline *) baselines;

            // Get wavenumber for first and last frequency
            float wavenumber_first = wavenumbers[0];
            float wavenumber_last  = wavenumbers[nr_channels-1];

            // Iterate all baselines
            int t = 0;
            for (int bl = 0; bl < nr_baselines; bl++) {
                Baseline baseline = baselines[bl];

                // Iterate all timeslots
                for (int timeslot = 0; timeslot < nr_timeslots; timeslot++) {
                    int time_offset = timeslot * nr_timesteps;

                    // Find mininmum and maximum u and v for current timeslot in pixels
                    float u_min =  std::numeric_limits<float>::infinity();
                    float u_max = -std::numeric_limits<float>::infinity();
                    float v_min =  std::numeric_limits<float>::infinity();
                    float v_max = -std::numeric_limits<float>::infinity();

                    // Iterate all timesteps
                    for (int timestep = 0; timestep < nr_timesteps; timestep++) {
                        UVW current = uvw[bl * (nr_time) + (time_offset + timestep)];

                        // U,V in meters
                        float u_meters = current.u;
                        float v_meters = current.v;

                        // Iterate all channels
                        for (int chan = 0; chan < nr_channels; chan++) {
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
                    }

                    // Compute middle point in pixels
                    int u_pixels = roundf((u_max + u_min) / 2);
                    int v_pixels = roundf((v_max + v_min) / 2);

                    // Shift center from middle of grid to top left
                    u_pixels += (grid_size/2);
                    v_pixels += (grid_size/2);

                    // Shift from middle of subgrid to top left
                    u_pixels -= (subgrid_size/2);
                    v_pixels -= (subgrid_size/2);

                    // Construct coordinate
                    Coordinate coordinate = { u_pixels, v_pixels };

                    // Compute subgrid number
                    int subgrid_nr = bl * nr_timeslots + timeslot;

                    // Set metadata
                    Metadata m = { timeslot, baseline, coordinate };
                    metadata.push_back(m);
                }
            }

            return metadata;
        }

        ostream& operator<<(ostream &out, idg::proxy::Baseline &b) {
            out << "("
                << b.station1 << ","
                << b.station2 << ")";
            return out;
        }

        ostream& operator<<(ostream &out, idg::proxy::Coordinate &c) {
            out << "("
                << c.x << ","
                << c.y << ")";
            return out;
        }

        ostream& operator<<(ostream &out, idg::proxy::Metadata &m) {
            out << m.time_nr << ", "
                << m.baseline << ", "
                << m.coordinate;
            return out;
        }

        ostream& operator<<(ostream &out, idg::proxy::UVW &uvw) {
            out << "("
                << uvw.u << ","
                << uvw.v << ","
                << uvw.w << ")";
            return out;
        }
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
    int Proxy_get_nr_timesteps(Proxy_t* p) {
        return p->get_nr_timesteps(); }
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

    int Proxy_get_nr_subgrids(Proxy_t* p, void *uvw, void *wavenumbers, void *baselines) {
        return p->init_metadata((float *) uvw, (float *) wavenumbers, (int *) baselines).size();
    }

    void Proxy_init_metadata(Proxy_t* p, void *metadata, void *uvw, void *wavenumbers, void *baselines) {
        auto _metadata = p->init_metadata((float *) uvw, (float *) wavenumbers, (int *) baselines);
        memcpy(metadata, _metadata.data(), _metadata.size() * sizeof(idg::proxy::Metadata));
    }
} // end extern "C"
