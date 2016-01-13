#include <exception> // runtime_error
#include "Proxy.h"

using namespace std;

namespace idg {
    namespace proxy{

        Plan Proxy::create_plan(
            const float *uvw,
            const float *wavenumbers,
            const int *baselines,
            const int *aterm_offsets,
            const int kernel_size)
        {
            #if defined(DEBUG)
            cout << __func__ << endl;
            #endif

            Plan plan(mParams, uvw, wavenumbers,
                      baselines, aterm_offsets,
                      kernel_size);

            return plan;
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
        idg::Plan plan = p->create_plan( (float *) uvw, (float *) wavenumbers,
                                         (int *) baselines, (int *) aterm_offsets,
                                         kernel_size);
        return plan.get_nr_subgrids();
    }

    void Proxy_init_metadata(
        Proxy_t* p, void *metadata, void *uvw, void *wavenumbers,
        void *baselines, void *aterm_offsets, int kernel_size)
    {
        idg::Plan plan = p->create_plan( (float *) uvw, (float *) wavenumbers,
                                    (int *) baselines, (int *) aterm_offsets,
                                    kernel_size);
        memcpy(metadata, plan.get_metadata_ptr(0),
               plan.get_nr_subgrids() * sizeof(idg::Metadata));
    }
} // end extern "C"
