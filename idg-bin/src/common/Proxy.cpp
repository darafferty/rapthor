#include "Proxy.h"

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
    int Proxy_get_nr_subgrids(Proxy_t* p) {
        return p->get_nr_subgrids(); }
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
} // end extern "C"
