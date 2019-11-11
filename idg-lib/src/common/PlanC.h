#include <limits>

extern "C" {
    const idg::Plan* Plan_init(
        const int kernel_size,
        const int subgrid_size,
        const int grid_size,
        const float cell_size,
        float* frequencies,
        const unsigned int frequencies_nr_channels,
        float* uvw,
        const unsigned int uvw_nr_baselines,
        const unsigned int uvw_nr_timesteps,
        const unsigned int uvw_nr_coordinates,
        unsigned int* baselines,
        const unsigned int baselines_nr_baselines,
        const unsigned int baselines_two,
        unsigned int* aterms_offsets,
        const unsigned int aterms_offsets_nr_timeslots_plus_one)
    {

        idg::Array1D<float> frequencies_(
            frequencies, frequencies_nr_channels);
        idg::Array2D<idg::UVW<float>> uvw_(
            (idg::UVW<float> *) uvw, uvw_nr_baselines, uvw_nr_timesteps);
        idg::Array1D<std::pair<unsigned int,unsigned int>> baselines_(
            (std::pair<unsigned int,unsigned int> *) baselines, baselines_nr_baselines);
        idg::Array1D<unsigned int> aterms_offsets_(
            aterms_offsets, aterms_offsets_nr_timeslots_plus_one);

        return new idg::Plan(
            kernel_size,
            subgrid_size,
            grid_size,
            cell_size,
            frequencies_,
            uvw_,
            baselines_,
            aterms_offsets_);
    }

    int Plan_get_nr_subgrids(idg::Plan* plan)
    {
        return plan->get_nr_subgrids();
    }

    void Plan_copy_metadata(
        idg::Plan* plan,
        void *ptr)
    {
        plan->copy_metadata(ptr);
    }

    void Plan_destroy(idg::Plan* plan)
    {
        delete plan;
    }
}
