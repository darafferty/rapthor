extern "C" {
    void* Proxy_get_grid(
        idg::proxy::Proxy* p,
        unsigned int nr_correlations,
        unsigned int grid_size)
    {
        const unsigned int nr_w_layers = 1;
        idg::Grid grid = p->get_grid(nr_w_layers, nr_correlations, grid_size, grid_size);
        return grid.data();
    }
}
