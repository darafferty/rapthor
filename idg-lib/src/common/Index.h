#ifndef IDG_INDEX_H_
#define IDG_INDEX_H_

//namespace idg {

    /* Index methods */
    inline FUNCTION_ATTRIBUTES size_t index_grid(
            long grid_size,
            int w_layer,
            int pol,
            int y,
            int x)
    {
        // grid: [nr_w_layers][NR_CORRELATIONS][grid_size][grid_size]
        return static_cast<size_t>(w_layer) * NR_CORRELATIONS * grid_size * grid_size +
               static_cast<size_t>(pol) * grid_size * grid_size +
               static_cast<size_t>(y) * grid_size +
               static_cast<size_t>(x);
    }

    inline FUNCTION_ATTRIBUTES size_t index_grid_tiling(
            int tile_size,
            size_t grid_size,
            int pol,
            int y,
            int x)
    {
        // grid: [NR_TILES][NR_TILES][NR_CORRELATIONS][TILE_SIZE][TILE_SIZE]
        assert(grid_size % tile_size == 0);
        const int NR_TILES  = grid_size / tile_size;
        size_t idx_tile_y = y / tile_size;
        size_t idx_tile_x = x / tile_size;
        size_t tile_y = y % tile_size;
        size_t tile_x = x % tile_size;

        return
               idx_tile_y * NR_TILES * NR_CORRELATIONS * tile_size * tile_size +
               idx_tile_x * NR_CORRELATIONS * tile_size * tile_size +
               pol * tile_size * tile_size +
               tile_y * tile_size +
               tile_x;
    }

    inline FUNCTION_ATTRIBUTES size_t index_grid(
            size_t grid_size,
            int pol,
            int y,
            int x)
    {
        // grid: [NR_CORRELATIONS][grid_size][grid_size]
        return pol * grid_size * grid_size +
               y * grid_size +
               x;
    }

    inline FUNCTION_ATTRIBUTES size_t index_subgrid(
        int subgrid_size,
        int s,
        int pol,
        int y,
        int x)
    {
        // subgrid: [nr_subgrids][NR_CORRELATIONS][subgrid_size][subgrid_size]
        return static_cast<size_t>(s) * NR_CORRELATIONS * subgrid_size * subgrid_size +
               static_cast<size_t>(pol) * subgrid_size * subgrid_size +
               static_cast<size_t>(y) * subgrid_size +
               static_cast<size_t>(x);
    }

    inline FUNCTION_ATTRIBUTES size_t index_visibility(
        int nr_channels,
        int time,
        int chan,
        int pol)
    {
        // visibilities: [nr_time][nr_channels][NR_CORRELATIONS]
        return static_cast<size_t>(time) * nr_channels * NR_CORRELATIONS +
               static_cast<size_t>(chan) * NR_CORRELATIONS +
               static_cast<size_t>(pol);
    }

    inline FUNCTION_ATTRIBUTES size_t index_aterm(
        int subgrid_size,
        int nr_stations,
        int aterm_index,
        int station,
        int y,
        int x,
        int pol)
    {
        // aterm: [nr_aterms][subgrid_size][subgrid_size][NR_CORRELATIONS]
        size_t aterm_nr = (aterm_index * nr_stations + station);
        return static_cast<size_t>(aterm_nr) * subgrid_size * subgrid_size * NR_CORRELATIONS +
               static_cast<size_t>(y) * subgrid_size * NR_CORRELATIONS +
               static_cast<size_t>(x) * NR_CORRELATIONS +
               static_cast<size_t>(pol);
    }

    inline FUNCTION_ATTRIBUTES size_t index_aterm_transposed(
        int subgrid_size,
        int nr_stations,
        int aterm_index,
        int station,
        int y,
        int x,
        int pol)
    {
        // aterm: [nr_aterms][NR_CORRELATIONS][subgrid_size][subgrid_size]
        size_t aterm_nr = (aterm_index * nr_stations + station);
        return static_cast<size_t>(aterm_nr) * NR_CORRELATIONS * subgrid_size * subgrid_size +
               static_cast<size_t>(pol) * subgrid_size * subgrid_size +
               static_cast<size_t>(y) * subgrid_size +
               static_cast<size_t>(x);
    }
//}

#endif
