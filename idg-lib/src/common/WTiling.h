#include <vector>
#include <complex>

#include "Types.h"
#include "WTiles.h"

namespace idg {

void find_patches_for_tiles(long grid_size, int tile_size, int padded_tile_size,
                            int patch_size, int nr_tiles,
                            const idg::Coordinate* tile_coordinates,
                            std::vector<idg::Coordinate>& patch_coordinates,
                            std::vector<int>& patch_nr_tiles,
                            std::vector<int>& patch_tile_ids,
                            std::vector<int>& patch_tile_id_offsets);

void sort_by_patches(long grid_size, int tile_size, int padded_tile_size,
                     int patch_size, int nr_tiles,
                     WTileUpdateInfo& update_info);

void run_adder_patch_to_grid(int nr_polarizations, long grid_size,
                             int patch_size, int nr_patches,
                             idg::Coordinate* __restrict__ patch_coordinates,
                             std::complex<float>* __restrict__ grid,
                             std::complex<float>* __restrict__ patches_buffer);

void run_splitter_patch_from_grid(
    int nr_polarizations, long grid_size, int patch_size, int nr_patches,
    idg::Coordinate* __restrict__ patch_coordinates,
    std::complex<float>* __restrict__ grid,
    std::complex<float>* __restrict__ patches_buffer);
}  // namespace idg