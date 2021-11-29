#include <algorithm>
#include <cassert>

#include "WTiling.h"
#include "Index.h"

namespace idg {

void find_patches_for_tiles(long grid_size, int tile_size, int padded_tile_size,
                            int patch_size, int nr_tiles,
                            const idg::Coordinate* tile_coordinates,
                            std::vector<idg::Coordinate>& patch_coordinates,
                            std::vector<int>& patch_nr_tiles,
                            std::vector<int>& patch_tile_ids,
                            std::vector<int>& patch_tile_id_offsets) {
  // Initialize patch_tile_ids and patch_coordinates
  // for all patches in the grid.
  std::vector<std::vector<int>> patch_tile_ids_;
  std::vector<idg::Coordinate> patch_coordinates_;
  patch_coordinates_.resize(0);

  // Determine the number of patches
  // needed to cover the entire grid.
  int nr_patches = 0;
  for (int y = 0; y < grid_size; y += patch_size) {
    for (int x = 0; x < grid_size; x += patch_size) {
      nr_patches++;
      patch_coordinates_.push_back({x, y});
    }
  }
  patch_tile_ids_.resize(nr_patches);

  // Iterate all patches to find the tiles that (partially) overlap
#pragma omp parallel for
  for (int i = 0; i < nr_patches; i++) {
    patch_tile_ids_[i].resize(0);

    int patch_x_start = patch_coordinates_[i].x;
    int patch_y_start = patch_coordinates_[i].y;
    int patch_x_end = patch_x_start + patch_size;
    int patch_y_end = patch_y_start + patch_size;

    for (int j = 0; j < nr_tiles; j++) {
      // Compute position of tile in grid
      idg::Coordinate coordinate = tile_coordinates[j];
      int x0 = coordinate.x * tile_size - (padded_tile_size - tile_size) / 2 +
               grid_size / 2;
      int y0 = coordinate.y * tile_size - (padded_tile_size - tile_size) / 2 +
               grid_size / 2;
      int x_start = std::max(0, x0);
      int y_start = std::max(0, y0);
      int x_end = x0 + padded_tile_size;
      int y_end = y0 + padded_tile_size;

      // Check whether the tile (partially) falls in the patch
      if (!(x_start > patch_x_end || y_start > patch_y_end ||
            x_end < patch_x_start || y_end < patch_y_start)) {
        patch_tile_ids_[i].push_back(j);
      }
    }
  }

  // Reset output vectors
  patch_coordinates.resize(0);
  patch_nr_tiles.resize(0);
  patch_tile_ids.resize(0);
  patch_tile_id_offsets.resize(0);

  // Fill in the output vectors for the non-empty patches
  for (int i = 0; i < nr_patches; i++) {
    int nr_tiles_patch = patch_tile_ids_[i].size();

    if (nr_tiles_patch > 0) {
      // Add the number of tiles for the current patch
      patch_nr_tiles.push_back(nr_tiles_patch);

      // Add the coordinate for the current patch
      patch_coordinates.push_back(patch_coordinates_[i]);

      // Add the tile id offset to the number of tile ids
      // in all previous non-empty patches.
      patch_tile_id_offsets.push_back(patch_tile_ids.size());

      // Add all tile ids for the current patch
      for (int j = 0; j < nr_tiles_patch; j++) {
        patch_tile_ids.push_back(patch_tile_ids_[i][j]);
      }
    }
  }
}

void sort_by_patches(long grid_size, int tile_size, int padded_tile_size,
                     int patch_size, int nr_tiles,
                     WTileUpdateInfo& update_info) {
  // Find the tile to patch mapping
  std::vector<idg::Coordinate> patch_coordinates;
  std::vector<int> patch_nr_tiles;
  std::vector<int> patch_tile_ids;
  std::vector<int> patch_tile_id_offsets;
  find_patches_for_tiles(
      grid_size, tile_size, padded_tile_size, patch_size,
      update_info.wtile_ids.size(), update_info.wtile_coordinates.data(),
      patch_coordinates, patch_nr_tiles, patch_tile_ids, patch_tile_id_offsets);

  // The patch_tile_ids now has a list of all tiles, sorted by patch.
  // Use this list to construct a new WTileUpdateInfo
  WTileUpdateInfo update_info_sorted;
  update_info_sorted.subgrid_index = update_info.subgrid_index;
  update_info_sorted.wtile_ids.resize(0);
  update_info_sorted.wtile_coordinates.resize(0);

  // Find unique tile ids
  std::vector<int> tile_ids_sorted;
  for (int i : patch_tile_ids) {
    if (std::find(tile_ids_sorted.begin(), tile_ids_sorted.end(), i) ==
        tile_ids_sorted.end()) {
      tile_ids_sorted.push_back(i);
    }
  }

  // Fill in the new update info
  for (int i : tile_ids_sorted) {
    update_info_sorted.wtile_ids.push_back(update_info.wtile_ids[i]);
    update_info_sorted.wtile_coordinates.push_back(
        update_info.wtile_coordinates[i]);
  }

  // Sanity check
  assert(update_info.wtile_ids.size() == update_info_sorted.wtile_ids.size());

  // Swap the update info
  std::swap(update_info.wtile_ids, update_info_sorted.wtile_ids);
  std::swap(update_info.wtile_coordinates,
            update_info_sorted.wtile_coordinates);
}

void run_adder_patch_to_grid(int nr_polarizations, long grid_size,
                             int patch_size, int nr_patches,
                             idg::Coordinate* __restrict__ patch_coordinates,
                             std::complex<float>* __restrict__ grid,
                             std::complex<float>* __restrict__ patches_buffer) {
  std::complex<float>* dst_ptr = grid;
  std::complex<float>* src_ptr = patches_buffer;

#pragma omp parallel for
  for (int y_ = 0; y_ < patch_size; y_++) {
    for (int i = 0; i < nr_patches; i++) {
      int x = patch_coordinates[i].x;
      int y = patch_coordinates[i].y;

      int width = std::min(patch_size, (int)grid_size - x);
      int height = std::min(patch_size, (int)grid_size - y);

      if (y_ >= height) {
        break;
      }

      for (int pol = 0; pol < nr_polarizations; pol++) {
        for (int x_ = 0; x_ < width; x_++) {
          size_t dst_idx = index_grid_3d(grid_size, pol, y + y_, x + x_);
          size_t src_idx =
              index_grid_4d(nr_polarizations, patch_size, i, pol, y_, x_);
          dst_ptr[dst_idx] += src_ptr[src_idx];
        }  // end for x_
      }    // end for pol
    }      // end for i
  }        // end for y_
}

void run_splitter_patch_from_grid(
    int nr_polarizations, long grid_size, int patch_size, int nr_patches,
    idg::Coordinate* __restrict__ patch_coordinates,
    std::complex<float>* __restrict__ grid,
    std::complex<float>* __restrict__ patches_buffer) {
  std::complex<float>* dst_ptr = patches_buffer;
  std::complex<float>* src_ptr = grid;

#pragma omp parallel for
  for (int y_ = 0; y_ < patch_size; y_++) {
    for (int i = 0; i < nr_patches; i++) {
      int x = patch_coordinates[i].x;
      int y = patch_coordinates[i].y;

      int width = std::min(static_cast<long>(patch_size), grid_size - x);
      int height = std::min(static_cast<long>(patch_size), grid_size - y);

      if (y_ >= height) {
        for (int pol = 0; pol < nr_polarizations; pol++) {
          for (int x_ = 0; x_ < patch_size; x_++) {
            size_t dst_idx =
                index_grid_4d(nr_polarizations, patch_size, i, pol, y_, x_);
            dst_ptr[dst_idx] = 0;
          }  // end for x_
        }
        continue;
      }

      for (int pol = 0; pol < nr_polarizations; pol++) {
        for (int x_ = 0; x_ < width; x_++) {
          size_t dst_idx =
              index_grid_4d(nr_polarizations, patch_size, i, pol, y_, x_);
          size_t src_idx = index_grid_3d(grid_size, pol, y + y_, x + x_);
          dst_ptr[dst_idx] = src_ptr[src_idx];
        }  // end for x_
        for (int x_ = width; x_ < patch_size; x_++) {
          size_t dst_idx =
              index_grid_4d(nr_polarizations, patch_size, i, pol, y_, x_);
          dst_ptr[dst_idx] = 0;
        }  // end for x_
      }    // end for pol
    }      // end for i
  }        // end for y_
}

}  // namespace idg