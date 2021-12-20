// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "WTiles.h"
#include "Math.h"

namespace idg {

int WTiles::add_subgrid(int subgrid_index, Coordinate wtile_coordinate) {
  // If this is a dummy WTiles instance, then addsubgrid will fail
  if (m_wtile_buffer_size == 0) return -1;

  // Get the w-tile from the map of active tiles
  // If the w-tile is not found, it will be default constructed, with wtile_id
  // = -1 and added to the map
  WTileInfo& wtile_info = m_wtile_map[wtile_coordinate];

  // Update the last access time, and increment the counter
  wtile_info.last_access = m_subgrid_count++;

  // If this is a newly constructed tile it needs to be initialized
  if (wtile_info.wtile_id == -1) {
    wtile_info.wtile_id = get_new_wtile(subgrid_index);
    if (m_initialize_set.size() == 0) {
      WTileUpdateInfo wtiles_to_initialize;
      wtiles_to_initialize.subgrid_index = subgrid_index;
      m_initialize_set.push_back(wtiles_to_initialize);
    }

    WTileUpdateInfo& wtiles_to_initialize = m_initialize_set.back();
    wtiles_to_initialize.wtile_coordinates.push_back(wtile_coordinate);
    wtiles_to_initialize.wtile_ids.push_back(wtile_info.wtile_id);
  }
  return wtile_info.wtile_id;
}

WTileUpdateInfo WTiles::clear() {
  WTileUpdateInfo wtiles_to_flush;
  wtiles_to_flush.subgrid_index = -1;
  for (auto wtile : m_wtile_map) {
    wtiles_to_flush.wtile_coordinates.push_back(wtile.first);
    wtiles_to_flush.wtile_ids.push_back(wtile.second.wtile_id);
    m_free_wtiles.push_back(wtile.second.wtile_id);
  }
  m_wtile_map.clear();
  return wtiles_to_flush;
}

int WTiles::get_new_wtile(int subgrid_index) {
  // If there are no more free w-tiles
  // a fraction of the active tiles needs to be retired
  if (!m_free_wtiles.size()) {
    WTileUpdateInfo wtiles_to_flush;
    wtiles_to_flush.subgrid_index = subgrid_index;
    // Sort the w-tiles by last access time
    WTilesOrderedByLastAccess wtiles_ordered_by_last_access(m_wtile_map.begin(),
                                                            m_wtile_map.end());
    unsigned int n = ceil(m_wtile_buffer_size * m_update_fraction);
    // Loop over the sorted w-tiles
    // starting at the least recently used w-tile
    for (auto wtile : wtiles_ordered_by_last_access) {
      // Remove w-tile from the active set
      m_wtile_map.erase(wtile.first);
      // Add w-tile to the flush set
      wtiles_to_flush.wtile_coordinates.push_back(wtile.first);
      wtiles_to_flush.wtile_ids.push_back(wtile.second.wtile_id);
      // Add w-tile to the set of free w-tiles
      m_free_wtiles.push_back(wtile.second.wtile_id);
      // Break out of the loop when the desired number of w-tiles have been
      // retired
      if (m_free_wtiles.size() == n) break;
    }

    // Add the update events to the flush and initialize sets
    WTileUpdateInfo wtiles_to_initialize;
    wtiles_to_initialize.subgrid_index = subgrid_index;
    m_initialize_set.push_back(wtiles_to_initialize);
    m_flush_set.push_back(wtiles_to_flush);
  }

  // Pop a w-tile from the set of free w-tiles
  int wtile_id = m_free_wtiles.back();
  m_free_wtiles.pop_back();

  return wtile_id;
}

int compute_w_padded_tile_size(const idg::Coordinate& coordinate,
                               const float w_step, const float image_size,
                               const float image_size_shift,
                               const int padded_tile_size) {
  float w = (coordinate.z + 0.5f) * w_step;
  float abs_w = std::abs(w);
  int w_padding = ceil(abs_w * image_size_shift * image_size);
  return next_composite(padded_tile_size + w_padding);
}

std::vector<int> compute_w_padded_tile_sizes(const idg::Coordinate* coordinates,
                                             const int nr_tiles,
                                             const float w_step,
                                             const float image_size,
                                             const float image_size_shift,
                                             const int padded_tile_size) {
  std::vector<int> w_padded_tile_sizes(nr_tiles);

  for (int i = 0; i < nr_tiles; i++) {
    w_padded_tile_sizes[i] = compute_w_padded_tile_size(
        coordinates[i], w_step, image_size, image_size_shift, padded_tile_size);
  }

  return w_padded_tile_sizes;
}

int compute_w_padded_tile_size_max(const WTileUpdateSet& wtile_set,
                                   const int tile_size, const int subgrid_size,
                                   const float image_size, const float w_step,
                                   const float shift_l, const float shift_m) {
  int w_padded_tile_size_max = 0;

  for (unsigned int i = 0; i < wtile_set.size(); i++) {
    const WTileUpdateInfo& wtile_info = wtile_set[i];
    const unsigned int nr_tiles = wtile_info.wtile_ids.size();
    const std::vector<Coordinate>& tile_coordinates =
        wtile_info.wtile_coordinates;
    const int padded_tile_size = tile_size + subgrid_size;
    const float image_size_shift =
        image_size + 2 * std::max(std::abs(shift_l), std::abs(shift_m));
    const std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
        tile_coordinates.data(), nr_tiles, w_step, image_size, image_size_shift,
        padded_tile_size);
    const int w_padded_tile_size = *std::max_element(
        w_padded_tile_sizes.begin(), w_padded_tile_sizes.end());
    w_padded_tile_size_max =
        std::max(w_padded_tile_size_max, w_padded_tile_size);
  }

  return w_padded_tile_size_max;
}

}  // namespace idg
