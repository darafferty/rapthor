// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_WTILES_H_
#define IDG_WTILES_H_

#include <iterator>
#include <limits>
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <stdexcept>  // runtime_error
#include <cmath>
#include <numeric>
#include <omp.h>

#include "Types.h"

namespace idg {

/**
 * @brief Structure to keep track of the last time a tile was accessed
 *
 */
struct WTileInfo {
  int wtile_id = -1;
  int last_access = 0;
};

/**
 * @brief A comparison operator to allow sorting by x,y,z coordinates
 *
 */
struct CompareCoordinate {
  bool operator()(const Coordinate& lhs, const Coordinate& rhs) const {
    return lhs.x < rhs.x ||
           (lhs.x == rhs.x &&
            (lhs.y < rhs.y || (lhs.y == rhs.y && lhs.z < rhs.z)));
  }
};

/**
 * @brief A comparison operator to allow sorting by last access time
 *
 */
struct CompareLastAccess {
  bool operator()(const std::pair<Coordinate, WTileInfo>& lhs,
                  const std::pair<Coordinate, WTileInfo>& rhs) const {
    return lhs.second.last_access < rhs.second.last_access;
  }
};

/**
 * @brief A map type to store the active W-tiles
 *
 */
typedef std::map<Coordinate, WTileInfo, CompareCoordinate> WTileMap;

/**
 * @brief A set type to sort W-tiles by last time accessed
 *
 */
typedef std::set<std::pair<Coordinate, WTileInfo>, CompareLastAccess>
    WTilesOrderedByLastAccess;

/**
 * @brief A structure to store the information needed for an update event
 *
 */
struct WTileUpdateInfo {
  int subgrid_index;
  std::vector<int> wtile_ids;
  std::vector<Coordinate> wtile_coordinates;
};

/**
 * @brief Double ended queue to store a list of update events
 *
 */
typedef std::deque<WTileUpdateInfo> WTileUpdateSet;

class WTiles {
  static constexpr float kUpdateFraction = 0.1;

 public:
  /**
   * @brief Dummy constructor to create an empty WTiles object
   *
   */
  WTiles() : m_wtile_buffer_size(0) {}

  /**
   * @brief Construct a new WTiles object
   *
   * @param wtile_buffer_size number of w-tiles to keep in the buffer
   *
   * The parameters below are not used by the WTile object
   * They are stored and can be queried users of the WTile object
   * @param grid_size - size of the grid
   * @param subgrid_size - size of the subgrid
   * @param image_size - size of the image (radians)
   * @param w_step - w step size
   *
   * @param update_fraction
   */
  WTiles(int wtile_buffer_size, int grid_size, int subgrid_size, int wtile_size,
         float image_size, float w_step,
         float update_fraction = kUpdateFraction)
      : m_subgrid_count(0),
        m_grid_size(grid_size),
        m_subgrid_size(subgrid_size),
        m_wtile_size(wtile_size),
        m_image_size(image_size),
        m_w_step(w_step),
        m_update_fraction(update_fraction),
        m_wtile_buffer_size(wtile_buffer_size),
        m_free_wtiles(wtile_buffer_size) {
    std::iota(m_free_wtiles.begin(), m_free_wtiles.end(), 0);
  }

  int get_grid_size() { return m_grid_size; }
  int get_subgrid_size() { return m_subgrid_size; }
  float get_image_size() { return m_image_size; }
  float get_w_step() { return m_w_step; }
  int get_wtile_buffer_size() { return m_wtile_buffer_size; }
  int get_wtile_size() { return m_wtile_size; }

  /**
   * @brief Main method used by Plan to obtain a wtile_index
   *
   * @param subgrid_index index of subgrid in Plan, if an update event is
   * needed, this is the index will be stored with the update event
   * @param wtile_coordinate coordinate of requested wtile
   * @return int wtile_index
   */
  int add_subgrid(int subgrid_index, Coordinate wtile_coordinate);

  /**
   * @brief Get the flush set object
   *
   * The Plan gives access to this function to provide the gridding method
   * with information on when what w-tiles need to be flushed
   *
   * @return WTileUpdateSet
   */
  WTileUpdateSet get_flush_set() { return std::move(m_flush_set); }

  /**
   * @brief Get the initialize set object
   *
   * The Plan gives access to this function to provide the degridding method
   * with information on when what w-tiles need to be initialized
   *
   * @return WTileUpdateSet
   */
  WTileUpdateSet get_initialize_set() { return std::move(m_initialize_set); }

  /**
   * @brief clear entire cache...
   *
   * @return * WTileUpdateInfo object with the w-tiles that need to be flushed
   */
  WTileUpdateInfo clear();

 private:
  /**
   * @brief Get a new wtile
   *
   * When the add_subgrid function does not find a w-tile in the cache
   * this function is called to get the index of a free w-tile.
   * If there are no more free w-tiles, an update event is created at the
   * provided subgrid_index to free up a number of w-tiles
   *
   * @param subgrid_index
   * @return int
   */
  int get_new_wtile(int subgrid_index);

  int m_subgrid_count;
  int m_grid_size;
  int m_subgrid_size;
  int m_wtile_size = 1;
  float m_image_size;
  float m_w_step;
  float m_update_fraction;
  WTileMap m_wtile_map;
  int m_wtile_buffer_size;
  std::vector<int> m_free_wtiles;
  WTileUpdateSet m_flush_set;
  WTileUpdateSet m_initialize_set;
};

}  // namespace idg

#endif
