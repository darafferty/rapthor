#ifndef IDG_WTILES_H_
#define IDG_WTILES_H_

#include <iterator>
#include <limits>
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <stdexcept> // runtime_error
#include <cmath>
#include <numeric>
#include <omp.h>

#include "Types.h"

namespace idg {

	struct WTileInfo
	{
		int wtile_id = -1;
        int last_access = 0;
	};

    struct CompareCoordinate
    {
        bool operator() (const Coordinate& lhs, const Coordinate& rhs) const
        {
            return lhs.x<rhs.x || (lhs.x==rhs.x && (lhs.y<rhs.y || (lhs.y==rhs.y && lhs.z<rhs.z)));
        }
    };

    struct CompareLastAccess
    {
        bool operator() (const std::pair<Coordinate,WTileInfo>& lhs, const std::pair<Coordinate, WTileInfo>& rhs) const
        {
            return lhs.second.last_access<rhs.second.last_access;
        }
    };


    typedef std::map<Coordinate, WTileInfo, CompareCoordinate> WTileMap;
    typedef std::set<std::pair<Coordinate, WTileInfo>, CompareLastAccess> WTilesOrderedByLastAccess;

    struct WTileUpdateInfo
    {
        int subgrid_index;
        std::vector<int> wtile_ids;
        std::vector<Coordinate> wtile_coordinates;
    };

    typedef std::deque<WTileUpdateInfo> WTileUpdateSet;

    class WTiles
    {
    public:
        WTiles(int wtile_buffer_size = 1000, float update_fraction=0.1) :
            m_subgrid_count(0),
            m_update_fraction(update_fraction),
            m_wtile_buffer_size(wtile_buffer_size),
            m_free_wtiles(wtile_buffer_size)
        {
            std::iota(m_free_wtiles.begin(), m_free_wtiles.end(), 0);
        }

        int add_subgrid(int subgrid_index, Coordinate wtile_coordinate)
        {
            if (m_wtile_buffer_size == 0) return -1;
            WTileInfo &wtile_info = m_wtile_map[wtile_coordinate];
            wtile_info.last_access = m_subgrid_count++;
            if (wtile_info.wtile_id == -1)
            {
                wtile_info.wtile_id = get_new_wtile(subgrid_index);
                m_wtiles_to_initialize.wtile_coordinates.push_back(wtile_coordinate);
                m_wtiles_to_initialize.wtile_ids.push_back(wtile_info.wtile_id);

            }
            m_wtiles_to_initialize.subgrid_index = subgrid_index;
            return wtile_info.wtile_id;
        }

        WTileUpdateSet get_flush_set()
        {
            return std::move(m_flush_set);
        }

        WTileUpdateSet get_initialize_set()
        {
            m_wtiles_to_initialize.subgrid_index++;
            m_initialize_set.push_back(std::move(m_wtiles_to_initialize));
            return std::move(m_initialize_set);
        }

        // clear entire cache...
        WTileUpdateInfo clear()
        {
            WTileUpdateInfo wtiles_to_flush;
            wtiles_to_flush.subgrid_index = -1;
            for(auto wtile : m_wtile_map)
            {
                wtiles_to_flush.wtile_coordinates.push_back(wtile.first);
                wtiles_to_flush.wtile_ids.push_back(wtile.second.wtile_id);
                m_free_wtiles.push_back(wtile.second.wtile_id);
            }
            m_flush_set.push_back(wtiles_to_flush);
            m_wtile_map.clear();
            return wtiles_to_flush;
        }

    private:

        int get_new_wtile(int subgrid_index)
        {
            if (!m_free_wtiles.size())
            {
                WTileUpdateInfo wtiles_to_flush;
                wtiles_to_flush.subgrid_index = subgrid_index;
                WTilesOrderedByLastAccess wtiles_ordered_by_last_access(m_wtile_map.begin(), m_wtile_map.end());
                unsigned int n = ceil(m_wtile_buffer_size*m_update_fraction);
                for(auto wtile : wtiles_ordered_by_last_access)
                {
                    m_wtile_map.erase(wtile.first);
                    wtiles_to_flush.wtile_coordinates.push_back(wtile.first);
                    wtiles_to_flush.wtile_ids.push_back(wtile.second.wtile_id);
                    m_free_wtiles.push_back(wtile.second.wtile_id);
                    if (m_free_wtiles.size() == n) break;
                }
                m_initialize_set.push_back(std::move(m_wtiles_to_initialize));
                m_flush_set.push_back(wtiles_to_flush);
            }
            int wtile_id = m_free_wtiles.back();
            m_free_wtiles.pop_back();
            return wtile_id;
        }

        int m_subgrid_count;
        float m_update_fraction;
        WTileMap m_wtile_map;
        int m_wtile_buffer_size;
        std::vector<int> m_free_wtiles;
        WTileUpdateSet m_flush_set;
        WTileUpdateSet m_initialize_set;
        WTileUpdateInfo m_wtiles_to_initialize;
    };

} // namespace idg

#endif
