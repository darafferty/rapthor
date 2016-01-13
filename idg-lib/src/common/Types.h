#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <ostream>

namespace idg {
    typedef struct { float u, v, w; } UVW;
    typedef struct { int x, y; } Coordinate;
    typedef struct { int station1, station2; } Baseline;

    // TODO: is int enough for offsets?
    typedef struct { int baseline_offset; int time_offset; int nr_timesteps;
                     int aterm_index; Baseline baseline; Coordinate coordinate; } Metadata;

    std::ostream& operator<<(std::ostream &out, Baseline &b);
    std::ostream& operator<<(std::ostream &out, Coordinate &c);
    std::ostream& operator<<(std::ostream &out, Metadata &m);
    std::ostream& operator<<(std::ostream &out, UVW &uvw);
}

#endif
