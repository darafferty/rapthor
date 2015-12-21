#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <ostream>

namespace idg {
    typedef struct { float u, v, w; } UVW;
    typedef struct { int x, y; } Coordinate;
    typedef struct { int station1, station2; } Baseline;
    typedef struct { int time_nr; Baseline baseline; Coordinate coordinate; } Metadata;

    std::ostream& operator<<(std::ostream &out, Baseline &b);
    std::ostream& operator<<(std::ostream &out, Coordinate &c);
    std::ostream& operator<<(std::ostream &out, Metadata &m);
    std::ostream& operator<<(std::ostream &out, UVW &uvw);
}

#endif
