#include "Types.h"

using namespace std;

namespace idg {
    ostream& operator<<(ostream &out, Baseline &b) {
        out << "("
            << b.station1 << ","
            << b.station2 << ")";
        return out;
    }

    ostream& operator<<(ostream &out, Coordinate &c) {
        out << "("
            << c.x << ","
            << c.y << ")";
        return out;
    }

    ostream& operator<<(ostream &out, Metadata &m) {
        out << m.offset << ", "
            << m.nr_timesteps << ", "
            << m.baseline << ", "
            << m.coordinate;
        return out;
    }

    ostream& operator<<(ostream &out, UVW &uvw) {
        out << "("
            << uvw.u << ","
            << uvw.v << ","
            << uvw.w << ")";
        return out;
    }
}
