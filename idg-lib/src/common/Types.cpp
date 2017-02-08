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
        out << "["
            << "baseline_offset = " << m.baseline_offset << ",\n"
            << "time_offset = "<< m.time_offset << ",\n"
            << "nr_timesteps = "<< m.nr_timesteps << ",\n"
            << "aterm_index = " << m.aterm_index << ",\n"
            << "baseline = "<< m.baseline << ",\n"
            << "coordinate = "<< m.coordinate
            << "]";
        return out;
    }

    template<class T>
    ostream& operator<<(ostream &out, UVWCoordinate<T> &uvw) {
        out << "("
            << uvw.u << ","
            << uvw.v << ","
            << uvw.w << ")";
        return out;
    }


    ostream& operator<<(ostream& os, const float2& x)
    {
        os << "(" << x.real << "," << x.imag << ")";
        return os;
    }

    ostream& operator<<(ostream& os, const double2& x)
    {
        os << "(" << x.real << "," << x.imag << ")";
        return os;
    }

}
