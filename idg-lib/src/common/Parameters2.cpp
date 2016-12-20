#include "Parameters2.h"

#include <cstdlib> // getenv, atoi
#include <iostream> // ostream
#include <iomanip> // setw
#include <sstream>
#include <cmath> // fabs
#include <cassert> // assert


using namespace std;

namespace idg {

    const string CompileConstants::ENV_NR_CORRELATIONS = "NR_CORRELATIONS";
    const string CompileConstants::ENV_SUBGRIDSIZE     = "SUBGRIDSIZE";

    // set methods
    void CompileConstants::set_nr_correlations(unsigned int n)
    {
        assert(n == 4);
        nr_correlations = n;
    }

    void CompileConstants::set_subgrid_size(unsigned int n)
    {
      subgrid_size = n;
    }

    // auxiliary functions
    void CompileConstants::print(ostream& os) const
    {
        const int fw1 = 30;
        const int fw2 = 10;

        os << "-----------" << endl;
        os << "PARAMETERS:" << endl;

        os << setw(fw1) << left << "Number of correlations" << "== "
           << setw(fw2) << right << nr_correlations << endl;

        os << setw(fw1) << left << "Subgrid size" << "== "
           << setw(fw2) << right << subgrid_size << endl;

        os << "-----------" << endl;
    }


    void CompileConstants::print() const
    {
        print(cout);
    }


    void CompileConstants::set_from_env()
    {
        const unsigned int DEFAULT_NR_CORRELATIONS = 4;
        const unsigned int DEFAULT_SUBGRIDSIZE     = 24;

        // nr_correlations
        char *cstr_nr_correlations = getenv(ENV_NR_CORRELATIONS.c_str());
        nr_correlations = cstr_nr_correlations ? atoi(cstr_nr_correlations) : DEFAULT_NR_CORRELATIONS;

        // subgrid_size
        char *cstr_subgrid_size = getenv(ENV_SUBGRIDSIZE.c_str());
        subgrid_size = cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;
    } // read_parameters_from_env()

    string CompileConstants::definitions(
        unsigned int nr_correlations,
        unsigned int subgrid_size)
    {
        stringstream parameters;
        parameters << " -DNR_POLARIZATIONS=" << nr_correlations;
        parameters << " -DSUBGRIDSIZE=" << subgrid_size;
        return parameters.str();
    }

    // helper functions
    ostream& operator<<(ostream& os, const CompileConstants& c)
    {
        c.print(os);
        return os;
    }
} // namespace idg
