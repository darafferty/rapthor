#ifndef IDG_PARAMETERS2_H_
#define IDG_PARAMETERS2_H_

#include "idg-config.h"

#include <iostream>

namespace idg {

    class CompileConstants
    {
        public:
            /// Define the environment names searched for
            static const std::string ENV_NR_CORRELATIONS;
            static const std::string ENV_SUBGRIDSIZE;

            /// Constructor: default reads values from ENV or sets default
            CompileConstants(
                unsigned int nr_correlations,
                unsigned int subgrid_size) :
                nr_correlations(nr_correlations),
                subgrid_size(subgrid_size) {}

            CompileConstants()
            {
                set_from_env();
            }

            // default copy constructor/assignment okay

            // default destructur
            ~CompileConstants() = default;

            // get methods
            unsigned int get_nr_correlations() const { return nr_correlations; }
            unsigned int get_subgrid_size() const { return subgrid_size; }

            // set methods
            void set_nr_correlations(unsigned int nr_correlations);
            void set_subgrid_size(unsigned int sgs);

            // auxiliary functions
            void print() const;
            void print(std::ostream& os) const;
            void set_from_env();

            private:
                unsigned int nr_correlations;
                unsigned int subgrid_size;
    };

    // helper functions
    std::ostream& operator<<(std::ostream& os, const CompileConstants& c);

} // namespace idg

#endif
