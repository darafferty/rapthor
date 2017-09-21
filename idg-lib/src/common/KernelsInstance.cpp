#include "KernelsInstance.h"

namespace idg {
    namespace kernel {

        void KernelsInstance::shift(
            Array3D<std::complex<float>>& data) const
        {
            int nr_polarizations = data.get_z_dim();
            int height = data.get_y_dim();
            int width = data.get_x_dim();
            assert(height == width);

            std::complex<float> tmp13, tmp24;

            // Dimensions
            int n = height;
            int n2 = n / 2;

            // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
            for (int pol = 0; pol < nr_polarizations; pol++) {
                #pragma omp parallel for private(tmp13, tmp24) collapse(2)
                for (int i = 0; i < n2; i++) {
                    for (int k = 0; k < n2; k++) {
                        tmp13                 = data(pol, i, k);
                        data(pol, i, k)       = data(pol, i+n2, k+n2);
                        data(pol, i+n2, k+n2) = tmp13;

                        tmp24                 = data(pol, i+n2, k);
                        data(pol, i+n2, k)    = data(pol, i, k+n2);
                        data(pol, i, k+n2)    = tmp24;
                     }
                }
            }
        }

        void KernelsInstance::scale(
            Array3D<std::complex<float>>& data,
            std::complex<float> scale) const
        {
            int nr_polarizations = data.get_z_dim();
            int height = data.get_y_dim();
            int width = data.get_x_dim();

            #pragma omp parallel for collapse(3)
            for (int pol = 0; pol < nr_polarizations; pol++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        std::complex<float> value = data(pol, y, x);
                        data(pol, y, x) = std::complex<float>(
                            value.real() * scale.real(),
                            value.imag() * scale.imag());
                    }
                }
            }
        }

    } // namespace kernel
} // namespace idg
