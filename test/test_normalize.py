import unittest
from rapthor.scripts.normalize_flux_scale import fit_sed
import numpy as np


class TestFacet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_fit_sed(self):
        # Test SED fitting
        frequencies = np.arange(120e6, 160e6, 5e6)  # Hz
        ref_flux = 1.0  # Jy
        ref_frequency = 140e6  # Hz
        alpha = -0.7
        fluxes = np.array([ref_flux * (frequency / ref_frequency) ** alpha for frequency in frequencies])
        errors = np.array([0.05] * len(fluxes))

        # Test single-flux case
        fit_fcn = fit_sed(fluxes[:1], errors[:1], frequencies[:1])
        self.assertAlmostEqual(fit_fcn(1.2e8), 0.0, places=5)

        # Test two-fluxes case
        fit_fcn = fit_sed(fluxes[:2], errors[:2], frequencies[:2])
        self.assertAlmostEqual(fit_fcn(1.2e8), fluxes[0], places=5)

        # Test many-fluxes case
        fit_fcn = fit_sed(fluxes, errors, frequencies)
        self.assertAlmostEqual(fit_fcn(1.2e8), fluxes[0], places=5)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFacet('test_fit_sed'))
    return suite


if __name__ == '__main__':
    unittest.main()
