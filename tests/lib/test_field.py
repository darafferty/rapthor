import os
import unittest

import requests
from rapthor.lib import miscellaneous as misc
from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read


class TestField(unittest.TestCase):
    @classmethod
    def downloadms(cls, filename):
        url = 'https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz'
        r = requests.get(url)
        f = open('downloaded.tgz', 'wb')
        f.write(r.content)
        f.close()

        os.system('tar xvf downloaded.tgz')
        os.system('rm downloaded.tgz')
        os.system('mv tDDECal.MS ' + filename)

    @classmethod
    def setUpClass(cls):
        # Change directory to the tests directory (one level up from this file),
        # because that's where these tests need to be run from.
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        testmsname = 'resources/test.ms'
        if (not os.path.exists(testmsname)):
            print('downloading ms file')
            cls.downloadms(testmsname)
        else:
            print('ms file found')

        cls.par = parset_read('resources/test.parset')
        cls.field = Field(cls.par)
        cls.field.fast_timestep_sec = 32.0  # needed for test_get_obs_parameters() below
        cls.field.scan_observations()
        cls.field.update_skymodels(1, True, target_flux=0.2)
        cls.field.set_obs_parameters()
        cls.field.define_imaging_sectors()
        cls.field.define_outlier_sectors(1)

    @classmethod
    def tearDownClass(cls):
        os.system('rm -r images/ logs/ pipelines/ regions/ skymodels/ solutions/ plots/')

    def test_scan_observations(self):
        self.assertEqual(self.field.fwhm_ra_deg, 4.500843683229519)

    def test_regular_frequency_spacing(self):
        self.assertTrue(all([obs.channels_are_regular for obs in self.field.observations]))

    def test_imaging_sectors(self):
        self.assertEqual(self.field.sector_bounds_deg, '[258.558431;57.961675;259.103519;56.885818]')

    def test_outlier_sectors(self):
        self.assertEqual(self.field.outlier_sectors, [])

    def test_chunk_observations(self):
        for obs in self.field.full_observations:
            obs.data_fraction = 0.8
        self.field.chunk_observations(600.0, prefer_high_el_periods=False)
        self.assertEqual(self.field.imaging_sectors[0].observations[0].starttime, 4871282392.90695)

    def test_chunk_observations_high_el(self):
        for obs in self.field.full_observations:
            obs.data_fraction = 0.2
        self.field.chunk_observations(600.0, prefer_high_el_periods=True)
        self.assertEqual(self.field.imaging_sectors[0].observations[0].starttime, 4871282392.90695)

    def test_get_obs_parameters(self):
        obsp = self.field.get_obs_parameters('starttime')
        self.assertEqual(obsp, ['29Mar2013/13:59:52.907'])

    def test_define_imaging_sectors(self):
        self.field.define_imaging_sectors()
        self.assertEqual(self.field.sector_bounds_mid_deg, '[258.841667;57.410833]')

    def test_define_outlier_sectors(self):
        self.field.define_outlier_sectors(1)
        self.assertEqual(self.field.outlier_sectors, [])

    def test_define_bright_source_sectors(self):
        self.field.define_bright_source_sectors(0)
        self.assertEqual(self.field.bright_source_sectors, [])

    def test_find_intersecting_sources(self):
        iss = self.field.find_intersecting_sources()
        self.assertAlmostEqual(iss[0].area, 18.37996802132365)

    def test_check_selfcal_progress(self):
        self.assertEqual(self.field.check_selfcal_progress(), (False, False, False))

    def test_plot_overview_patches(self):
        self.field.plot_overview('field_overview_1.png', show_calibration_patches=True)
        self.assertTrue(os.path.exists(os.path.join('plots', 'field_overview_1.png')))

    def test_plot_overview_initial(self):
        self.field.plot_overview('initial_field_overview.png', show_initial_coverage=True)
        self.assertTrue(os.path.exists(os.path.join('plots', 'initial_field_overview.png')))
        os.system('rm plots/initial_field_overview.png')

    def test_plot_overview_initial_near_pole(self):
        self.field.dec = 89.5  # test behavior near pole
        self.field.plot_overview('initial_field_overview.png', show_initial_coverage=True)
        self.assertTrue(os.path.exists(os.path.join('plots', 'initial_field_overview.png')))
        os.system('rm plots/initial_field_overview.png')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestField('test_scan_observations'))
    suite.addTest(TestField('test_regular_frequency_spacing'))
    suite.addTest(TestField('test_imaging_sectors'))
    suite.addTest(TestField('test_outlier_sectors'))
    suite.addTest(TestField('test_radec2xy'))
    suite.addTest(TestField('test_chunk_observations'))
    suite.addTest(TestField('test_chunk_observations_high_el'))
    suite.addTest(TestField('test_get_obs_parameters'))
    suite.addTest(TestField('test_define_imaging_sectors'))
    suite.addTest(TestField('test_define_outlier_sectors'))
    suite.addTest(TestField('test_define_bright_source_sectors'))
    suite.addTest(TestField('test_find_intersecting_sources'))
    suite.addTest(TestField('test_check_selfcal_progress'))
    suite.addTest(TestField('test_plot_overview_patches'))
    suite.addTest(TestField('test_plot_overview_initial'))
    suite.addTest(TestField('test_plot_overview_initial_near_pole'))
    return suite


if __name__ == '__main__':
    unittest.main()
