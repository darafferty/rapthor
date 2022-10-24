import unittest
import os
import requests
from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read

class TestField(unittest.TestCase):
    @classmethod
    def downloadms(self, filename):
        url = 'https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz'
        r = requests.get(url)
        f = open('downloaded.tgz', 'wb')
        f.write(r.content)
        f.close()

        os.system('tar xvf downloaded.tgz')
        os.system('rm downloaded.tgz')
        os.system('mv tDDECal.MS ' + filename)

    @classmethod
    def setUpClass(self):
        cwd = os.getcwd()
        if not cwd.endswith('test'):
            raise SystemExit('Please run this test from the test directory!')
        testmsname = 'resources/test.ms'
        if (not os.path.exists(testmsname)):
            print('downloading ms file')
            self.downloadms(testmsname)
        else:
            print('ms file found')

        self.par = parset_read('resources/test.parset')
        self.field = Field(self.par)
        self.field.scan_observations()
        self.field.make_skymodels('resources/test_true_sky.txt', skymodel_apparent_sky='resources/test_apparent_sky.txt', target_flux=0.1)
        self.field.set_obs_parameters()
        self.field.define_imaging_sectors()
        self.field.define_outlier_sectors(1)

    @classmethod
    def tearDownClass(self):
        os.system('rm -r images/ logs/ pipelines/ regions/ scratch/ skymodels/ solutions/')

    def test_scan_observations(self):
        self.assertEqual(self.field.fwhm_ra_deg, 4.500843683229519)


    def test_imaging_sectors(self):
        self.assertEqual(self.field.sector_bounds_deg, '[258.558431;57.961675;259.103519;56.885818]')


    def test_outlier_sectors(self):
        self.assertEqual(self.field.outlier_sectors, [])


    def test_radec2xy(self):
        self.assertEqual(self.field.radec2xy([0.0], [0.0]), ([12187.183569042127], [-12477.909993882473]))

    def test_xy2radec(self):
        self.assertEqual(self.field.xy2radec([12187.183569042127], [-12477.909993882473]), ([1.4210854715202004e-14],[0.0]))

    def test_chunk_observations(self):
        self.field.chunk_observations(data_fraction=0.8)
        self.assertEqual(self.field.imaging_sectors[0].observations[0].starttime, 4871282392.90695)

    def test_get_obs_parameters(self):
        obsp = self.field.get_obs_parameters('starttime')
        self.assertEqual(obsp, ['29Mar2013/13:59:52.907', '29Mar2013/14:00:22.949'])

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
        self.assertEqual(iss[0].area, 18.37996802132365)

    def test_check_selfcal_progress(self):
        self.assertEqual(self.field.check_selfcal_progress(), (False, False))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestField('test_scan_observations'))
    suite.addTest(TestField('test_imaging_sectors'))
    suite.addTest(TestField('test_outlier_sectors'))
    suite.addTest(TestField('test_radec2xy'))
    suite.addTest(TestField('test_xy2radec'))
    suite.addTest(TestField('test_chunk_observations'))
    suite.addTest(TestField('test_get_obs_parameters'))
    suite.addTest(TestField('test_define_imaging_sectors'))
    suite.addTest(TestField('test_define_outlier_sectors'))
    suite.addTest(TestField('test_define_bright_source_sectors'))
    suite.addTest(TestField('test_find_intersecting_sources'))
    suite.addTest(TestField('test_check_selfcal_progress'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())


