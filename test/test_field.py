import unittest
import os
import requests
from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read

class TestField(unittest.TestCase):

    def downloadms(self, filename):
        url = 'https://git.astron.nl/RD/DP3/-/raw/master/DDECal/test/integration/tDDECal.in_MS.tgz?inline=false'

        r = requests.get(url)

        f = open('downloaded.tgz', 'wb')
        f.write(r.content)
        f.close()

        os.system('tar xvf downloaded.tgz')

        os.system('rm downloaded.tgz')
        os.system('mv tDDECal.MS ' + filename)

    def setUp(self):
        testmsname = 'resources/test.ms'
        if (not os.path.exists(testmsname)):
            print('downloading ms file')
            self.downloadms(testmsname)
        else:
            print('ms file found')

        self.par = parset_read('resources/test.parset')
        self.field = Field(self.par)
        self.field.scan_observations()
#        self.field.make_skymodels('resources/test_true_sky.txt')
#        self.field.set_obs_parameters()
        self.field.define_imaging_sectors()
        self.field.define_outlier_sectors(1)

    def test_scan_observations(self):
        self.assertEqual(self.field.fwhm_ra_deg, 4.500843683229519)


    def test_imaging_sectors(self):
        self.assertEqual(self.field.sector_bounds_deg, '[258.558431;57.961675;259.103519;56.885818]')        


    def test_outlier_sectors(self):
        self.assertEqual(self.field.outlier_sectors, [])


    def test_radec2xy(self):
        self.assertEqual(self.field.radec2xy([0.0], [0.0]), ([12187.183569042127], [-12477.909993882473]))


if __name__ == '__main__':
    unittest.main()


