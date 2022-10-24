import unittest
import os
import requests
from rapthor.lib.screen import KLScreen

class TestScreen(unittest.TestCase):
    @classmethod
    def downloadms(self, filename):
        url = 'https://support.astron.nl/software/ci_data/rapthor/' + filename
        r = requests.get(url)
        f = open('resources/' + filename, 'wb')
        f.write(r.content)
        f.close()

    @classmethod
    def setUpClass(self):
        cwd = os.getcwd()
        if not cwd.endswith('test'):
            raise SystemExit('Please run this test from the test directory!')
        testh5name = 'split_solutions_0.h5'
        if (not os.path.exists('resources/' + testh5name)):
            print('downloading h5 file')
            self.downloadms(testh5name)
        else:
            print('h5 file found')

        self.screen = KLScreen('testscreen', 'resources/split_solutions_0.h5', 'resources/calibration_skymodel.txt', 0.0, 0.0, 1.0, 1.0)

    @classmethod
    def tearDownClass(self):
        pass

    def test_screen(self):
        self.assertEqual(self.screen.name, 'testscreen')
        self.assertEqual(self.screen.width_ra, 1.0)

    def test_fit(self):
        self.screen.ncpu = 1
        self.screen.fit()
        self.assertAlmostEqual(self.screen.midDec, 57.2615, delta=1.0E-4)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestScreen('test_screen'))
    suite.addTest(TestScreen('test_fit'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())


