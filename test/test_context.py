import unittest
import os
import requests
from rapthor.lib.context import Timer
from rapthor.lib.context import RedirectStdStreams

class TestField(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_timer(self):
        t = Timer()

    def test_streams(self):
        s = RedirectStdStreams()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestField('test_timer'))
    suite.addTest(TestField('test_streams'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())


