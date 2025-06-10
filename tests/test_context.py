import unittest
import os
import requests
from rapthor.lib.context import Timer
from rapthor.lib.context import RedirectStdStreams

class TestContext(unittest.TestCase):

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
    suite.addTest(TestContext('test_timer'))
    suite.addTest(TestContext('test_streams'))
    return suite

if __name__ == '__main__':
    unittest.main()
