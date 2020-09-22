#!/usr/bin/env python3
#
# Gridding/Degridding integration tests. Integration testing of gridding/degridding with
# wsclean-idg. The degridding test compares DPPP generated "DATA" column against wsclean
# generated "MODEL_DATA" column. The gridding tests checks whether the specified flux of the source
# is accurately reconstructed.
#
# This test was extracted from test-main.py in the https://gitlab.com/astron-idg/idg-test-wsclean
# repository.
#
# Please note the following:
#   - Despite the name, this test includes more than just gridding, there is a degridding test as well, and a (disbabled) clean test
#   - To run this test, idg needs to be compiled in DEBUG mode, or with WRITE_OUT_SCALAR_BEAM=ON, in order to write the scalar_beam.py file.
#   - The different test cases are prepared in the load_tests method. As a future improvement, we might consider using
#     pytest to parametrize the different test cases. This probably gives a nicer/cleaner/clearer overview of the different
#     tests compared to the unittest framework.
#   - To limit the computational runtime, this test only checks the Stokes I component. We may want to re-enable the
#     other Stokes components in the future by extending stokes_list.
#   - To limit the computational runtime, this test only checks for the more complex case "grid_with_beam=True" and
#     "differential_beam=True"
#   - Currently, this test only runs on the cpu. However, we can once IDG is compiled with GPU enabled, the ["cpu"] list in load_tests
#     can be changed to ["cpu", "hybrid"] to test on different hardware configurations.
#   - The python tests were inactive in test-main.py, and are therefore not included in this integration test.
#     We may want to re-enable these test cases.
#   - The same holds for "createTestClean". We may want to re-enable this in the future.
#   - As a future improvement, we might consider to eliminate the overlap between preparetestset.py/test_pointsource.py and
#     the test_gridding.py

import unittest
import os
from subprocess import check_call

import numpy as np
from astropy.io import fits
import casacore.tables

datadir = os.environ["DATADIR"]
common_dir = os.environ["COMMON"]
msname = os.environ["MSNAME"]
ms = os.path.join(datadir, msname)

interval_start = 0
interval_end = 100
startchan = 0
nchan = 0

cellsize = 2.0  # asecs
imagesize = 512  # pixels

# Only Stokes I, we may want to re-enable the other components in the future
stokes_list = ["I"]
# stokes_list = ["I","Q", "U", "V"]


class TestSuiteTemplateImage(unittest.TestSuite):
    def run(self, result):

        self.ms = os.path.join(datadir, ms)
        print(f"Creating template image from {ms}")
        check_call(
            [
                "wsclean",
                "-quiet",
                "-size",
                str(imagesize),
                str(imagesize),
                "-scale",
                f"{cellsize}asec",
                "-interval",
                "0",
                "1",
                "-no-reorder",
                self.ms,
            ]
        )
        template = os.path.join(datadir, "template.fits")
        check_call(["mv", "wsclean-image.fits", template])
        super(TestSuiteTemplateImage, self).run(result)


def createTestPointSource(
    stokes, offset, grid_with_beam, differential_beam, idgmode="cpu"
):
    class TestPointSource(unittest.TestCase):
        def __init__(self, methodName="runTest"):
            super(TestPointSource, self).__init__(methodName)

        def shortDescription(self):
            return f"stokes={stokes}, offset={offset}, grid_with_beam={grid_with_beam}, differential_beam={differential_beam}, idgmode={idgmode}"

        @classmethod
        def setUpClass(self):

            super(TestPointSource, self).setUpClass()
            print(f"Setting up test for Stokes {stokes}")

            self.setupok = False

            template = os.path.join(datadir, "template.fits")
            self.ms = os.path.join(datadir, ms)

            stokes1 = stokes
            with fits.open(template) as img:
                N = img[0].data.shape[-1]

                print(N)

                img[0].data[:] = 0.0

                for stokes2 in ["Q", "U", "V"]:
                    if stokes1 != stokes2:
                        img.writeto(
                            f"pointsource-{stokes1}-{stokes2}-model.fits",
                            overwrite=True,
                        )

                img[0].data[0, 0, int(N / 2 + offset[0]), int(N / 2 + offset[1])] = 1.0

                img.writeto(f"pointsource-{stokes1}-I-model.fits", overwrite=True)
                if stokes1 != "I":
                    img.writeto(
                        f"pointsource-{stokes1}-{stokes1}-model.fits", overwrite=True
                    )

            check_call(
                [
                    "casapy2bbs.py",
                    "--no-patches",
                    f"pointsource-{stokes1}-{stokes1}-model.fits",
                    "temp.cat",
                ]
            )

            # TODO: this probably needs generalization in case we do not want to test all the Stokes components
            stokesI_idx = 4

            stokes_idx = 4 + stokes_list.index(stokes)

            file_in = "temp.cat"
            file_out = f"pointsource-{stokes}.cat"

            with open(file_in) as f_in, open(file_out, "w") as f_out:
                for cnt, line in enumerate(f_in):
                    line = line[:-1]
                    s = line.split(",")
                    if cnt > 4 and len(s) == 8:
                        s[stokes_idx] = s[stokesI_idx]
                        line = ",".join(s)
                    print(line)
                    f_out.write(line + "\n")

            sourcedb = f"pointsource-{stokes}.sourcedb"
            check_call(["rm", "-rf", sourcedb])
            check_call(
                    [
                        "makesourcedb",
                        f"in={file_out}",
                        "format=Name, Type, Ra, Dec, I, Q, U, V",
                        f"out={sourcedb}",
                    ]
                )

            T = casacore.tables.taql(
                "SELECT TIME, cdatetime(TIME-.1) AS TIMESTR FROM $(self.ms) GROUPBY TIME"
            )

            global interval_end

            if interval_end == 0:
                interval_end = len(T) - 1

            self.starttime = T[interval_start]["TIME"] - 0.1
            self.endtime = T[interval_end]["TIME"] - 0.1
            starttimestr = T[interval_start]["TIMESTR"]
            endtimestr = T[interval_end]["TIMESTR"]

            check_call(
                [
                    "DPPP",
                    os.path.join(
                        common_dir,
                        ["dppp-predict.parset", "dppp-predict-correct.parset"][
                            grid_with_beam and differential_beam
                        ],
                    ),
                    f"msin={self.ms}",
                    f"msin.starttime={starttimestr}",
                    f"msin.endtime={endtimestr}",
                    f"msin.startchan={startchan}",
                    f"msin.nchan={nchan}",
                    f"predict.sourcedb={sourcedb}",
                    f"predict.usebeammodel={grid_with_beam}",
                ]
            )

            # Hack to create a pointsource-I-beam-I.fits
            cmd = [
                "wsclean",
                "-quiet",
                "-name",
                "pointsource-{0}".format(stokes),
                "-size",
                str(imagesize),
                str(imagesize),
                "-scale",
                "{0}asec".format(cellsize),
                "-interval",
                str(interval_start),
                str(interval_end),
                "-no-reorder",
                "-use-idg",
                "-idg-mode",
                idgmode,
                "-no-dirty",
                "-niter",
                str(1),
                "-grid-with-beam",
                self.ms,
            ]
            print(" ".join(cmd))
            check_call(cmd)
            self.setupok = True

        def setUp(self):
            pass

        def test_degridding(self):

            if not self.setupok:
                self.fail("test setup failed")

            cmd = (
                [
                    "wsclean",
                    "-quiet",
                    "-name",
                    f"pointsource-{stokes}",
                    "-predict",
                    "-interval",
                    str(interval_start),
                    str(interval_end),
                    "-pol",
                    "IQUV",
                    "-use-idg",
                    "-idg-mode",
                    idgmode,
                    "-no-reorder",
                ]
                + ([], ["-channel-range", str(startchan), str(startchan + nchan)])[
                    nchan > 0
                ]
                + ["-beam-aterm-update", "30"]
                + ([], ["-grid-with-beam"])[grid_with_beam]
                + ([], ["-use-differential-lofar-beam"])[
                    grid_with_beam and differential_beam
                ]
                + [self.ms]
            )
            print(" ".join(cmd))
            check_call(cmd)

            ms = self.ms
            starttime = self.starttime
            endtime = self.endtime
            t = casacore.tables.taql(
                "SELECT * FROM $ms WHERE TIME>$starttime AND TIME<$endtime  AND ANTENNA1!=ANTENNA2"
            )

            data = t.getcol("DATA")  # generated by DPPP
            if nchan > 0:
                data = data[:, startchan : startchan + nchan, :]

            model_data = t.getcol("MODEL_DATA")  # generated by wsclean
            if nchan > 0:
                model_data = model_data[:, startchan : startchan + nchan, :]

            print(model_data.shape)
            flag = t.getcol("FLAG")
            flag_row = t.getcol("FLAG_ROW")

            print(f"Infinity norm data-model_data {np.amax(abs(data-model_data))}")
            if grid_with_beam:
                self.assertTrue(np.allclose(data, model_data, rtol=5e-2, atol=5e-2))
            else:
                self.assertTrue(np.allclose(data, model_data, rtol=1e-3, atol=1e-3))

        def test_gridding(self):
            cmd = (
                [
                    "wsclean",
                    "-quiet",
                    "-name",
                    f"pointsource-{stokes}",
                    "-data-column",
                    "DATA",
                    "-size",
                    str(imagesize),
                    str(imagesize),
                    "-scale",
                    f"{cellsize}asec",
                    "-interval",
                    str(interval_start),
                    str(interval_end),
                    "-no-reorder",
                    "-pol",
                    "IQUV",
                    "-use-idg",
                    "-idg-mode",
                    idgmode,
                    "-no-dirty",
                ]
                + ([], ["-channel-range", str(startchan), str(startchan + nchan)])[
                    nchan > 0
                ]
                + ([], ["-grid-with-beam"])[grid_with_beam]
                + ([], ["-use-differential-lofar-beam"])[
                    grid_with_beam and differential_beam
                ]
                + [self.ms]
            )
            print(cmd)
            check_call(cmd)

            if grid_with_beam:
                beam = np.load("scalar_beam.npy")

            for stokes1 in stokes_list:
                imgname = f'pointsource-{stokes}-{stokes1}-image{["", "-pb"][grid_with_beam]}.fits'
                with fits.open(imgname) as img:
                    N = img[0].data.shape[-1]
                    flux = img[0].data[
                        0, 0, int(N / 2 + offset[0]), int(N / 2 + offset[1])
                    ]

                    if grid_with_beam:
                        flux /= beam[int(N / 2 + offset[0]), int(N / 2 + offset[1])]

                    if (stokes1 == "I") or (stokes == stokes1):
                        expected_flux = 1.0
                    else:
                        expected_flux = 0.0
                    self.assertTrue(
                        np.allclose(flux, expected_flux, rtol=1e-2, atol=1e-2),
                        f"Expected flux {stokes1} in Stokes {expected_flux}, found {flux}. Grid with beam? {grid_with_beam}",
                    )

    return TestPointSource


def load_tests(loader, tests, pattern):
    suite = TestSuiteTemplateImage()

    # Loop over cases
    for x in [-128, 0]:
        for y in [0]:
            for grid_with_beam in [True]:  # , True]:
                for differential_beam in [
                    True
                ]:  # [[False],[False, True]][grid_with_beam]: # only when grid_with_beam is True, iterate with differential_beam over True and False
                    offset = (y, x)
                    for stokes in stokes_list:
                        for idgmode in ["cpu"]:
                            # TODO: re-enable hybrid/gpu code once IDG compiled with BUILD_LIB_CUDA=ON
                            # for idgmode in ["cpu", "hybrid"]:
                            tests = loader.loadTestsFromTestCase(
                                createTestPointSource(
                                    stokes,
                                    offset,
                                    grid_with_beam,
                                    differential_beam,
                                    idgmode,
                                )
                            )
                            suite.addTests(tests)
    return suite


if __name__ == "__main__":
    unittest.main(failfast=True)
