"""
Tests for the adjust_h5parm_sources script.
"""

import os
import tempfile
from pathlib import Path

import losoto.h5parm
import numpy
import pytest
from rapthor.scripts.adjust_h5parm_sources import main


def test_main(): #skymodel, h5parm, solset):
    # # We would normally check the output or state after running main.
    # # Here we just ensure it runs without error.
    # with open("test.sky", "w") as f:
    #     f.write("FORMAT = Name, Type, Patch, Ra, Dec, I, SpectralIndex='[]', LogarithmicSI, ReferenceFrequency='140349138.906485', MajorAxis, MinorAxis, Orientation\n")
    #     f.write(" , , Patch_16, 17:15:01.0282, 56.45.51.4051\n")
    #     f.write("s0c437_sector_3, POINT, Patch_16, 17:15:01.724, 56.45.34.113, 0.000689301627026287, [-0.000400512540633508, 0.00906344920554066], false, 140335482.064951, 0, 0, 0\n")

    # with losoto.h5parm.h5parm("test.h5", readonly=False) as h5parm:
    #     solset = h5parm.makeSolset("sol000")
    #     # Create a dummy soltab with some directions
    #     directions = numpy.array(['[Patch_16]']) #, '[Patch_19]', '[Patch_42]', '[Patch_43]',
    #     #'[Patch_49]', '[Patch_77]', '[Patch_84]'])
    #     frequencies = numpy.array([140349138.906485])  # Hz
    #     amplitudes = numpy.array([4.0])
    #     solset.makeSoltab(
    #         soltype="amplitude",
    #         soltabName="amplitude000",
    #         axesNames=["dir"],
    #         axesVals=[frequencies],
    #         vals=amplitudes,
    #         weights=numpy.ones(len(directions))
    #     )
        
    # # print("\n===>", dir(solset.getSoltabs()[0]))
    # main(skymodel="test.sky", h5parm_file="test.h5") #, solset.name)
    pass