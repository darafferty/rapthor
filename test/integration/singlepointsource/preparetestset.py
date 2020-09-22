#!/usr/bin/env python3

import os
from astropy.io import fits
from subprocess import call, check_call

# Extract some environment variables
source_dir = os.path.dirname(os.path.realpath(__file__))
datadir = os.environ["DATADIR"]
msname = os.environ["MSNAME"]

# Write template fits image
ms = os.path.join(datadir, msname)
check_call(
    [
        "wsclean",
        "-size",
        "1000",
        "1000",
        "-scale",
        "1asec",
        "-interval",
        "0",
        "1",
        "-no-reorder",
        ms,
    ]
)
template = os.path.join(datadir, "template.fits")
check_call(["mv", "wsclean-image.fits", template])

with fits.open(template) as img:
    N = img[0].data.shape[-1]
    img[0].data[:] = 0.0
    for stokes1 in ["I", "Q", "U", "V"]:
        for stokes2 in ["I", "Q", "U", "V"]:
            if stokes1 != stokes2:
                img.writeto(
                    f"pointsource-{stokes1}-{stokes2}-model.fits", overwrite=True,
                )

    img[0].data[0, 0, int(N / 2), int(N / 2)] = 1.0

    for stokes in ["I", "Q", "U", "V"]:
        img.writeto(f"pointsource-{stokes}-{stokes}-model.fits", overwrite=True)

# Convert the non-zero (center) pixel to ra-dec coordinates and write cat file
check_call(["casapy2bbs.py", "pointsource-I-I-model.fits", "pointsource.cat"])

stokes = ["I", "Q", "U", "V"]
stokes1 = "I"
stokes1_idx = 5 + stokes.index(stokes1)

# This loop takes pointsource.cat and:
# - copies/writes its contents into pointsource-[stokes2].cat files (only line 8 is modified)
# - writes the sourcedb directories based on the .cat files
# TODO: once the integration test is up-and-running, this loop probably can
# be cleaned up a bit.
for stokes2 in stokes:
    stokes2_idx = 5 + stokes.index(stokes2)

    file_in = "pointsource.cat"
    file_out = f"pointsource-{stokes2}.cat"

    with open(file_in) as f_in, open(file_out, "w") as f_out:
        for cnt, line in enumerate(f_in):
            line = line[:-1]
            s = line.split(",")
            if cnt > 6 and len(s) == 9:
                s[stokes1_idx], s[stokes2_idx] = "0.0", s[stokes1_idx]
                line = ",".join(s)
            print(line)
            f_out.write(line + "\n")

    sourcedb = f"pointsource-{stokes2}.sourcedb"
    check_call(["rm", "-rf", sourcedb])
    check_call(
        [
            "makesourcedb",
            f"in={file_out}",
            "format=Name, Type, Patch, Ra, Dec, I, Q, U, V",
            f"out={sourcedb}",
        ]
    )
exit(0)
