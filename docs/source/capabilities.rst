.. _capabilities:

Capabilities of Rapthor
=======================

Rapthor corrects for direction-dependent effects in HBA LOFAR (and SKA-Low) 
data, including ionospheric effects and beam-model errors. These corrections 
are essential to obtaining instrumental-noise limited (~ 0.1 mJy/beam for an 
8-hour observation), high-resolution (~ 5 arcsec FWHM) images. Rapthor can 
perform both full-field and target-only processing. Full-field processing is 
recommended if you want to image the full field or do not have a sky model of 
sufficient quality for field subtraction. If you have a good sky model of the 
field already, Rapthor can subtract off the field sources and do self 
calibration on the targets only (and/or provide calibrated data for each 
target for imaging or self calibration outside of Rapthor).
