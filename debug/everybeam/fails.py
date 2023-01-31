import lsmtool
s = lsmtool.load(
    '/project/rapthor/Share/rapthor/run_480ch_5chunks_flag/skymodels/calibrate_1/calibration_skymodel_small.txt',
    beamMS='/project/rapthor/Share/rapthor/L632477_480ch_50min/allbands.ms.mjd5020562823_field.ms')
s.getColValues('I', applyBeam=True)   
