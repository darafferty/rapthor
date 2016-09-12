#!/usr/bin/env python

# Converts a LOFAR MS to an simplified HDF5 MS
#
# TODO:
# - do not read all data into memory
# - better ways to detremine NR_BASELINES, NR_TIMESTEPS

import os
import argparse
import numpy as np
import pyrap.tables
import h5py

# Command line argument parsing
parser = argparse.ArgumentParser(
    description='Convert a LOFAR MS to an HDF5 file')
parser.add_argument(dest='msin', nargs=1, type=str,
                    help='measurement set')
parser.add_argument('-c', '--column',
                    help='''Data column used, such as
                    DATA or CORRECTED_DATA (default: DATA)''',
                    required=False, default="DATA")
parser.add_argument('-o', '--output',
                    help='Output file',
                    required=False, default="")
args = parser.parse_args()
msin = args.msin[0]
datacolumn = args.column
out_filename = args.output

if (not out_filename):
    ms_filename = os.path.split(msin)[1]
    filename = os.path.splitext(ms_filename)[0]
    out_filename = filename + ".h5"

print "Input parameters:"
print "-----------------"
print "{0:30}{1:<40}".format("Measurement set:", msin)
print "{0:30}{1:<40}".format("Data column:", datacolumn)
print "{0:30}{1:<40}".format("Output:", out_filename)
print ""

# Open measurement set and read meta data
table = pyrap.tables.table(msin)
t_ant = pyrap.tables.table(table.getkeyword("ANTENNA"))
t_spw = pyrap.tables.table(table.getkeyword("SPECTRAL_WINDOW"))

nr_rows = table.nrows()
frequencies = t_spw[0]['CHAN_FREQ']

# Read nr_time samples for all baselines including auto correlations
timestamps   = table.getcol('TIME')
antenna1     = table.getcol('ANTENNA1')
antenna2     = table.getcol('ANTENNA2')
uvw          = table.getcol('UVW')
visibilities = table.getcol(datacolumn)
flags        = table.getcol('FLAG')

nr_antennas     = t_ant.nrows()
nr_channels     = frequencies.shape[0]
nr_correlations = visibilities.shape[2]

# TODO: make this cleaner
have_auto_correlations = False
for ant1, ant2 in zip(antenna1,antenna2):
    if (ant1 == ant2):
        have_auto_correlations = True

if have_auto_correlations == True:
    nr_baselines = ( (nr_antennas + 1) * nr_antennas ) / 2
else:
    nr_baselines = ( (nr_antennas - 1) * nr_antennas ) / 2

if (nr_rows % nr_baselines != 0):
    raise RuntimeError("The total number of rows ({}) is not divisible by the number of baselines ({})".format(nr_rows, nr_baselines))
else:
    nr_timesteps = nr_rows / nr_baselines

print "Writing:"
print "--------"
print "{0:30}{1:<40}".format("NR_ANTENNAS:", nr_antennas)
print "{0:30}{1:<40}".format("NR_BASELINES:", nr_baselines)
print "{0:30}{1:<40}".format("NR_TIMESTEPS:", nr_timesteps)
print "{0:30}{1:<40}".format("NR_CHANNELS:", nr_channels)
print "{0:30}{1:<40}".format("NR_CORRELATIONS:", nr_correlations)
print ""

# put NR_TIMESTEPS as outer dimension
timestamps = timestamps.reshape((nr_timesteps, nr_baselines))
antenna1 = antenna1.reshape((nr_timesteps, nr_baselines))
antenna2 = antenna2.reshape((nr_timesteps, nr_baselines))
uvw = uvw.reshape((nr_timesteps, nr_baselines, 3))
visibilities = visibilities.reshape((nr_timesteps, nr_baselines,
                                     nr_channels, nr_correlations))
flags = flags.reshape((nr_timesteps, nr_baselines,
                       nr_channels, nr_correlations))

# Open file to write to
f = h5py.File(out_filename, "w")
f.attrs['default']      = 'data'
f.attrs['HDF5_Version'] = h5py.version.hdf5_version
f.attrs['h5py_version'] = h5py.version.version

data_group = f.create_group('data')
data_group.attrs['NR_ANTENNAS']     = nr_antennas
data_group.attrs['NR_BASELINES']    = nr_baselines
data_group.attrs['NR_TIMESTEPS']    = nr_timesteps
data_group.attrs['NR_CHANNELS']     = nr_channels
data_group.attrs['NR_CORRELATIONS'] = nr_correlations

frequencies_dataset = data_group.create_dataset("FREQUENCIES",
                                                frequencies.shape,
                                                dtype=frequencies.dtype)
# TODO: what is the unit of the frequencies given?
frequencies_dataset.attrs['unit'] = 'hertz'
frequencies_dataset[:] = frequencies

timestamps_dataset = data_group.create_dataset("TIME",
                                               timestamps.shape,
                                               dtype=timestamps.dtype)
timestamps_dataset.attrs['unit'] = 'none'
timestamps_dataset[:] = timestamps

antenna1_dataset = data_group.create_dataset("ANTENNA1",
                                             antenna1.shape,
                                             dtype=antenna1.dtype)
antenna1_dataset.attrs['unit'] = 'none'
antenna1_dataset[:] = antenna1

antenna2_dataset = data_group.create_dataset("ANTENNA2",
                                             antenna2.shape,
                                             dtype=antenna2.dtype)
antenna2_dataset.attrs['unit'] = 'none'
antenna2_dataset[:] = antenna2

uvw_dataset = data_group.create_dataset("UVW",
                                        uvw.shape,
                                        dtype=uvw.dtype)
uvw_dataset.attrs['unit'] = 'meters'
uvw_dataset[:] = uvw

visibilities_dataset = data_group.create_dataset("DATA",
                                                 visibilities.shape,
                                                 dtype=visibilities.dtype)
visibilities_dataset.attrs['unit'] = 'none'
visibilities_dataset[:] = visibilities

flags_dataset = data_group.create_dataset("FLAG",
                                          flags.shape,
                                          dtype=flags.dtype)
flags_dataset.attrs['unit'] = 'none'
flags_dataset[:] = flags

table.close()
f.close()
