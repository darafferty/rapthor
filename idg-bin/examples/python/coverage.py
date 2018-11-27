#!/usr/bin/env python

import idg
import util
from data import Data
import numpy
import matplotlib.pyplot as plt

############
# paramaters
############
nr_channels           = 8
nr_timesteps          = 128          # samples per baseline
nr_timeslots          = 1             # A-term time slots
subgrid_size          = 24
integration_time      = 0.9
kernel_size           = (subgrid_size / 2) + 1
nr_correlations       = 4
layout_file           = "SKA1_low_ecef"
nr_stations_limit     = 100
baseline_length_limit = 0 # m
w_step                = 0.0
max_nr_timesteps      = numpy.iinfo(numpy.int32).max # per subgrid

######################################################################
# initialize data generator
######################################################################
data = Data(nr_stations_limit, baseline_length_limit, layout_file)

# option 1, choose image_size and compute corresponding grid_size
#image_size      = 0.05
#grid_size       = int(data.compute_grid_size(image_size))

# option 2, choose grid_size and compute image_size
#           such that all baselines fit in the grid
grid_size       = 2048
image_size      = round(data.compute_image_size(grid_size), 3)

# option 3, choose both grid_size and image_size
#           filter baselines to match parameters
#grid_size       = 2048
#image_size      = 0.05
#data.filter_baselines(grid_size, image_size)

# get remaining parameters
cell_size       = image_size / grid_size
nr_stations     = data.get_nr_stations()
nr_baselines    = data.get_nr_baselines()

######################################################################
# print parameters
######################################################################
print "nr_stations           = ", nr_stations
print "nr_baselines          = ", nr_baselines
print "nr_channels           = ", nr_channels
print "nr_timesteps          = ", nr_timesteps
print "nr_timeslots          = ", nr_timeslots
print "nr_correlations       = ", nr_correlations
print "subgrid_size          = ", subgrid_size
print "grid_size             = ", grid_size
print "image_size            = ", image_size
print "kernel_size           = ", kernel_size
print "integration_time      = ", integration_time

######################################################################
# initialize data
######################################################################
channel_offset  = 0
baseline_offset = 0
time_offset     = 0
uvw             = numpy.zeros((nr_baselines, nr_timesteps), dtype=idg.uvwtype)
frequencies     = numpy.zeros((nr_channels), dtype=idg.frequenciestype)
data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
data.get_uvw(uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time)

######################################################################
# plot data
######################################################################
util.plot_uvw_meters(uvw)
util.plot_uvw_pixels(uvw, frequencies, image_size)
plt.show()
