#!/usr/bin/env python

import idg
import idg.util
from idg.data import Data
import numpy
import matplotlib.pyplot as plt

############
# paramaters
############
nr_channels      = 1
nr_timeslots     = 1
subgrid_size     = 32
kernel_size      = int((subgrid_size / 2) + 1)
nr_correlations  = 4
layout_file      = "LOFAR_hba.txt"
w_step           = 0.0


######################################################################
# initialize data generator
######################################################################

# Reduce number of timesteps
_integration_time = 0.9
time_factor = 100

for _nr_timesteps in [1*60*60, 6*60*60, 12*60*60]:
    integration_time = _integration_time * time_factor
    nr_timesteps = int(_nr_timesteps/time_factor)

    for grid_size in [8192, 16384]:
        for nr_stations in [20, 30]:
            nr_baselines = int((nr_stations * (nr_stations - 1)) / 2)

            ######################################################################
            # initialize data
            ######################################################################
            data = Data(layout_file)
            max_uv = data.compute_max_uv(grid_size) # m
            data.limit_max_baseline_length(max_uv)
            data.limit_nr_baselines(nr_baselines)
            image_size = round(data.compute_image_size(grid_size), 4)
            cell_size  = image_size / grid_size

            channel_offset  = 0
            baseline_offset = 0
            time_offset     = 0

            uvw            = numpy.zeros((nr_baselines, nr_timesteps), dtype=idg.uvwtype)
            frequencies    = numpy.zeros((nr_channels), dtype=idg.frequenciestype)
            data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
            data.get_uvw(uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time)
            baselines      = idg.util.get_example_baselines(nr_baselines)
            aterms_offsets = idg.util.get_example_aterms_offset(nr_timeslots, nr_timesteps)

            ######################################################################
            # create plan
            ######################################################################
            plan = idg.Plan(
                kernel_size, subgrid_size, grid_size, cell_size,
                frequencies, uvw, baselines, aterms_offsets)
            nr_subgrids = plan.get_nr_subgrids()
            metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
            plan.copy_metadata(metadata)

            ######################################################################
            # plot data
            ######################################################################
            tile_size = 256
            #percentage_used = idg.util.plot_tiles(uvw, frequencies, image_size, grid_size, tile_size)
            percentage_used = idg.util.plot_tiles(metadata, image_size, grid_size, tile_size)
            filename = "{}_{}_{}_{}_{}.png".format(nr_stations, grid_size, tile_size, nr_timesteps/3.6, percentage_used)
            plt.savefig(filename)
