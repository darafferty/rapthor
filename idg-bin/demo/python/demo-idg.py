#!/usr/bin/env python
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pyrap.tables
import signal
import argparse
import time
import idg
import util

# Enable interactive plotting and create figure to plot into
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

# Set signal handler to exit when ctrl-c is pressed
def signal_handler(signal, frame):
    exit()
signal.signal(signal.SIGINT, signal_handler)


######################################################################
# Command line argument parsing
######################################################################
parser = argparse.ArgumentParser(description='Run image domain gridding on a measurement set')
parser.add_argument(dest='msin', nargs=1, type=str,
                    help='path to measurement set')
parser.add_argument(dest='percentage',
                    nargs='?', type=int,
                    help='percentage of data to process',
                    default=100)
parser.add_argument('-c', '--column',
                    help='Data column used, such as DATA or CORRECTED_DATA (default: CORRECTED_DATA)',
                    required=False, default="CORRECTED_DATA")
parser.add_argument('--imagesize',
                    help='Image size (cell size / grid size)',
                    required=False, type=float, default=0.1)
parser.add_argument('--use-cuda',
                    help='Use CUDA proxy',
                    required=False, action='store_true')
args = parser.parse_args()
msin = args.msin[0]
percentage = args.percentage
image_size = args.imagesize
datacolumn = args.column
use_cuda   = args.use_cuda


######################################################################
# Open measurementset
######################################################################
table = pyrap.tables.table(msin)

# Read parameters from measurementset
t_ant = pyrap.tables.table(table.getkeyword("ANTENNA"))
t_spw = pyrap.tables.table(table.getkeyword("SPECTRAL_WINDOW"))
frequencies = np.asarray(t_spw[0]['CHAN_FREQ'], dtype=np.float32)


######################################################################
# Parameters
######################################################################
nr_stations      = len(t_ant)
nr_baselines     = (nr_stations * (nr_stations - 1)) / 2
nr_channels      = table[0][datacolumn].shape[0]
nr_timesteps     = 256
nr_timeslots     = 1
nr_correlations  = 4
grid_size        = 1024
subgrid_size     = 32
kernel_size      = 16
cell_size        = image_size / grid_size


######################################################################
# Initialize data
######################################################################
grid           = util.get_example_grid(nr_correlations, grid_size)
aterms         = util.get_identity_aterms(
                    nr_timeslots, nr_stations, subgrid_size, nr_correlations)
aterms_offsets = util.get_example_aterms_offset(
                    nr_timeslots, nr_timesteps)

# Initialize spheroidal
spheroidal = util.get_example_spheroidal(subgrid_size)
spheroidal_grid = util.get_identity_spheroidal(grid_size)

######################################################################
# Initialize proxy
######################################################################
if use_cuda:
    proxy = idg.CUDA.Generic(nr_correlations, subgrid_size)
else:
    proxy = idg.CPU.Optimized(nr_correlations, subgrid_size)


######################################################################
# Process entire measurementset
######################################################################
nr_rows = table.nrows()
nr_rows_read = 0
nr_rows_per_batch = (nr_baselines + nr_stations) * nr_timesteps
nr_rows_to_process = min( int( nr_rows * percentage / 100. ), nr_rows)

# Initialize empty buffers
uvw          = np.zeros(shape=(nr_baselines, nr_timesteps),
                        dtype=idg.uvwtype)
visibilities = np.zeros(shape=(nr_baselines, nr_timesteps, nr_channels,
                               nr_correlations),
                        dtype=idg.visibilitiestype)
baselines    = np.zeros(shape=(nr_baselines),
                        dtype=idg.baselinetype)
img          = np.zeros(shape=(nr_correlations, grid_size, grid_size),
                        dtype=idg.gridtype)

iteration = 0
while (nr_rows_read + nr_rows_per_batch) < nr_rows_to_process:
    # Reset buffers
    uvw.fill(0)
    visibilities.fill(0)
    baselines.fill(0)

    # Start timing
    time_total = -time.time()
    time_read = -time.time()

    # Read nr_timesteps samples for all baselines including auto correlations
    timestamp_block = table.getcol('TIME',
                                   startrow = nr_rows_read,
                                   nrow = nr_rows_per_batch)
    antenna1_block  = table.getcol('ANTENNA1',
                                   startrow = nr_rows_read,
                                   nrow = nr_rows_per_batch)
    antenna2_block  = table.getcol('ANTENNA2',
                                   startrow = nr_rows_read,
                                   nrow = nr_rows_per_batch)
    uvw_block       = table.getcol('UVW',
                                   startrow = nr_rows_read,
                                   nrow = nr_rows_per_batch)
    vis_block       = table.getcol(datacolumn,
                                   startrow = nr_rows_read,
                                   nrow = nr_rows_per_batch)
    flags_block     = table.getcol('FLAG',
                                   startrow = nr_rows_read,
                                   nrow = nr_rows_per_batch)
    vis_block = vis_block * -flags_block
    vis_block[np.isnan(vis_block)] = 0

    nr_rows_read += nr_rows_per_batch
    time_read += time.time()

    time_transpose = -time.time()

    # Change precision
    uvw_block = uvw_block.astype(np.float32)
    vis_block = vis_block.astype(np.complex64)

    # Remove autocorrelations
    flags = antenna1_block != antenna2_block
    antenna1_block = antenna1_block[flags]
    antenna2_block = antenna2_block[flags]
    uvw_block      = uvw_block[flags]
    vis_block      = vis_block[flags]

    # Reshape data
    antenna1_block = np.reshape(antenna1_block,
                                newshape=(nr_timesteps, nr_baselines))
    antenna2_block = np.reshape(antenna2_block,
                                newshape=(nr_timesteps, nr_baselines))
    uvw_block = np.reshape(uvw_block,
                           newshape=(nr_timesteps, nr_baselines, 3))
    vis_block = np.reshape(vis_block,
                           newshape=(nr_timesteps, nr_baselines,
                                     nr_channels, nr_correlations))

    # Transpose data
    for t in range(nr_timesteps):
        for bl in range(nr_baselines):
            # Set baselines
            antenna1 = antenna1_block[t][bl]
            antenna2 = antenna2_block[t][bl]

            baselines[bl] = (antenna1, antenna2)

            # Set uvw
            uvw_ = uvw_block[t][bl]
            uvw[bl][t] = uvw_

            # Set visibilities
            visibilities[bl][t] = vis_block[t][bl]
    time_transpose += time.time()

    # Grid visibilities
    w_offset = 0.0
    time_gridding = -time.time()

    proxy.gridding(
        w_offset, cell_size, kernel_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

    time_gridding += time.time()

    # Compute fft over grid
    time_fft = -time.time()

    # Using fft from library
    np.copyto(img, grid)
    proxy.transform(idg.FourierDomainToImageDomain, img)
    img_real = np.real(img[0,:,:])
    time_fft += time.time()

    time_plot = -time.time()

    # Remove spheroidal from grid
    img_real = img_real/spheroidal_grid

    # Crop image
    img_crop = img_real[int(grid_size*0.1):int(grid_size*0.9),int(grid_size*0.1):int(grid_size*0.9)]

    # Set plot properties
    colormap_grid=plt.get_cmap('hot')
    colormap_img=plt.get_cmap('hot')
    font_size = 16

    # Make first plot (raw grid)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(np.log(np.abs(grid[0,:,:]) + 1), cmap=colormap_grid)
    time1 = timestamp_block[0]
    ax1.set_title("UV Data: %2.2i:%2.2i\n" % (np.mod(int(time1/3600 ),24), np.mod(int(time1/60),60)), fontsize=font_size)

    # Make second plot (processed grid)
    m = np.amax(img_crop)
    ax2.imshow(img_crop, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=colormap_img)
    ax2.set_title("Sky image\n", fontsize=font_size)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Draw figure
    plt.pause(0.01)

    time_plot += time.time()

    # Print timings
    time_total += time.time()
    print ">>> Iteration %d" % iteration
    print "Runtime total:     %5d ms"            % (time_total*1000)
    print "Runtime reading:   %5d ms (%5.2f %%)" % (time_read*1000,      100.0 * time_read/time_total)
    print "Runtime transpose: %5d ms (%5.2f %%)" % (time_transpose*1000, 100.0 * time_transpose/time_total)
    print "Runtime gridding:  %5d ms (%5.2f %%)" % (time_gridding*1000,  100.0 * time_gridding/time_total)
    print "Runtime fft:       %5d ms (%5.2f %%)" % (time_fft*1000,       100.0 * time_fft/time_total)
    print "Runtime plot:      %5d ms (%5.2f %%)" % (time_plot*1000,      100.0 * time_plot/time_total)
    print ""
    iteration += 1

    plt.show()

# Do not close window at the end?
plt.show(block=True)
