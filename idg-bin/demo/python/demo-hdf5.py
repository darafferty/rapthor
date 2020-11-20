#!/usr/bin/env python
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import signal
import argparse
import time
import h5py
import idg

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
use_cuda   = args.use_cuda


######################################################################
# Open measurementset
######################################################################
f = h5py.File(msin,  "r")
data = f['data']
nr_antennas     = data.attrs['NR_ANTENNAS']
nr_baselines_w  = data.attrs['NR_BASELINES']
nr_baselines    = ((nr_antennas - 1) * nr_antennas ) / 2   # without autocorrelations
nr_timesteps    = data.attrs['NR_TIMESTEPS']
nr_channels     = data.attrs['NR_CHANNELS']
nr_correlations = data.attrs['NR_CORRELATIONS']
frequencies     = data['FREQUENCIES'][()]

######################################################################
# Parameters
######################################################################
nr_timeslots       = 1
buffered_timesteps = 256
grid_size          = 1024
subgrid_size       = 32
kernel_size        = 16
cell_size          = image_size / grid_size

######################################################################
# Initialize data
######################################################################
grid           = util.get_example_grid(nr_correlations, grid_size)
aterms         = util.get_identity_aterms(
                    nr_timeslots, nr_stations, subgrid_size, nr_correlations)
aterms_offsets = util.get_identity_aterms_offset(
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
# TODO: do not read antenna ID and timestamps in every iteration
# TODO: for now, only multiple of buffered_timesteps
num_iter = nr_timesteps / buffered_timesteps

for iteration in np.arange(num_iter):
    time_total = -time.time()

    start_index = iteration * buffered_timesteps
    end_index   = start_index + buffered_timesteps

    ### READ THE DATA
    time_read = -time.time()

    timestamps   = data['TIME'][start_index:end_index,:]
    antenna1     = data['ANTENNA1'][start_index:end_index,:]
    antenna2     = data['ANTENNA2'][start_index:end_index,:]
    uvw          = data['UVW'][start_index:end_index,:,:]
    visibilities = data['DATA'][start_index:end_index,:,:,:]
    flags        = data['FLAG'][start_index:end_index,:,:,:]

    time_read += time.time()

    ### TRANSPOSE DATA, AND OTHER UNNECESSARY STUFF
    # TODO: make this cleaner
    time_transpose = -time.time()

    # apply flags
    visibilities[flags] = 0

    # convert to float
    uvw = uvw.astype(np.float32)

    # Remove autocorrelations
    mask = antenna1 != antenna2
    antenna1     = antenna1[mask].copy()
    antenna2     = antenna2[mask].copy()
    uvw          = uvw[mask,:].copy()
    uvw          = uvw.reshape((buffered_timesteps, nr_baselines, 3))
    visibilities = visibilities[mask, :, :].copy()
    visibilities = visibilities.reshape((buffered_timesteps, nr_baselines,
                                         nr_channels, nr_correlations))

    # Construct baselines array
    baselines    = np.zeros(shape=(nr_baselines),
                            dtype=idg.baselinetype)
    for k in range(nr_baselines):
        baselines[k] = (antenna1[k], antenna2[k])

    # transpose
    uvw          = uvw.transpose(1,0,2).copy()
    visibilities = visibilities.transpose(1,0,2,3).copy()

    # make view of uvw data type
    uvw = uvw.view(idg.uvwtype)[:,:,0]

    time_transpose += time.time()


    ### Grid visibilities
    time_gridding = -time.time()

    w_offset = 0.0

    proxy.gridding(
        w_offset, cell_size, kernel_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

    time_gridding += time.time()

    ### Compute fft over grid
    time_fft = -time.time()

    # Using fft from library
    img = grid.copy()
    proxy.transform(idg.FourierDomainToImageDomain, img)
    img = np.real(img[0,:,:])
    time_fft += time.time()

    time_plot = -time.time()

    # Remove spheroidal from grid
    img = img/spheroidal_grid

    # Crop image
    # img = img[int(grid_size*0.1):int(grid_size*0.9),int(grid_size*0.1):int(grid_size*0.9)]

    # Set plot properties
    colormap=plt.get_cmap("hot")
    font_size = 16

    # Make first plot (raw grid)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_bgcolor(colormap(0))
    ax1.imshow(np.log(np.abs(grid[0,:,:]) + 1), interpolation='nearest')
    time1 = timestamps[0][0]
    ax1.set_title("UV Data: %2.2i:%2.2i\n" % (np.mod(int(time1/3600 ),24), np.mod(int(time1/60),60)), fontsize=font_size)

    # Make second plot (processed grid)
    m = np.amax(img)
    ax2.imshow(img, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=colormap)
    ax2.set_title("Sky image\n", fontsize=font_size)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Draw figure
    plt.pause(0.05)

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

# Do not close window at the end?
plt.show(block=True)

######################################################################
# Clean up
######################################################################
f.close()
