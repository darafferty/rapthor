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
import threading

# Enable interactive plotting and create figure to plot into
plt.ion()
fig, (axes) = plt.subplots(2, 2, figsize=(20,10))
ax1 = axes[0][0]
ax2 = axes[0][1]
ax3 = axes[1][0]
ax4 = axes[1][1]

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
args = parser.parse_args()
msin = args.msin[0]
percentage = args.percentage
image_size = args.imagesize
datacolumn = args.column


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
nr_timesteps     = 128
nr_timeslots     = 1
nr_correlations  = 4
grid_size        = 1024
subgrid_size     = 32
kernel_size      = 16
cell_size        = image_size / grid_size


######################################################################
# Plot properties
######################################################################
colormap_grid   = plt.get_cmap('hot')
colormap_img    = plt.get_cmap('hot')
font_size       = 16


######################################################################
# Initialize data
######################################################################
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
proxy1 = idg.CPU.Optimized(nr_correlations, subgrid_size)
proxy2 = idg.CUDA.Generic(nr_correlations, subgrid_size)


######################################################################
# Prepare data
######################################################################
nr_rows = table.nrows()
nr_rows_read = 0
nr_rows_per_batch = (nr_baselines + nr_stations) * nr_timesteps
nr_rows_to_process = min( int( nr_rows * percentage / 100. ), nr_rows)

# Storage for all blocks of data
jobs_uvw          = list()
jobs_visibilities = list()
jobs_baselines    = list()

# Iterate all rows
while (nr_rows_read + nr_rows_per_batch) < nr_rows_to_process:
    # Info
    print "Prepare data for row %d - %d" % (nr_rows_read, nr_rows_read + nr_rows_per_batch)

    # Initialize empty buffers
    uvw          = np.zeros(shape=(nr_baselines, nr_timesteps),
                            dtype=idg.uvwtype)
    visibilities = np.zeros(shape=(nr_baselines, nr_timesteps, nr_channels,
                                   nr_correlations),
                            dtype=idg.visibilitiestype)
    baselines    = np.zeros(shape=(nr_baselines),
                        dtype=idg.baselinetype)

    # Start timing
    time_total = -time.time()

    # Read nr_timesteps samples for all baselines including auto correlations
    time_read = -time.time()
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

    # Store job
    jobs_uvw.append(uvw)
    jobs_visibilities.append(visibilities)
    jobs_baselines.append(baselines)

    # Update number of rows read for current job
    nr_rows_read += nr_rows_per_batch

    # Print timings
    time_total += time.time()
    print "Runtime reading:   %5d ms (%5.2f %%)" % (time_read*1000,      100.0 * time_read/time_total)
    print "Runtime transpose: %5d ms (%5.2f %%)" % (time_transpose*1000, 100.0 * time_transpose/time_total)

# Set total number of jobs
nr_jobs = len(jobs_baselines)


######################################################################
# Process entire measurementset
######################################################################
def plot_grid(axis, data, title=None):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.imshow(np.log(np.abs(data[0,:,:]) + 1), cmap=colormap_grid)
    if (title is not None):
        axis.set_title(title, fontsize=font_size)

def plot_image(axis, data, title=None):
    m = np.amax(data)
    axis.imshow(data, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=colormap_img)
    axis.set_title("Sky image\n", fontsize=font_size)
    axis.set_xticks([])
    axis.set_yticks([])
    if (title is not None):
        axis.set_title(title, fontsize=font_size)

class IDGThread(threading.Thread):
    def __init__(self, proxy, axis_grid, axis_image, title_image):
        threading.Thread.__init__(self)
        self.proxy = proxy
        self.axis_grid = axis_grid
        self.axis_image = axis_image
        self.title_image = title_image
        self.grid = util.get_example_grid(nr_correlations, grid_size)

    def run(self):
        nr_rows = table.nrows()
        nr_rows_read = 0
        nr_rows_per_batch = (nr_baselines + nr_stations) * nr_timesteps
        nr_rows_to_process = min( int( nr_rows * percentage / 100. ), nr_rows)

        # Initialize image
        img          = np.zeros(shape=(nr_correlations, grid_size, grid_size),
                                dtype=idg.gridtype)

        # Show initial plots
        grid = self.grid
        plot_grid(self.axis_grid, grid)
        plot_image(self.axis_image, np.real(img[0,:,:]), self.title_image)

        iteration = 0
        for job in range(nr_jobs):
            # Get data
            uvw          = jobs_uvw[job]
            visibilities = jobs_visibilities[job]
            baselines    = jobs_baselines[job]

            # Start timing
            time_total = -time.time()

            # Grid visibilities
            w_offset = 0.0
            time_gridding = -time.time()

            self.proxy.gridding(
                w_offset, cell_size, kernel_size, frequencies, visibilities,
                uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

            time_gridding += time.time()

            # Compute fft over grid
            time_fft = -time.time()

            # Using fft from library
            np.copyto(img, grid)
            self.proxy.transform(idg.FourierDomainToImageDomain, img)
            img_real = np.real(img[0,:,:])
            time_fft += time.time()

            time_plot = -time.time()

            # Remove spheroidal from grid
            img_real = img_real/spheroidal_grid

            # Crop image
            img_crop = img_real[int(grid_size*0.1):int(grid_size*0.9),int(grid_size*0.1):int(grid_size*0.9)]

            # Make first plot (raw grid)
            progress = int(float(job+1) / nr_jobs * 100.0)
            plot_grid(self.axis_grid, grid, "Imaging: %d%%" % (progress))

            # Make second plot (processed grid)
            plot_image(self.axis_image, img_crop, self.title_image)

            # Draw figure
            plt.pause(0.01)
            time_plot += time.time()

            # Print timings
            time_total += time.time()
            print "Runtime total:     %5d ms"            % (time_total*1000)
            print "Runtime gridding:  %5d ms (%5.2f %%)" % (time_gridding*1000,  100.0 * time_gridding/time_total)
            print "Runtime fft:       %5d ms (%5.2f %%)" % (time_fft*1000,       100.0 * time_fft/time_total)
            print "Runtime plot:      %5d ms (%5.2f %%)" % (time_plot*1000,      100.0 * time_plot/time_total)
            print ""

            plt.show()

while (True):
    thread1 = IDGThread(proxy1, ax1, ax2, "CPU")
    thread2 = IDGThread(proxy2, ax3, ax4, "GPU")

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    plt.pause(10)

# Do not close window at the end?
plt.show(block=True)
