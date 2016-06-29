#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pyrap.tables
import signal
import argparse
import time
import idg
import IDG

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
parser.add_argument(dest='msin', nargs=1, type=str, help='path to measurement set')
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
frequencies = t_spw[0]['CHAN_FREQ']


######################################################################
# Parameters
######################################################################
nr_stations      = len(t_ant)
nr_baselines     = (nr_stations * (nr_stations - 1)) / 2
nr_channels      = table[0][datacolumn].shape[0]
nr_time          = 256
nr_timeslots     = 1
nr_polarizations = 4
grid_size        = 1024
subgrid_size     = 32
kernel_size      = 16


######################################################################
# Initialize data
######################################################################
grid = idg.utils.get_zero_grid(nr_polarizations, grid_size, dtype=np.complex128)
# aterms = idg.utils.get_example_aterms(nr_timeslots, nr_stations, subgrid_size, nr_polarizations, dtype=np.complex128)
# aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots, nr_time)

# Initialize spheroidal
# Option 1: example spherodail
# spheroidal = idg.utils.get_example_spheroidal(subgrid_size, dtype=np.float64)
# Option 2: identity
spheroidal = np.ones(shape=(subgrid_size, subgrid_size), dtype=np.float64)
spheroidal_grid = np.ones(shape=(grid_size, grid_size), dtype=np.float64)


######################################################################
# Initialize proxy
######################################################################
plan = IDG.GridderPlan(IDG.Type.CPU_OPTIMIZED, nr_time)
plan.set_stations(nr_stations);
plan.set_frequencies(frequencies);
plan.set_grid(grid);
plan.set_spheroidal(spheroidal);
plan.set_image_size(image_size);
plan.set_w_kernel_size(kernel_size);
plan.internal_set_subgrid_size(subgrid_size);
plan.bake();


######################################################################
# Process entire measurementset
######################################################################
nr_rows = table.nrows()
nr_rows_read = 0
nr_rows_per_batch = (nr_baselines + nr_stations) * nr_time
nr_rows_to_process = min( int( nr_rows * percentage / 100. ), nr_rows)

iteration = 0
timeIndex = -1
t_previous = -1
while (nr_rows_read + nr_rows_per_batch) < nr_rows_to_process:
    time_total = -time.time()

    time_read = -time.time()
    # Read nr_time samples for all baselines including auto correlations
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

    vis_block       = vis_block * -flags_block
    vis_block[np.isnan(vis_block)] = 0
    nr_rows_read += nr_rows_per_batch
    time_read += time.time()

    time_buffer = -time.time()

    # Change precision (necessary?)
    uvw_block = uvw_block.astype(np.float64)
    vis_block = vis_block.astype(np.complex64)

    # Fill buffer
    for row in range(nr_rows_per_batch):

        # Set time index
        if (timestamp_block[row] != t_previous):
            timeIndex += 1

        # Set antenna indices
        antenna1 = antenna1_block[row]
        antenna2 = antenna2_block[row]

        # Set uvw
        uvw_coordinate = uvw_block[row]

        # Set visibilities
        visibilities = vis_block[row]

        # Add visibilities to the buffer
        plan.grid_visibilities(
            timeIndex,
            antenna1,
            antenna2,
            uvw_coordinate,
            visibilities
        )

        t_previous = timestamp_block[row]

    time_buffer += time.time()
                    
    # Grid visibilities
    time_gridding = -time.time()
    plan.flush()
    time_gridding += time.time()

    # Compute fft over grid
    time_fft = -time.time()

    # Using numpy
    grid_copy = plan.get_copy_grid()
    plan.transform_grid(grid_copy)
    img = np.real(grid_copy[0,:,:])

    time_fft += time.time()

    time_plot = -time.time()

    # Set plot properties
    colormap=plt.get_cmap("hot")
    font_size = 16

    # Make first plot (raw grid)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_bgcolor(colormap(0))
    ax1.imshow(np.log(np.abs(grid[0,:,:]) + 1), interpolation='nearest')
    time1 = timestamp_block[0]
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
    print "Runtime buffer:    %5d ms (%5.2f %%)" % (time_buffer*1000, 100.0 * time_buffer/time_total)
    print "Runtime gridding:  %5d ms (%5.2f %%)" % (time_gridding*1000,  100.0 * time_gridding/time_total)
    print "Runtime fft:       %5d ms (%5.2f %%)" % (time_fft*1000,       100.0 * time_fft/time_total)
    print "Runtime plot:      %5d ms (%5.2f %%)" % (time_plot*1000,      100.0 * time_plot/time_total)
    print ""
    iteration += 1

# TODO: need to process the remaining visibilities

# Do not close window at the end?
# plt.show()
