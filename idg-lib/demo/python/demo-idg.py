#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pyrap.tables
import signal
import argparse
import time
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
grid = idg.utils.get_zero_grid(nr_polarizations, grid_size,
                               dtype=np.complex64)
aterms = idg.utils.get_identity_aterms(nr_timeslots, nr_stations,
                                       subgrid_size, nr_polarizations)
aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots, nr_time)

# Initialize spheroidal
# spheroidal = idg.utils.get_example_spheroidal(subgrid_size, dtype=np.float64)
# spheroidal_grid = idg.fft.resize2f_r2r(spheroidal, grid_size, grid_size)

# Dummy spheroidal
spheroidal = np.ones(shape=(subgrid_size, subgrid_size), dtype=np.float32)
spheroidal_grid = np.ones(shape=(grid_size, grid_size), dtype=np.float32)

# Inialize wavenumbers
wavelengths = np.array(sc.speed_of_light / frequencies, dtype=np.float32)
wavenumbers = np.array(2*np.pi / wavelengths, dtype=np.float32)


######################################################################
# Initialize proxy
######################################################################
proxy = idg.CPU.Optimized(
    nr_stations, nr_channels,
    nr_time, nr_timeslots,
    image_size, grid_size, subgrid_size)


######################################################################
# Process entire measurementset
######################################################################
nr_rows = table.nrows()
nr_rows_read = 0
nr_rows_per_batch = (nr_baselines + nr_stations) * nr_time
nr_rows_to_process = min( int( nr_rows * percentage / 100. ), nr_rows)

iteration = 0
while (nr_rows_read + nr_rows_per_batch) < nr_rows_to_process:
    time_total = -time.time()

    # Initialize empty buffers
    uvw          = np.zeros(shape=(nr_baselines, nr_time),
                            dtype=idg.uvwtype)
    visibilities = np.zeros(shape=(nr_baselines, nr_time, nr_channels,
                                   nr_polarizations),
                            dtype=idg.visibilitiestype)
    baselines    = np.zeros(shape=(nr_baselines),
                            dtype=idg.baselinetype)

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
                                newshape=(nr_time, nr_baselines))
    antenna2_block = np.reshape(antenna2_block,
                                newshape=(nr_time, nr_baselines))
    uvw_block = np.reshape(uvw_block,
                           newshape=(nr_time, nr_baselines, 3))
    vis_block = np.reshape(vis_block,
                           newshape=(nr_time, nr_baselines,
                                     nr_channels, nr_polarizations))

    # Transpose data
    for t in range(nr_time):
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

    proxy.grid_visibilities(
        visibilities,
        uvw,
        wavenumbers,
        baselines,
        grid,
        w_offset,
        kernel_size,
        aterms,
        aterms_offset,
        spheroidal)

    time_gridding += time.time()

    # Compute fft over grid
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
    print "Runtime transpose: %5d ms (%5.2f %%)" % (time_transpose*1000, 100.0 * time_transpose/time_total)
    print "Runtime gridding:  %5d ms (%5.2f %%)" % (time_gridding*1000,  100.0 * time_gridding/time_total)
    print "Runtime fft:       %5d ms (%5.2f %%)" % (time_fft*1000,       100.0 * time_fft/time_total)
    print "Runtime plot:      %5d ms (%5.2f %%)" % (time_plot*1000,      100.0 * time_plot/time_total)
    print ""
    iteration += 1

# TODO: need to process the remaining visibilities

# Do not close window at the end?
# plt.show()
