#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
import pyrap.tables
import signal
import argparse
import time
import idg

######################################################################
# Command line argument parsing
######################################################################
parser = argparse.ArgumentParser(description='Run image domain gridding on a measurement set')
parser.add_argument(dest='msin', nargs=1, type=str, help='path to measurement set')
parser.add_argument(dest='percentage', nargs='?', type=int, help='percentage of data to process', default=100)
args = parser.parse_args()
msin = args.msin[0]
percentage = args.percentage

# Set signal handler to exit when ctrl-c is pressed
def signal_handler(signal, frame):
    exit()
signal.signal(signal.SIGINT, signal_handler)


######################################################################
# Utility functions
######################################################################
def plot_uvw(uvw):
    """Plot UVW data as (u,v)-plot
    Input:
    uvw - numpy.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    u = uvw.flatten()['u']
    v = uvw.flatten()['v']
    uvlim = 1.2*max(max(abs(u)), max(abs(v)))
    fig = plt.figure(get_figure_name("uvw"))
    plt.plot(numpy.append(u,-u),numpy.append(v,-v),'.')
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.grid(True)
    plt.axes().set_aspect('equal')



# Function to compute spheroidal
def func_spheroidal(nu):
  P = numpy.array([[ 8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1,  2.312756e-1],
              [ 4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
  Q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
              [1.0000000e0, 9.599102e-1, 2.918724e-1]])
  part = 0;
  end = 0.0;
  if (nu >= 0.0 and nu < 0.75):
    part = 0
    end = 0.75
  elif (nu >= 0.75 and nu <= 1.00):
    part = 1
    end = 1.00
  else:
    return 0.0

  nusq = nu * nu
  delnusq = nusq - end * end
  delnusqPow = delnusq
  top = P[part][0]
  for k in range(1,5):
    top += P[part][k] * delnusqPow
    delnusqPow *= delnusq

  bot = Q[part][0]
  delnusqPow = delnusq
  for k in range(1,3):
    bot += Q[part][k] * delnusqPow
    delnusqPow *= delnusq

  if bot == 0:
    result = 0
  else:
    result = (1.0 - nusq) * (top / bot)
  return result


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
nr_channels      = table[0]["CORRECTED_DATA"].shape[0]
nr_time          = 128
nr_timeslots     = 1
nr_polarizations = 4
image_size       = 0.05
grid_size        = 1024
subgrid_size     = 32
kernel_size      = 16


######################################################################
# Initialize data
######################################################################
grid          = idg.utils.get_zero_grid(nr_polarizations, grid_size, dtype=numpy.complex64)
aterms        = idg.utils.get_example_aterms(nr_timeslots, nr_stations, subgrid_size, nr_polarizations)
aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots, nr_time)

# Initialize spheroidal
# Real spheroidal
x = numpy.array([func_spheroidal(abs(a)) for a in 2*numpy.arange(subgrid_size, dtype=numpy.float32) / (subgrid_size-1) - 1.0], dtype = numpy.float32)
spheroidal = x[numpy.newaxis,:] * x[:, numpy.newaxis]
s = numpy.fft.fft2(spheroidal)
s = numpy.fft.fftshift(s)
s1 = numpy.zeros((grid_size, grid_size), dtype = numpy.complex64)
s1[(grid_size-subgrid_size)/2:(grid_size+subgrid_size)/2, (grid_size-subgrid_size)/2:(grid_size+subgrid_size)/2] = s
s1 = numpy.fft.ifftshift(s1)
spheroidal_grid = numpy.real(numpy.fft.ifft2(s1))

# Dummy spheroidal
#spheroidal = numpy.ones(shape=spheroidal.shape, dtype=numpy.float32)
#spheroidal_grid = numpy.ones(shape=spheroidal_grid.shape, dtype=numpy.float32)

# Inialize wavenumbers
speed_of_light = 299792458.0
wavelengths = numpy.array(speed_of_light / frequencies, dtype=numpy.float32)
wavenumbers = numpy.array(2*numpy.pi / wavelengths, dtype=numpy.float32)


######################################################################
# Initialize proxy
######################################################################
proxy = idg.CPU.HaswellEP(
    nr_stations, nr_channels,
    nr_time, nr_timeslots,
    image_size, grid_size, subgrid_size)


######################################################################
# Process entire measurementset
######################################################################
nr_rows = table.nrows()
nr_rows_read = 0
nr_rows_per_batch = (nr_baselines + nr_stations) * nr_time
iteration = 0
while (nr_rows_read + nr_rows_per_batch) < nr_rows:
    time_total = -time.time()

    # Enable interactive plotting
    plt.ion()

    # Initialize empty buffers
    uvw          = numpy.zeros(shape=(nr_baselines, nr_time), dtype=idg.uvwtype)
    visibilities = numpy.zeros(shape=(nr_baselines, nr_time, nr_channels, nr_polarizations), dtype=idg.visibilitiestype)
    baselines    = numpy.zeros(shape=(nr_baselines), dtype=idg.baselinetype)

    time_read = -time.time()
    # Read nr_time samples for all baselines including auto correlations
    timestamp_block = table.getcol('TIME',           startrow = nr_rows_read, nrow = nr_rows_per_batch)
    antenna1_block  = table.getcol('ANTENNA1',       startrow = nr_rows_read, nrow = nr_rows_per_batch)
    antenna2_block  = table.getcol('ANTENNA2',       startrow = nr_rows_read, nrow = nr_rows_per_batch)
    uvw_block       = table.getcol('UVW',            startrow = nr_rows_read, nrow = nr_rows_per_batch)
    vis_block       = table.getcol('CORRECTED_DATA', startrow = nr_rows_read, nrow = nr_rows_per_batch)
    flags_block     = table.getcol('FLAG',           startrow = nr_rows_read, nrow = nr_rows_per_batch)
    vis_block       = vis_block * -flags_block
    nr_rows_read += nr_rows_per_batch
    time_read += time.time()

    time_transpose = -time.time()
    # Change precision
    uvw_block = uvw_block.astype(numpy.float32)
    vis_block = vis_block.astype(numpy.complex64)

    # Remove autocorrelations
    flags = antenna1_block != antenna2_block
    antenna1_block = antenna1_block[flags]
    antenna2_block = antenna2_block[flags]
    uvw_block      = uvw_block[flags]
    vis_block      = vis_block[flags]

    # Reshape data
    antenna1_block = numpy.reshape(antenna1_block, newshape=(nr_time, nr_baselines))
    antenna2_block = numpy.reshape(antenna1_block, newshape=(nr_time, nr_baselines))
    uvw_block = numpy.reshape(uvw_block, newshape=(nr_time, nr_baselines, 3))
    vis_block = numpy.reshape(vis_block, newshape=(nr_time, nr_baselines, nr_channels, nr_polarizations))

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
    w_offset = 0
    kernel_size = (proxy.get_subgrid_size() / 2) + 1
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
    # Using numpy
    #img = numpy.real(numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(grid[0,:,:]))))

    # Using fft from library
    img = grid.copy()
    proxy.transform(idg.FourierDomainToImageDomain, img)
    img = numpy.real(img[0,:,:])
    time_fft += time.time()

    time_plot = -time.time()

    # Remove spheroidal from grid
    img = img/spheroidal_grid

    # Crop image
    img = img[int(grid_size*0.9):int(grid_size*0.1):-1,int(grid_size*0.9):int(grid_size*0.1):-1]


    # Set plot properties
    colormap=plt.get_cmap("hot")
    font_size = 22

    # Make first plot (raw grid)
    plt.figure(1, figsize=(20,10))
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.cla()
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_bgcolor(colormap(0))
    plt.imshow(numpy.log(numpy.abs(grid[0,:,:]) + 1), interpolation='nearest')
    time1 = timestamp_block[0]
    plt.title("UV Data: %2.2i:%2.2i\n" % (numpy.mod(int(time1/3600 ),24), numpy.mod(int(time1/60),60)))
    ax.title.set_fontsize(font_size)

    # Make second plot (processed grid)
    plt.subplot(1,2,2)
    plt.cla()
    m = numpy.amax(img)
    plt.imshow(img, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=colormap)
    plt.title("Sky image\n")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_fontsize(font_size)
    time_plot += time.time()

    # Draw figure
    plt.show()
    plt.draw()

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
