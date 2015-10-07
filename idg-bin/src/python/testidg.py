#!/usr/bin/env python

import numpy
import time
import idg
import pyrap.tables
import matplotlib.pyplot as plt
import signal
import argparse

# Parse command line arguments
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

# Class to store baseline data
class BaselineBuffer :
  def __init__(self, antenna1, antenna2, parameters) :
    self.antenna1 = antenna1
    self.antenna2 = antenna2
    self.count = 0
    self.uvw = numpy.zeros((parameters.nr_timesteps, 3), dtype = numpy.float32)
    self.visibilities =  numpy.zeros((parameters.nr_timesteps, p.nr_channels, p.nr_polarizations), dtype = numpy.complex64)
    self.N_timesteps = parameters.nr_timesteps

  def append(self, row):
    self.uvw[self.count, :] = row['UVW']
    self.visibilities[self.count, :, :] = row['DATA']
    self.count += 1
    if self.count == self.N_timesteps:
      return True

  def clear(self):
    self.count = 0

# Class to store misc data
class DataBuffer :
  def __init__(self, parameters, nr_subgrids, freqs, proxy) :
    self.data = []
    self.parameters = parameters
    self.proxy = proxy
    self.count = 0
    self.nr_subgrids = nr_subgrids
    N_ant = parameters.nr_stations
    N_timesteps = parameters.nr_timesteps

    # Initialize uvw to zero
    self.uvw = numpy.zeros((nr_subgrids, p.nr_timesteps, 3), dtype = numpy.float32)

    # Inialize wavenumbers
    speed_of_light = 299792458.0
    self.wavelengths = numpy.array(speed_of_light / freqs, dtype=numpy.float32)
    self.wavenumbers = numpy.array(2*numpy.pi / self.wavelengths, dtype=numpy.float32)

    # Initialize visibilities to zero
    self.visibilities =  numpy.zeros((nr_subgrids, parameters.nr_timesteps, parameters.nr_channels, parameters.nr_polarizations), dtype = numpy.complex64)

    # Initialize aterm to zero
    self.aterm = numpy.zeros((p.nr_stations, p.nr_timeslots, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

    # Set aterm to one (so that it has no effect during imaging)
    self.aterm[:,:,0,:,:] = 1.0
    self.aterm[:,:,3,:,:] = 1.0

    # Initialize spheroidal to ones (so that it has no effect during imaging)
    #self.spheroidal = numpy.ones((p.subgrid_size, p.subgrid_size), dtype = numpy.float32)

    # Initialize spheroidal
    x = numpy.array([func_spheroidal(abs(a)) for a in 2*numpy.arange(p.subgrid_size, dtype=numpy.float32) / (p.subgrid_size-1) - 1.0], dtype = numpy.float32)
    self.spheroidal = x[numpy.newaxis,:] * x[:, numpy.newaxis]
    s = numpy.fft.fft2(self.spheroidal)
    s = numpy.fft.fftshift(s)
    s1 = numpy.zeros((p.grid_size, p.grid_size), dtype = numpy.complex64)
    s1[(p.grid_size-p.subgrid_size)/2:(p.grid_size+p.subgrid_size)/2, (p.grid_size-p.subgrid_size)/2:(p.grid_size+p.subgrid_size)/2] = s
    s1 = numpy.fft.ifftshift(s1)
    self.spheroidal1 = numpy.real(numpy.fft.ifft2(s1))

    # Initialize subgrids to zero
    self.subgrids = numpy.zeros((nr_subgrids, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

    # Initialize grid to zero
    self.grid = numpy.zeros((p.nr_polarizations, p.grid_size, p.grid_size), dtype = numpy.complex64)

    # Initialize baselines
    baselinetype = numpy.dtype([('station1', numpy.int32), ('station2', numpy.int32)])

    # Initialize coordinates
    coordinatetype = numpy.dtype([('x', numpy.int32), ('y', numpy.int32)])

    # Initialize metadata to zero
    metadatatype = numpy.dtype([ ('time_nr', numpy.int32), ('baseline', baselinetype), ('coordinate', coordinatetype)])
    self.metadata = numpy.zeros(nr_subgrids, dtype=metadatatype)

    # Initialize baseline buffers
    self.baselinebuffers = numpy.zeros((N_ant, N_ant), dtype = object)
    for i in range(N_ant):
      for j in range(N_ant):
        self.baselinebuffers[i,j] = BaselineBuffer(i,j,parameters)


  def clear(self):
    self.grid[:] = 0
    for i in range(self.parameters.nr_stations):
      for j in range(self.parameters.nr_stations):
        self.baselinebuffers[i,j].clear()
    self.data = []

  def append(self, row):
    self.time = row['TIME']

    if row['ANTENNA1'] == row['ANTENNA2']:
      return
    baselinebuffer = self.baselinebuffers[row['ANTENNA1'], row['ANTENNA2']]
    if baselinebuffer.append(row):
      self.append_subgrid(baselinebuffer)
      baselinebuffer.clear()

  def append_subgrid(self, baselinebuffer):
    # Set u and v for middle channel
    u = numpy.mean(baselinebuffer.uvw[:,0])*numpy.mean(self.wavenumbers)/(2*numpy.pi)
    v = numpy.mean(baselinebuffer.uvw[:,1])*numpy.mean(self.wavenumbers)/(2*numpy.pi)

    # Compute x and y coordinate in grid
    coordinate_x = round(u * self.parameters.imagesize) + self.parameters.grid_size/2 - self.parameters.subgrid_size/2
    coordinate_y = round(v * self.parameters.imagesize) + self.parameters.grid_size/2 - self.parameters.subgrid_size/2

    # Check whether this subgrid will fit
    if ((coordinate_x>=0) and
      (coordinate_y>=0) and
      ((coordinate_x+self.parameters.subgrid_size) < self.parameters.grid_size) and
      ((coordinate_y+self.parameters.subgrid_size) < self.parameters.grid_size)):

      # Store data
      self.uvw[self.count, :, :] = baselinebuffer.uvw
      self.visibilities[self.count, :, :, :] = baselinebuffer.visibilities
      self.metadata[self.count] = (0, (baselinebuffer.antenna1, baselinebuffer.antenna2), (coordinate_x,coordinate_y))

      # Stop condition
      self.count += 1
      if self.count == self.nr_subgrids:
        self.flush()
        return True

  def flush(self):
    jobsize = 0
    w_offset = 0
    self.data.append((self.time, self.count, self.uvw.copy(), self.visibilities.copy(), self.metadata.copy()))
    self.count = 0

# Open measurementset
t = pyrap.tables.table(msin)

# Read parameters from measurementset
t_ant = pyrap.tables.table(t.getkeyword("ANTENNA"))
t_spw = pyrap.tables.table(t.getkeyword("SPECTRAL_WINDOW"))
freqs = t_spw[0]['CHAN_FREQ']

# Set parameters
w_offset = 0.0
nr_subgrids = 500
jobsize = 100
w_offset = 0

# Initialize parameters
p = idg.Parameters()
p.nr_stations = len(t_ant)
p.nr_channels = t[0]["CORRECTED_DATA"].shape[0]
p.nr_timesteps = 16
p.nr_timeslots = 1
p.imagesize = 0.12
p.grid_size = 1000
p.subgrid_size = 32
p.job_size = jobsize

# Initialize proxy
proxy = idg.HaswellEP(p)

# Initialize databuffer
databuffer = DataBuffer(p, nr_subgrids, freqs, proxy)

# Number of samples to read in a single block
N = 40000

# Data row description
rowtype = numpy.dtype([
  ('TIME', numpy.float32),
  ('ANTENNA1', int),
  ('ANTENNA2', int),
  ('UVW', numpy.float32, (3,)),
  ('DATA', complex, (p.nr_channels, p.nr_polarizations))
])

# Read measurementset one block at a time
block = numpy.zeros(N, dtype = rowtype)
nr_rows = int(t.nrows() * (percentage/100.))
for i in range(0, nr_rows, N):
    print("Reading data: %.1f %%" % (float(i) / nr_rows * 100))
    block[:]['TIME'] = t.getcol('TIME', startrow = i, nrow = N)
    block[:]['ANTENNA1'] = t.getcol('ANTENNA1', startrow = i, nrow = N)
    block[:]['ANTENNA2'] = t.getcol('ANTENNA2', startrow = i, nrow = N)
    block[:]['UVW'] = t.getcol('UVW', startrow = i, nrow = N)
    block[:]['DATA'] = t.getcol('CORRECTED_DATA', startrow = i, nrow = N) * -t.getcol('FLAG', startrow = i, nrow = N)
    for j in range(N):
        databuffer.append(block[j])
databuffer.flush()

# Enable interactive plotting
plt.ion()

# Repeat forever
while True:
  frame_number = 0
  for time1, count, uvw, visibilities, metadata in databuffer.data:
    # Print progress
    print "Imaging: %.1f%%"  %(float(frame_number) / len(databuffer.data) * 100)

    # Grid visibilities onto subgrids
    proxy.grid_onto_subgrids(
      jobsize,
      count,
      w_offset,
      uvw,
      databuffer.wavenumbers,
      visibilities,
      databuffer.spheroidal,
      databuffer.aterm,
      metadata,
      databuffer.subgrids)

    # Add subgrids to grid
    proxy.add_subgrids_to_grid(
      jobsize,
      count,
      metadata,
      databuffer.subgrids,
      databuffer.grid)

    # Compute fft over grid
    # Using numpy#
    #img = numpy.real(numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(databuffer.grid[0,:,:]))))

    # Using fft from library (not working)
    #databuffer.grid = numpy.zeros((p.nr_polarizations, p.grid_size, p.grid_size), dtype = numpy.complex64)
    #databuffer.grid[0,500,400]=100
    img_ref = numpy.copy(databuffer.grid[0,:,:])
    img_ref = numpy.real(numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img_ref))))
    img = numpy.copy(databuffer.grid[:,:,:])
    for i in range(4):
        img[i,:,:] = numpy.fft.fftshift(img[i,:,:])
    proxy.transform(0, img)
    for i in range(4):
        img[i,:,:] = numpy.fft.fftshift(img[i,:,:])
    img = numpy.real(img[0,:,:])

    # Remove spheroidal from grid
    img = img/databuffer.spheroidal1
    img_ref = img_ref/databuffer.spheroidal1

    # Crop image
    img = img[int(databuffer.parameters.grid_size*0.9):int(databuffer.parameters.grid_size*0.1):-1,int(databuffer.parameters.grid_size*0.9):int(databuffer.parameters.grid_size*0.1):-1]
    img_ref = img_ref[int(databuffer.parameters.grid_size*0.9):int(databuffer.parameters.grid_size*0.1):-1,int(databuffer.parameters.grid_size*0.9):int(databuffer.parameters.grid_size*0.1):-1]

    # Set plot properties
    colormap=plt.get_cmap("YlGnBu_r")
    font_size = 22

    # Make first plot (raw grid)
    plt.figure(1, figsize=(30,15))
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.cla()
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_bgcolor(colormap(0))
    plt.imshow(numpy.log(numpy.abs(databuffer.grid[0,:,:])), interpolation='nearest')
    plt.title("UV Data: %2.2i:%2.2i\n" % (numpy.mod(int(time1/3600 ),24), numpy.mod(int(time1/60),60) ))
    ax.title.set_fontsize(font_size)

    # Make second plot (processed grid)
    plt.subplot(1,3,2)
    plt.cla()
    m = numpy.amax(img)
    print("    max: %f" % m)
    plt.imshow(img, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=colormap)
    plt.title("Sky image\n")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_fontsize(font_size)

    # Make third plot (processed grid)
    plt.subplot(1,3,3)
    plt.cla()
    m = numpy.amax(img_ref)
    print("ref max: %f" % m)
    plt.imshow(img_ref, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=colormap)
    plt.title("Sky image reference\n")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_fontsize(font_size)

    # Draw figure
    plt.show()
    plt.draw()
    frame_number += 1

  # Wait for some time
  time.sleep(30)

  # Reset grid
  databuffer.grid[:] = 0
