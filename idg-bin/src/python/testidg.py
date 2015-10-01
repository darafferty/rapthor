#!/usr/bin/env python

import numpy
import time
import idg
import pyrap.tables
import matplotlib.pyplot as plt

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

class DataBuffer :
  
  def __init__(self, parameters, nr_subgrids, freqs, proxy) :
    self.parameters = parameters
    self.proxy = proxy
    self.count = 0
    self.nr_subgrids = nr_subgrids
    
    N_ant = parameters.nr_stations
    N_timesteps = parameters.nr_timesteps
    
    self.uvw = numpy.zeros((nr_subgrids, p.nr_timesteps, 3), dtype = numpy.float32)
    
    speed_of_light = 299792458.0
    self.wavelengths = numpy.array(speed_of_light / freqs, dtype=numpy.float32)
    self.wavenumbers = numpy.array(2*numpy.pi / self.wavelengths, dtype=numpy.float32)

    self.visibilities =  numpy.zeros((nr_subgrids, parameters.nr_timesteps, parameters.nr_channels, parameters.nr_polarizations), dtype = numpy.complex64)


    self.aterm = numpy.zeros((p.nr_stations, p.nr_timeslots, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

    self.aterm[:,:,0,:,:] = 1.0
    self.aterm[:,:,3,:,:] = 1.0
    
    #self.spheroidal = numpy.ones((p.subgrid_size, p.subgrid_size), dtype = numpy.float32)
    x = numpy.array([func_spheroidal(abs(a)) for a in 2*numpy.arange(p.subgrid_size, dtype=numpy.float32) / (p.subgrid_size-1) - 1.0], dtype = numpy.float32)
    self.spheroidal = x[numpy.newaxis,:] * x[:, numpy.newaxis]
    
    s = numpy.fft.fft2(self.spheroidal)
    s = numpy.fft.fftshift(s)
    s1 = numpy.zeros((p.grid_size, p.grid_size), dtype = numpy.complex64)
    s1[(p.grid_size-p.subgrid_size)/2:(p.grid_size+p.subgrid_size)/2, (p.grid_size-p.subgrid_size)/2:(p.grid_size+p.subgrid_size)/2] = s
    s1 = numpy.fft.ifftshift(s1)
    self.spheroidal1 = numpy.real(numpy.fft.ifft2(s1))


    self.subgrids = numpy.zeros((nr_subgrids, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

    self.grid = numpy.zeros((p.nr_polarizations, p.grid_size, p.grid_size), dtype = numpy.complex64)

    baselinetype = numpy.dtype([('station1', numpy.int32), ('station2', numpy.int32)])
    coordinatetype = numpy.dtype([('x', numpy.int32), ('y', numpy.int32)])

    metadatatype = numpy.dtype([ ('time_nr', numpy.int32), ('baseline', baselinetype), ('coordinate', coordinatetype)])

    self.metadata = numpy.zeros(nr_subgrids, dtype=metadatatype)


    self.baselinebuffers = numpy.zeros((N_ant, N_ant), dtype = object)
    for i in range(N_ant):
      for j in range(N_ant):
        self.baselinebuffers[i,j] = BaselineBuffer(i,j,parameters)
    
  def clear(self):
    self.grid[:] = 0
    
  def append(self, row):
    self.time = row['TIME']

    if row['ANTENNA1'] == row['ANTENNA2']:
      return
    baselinebuffer = self.baselinebuffers[row['ANTENNA1'], row['ANTENNA2']]
    if baselinebuffer.append(row) :
      r = self.append_subgrid(baselinebuffer)
      baselinebuffer.clear()
      return r
      
  def append_subgrid(self, baselinebuffer):
    u = numpy.mean(baselinebuffer.uvw[:,0])*numpy.mean(self.wavenumbers)/(2*numpy.pi)
    v = numpy.mean(baselinebuffer.uvw[:,1])*numpy.mean(self.wavenumbers)/(2*numpy.pi)
    coordinate_x = round(u * self.parameters.imagesize) + self.parameters.grid_size/2 - self.parameters.subgrid_size/2
    coordinate_y = round(v * self.parameters.imagesize) + self.parameters.grid_size/2 - self.parameters.subgrid_size/2
    
    if ((coordinate_x>=0) and 
      (coordinate_y>=0) and 
      ((coordinate_x+self.parameters.subgrid_size) < self.parameters.grid_size) and
      ((coordinate_y+self.parameters.subgrid_size) < self.parameters.grid_size)):
         
      self.uvw[self.count, :, :] = baselinebuffer.uvw
      self.visibilities[self.count, :, :, :] = baselinebuffer.visibilities
      self.metadata[self.count] = (0, (baselinebuffer.antenna1, baselinebuffer.antenna2), (coordinate_x,coordinate_y))
      
      self.count += 1
      if self.count == self.nr_subgrids:
        self.flush()
        return True
    
  
  def flush(self):
    jobsize = 10000
    w_offset = 0

    proxy.grid_onto_subgrids(
      jobsize, 
      self.count, 
      w_offset, 
      self.uvw, 
      self.wavenumbers, 
      self.visibilities, 
      self.spheroidal, 
      self.aterm, 
      self.metadata, 
      self.subgrids)
    
    proxy.add_subgrids_to_grid(
      jobsize,
      self.count,
      self.metadata,
      self.subgrids,
      self.grid)
    
    print "***"
    plt.ion()
    plt.figure(1, figsize=(20,10))
    plt.subplot(1,2,1)
    plt.cla()
    plt.imshow(numpy.log(numpy.abs(self.grid[0,:,:])), interpolation='nearest')
    plt.title("UV Data - tijd %2.2i:%2.2i" % (numpy.mod(int(self.time/3600 ),24), numpy.mod(int(self.time/60),60) ))
    plt.subplot(1,2,2)
    plt.cla()
    img = numpy.real(numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(databuffer.grid[0,:,:]))))
    img = img/self.spheroidal1
    img = img[int(self.parameters.grid_size*0.9):int(self.parameters.grid_size*0.1):-1,int(self.parameters.grid_size*0.9):int(self.parameters.grid_size*0.1):-1]
    m = numpy.amax(img)
    plt.imshow(img, interpolation='nearest', clim = (-0.01*m, 0.3*m), cmap=plt.get_cmap("YlGnBu_r"))
    plt.title("Radio kaart")
    plt.show()
    plt.draw()
    
    self.count = 0
    
  
p = idg.Parameters()

w_offset = 0.0

nr_subgrids = 3000

#t = pyrap.tables.table('/home/vdtol/cep1home/imagtest2/COV.beam_on.one.MS')
t = pyrap.tables.table('/data/scratch/vdtol/idg_test1/RX42_SB100-109.2ch10s.ms')
t_ant = pyrap.tables.table(t.getkeyword("ANTENNA"))
t_spw = pyrap.tables.table(t.getkeyword("SPECTRAL_WINDOW"))
freqs = t_spw[0]['CHAN_FREQ']
p.nr_stations = len(t_ant)
p.nr_channels = t[0]["CORRECTED_DATA"].shape[0]
p.nr_timesteps = 10
p.nr_timeslots = 1
p.imagesize = 0.12
p.grid_size = 1000
p.subgrid_size = 32
p.job_size = 10000

proxy = idg.HaswellEP(p)

databuffer = DataBuffer(p, nr_subgrids, freqs, proxy)



N = 1000

t0 = time.time()

rowtype = numpy.dtype([
  ('TIME', numpy.float32), 
  ('ANTENNA1', int), 
  ('ANTENNA2', int), 
  ('UVW', numpy.float32, (3,)),
  ('DATA', complex, (p.nr_channels, p.nr_polarizations))
])


while True:
  block = numpy.zeros(N, dtype = rowtype)
  j = 0
  k = 0
  for i in range(t.nrows()):
    if (j == 0):
      block = block[0:min(t.nrows() - i, N)]
      block[:]['TIME'] = t.getcol('TIME', startrow = i, nrow = N)
      block[:]['ANTENNA1'] = t.getcol('ANTENNA1', startrow = i, nrow = N)
      block[:]['ANTENNA2'] = t.getcol('ANTENNA2', startrow = i, nrow = N)
      block[:]['UVW'] = t.getcol('UVW', startrow = i, nrow = N)
      block[:]['DATA'] = t.getcol('CORRECTED_DATA', startrow = i, nrow = N) * -t.getcol('FLAG', startrow = i, nrow = N)
    databuffer.append(block[j])
    j += 1
    if j == N:
      j = 0
      k += 1
  t1 = time.time()
  databuffer.flush()
  time.sleep(30)
  databuffer.clear()
  



#for i in range(150,160):
  #plt.figure()
  #plt.imshow(numpy.fft.fftshift(abs(databuffer.subgrids[i,0,:,:])), interpolation='nearest')
#plt.show()



#proxy.grid_onto_subgrids(jobsize, nr_subgrids, w_offset, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids)


#typedef struct { float u, v, w; } UVW;
#typedef struct { int x, y; } Coordinate;
#typedef struct { int station1, station2; } Baseline;
#typedef struct { int time_nr; Baseline baseline; Coordinate coordinate; } Metadata;

#/*
    #Complex numbers
#*/
##define FLOAT_COMPLEX std::complex<float>

#/*
    #Datatypes
#*/
#typedef UVW UVWType[1][NR_TIMESTEPS];
#typedef FLOAT_COMPLEX VisibilitiesType[1][NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
#typedef float WavenumberType[NR_CHANNELS];
#typedef FLOAT_COMPLEX ATermType[NR_STATIONS][NR_TIMESLOTS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
#typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
#typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];
#typedef FLOAT_COMPLEX SubGridType[1][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
#typedef Metadata MetadataType[1];
