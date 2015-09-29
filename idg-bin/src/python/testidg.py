#!/usr/bin/env python

import numpy
import time
import idg
import pyrap.tables


class BaselineBuffer :
  def __init__(self, antenna1, antenna2, parameters) :
    self.antenna1 = antenna1
    self.antenna2 = antenna2
    self.count = 0
    self.uvw = numpy.zeros((parameters.nr_timesteps, 3), dtype = float32)
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
    
    self.uvw = numpy.zeros((nr_subgrids, p.nr_timesteps, 3), dtype = float32)
    
    speed_of_light = 299792458.0
    self.wavelengths = speed_of_light / freqs
    self.wavenumbers = 2*numpy.pi / self.wavelengths

    self.visibilities =  numpy.zeros((nr_subgrids, p.nr_timesteps, p.nr_channels, p.nr_polarizations), dtype = numpy.complex64)

    self.spheroidal = numpy.ones((p.subgrid_size, p.subgrid_size), dtype = float32)

    self.aterm = numpy.zeros((p.nr_stations, p.nr_timeslots, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

    self.aterm[:,:,0,:,:] = 1.0
    self.aterm[:,:,3,:,:] = 1.0

    self.subgrids = numpy.zeros((nr_subgrids, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

    baselinetype = numpy.dtype([('station1', numpy.int32), ('station2', numpy.int32)])
    coordinatetype = numpy.dtype([('x', numpy.int32), ('y', numpy.int32)])

    metadatatype = numpy.dtype([ ('time_nr', numpy.int32), ('baseline', baselinetype), ('coordinate', coordinatetype)])

    self.metadata = numpy.zeros(nr_subgrids, dtype=metadatatype)


    self.baselinebuffers = numpy.zeros((N_ant, N_ant), dtype = object)
    for i in range(N_ant):
      for j in range(N_ant):
        self.baselinebuffers[i,j] = BaselineBuffer(i,j,parameters)
  
  def append(self, row):
    baselinebuffer = self.baselinebuffers[row['ANTENNA1'], row['ANTENNA2']]
    if baselinebuffer.append(row) :
      self.append_subgrid(baselinebuffer)
      baselinebuffer.clear()
      
  def append_subgrid(self, baselinebuffer):
    u = numpy.mean(baselinebuffer.uvw[:,0])*numpy.mean(self.wavenumbers)/(2*numpy.pi)
    v = numpy.mean(baselinebuffer.uvw[:,1])*numpy.mean(self.wavenumbers)/(2*numpy.pi)
    coordinate_x = int(u * self.parameters.imagesize) + self.parameters.grid_size/2 - self.parameters.subgrid_size/2
    coordinate_y = int(v * self.parameters.imagesize) + self.parameters.grid_size/2 - self.parameters.subgrid_size/2
    
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
    self.count = 0
    
  
p = idg.Parameters()

w_offset = 0.0

nr_subgrids = 100000

#t = pyrap.tables.taql('SELECT TIME, ANTENNA1, ANTENNA2 FROM /home/vdtol/cep1home/imagtest2/COV.beam_on.one.MS')
t = pyrap.tables.table('/home/vdtol/cep1home/imagtest2/COV.beam_on.one.MS')
t_ant = pyrap.tables.table(t.getkeyword("ANTENNA"))
t_spw = pyrap.tables.table(t.getkeyword("SPECTRAL_WINDOW"))
freqs = t_spw[0]['CHAN_FREQ']
p.nr_stations = len(t_ant)
p.nr_channels = t[0]["CORRECTED_DATA"].shape[0]
p.nr_timesteps = 1
p.nr_timeslots = 1
p.imagesize = 0.1
p.grid_size = 1000
p.subgrid_size = 32
p.job_size = 10000

proxy = idg.CPU(p)

databuffer = DataBuffer(p, nr_subgrids, freqs, proxy)


j = 0
k = 0

N = 1000

t0 = time.time()

rowtype = numpy.dtype([
  ('TIME', float32), 
  ('ANTENNA1', int), 
  ('ANTENNA2', int), 
  ('UVW', float32, (3,)),
  ('DATA', complex, (p.nr_channels, p.nr_polarizations))
])

block = numpy.zeros(N, dtype = rowtype)

#for i in range(t.nrows()):
for i in range(10000):
  if (j == 0):
    block = block[0:min(t.nrows() - i, N)]
    block[:]['TIME'] = t.getcol('TIME', startrow = i, nrow = N)
    block[:]['ANTENNA1'] = t.getcol('ANTENNA1', startrow = i, nrow = N)
    block[:]['ANTENNA2'] = t.getcol('ANTENNA2', startrow = i, nrow = N)
    block[:]['UVW'] = t.getcol('UVW', startrow = i, nrow = N)
    block[:]['DATA'] = t.getcol('CORRECTED_DATA', startrow = i, nrow = N)
  databuffer.append(block[j])
  j += 1
  if j == N:
    j = 0
    k += 1
t1 = time.time()
databuffer.flush()
print t1-t0

  


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
