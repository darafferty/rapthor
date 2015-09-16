#!/usr/bin/env python

import numpy
import time
import idg
import pyrap.tables


class BaselineBuffer :
  def __init__(self, N_timesteps) :
    self.count = 0
    self.N_timesteps = N_timesteps
    
  def append(self, row):
    self.count += 1
    if self.count == self.N_timesteps:
      return True

class SubGridBuffer :
  
  def __init__(self, N_ant) :
    self.baselinebuffers = numpy.zeros((N_ant, N_ant), dtype = object)
    for i in range(N_ant):
      for j in range(N_ant):
        self.baselinebuffers[i,j] = BaselineBuffer(N_timesteps)
  
  def append(self, row):
    baselinebuffer = self.baselinebuffers[row['ANTENNA1'], row['ANTENNA2']]
    if baselinebuffer.append(row) :
      self.append_subgrid(baselinebuffer)
      baselinebuffer.clear()
      
  def append_subgrid(self, subgrid):
    pass
  
  def flush():
    pass
  
p = idg.Parameters()

p.timeslots = 1

p.print0()

#proxy = idg.CPU(p)


w_offset = 0.0

nr_subgrids = 1000
jobsize = 1000

uvw = numpy.zeros((nr_subgrids, p.nr_timesteps, 3), dtype = float)
wavenumbers =  numpy.zeros((p.nr_channels), dtype = float)

visibilities =  numpy.zeros((nr_subgrids, p.nr_timesteps, p.nr_channels, p.nr_polarizations), dtype = numpy.complex64)
visibilities[:,:,:,0] = 1.0
visibilities[:,:,:,3] = 1.0

spheroidal = numpy.ones((p.subgrid_size, p.subgrid_size), dtype = float)


aterm = numpy.zeros((p.nr_timeslots, p.nr_stations, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

aterm[:,:,0,:,:] = 1.0
aterm[:,:,3,:,:] = 1.0

subgrids = numpy.zeros((nr_subgrids, p.nr_polarizations, p.subgrid_size, p.subgrid_size), dtype = numpy.complex64)

baselinetype = numpy.dtype([('station1', int), ('station2', int)])
coordinatetype = numpy.dtype([('x', int), ('y', int)])

metadatatype = numpy.dtype([ ('time_nr', int), ('baseline', baselinetype), ('coordinate', coordinatetype)])

metadata = numpy.zeros(nr_subgrids, dtype=metadatatype)


t = pyrap.tables.taql('SELECT TIME, ANTENNA1, ANTENNA2 FROM /home/vdtol/cep1home/imagtest2/COV.beam_on.one.MS')
t_ant = pyrap.tables.table(t.getkeyword("ANTENNA"))
N_ant = len(t_ant)

subgridbuffer = SubGridBuffer(N_ant)

j = 0
k = 0

N = 10000

t0 = time.time()

rowtype = numpy.dtype([('TIME', float), ('ANTENNA1', int), ('ANTENNA2', int), ('UVW', complex, (10,4))])

block = numpy.zeros(N, dtype = rowtype)

for i in range(t.nrows()):
  if (j == 0):
    block = block[0:min(t.nrows() - i, N)]
    block[:]['TIME'] = t.getcol('TIME', startrow = i, nrow = N)
    block[:]['ANTENNA1'] = t.getcol('ANTENNA1', startrow = i, nrow = N)
    block[:]['ANTENNA2'] = t.getcol('ANTENNA2', startrow = i, nrow = N)
  subgridbuffer.append(block[j])
  j += 1
  if j == N:
    j = 0
    k += 1
t1 = time.time()
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
