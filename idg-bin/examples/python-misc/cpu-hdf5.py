#!/usr/bin/env python
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

#########
# imports
#########
import argparse
import h5py

from common import *

######################################################################
# command line argument parsing
######################################################################
parser = argparse.ArgumentParser(description='Run image domain gridding on a hdf5 dataset')
parser.add_argument(dest='filename', type=str, help='path to dataset')
args = parser.parse_args()
filename = args.filename

##############
# read dataset
##############
hf = h5py.File(filename, 'r')
data = hf.get('vis')[0]

# extract data columns
uv       = data['uv']
sub_uv   = data['sub_uv']
weights  = data['weights']
vis      = data['vis']
w_planes = data['w_plane']
nr_w_planes = len(w_planes)


############
# paramaters
############
nr_stations      = 2
nr_baselines     = 1
nr_channels      = 1
nr_time          = len(uv[0])
nr_timeslots     = 1
image_size       = 0.5
subgrid_size     = 24
grid_size        = 512
kernel_size      = (subgrid_size / 2) + 1
nr_correlations = 4

# dummy data
frequencies = idg.utils.get_example_frequencies(nr_channels)
baselines = numpy.zeros(shape=(nr_baselines), dtype=idg.baselinetype)
baselines[0]['station1'] = 0
baselines[0]['station2'] = 1
grid = numpy.zeros(shape=(nr_correlations, grid_size, grid_size), dtype=numpy.complex64)
aterms = idg.utils.get_identity_aterms(nr_timeslots, nr_stations, subgrid_size, nr_correlations)
aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots, nr_time)
spheroidal = idg.utils.get_identity_spheroidal(subgrid_size)
visibilities = numpy.ones(shape=(nr_baselines, nr_time, nr_channels, nr_correlations), dtype=idg.visibilitiestype)



######################################################################
# initialize proxy
######################################################################
proxyname = idg.CPU.Reference
p = proxyname(nr_stations, nr_channels,
              nr_time, nr_timeslots,
              image_size, grid_size,
              subgrid_size)


######################################################################
# print parameters
######################################################################
print "nr_stations           = ", p.get_nr_stations()
print "nr_baselines          = ", p.get_nr_baselines()
print "nr_channels           = ", p.get_nr_channels()
print "nr_timesteps          = ", p.get_nr_time()
print "nr_timeslots          = ", p.get_nr_timeslots()
print "nr_correlations      = ", p.get_nr_correlations()
print "subgrid_size          = ", p.get_subgrid_size()
print "grid_size             = ", p.get_grid_size()
print "image_size            = ", p.get_image_size()
print "kernel_size           = ", kernel_size
print "job size (gridding)   = ", p.get_job_size_gridding()
print "job size (degridding) = ", p.get_job_size_degridding()


######
# main
######

# image for all planes
image = numpy.zeros(shape=(nr_correlations, grid_size, grid_size), dtype=numpy.complex64)

for plane in range(len(uv)):
    # reset grid
    grid = numpy.zeros(shape=(nr_correlations, grid_size, grid_size), dtype=numpy.complex64)

    # get data for current plane
    uv_plane     = uv[plane]
    sub_uv_plane = sub_uv[plane]
    weight_plane = weights[plane]
    vis_plane    = vis[plane]
    w_plane      = w_planes[plane]

    # fill idg datastructures
    uvw = numpy.zeros(shape=(nr_baselines, nr_time, 3), dtype=numpy.float32)
    uvw[:,:,0] = uv_plane[:,0]
    uvw[:,:,1] = uv_plane[:,1]
    uvw[:,:,2] = w_plane
    uvw = uvw.view(idg.uvwtype)[:,:,0]
    visibilities = numpy.repeat(vis_plane, nr_correlations, axis=0).reshape((nr_baselines, nr_time, nr_channels, nr_correlations))

    # grid visibilities
    w_offset = float(plane)
    p.grid_visibilities(
        visibilities, uvw, frequencies, baselines, grid,
        w_offset, kernel_size, aterms, aterms_offset, spheroidal)

    # create image
    p.transform(idg.FourierDomainToImageDomain, grid)

    # add image to master image
    image += grid

# show final image
idg.utils.plot_grid(image, pol=0)
plt.show()
