#!/usr/bin/env python

import numpy as np
import casacore.tables
import signal
import argparse
import time
import idg
import idg.util as util
import astropy.io.fits as fits
from idg.idgcalutils import next_composite, idgwindow

MS = "/var/scratch/maljaars/data/L258627_tsteps150_RStations_3.dppp"
IMAGENAME = "/home/maljaars/Documents/tests/idgcalstep/modelimage.fits"

# ms = "/var/scratch/maljaars/data/L258627_tsteps150_RStations_3.dppp"
# imagename = "/home/maljaars/Documents/tests/idgcalstep/modelimage.fits"

h = fits.getheader(IMAGENAME)

padding = 1.2

N0 = h["NAXIS1"]
N = next_composite(int(N0 * padding))

cell_size = abs(h["CDELT1"]) / 180 * np.pi
image_size = N * cell_size
print(N0, N, image_size)

datacolumn = "DATA"

######################################################################
# Open measurementset
######################################################################
table = casacore.tables.taql("SELECT * FROM $MS WHERE ANTENNA1 != ANTENNA2")

# Read parameters from measurementset
t_ant = casacore.tables.table(table.getkeyword("ANTENNA"))
t_spw = casacore.tables.table(table.getkeyword("SPECTRAL_WINDOW"))


######################################################################
# Parameters
######################################################################
nr_stations      = len(t_ant)
nr_baselines     = (nr_stations * (nr_stations - 1)) // 2
nr_channels      = table[0][datacolumn].shape[0]

nr_timesteps = 4

nr_timeslots     = 2
nr_correlations  = 4
subgrid_size     = 40

taper_support    = 7
wterm_support    = 7
aterm_support    = 11

kernel_size      = taper_support + wterm_support + aterm_support

nr_parameters_ampl = 6
nr_parameters_phase = 3
nr_parameters0 = nr_parameters_ampl + nr_parameters_phase
nr_parameters = nr_parameters_ampl + nr_parameters_phase*nr_timeslots

# solver_update_gain = 0.3

# use_fits = False

frequencies = np.asarray(t_spw[0]['CHAN_FREQ'], dtype=np.float32)[:nr_channels]

# Initialize empty buffers
uvw          = np.zeros(shape=(nr_baselines, nr_timesteps),
                        dtype=idg.uvwtype)
visibilities = np.zeros(shape=(nr_baselines, nr_timesteps, nr_channels,
                            nr_correlations),
                        dtype=idg.visibilitiestype)
weights      = np.zeros(shape=(nr_baselines, nr_timesteps, nr_channels,
                            nr_correlations),
                        dtype=np.float32)
baselines    = np.zeros(shape=(nr_baselines),
                        dtype=idg.baselinetype)

# Init proxy
proxy = idg.CPU.Optimized()

# Initialize taper
taper = idgwindow(subgrid_size, taper_support, padding)
taper2 = np.outer(taper, taper).astype(np.float32)

N0 = 1000
N = next_composite(int(N0 * padding))
grid_size = N

d = np.zeros(shape=(N0, N0), dtype=np.float32)

for x in range(200, N0-200, 200):
    for y in range(200, N0-200, 200):
        d[x,y] = 1.0

taper_ = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(taper)))
taper_grid = np.zeros(grid_size, dtype=np.complex128)
taper_grid[(grid_size-subgrid_size)//2:(grid_size+subgrid_size)//2] = taper_ * np.exp(-1j*np.linspace(-np.pi/2, np.pi/2, subgrid_size, endpoint=False))
taper_grid = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(taper_grid))).real*grid_size/subgrid_size
taper_grid0 = taper_grid[(N-N0)//2:(N+N0)//2]

img = np.zeros(shape=(nr_correlations, grid_size, grid_size), dtype=idg.gridtype)
img[0, (N-N0)//2:(N+N0)//2, (N-N0)//2:(N+N0)//2] = d[:,:]/np.outer(taper_grid0, taper_grid0)
img[3, (N-N0)//2:(N+N0)//2, (N-N0)//2:(N+N0)//2] = d[:,:]/np.outer(taper_grid0, taper_grid0)

# Initialize grid in proxy
grid = img.copy()
proxy.set_grid(grid)
proxy.transform(idg.ImageDomainToFourierDomain)

p = []

nr_timesteps_in_ms = len(casacore.tables.taql("SELECT UNIQUE TIME FROM $table"))

interval_start = 0

start_row = nr_baselines * interval_start
nr_rows = nr_baselines * nr_timesteps

# Read nr_timesteps samples for all baselines including auto correlations
timestamp_block = table.getcol('TIME',
                                startrow = start_row,
                                nrow = nr_rows)
antenna1_block  = table.getcol('ANTENNA1',
                                startrow = start_row,
                                nrow = nr_rows)
antenna2_block  = table.getcol('ANTENNA2',
                                startrow = start_row,
                                nrow = nr_rows)
uvw_block       = table.getcol('UVW',
                                startrow = start_row,
                                nrow = nr_rows)
vis_block       = table.getcol(datacolumn,
                                startrow = start_row,
                                nrow = nr_rows)[:,:nr_channels, :]
weight_block    = table.getcol("WEIGHT_SPECTRUM",
                                startrow = start_row,
                                nrow = nr_rows)[:,:nr_channels, :]
flags_block     = table.getcol('FLAG',
                                startrow = start_row,
                                nrow = nr_rows)[:,:nr_channels, :]

weight_block[:] = 1.0

weight_block = weight_block * ~flags_block
vis_block[np.isnan(vis_block)] = 0
rowid_block     = np.arange(nr_rows)

uvw_block[:,1:3] = -uvw_block[:,1:3]

print(start_row, nr_rows)

# Change precision
uvw_block = uvw_block.astype(np.float32)

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
weight_block = np.reshape(weight_block,
                        newshape=(nr_timesteps, nr_baselines,
                                    nr_channels, nr_correlations))
rowid_block = np.reshape(rowid_block,
                            newshape=(nr_timesteps, nr_baselines))

# Transpose data
for t in range(nr_timesteps):
    for bl in range(nr_baselines):
        # Set baselines
        antenna1 = antenna1_block[t][bl]
        antenna2 = antenna2_block[t][bl]

        baselines[bl] = (antenna1, antenna2)

        # Set uvw
        uvw_ = uvw_block[t][bl]
        uvw[bl][t]["u"] = uvw_[0]
        uvw[bl][t]["v"] = uvw_[1]
        uvw[bl][t]["w"] = uvw_[2]

        # Set visibilities
        visibilities[bl][t] = vis_block[t][bl]

        weights[bl][t] = weight_block[t][bl]


# Grid visibilities
w_step = 400.0
shift = np.array((0.0, 0.0, 0.0), dtype=np.float32)
aterms         = util.get_identity_aterms(
                    nr_timeslots, nr_stations, subgrid_size, nr_correlations)
aterms_offsets = util.get_example_aterms_offset(
                    nr_timeslots, nr_timesteps)

B0 = np.ones((1, subgrid_size, subgrid_size,1))

x = np.linspace(-0.5, 0.5, subgrid_size)

B1,B2 = np.meshgrid(x,x)
B1 = B1[np.newaxis, :, :, np.newaxis]
B2 = B2[np.newaxis, :, :, np.newaxis]
B3 = B1*B1
B4 = B2*B2
B5 = B1*B2

BB = np.concatenate((B0,B1,B2,B3,B4,B5))
B = np.kron(BB, np.array([1.0, 0.0, 0.0, 1.0]))

Bampl = B[:nr_parameters_ampl]
Bphase = B[:nr_parameters_phase]

proxy.init_cache(subgrid_size, cell_size, w_step, shift)
proxy.calibrate_init(
    kernel_size,
    frequencies,
    visibilities,
    weights,
    uvw,
    baselines,
    aterms_offsets,
    taper2)

X0 =  np.zeros((nr_stations, 1))
X1 =  np.ones((nr_stations, 1))
X2 = -0.2*np.ones((nr_stations, 1)) + 0.4*np.random.random((nr_stations, 1))

parameters = np.concatenate((X1,) + (nr_parameters_ampl-1)*(X0,) + (X2,) + (nr_parameters_phase * nr_timeslots - 1)* (X0,), axis=1)

aterm_ampl = np.tensordot(parameters[:,:nr_parameters_ampl] , Bampl, axes = ((1,), (0,)))
aterm_phase = np.exp(1j*np.tensordot(parameters[:,nr_parameters_ampl:].reshape((nr_stations, nr_timeslots, nr_parameters_phase)), Bphase, axes = ((2,), (0,))))
aterms[:,:,:,:,:] = aterm_phase.transpose((1,0,2,3,4))*aterm_ampl

uvw1          = np.zeros(shape=(nr_stations, nr_stations-1, nr_timesteps),
                        dtype=idg.uvwtype)
visibilities1 = np.zeros(shape=(nr_stations, nr_stations-1, nr_timesteps, nr_channels,
                               nr_correlations),
                        dtype=idg.visibilitiestype)
weights1      = np.zeros(shape=(nr_stations, nr_stations-1, nr_timesteps, nr_channels,
                            nr_correlations),
                        dtype=np.float32)
baselines1    = np.zeros(shape=(nr_stations, nr_stations-1),
                        dtype=idg.baselinetype)

for bl in range(nr_baselines):
    # Set baselines
    antenna1 = antenna1_block[0][bl]
    antenna2 = antenna2_block[0][bl]

    bl1 = antenna2 - (antenna2>antenna1)

    # print bl, antenna1, antenna2, bl1
    baselines1[antenna1][bl1] = (antenna1, antenna2)

    # Set uvw
    uvw1[antenna1][bl1] = uvw[bl]

    # Set visibilities
    visibilities1[antenna1][bl1] = visibilities[bl]
    weights1[antenna1][bl1] = weights[bl]

    antenna1, antenna2 = antenna2, antenna1

    bl1 = antenna2 - (antenna2>antenna1)

    baselines1[antenna1][bl1] = (antenna1, antenna2)

    # Set uvw
    uvw1[antenna1][bl1]['u'] = -uvw[bl]['u']
    uvw1[antenna1][bl1]['v'] = -uvw[bl]['v']
    uvw1[antenna1][bl1]['w'] = -uvw[bl]['w']

    # Set visibilities
    visibilities1[antenna1][bl1] = np.conj(visibilities[bl,:,:,(0,2,1,3)].transpose((1,2,0)))
    weights1[antenna1][bl1] = weights[bl,:,:,(0,2,1,3)].transpose((1,2,0))


predicted_visibilities = np.zeros_like(visibilities)
print(f"Shape predicted visibilities {predicted_visibilities.shape}")
proxy.degridding(
    kernel_size,
    frequencies,
    predicted_visibilities,
    uvw,
    baselines,
    aterms,
    aterms_offsets,
    taper2)
residual_visibilities = visibilities - predicted_visibilities

for i in range(nr_stations):
    bl_sel = [i in bl for bl in baselines]
    # Predict visibilities for current solution
    hessian  = np.zeros((nr_timeslots, nr_parameters0, nr_parameters0), dtype = np.float64)
    gradient = np.zeros((nr_timeslots, nr_parameters0), dtype = np.float64)
    residual = np.zeros((1, ), dtype = np.float64)

    aterm_ampl = np.repeat(np.tensordot(parameters[i,:nr_parameters_ampl], Bampl, axes = ((0,), (0,)))[np.newaxis,:], nr_timeslots, axis=0)
    aterm_phase = np.exp(1j * np.tensordot(parameters[i,nr_parameters_ampl:].reshape((nr_timeslots, nr_parameters_phase)), Bphase, axes = ((1,), (0,))))

    aterm_derivatives_ampl = aterm_phase[:, np.newaxis,:,:,:]*Bampl[np.newaxis,:,:,:,:]

    aterm_derivatives_phase = 1j*aterm_ampl[:, np.newaxis,:,:,:] * aterm_phase[:, np.newaxis,:,:,:] * Bphase[np.newaxis,:,:,:,:]

    aterm_derivatives = np.concatenate((aterm_derivatives_ampl, aterm_derivatives_phase), axis=1)
    aterm_derivatives = np.ascontiguousarray(aterm_derivatives, dtype=np.complex64)

    proxy.calibrate_update(i, aterms, aterm_derivatives, hessian, gradient, residual)

    # Predict visibilities for current solution
    predicted_visibilities1 = np.zeros(shape=(nr_parameters+1, nr_stations-1, nr_timesteps, nr_channels,
                                nr_correlations),
                            dtype=idg.visibilitiestype)

    aterms_local = aterms.copy()

    # iterate over degrees of freedom
    for j in range(nr_parameters0+1):
        # fill a-term with derivative
        if j>0:
            aterms_local[:,i,:,:,:] = aterm_derivatives[:,j-1,:,:,:]

        proxy.degridding(
            kernel_size,
            frequencies,
            predicted_visibilities1[j],
            uvw1[i],
            baselines1[i],
            aterms_local,
            aterms_offsets,
            taper2)

    ## compute residual visibilities
    residual_visibilities1 = visibilities1[i] - predicted_visibilities1[0]

    residual_ref1 = np.sum(weights1[i]*residual_visibilities1*np.conj(residual_visibilities1)).real
    residual_ref = np.sum(residual_visibilities[bl_sel] * residual_visibilities[bl_sel].conj()*weights[bl_sel]).real
    # compute vector and  matrix
    v = np.zeros((nr_timeslots, nr_parameters0, 1), dtype = np.float64)
    M = np.zeros((nr_timeslots, nr_parameters0, nr_parameters0), dtype = np.float64)

    for l in range(nr_timeslots):
        time_idx = slice(aterms_offsets[l], aterms_offsets[l+1])
        for j in range(nr_parameters0):
            v[l,j] = np.sum(weights1[i, :, time_idx ,:,:] * residual_visibilities1[:, time_idx ,:,:] * np.conj(predicted_visibilities1[j+1, :,time_idx,:,: ]))
            for k in range(j+1):
                M[l,j,k] = np.sum(weights1[i, :,time_idx,:,: ] * predicted_visibilities1[j+1, :,time_idx,:,: ] * np.conj(predicted_visibilities1[k+1, :,time_idx,:,: ]))
                M[l,k,j] = np.conj(M[l,j,k])

    hessian_err = np.amax((abs(hessian)>1e-3)*abs(hessian - M)/np.maximum(abs(hessian), 1.0))
    gradient_err = np.amax(abs(gradient - v[:,:,0])/np.maximum(abs(gradient), 1.0))
    residual_err = abs(residual[0] - residual_ref)/residual[0]

    assert hessian_err < 1e-4, f"Hessian error {hessian_err:.2e} for station {i} larger than threshold of 1e-4"
    assert gradient_err < 1e-4, f"Gradient error {gradient_err:.2e} for station {i} larger than threshold of 1e-4"
    assert residual_err < 1e-4, f"Residual error {residual_err:.2e} for station {i} larger than threshold of 1e-4"