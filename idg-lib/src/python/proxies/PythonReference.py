# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

from .Python import *

class Reference(Proxy):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):

        self.nr_correlations = nr_correlations
        self.subgrid_size = subgrid_size

    def compute_wavenumber(
        self,
        frequencies):
        speed_of_light = 299792458.0
        return (frequencies * 2 * np.pi) / speed_of_light

    def grid_onto_subgrids(
        self,
        plan,
        w_step,
        grid_size,
        image_size,
        wavenumbers,
        visibilities,
        uvw,
        spheroidal,
        aterms,
        subgrids):

        # Parameters
        nr_correlations = subgrids.shape[1]
        subgrid_size = subgrids.shape[2]
        nr_channels = visibilities.shape[2]

        # Get metadata
        nr_subgrids = plan.get_nr_subgrids()
        metadata = numpy.zeros(nr_subgrids, dtype = idgtypes.metadatatype)
        plan.copy_metadata(metadata)

        # Flatten uvw
        u_ = uvw['u'].flatten()
        v_ = uvw['v'].flatten()
        w_ = uvw['w'].flatten()

        # Iterate all subgrids
        for s,m in enumerate(metadata):
            # Load metadata
            offset       = m['time_index']
            nr_timesteps = m['nr_timesteps']
            aterm_index  = m['aterm_index']
            station1     = m['baseline']['station1']
            station2     = m['baseline']['station2']
            x_coordinate = m['coordinate']['x']
            y_coordinate = m['coordinate']['y']
            w_offset_in_lambda = w_step * (m['coordinate']['z'] + 0.5)

            # Derive baseline
            total_nr_timesteps = visibilities.shape[1]
            bl = offset /  total_nr_timesteps

            # Compute u and v offset in wavelenghts
            u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*np.pi / image_size)
            v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*np.pi / image_size)
            w_offset = 2*np.pi * w_offset_in_lambda

            # Iterate all pixels in subgrid
            for y in range(subgrid_size):
                for x in range(subgrid_size):
                    # Initialize pixel for every polarization
                    pixels = np.zeros(nr_correlations, dtype=np.complex64)

                    # Compute l,m,n
                    l = np.float32((x+0.5-(subgrid_size/2)) * image_size/subgrid_size)
                    m = np.float32((y+0.5-(subgrid_size/2)) * image_size/subgrid_size)
                    # evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m))
                    # accurately for small values of l and m
                    tmp = np.float32((l * l) + (m * m))
                    n = np.float32(tmp / (1.0 + np.sqrt(1.0 - tmp)))

                    # Iterate all timesteps
                    for time in range(nr_timesteps):
                        # Load UVW coordinates
                        u = u_[offset + time]
                        v = v_[offset + time]
                        w = w_[offset + time]

                        # Compute phase index
                        phase_index = np.float32(u*l + v*m + w*n)

                        # Compute phase offset
                        phase_offset = np.float32(u_offset*l + v_offset*m + w_offset*n)

                        # Update pixel for every channel
                        for chan in range(nr_channels):
                            # Compute phase
                            phase = np.float32(phase_offset - (phase_index * wavenumbers[chan]))

                            # Compute phasor
                            phasor_real = np.float32(np.cos(phase))
                            phasor_imag = np.float32(np.sin(phase))
                            phasor      = np.complex64(complex(phasor_real, phasor_imag))

                            # Update pixel for every polarization
                            for pol in range(nr_correlations):
                                pixels[pol] += visibilities[bl,time,chan,pol] * phasor

                    # Load a term for station1
                    aXX1 = aterms[aterm_index, station1, y, x, 0]
                    aXY1 = aterms[aterm_index, station1, y, x, 1]
                    aYX1 = aterms[aterm_index, station1, y, x, 2]
                    aYY1 = aterms[aterm_index, station1, y, x, 3]

                    # Load aterm for station2
                    aXX2 = np.conj(aterms[aterm_index, station2, y, x, 0])
                    aXY2 = np.conj(aterms[aterm_index, station2, y, x, 1])
                    aYX2 = np.conj(aterms[aterm_index, station2, y, x, 2])
                    aYY2 = np.conj(aterms[aterm_index, station2, y, x, 3])

                    # Apply aterm to subgrid: P*A1
                    # [ pixels[0], pixels[1];    [ aXX1, aXY1;
                    #   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
                    pixelsXX = pixels[0]
                    pixelsXY = pixels[1]
                    pixelsYX = pixels[2]
                    pixelsYY = pixels[3]
                    pixels[0]  = (pixelsXX * aXX1)
                    pixels[0] += (pixelsXY * aYX1)
                    pixels[1]  = (pixelsXX * aXY1)
                    pixels[1] += (pixelsXY * aYY1)
                    pixels[2]  = (pixelsYX * aXX1)
                    pixels[2] += (pixelsYY * aYX1)
                    pixels[3]  = (pixelsYX * aXY1)
                    pixels[3] += (pixelsYY * aYY1)

                    # Apply aterm to subgrid: A2^H*P
                    # [ aXX2, aYX1;      [ pixels[0], pixels[1];
                    #   aXY1, aYY2 ]  *    pixels[2], pixels[3] ]
                    pixelsXX = pixels[0]
                    pixelsXY = pixels[1]
                    pixelsYX = pixels[2]
                    pixelsYY = pixels[3]
                    pixels[0]  = (pixelsXX * aXX2)
                    pixels[0] += (pixelsYX * aYX2)
                    pixels[1]  = (pixelsXY * aXX2)
                    pixels[1] += (pixelsYY * aYX2)
                    pixels[2]  = (pixelsXX * aXY2)
                    pixels[2] += (pixelsYX * aYY2)
                    pixels[3]  = (pixelsXY * aXY2)
                    pixels[3] += (pixelsYY * aYY2)

                    # Load spheroidal
                    sph = spheroidal[y, x]

                    # Compute shifted position in subgrid
                    x_dst = (x + (subgrid_size/2)) % subgrid_size
                    y_dst = (y + (subgrid_size/2)) % subgrid_size

                    # Set subgrid value
                    for pol in range(nr_correlations):
                        subgrids[s,pol,y_dst,x_dst] = np.complex64(pixels[pol] * sph)

        # Apply FFT to each subgrid
        subgrids[:] = np.fft.ifft2(subgrids, axes=(2,3))


    def add_subgrids_to_grid(
        self,
        plan,
        w_step,
        subgrids,
        grid):

        nr_correlations = grid.shape[0]
        grid_size = grid.shape[1]
        subgrid_size = subgrids.shape[2]

        # Compute phasor
        phasor = np.zeros(shape=(subgrid_size, subgrid_size), dtype=np.complex64)
        for y in range(subgrid_size):
            for x in range(subgrid_size):
                phase = np.float32(np.pi*(x+y-subgrid_size)/subgrid_size)
                phasor_real = np.float32(np.cos(phase))
                phasor_imag = np.float32(np.sin(phase))
                phasor[y,x] = np.complex64(complex(phasor_real, phasor_imag))

        # Get metadata
        nr_subgrids = plan.get_nr_subgrids()
        metadata = numpy.zeros(nr_subgrids, dtype = idgtypes.metadatatype)
        plan.copy_metadata(metadata)

        # Iterate all subgrids
        for s,m in enumerate(metadata):
            # Load position in grid
            coordinate = m['coordinate']
            grid_x = coordinate['x']
            grid_y = coordinate['y']

            # Check wheter subgrid fits in grid
            if (grid_x >= 0 and grid_x < grid_size-subgrid_size and
                grid_y >= 0 and grid_y < grid_size-subgrid_size):
                for y in range(subgrid_size):
                    for x in range(subgrid_size):
                        # Compute shifted position in subgrid
                        x_src = (x + (subgrid_size/2)) % subgrid_size
                        y_src = (y + (subgrid_size/2)) % subgrid_size

                        # Add subgrid value to grid
                        for p in range(nr_correlations):
                            grid[p,grid_y+y,grid_x+x] += np.complex64(subgrids[s,p,y_src,x_src] * phasor[y,x])


    def gridding(
        self,
        w_step,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offsets,
        spheroidal):
        """
        Grid visibilities onto grid.

        :param frequencies: numpy.ndarray(
                shapenr_channels,
                dtype = idg.frequenciestype)
        :param visibilities: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                dtype=idg.visibilitiestype)
        :param uvw: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps),
                dtype = idg.uvwtype)
        :param baselines: numpy.ndarray(
                shape=(nr_baselines),
                dtype=idg.baselinetype)
        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        :param aterms: numpy.ndarray(
                shape=(nr_timeslots, nr_stations, height, width, nr_correlations),
                dtype = idg.atermtype)
        :param aterms_offsets: numpy.ndarray(
                shape=(nr_timeslots+1),
                dtype = idg.atermoffsettype)
        :param spheroidal: numpy.ndarray(
                shape=(height, width),
                dtype = idg.tapertype)
        """
        # extract dimensions
        nr_channels = frequencies.shape[0]
        visibilities_nr_baselines    = visibilities.shape[0]
        visibilities_nr_timesteps    = visibilities.shape[1]
        visibilities_nr_channels     = visibilities.shape[2]
        visibilities_nr_correlations = visibilities.shape[3]
        uvw_nr_baselines             = uvw.shape[0]
        uvw_nr_timesteps             = uvw.shape[1]
        uvw_nr_coordinates           = 3
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = 2
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[1]
        grid_width                   = grid.shape[2]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offsets.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # convert frequencies into wavenumbers
        wavenumbers = self.compute_wavenumber(frequencies)

        # check parameters
        assert(grid_nr_correlations == self.nr_correlations)
        assert(grid_height == grid_width)

        # get parameters
        nr_correlations = self.nr_correlations
        subgrid_size = self.subgrid_size
        grid_size = grid_height
        nr_timesteps = visibilities_nr_timesteps
        image_size = cell_size * grid_size

        # create plan
        plan = idg.Plan(
                    kernel_size, subgrid_size, grid_size, cell_size,
                    frequencies, uvw, baselines, aterms_offsets, nr_timesteps)

        # allocate subgrids
        nr_subgrids = plan.get_nr_subgrids()
        subgrids = np.zeros(shape=(nr_subgrids, nr_correlations, subgrid_size, subgrid_size), dtype=idgtypes.gridtype)

        # run subroutines
        self.grid_onto_subgrids(
                plan, w_step, grid_size, image_size,
                wavenumbers, visibilities, uvw, spheroidal, aterms, subgrids)

        self.add_subgrids_to_grid(
                plan, w_step, subgrids, grid)


    def split_grid_into_subgrids(
        self,
        plan,
        w_step,
        subgrids,
        grid):
        pass


    def degrid_from_subgrids(
        self,
        plan,
        w_step,
        grid_size,
        image_size,
        wavenumbers,
        visibilities,
        uvw,
        spheroidal,
        aterms,
        subgrids):
        pass


    def degridding(
        self,
        w_step,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offsets,
        spheroidal):
        """
        Degrid visibilities from grid.

        :param frequencies: numpy.ndarray(
                shapenr_channels,
                dtype = idg.frequenciestype)
        :param visibilities: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                dtype=idg.visibilitiestype)
        :param uvw: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps),
                dtype = idg.uvwtype)
        :param baselines: numpy.ndarray(
                shape=(nr_baselines),
                dtype=idg.baselinetype)
        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        :param aterms: numpy.ndarray(
                shape=(nr_timeslots, nr_stations, height, width, nr_correlations),
                dtype = idg.atermtype)
        :param aterms_offsets: numpy.ndarray(
                shape=(nr_timeslots+1),
                dtype = idg.atermoffsettype)
        :param spheroidal: numpy.ndarray(
                shape=(height, width),
                dtype = idg.tapertype)
        """
        # extract dimensions
        nr_channels = frequencies.shape[0]
        visibilities_nr_baselines    = visibilities.shape[0]
        visibilities_nr_timesteps    = visibilities.shape[1]
        visibilities_nr_channels     = visibilities.shape[2]
        visibilities_nr_correlations = visibilities.shape[3]
        uvw_nr_baselines             = uvw.shape[0]
        uvw_nr_timesteps             = uvw.shape[1]
        uvw_nr_coordinates           = 3
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = 2
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[1]
        grid_width                   = grid.shape[2]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offsets.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # convert frequencies into wavenumbers
        wavenumbers = self.compute_wavenumber(frequencies)

        # check parameters
        assert(grid_nr_correlations == self.nr_correlations)
        assert(grid_height == grid_width)

        # get parameters
        nr_correlations = self.nr_correlations
        subgrid_size = self.subgrid_size
        grid_size = grid_height
        nr_timesteps = visibilities_nr_timesteps
        image_size = cell_size * grid_size

        # create plan
        plan = idg.Plan(
                    kernel_size, subgrid_size, grid_size, cell_size,
                    frequencies, uvw, baselines, aterms_offsets, nr_timesteps)

        # allocate subgrids
        nr_subgrids = plan.get_nr_subgrids()
        subgrids = np.zeros(shape=(nr_subgrids, nr_correlations, subgrid_size, subgrid_size), dtype=idgtypes.gridtype)

        # run subroutines
        self.split_grid_into_subgrids(
                plan, w_step, subgrids, grid)

        self.degrid_from_subgrids(
                plan, w_step, grid_size, image_size,
                wavenumbers, visibilities, uvw, spheroidal, aterms, subgrids)


    def transform(
        self,
        direction,
        grid):
        """
        Transform Fourier Domain<->Image Domain.

        :param direction: idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        """
        # extract dimesions
        nr_correlations = grid.shape[0]
        height          = grid.shape[1]
        width           = grid.shape[2]

        # FFT shift
        grid[:] = np.fft.fftshift(grid, axes=(1,2))

        # FFT
        if direction == idgtypes.FourierDomainToImageDomain:
            grid[:] = np.fft.ifft2(grid, axes=(1,2))
        else:
            grid[:] = np.fft.fft2(grid, axes=(1,2))

        # FFT shift
        grid[:] = np.fft.fftshift(grid, axes=(1,2))

        # Scaling
        scale = (2 + 0j)
        if direction == idgtypes.FourierDomainToImageDomain:
            grid *= scale
        else:
            grid /= scale
