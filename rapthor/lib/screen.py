"""
Module that holds screen-related functions
"""
from numpy import kron, concatenate, newaxis
from numpy.linalg import pinv, norm
import numpy as np
import os
from rapthor.lib import miscellaneous as misc
from losoto.operations.stationscreen import _makeWCS, _circ_chi2


def calculate_kl_screen(inscreen, residuals, pp, N_piercepoints, k, east, north, up,
    T, Nx, Ny, sindx, height, beta_val, r_0, is_phase, outQueue):
    """
    Calculates screen images

    Parameters
    ----------
    inscreen : array
        Array of screen values at the piercepoints
    residuals : array
        Array of screen residuals at the piercepoints
    pp : array
        Array of piercepoint locations
    N_piercepoints : int
        Number of pierce points
    k : int
        Time index
    east : array
        East array
    north : array
        North array
    up : array
        Up array
    T : array
        T array
    Nx : int
        Number of pixels in x for screen
    Ny : int
        Number of pixels in y for screen
    sindx : int
        Station index
    height : float
        height of screen (m)
    beta_val : float
        power-law index for phase structure function (5/3 =>
        pure Kolmogorov turbulence)
    r_0 : float
        scale size of phase fluctuations
    is_phase : bool
        input screen is a phase screen

    """
    from numpy import kron, concatenate, newaxis
    from numpy.linalg import pinv, norm
    import numpy as np

    screen = np.zeros((Nx, Ny))
    pp1 = pp[:, :]

    min_xy = np.amin(pp1, axis=0)
    max_xy = np.amax(pp1, axis=0)
    extent = max_xy - min_xy
    lowerk = min_xy - 0.1 * extent
    upperk = max_xy + 0.1 * extent
    im_extent_mk = upperk - lowerk
    pix_per_mk = Nx / im_extent_mk[0]
    m_per_pixk = 1.0 / pix_per_mk

    xr = np.arange(lowerk[0], upperk[0], m_per_pixk)
    yr = np.arange(lowerk[1], upperk[1], m_per_pixk)
    D = np.resize(pp, (N_piercepoints, N_piercepoints, 3))
    D = np.transpose(D, (1, 0, 2)) - D
    D2 = np.sum(D**2, axis=2)
    C = -(D2 / r_0**2)**(beta_val / 2.0) / 2.0
    f = inscreen.reshape(N_piercepoints)
    fitted_tec = np.dot(C, f) + residuals
    if is_phase:
        fitted_tec = normalize_phase(fitted_tec)
    for i, xi in enumerate(xr[0: Nx]):
        for j, yi in enumerate(yr[0: Ny]):
            if height == 0.0:
                p = np.array([xi, yi, 0.0])
            else:
                p, airmass = _calc_piercepoint(np.dot(np.array([xi, yi]), np.array([east, north])), up, height)
            d2 = np.sum(np.square(pp - p), axis=1)
            c = -(d2 / ( r_0**2 ))**(beta_val / 2.0) / 2.0
            screen[i, j] = np.dot(c, f)

    # Calculate the piercepoint coords
    x = pp1[:, 0]
    y = pp1[:, 1]

    outQueue.put([k, fitted_tec, screen, x, y])


def plot_frame(screen, fitted_phase1, residuals, weights, x, y, k, lower,
    upper, vmin, vmax, source_names, show_source_names, station_names, sindx,
    root_dir, prestr, is_image_plane,  midRA, midDec, order, is_phase, outQueue):
    """
    Plots screen images

    Parameters
    ----------
    screen : array
        Image of screen values
    fitted_phase1 : array
        Array of fitted phase values
    residuals : array
        Array of phase residuals at the piercepoints
    weights : array
        Array of weights at the piercepoints
    x : array
        Array of piercepoint x locations
    y : array
        Array of piercepoint y locations
    k : int
        Time index
    lower : array
        Array of lower limits for plot
    upper : array
        Array of upper limits for plot
    vmin : float
        minimum value for plot range
    vmax : float
        maximum value for plot range
    source_names : list
        List of source (direction) names
    show_source_names : bool
        label sources on screen plots
    order : int
        order of screen
    is_phase : bool
        True if screens are phase screens

    """
    # Set colormap
    if is_phase:
        cmap = _phase_cm()
    else:
        cmap = plt.cm.jet
    sm = plt.cm.ScalarMappable(cmap=cmap,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    if is_image_plane and hasWCSaxes:
        wcs = _makeWCS(midRA, midDec)
        ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs)
    else:
        plt.gca().set_aspect('equal')
        ax = plt.gca()

    s = []
    c = []
    xf = []
    yf = []
    weights = np.array(weights, dtype=float)
    nonflagged = np.where(weights > 0.0)
    for j in range(fitted_phase1.shape[0]):
        if weights[j] > 0.0:
            s.append(max(20, 200*np.sqrt(weights[j]/np.median(weights[nonflagged]))))
        else:
            s.append(120)
            xf.append(x[j])
            yf.append(y[j])
        c.append(sm.to_rgba(fitted_phase1[j]))

    if is_image_plane:
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        extent_x = max_x - min_x
        extent_y = max_y - min_y
        lower = [min_x - 0.1 * extent_x, min_y - 0.1 * extent_y]
        upper = [max_x + 0.1 * extent_x, max_y + 0.1 * extent_y]
        Nx = screen.shape[0]
        pix_per_m = Nx / (upper[0] - lower[0])
        m_per_pix = 1.0 / pix_per_m
        xr = np.arange(lower[0], upper[0], m_per_pix)
        yr = np.arange(lower[1], upper[1], m_per_pix)
        lower = np.array([xr[0], yr[0]])
        upper = np.array([xr[-1], yr[-1]])
    else:
        # convert from m to km
        lower /= 1000.0
        upper /= 1000.0

    im = ax.imshow(screen.transpose([1, 0])[:, :],
        cmap = cmap,
        origin = 'lower',
        interpolation = 'nearest',
        extent = (lower[0], upper[0], lower[1], upper[1]),
        vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im)
    cbar.set_label('Value', rotation=270)

    ax.scatter(np.array(x), np.array(y), s=np.array(s), c=np.array(c), alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='black')
    if len(xf) > 0:
        ax.scatter(xf, yf, s=120, c='k', marker='x')
    if show_source_names:
        labels = source_names
        for label, xl, yl in zip(labels, x, y):
            plt.annotate(
                label,
                xy = (xl, yl), xytext = (-2, 2),
                textcoords = 'offset points', ha = 'right', va = 'bottom')

    nsrcs = np.where(weights > 0.0)[0].size
    if is_phase:
        redchi2 =  _circ_chi2(residuals, weights) / (nsrcs-order)
    else:
        redchi2 =  np.sum(np.square(residuals) * weights) / (nsrcs-order)
    if sindx >= 0:
        plt.title('Station {0}, Time {1} (red. chi2 = {2:0.3f})'.format(station_names[sindx], k, redchi2))
    else:
        plt.title('Time {0}'.format(k))
    if is_image_plane:
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect('equal')
        if hasWCSaxes:
            RAAxis = ax.coords['ra']
            RAAxis.set_axislabel('RA', minpad=0.75)
            RAAxis.set_major_formatter('hh:mm:ss')
            DecAxis = ax.coords['dec']
            DecAxis.set_axislabel('Dec', minpad=0.75)
            DecAxis.set_major_formatter('dd:mm:ss')
            ax.coords.grid(color='black', alpha=0.5, linestyle='solid')
            plt.xlabel("RA")
            plt.ylabel("Dec")
        else:
            plt.xlabel("RA (arb. units)")
            plt.ylabel("Dec (arb. units)")
    else:
        # Reverse the axis so that RA coord increases to left
        plt.xlim(upper[0], lower[0])
        plt.ylim(lower[1], upper[1])
        plt.xlabel('Projected Distance East-West (km)')
        plt.ylabel('Projected Distance North-South (km)')
    if sindx >= 0:
        plt.savefig(root_dir + '/' + prestr + '_station%0.4i' % sindx + '_frame%0.4i.png' % k, bbox_inches='tight')
    else:
        plt.savefig(root_dir + '/' + prestr + '_frame%0.4i.png' % k, bbox_inches='tight')
    plt.close(fig)


def make_kl_screen_images(soltab_ph, resSoltab_ph, soltab_amp=None, resSoltab_amp=None,
                          prefix='', ncpu=0):
    """
    Make FITS images from KL screens

    Parameters
    ----------
    soltab_ph : solution table
        Soltab containing the phase screen.
    resSoltab_ph : solution table, optional
        Soltab containing the phase screen residuals.
    soltab_amp : solution table
        Soltab containing the amplitude screen.
    resSoltab_amp : solution table, optional
        Soltab containing the amplitude screen residuals.
    prefix : str, optional
        String to prepend to output files.
    ncpu : int, optional
        Number of CPUs to use. If 0, all are used.
    """
    screen_type = soltab.getType()
    if screen_type == 'phasescreen':
        is_phase = True
    else:
        is_phase = False

    # Get soltabs
    solset = soltab_ph.getSolset()
    pols_ph = soltab_ph.pol
    times = soltab.time
    freqs = soltab.freq

    # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
    outfile = '{0}_{1}.fits'.format(outroot, gnum)
    misc.make_template_image(temp_image, midRA, midDec, ximsize=imsize,
                             yimsize=imsize, cellsize_deg=cellsize_deg,
                             times=times[g_start:g_stop],
                             freqs=freqs, antennas=soltab.ant,
                             aterm_type='gain')
    hdu = pyfits.open(temp_image, memmap=False)
    data = hdu[0].data
    w = wcs.WCS(hdu[0].header)

    # Make the screen images, iterating over frequencies and polarizations
    # Axes are ['time', 'freq', 'ant', 'dir', 'pol']
    for freq in freqs:
        for pol in pols:
            screen = np.array(soltab.val[:, freq, :, :, pol])
            weights = np.array(soltab.weight[:, freq, :, :, pol])
            residuals = np.array(ressoltab.val[:, freq, :, :, pol])
            orders = np.array(ressoltab.weight[:, freq, :, :, pol])
            axis_names = soltab.getAxesNames()

            # Interpolate the slow amps to the fast time and frequency grid
            soltab_amp_screen = solset.getSoltab('amplitude_screen000')
            vals_amp = interpolate_amps(soltab_amp_screen, soltab_ph, interp_kind='nearest')

            # Make a new soltab with the interpolated amps
            soltab_amp_screen = solset.makeSoltab('amplitudescreen', 'amplitude_screen001',
                                          axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                                          axesVals=[soltab_ph.time, soltab_ph.freq,
                                                    soltab_amp.ant, soltab_amp.dir,
                                                    soltab_amp.pol],
                                          vals=vals_amp, weights=np.ones_like(vals_amp))


            # Rearrange to get order [dir, time, ant]
            dir_ind = 2
            time_ind = 0
            ant_ind = 1
            screen = screen.transpose([dir_ind, time_ind, ant_ind])
            weights = weights.transpose([dir_ind, time_ind, ant_ind])
            residuals = residuals.transpose([dir_ind, time_ind, ant_ind])
            orders = orders.transpose([dir_ind, time_ind, ant_ind])

            # Collect station and source names and positions and times, making sure
            # that they are ordered correctly.
            source_names = soltab.dir[:]
            source_dict = solset.getSou()
            source_positions = []
            for source in source_names:
                source_positions.append(source_dict[source])
            station_names = soltab.ant
            station_dict = solset.getAnt()
            station_positions = []
            for station in station_names:
                station_positions.append(station_dict[station])
            height = soltab.obj._v_attrs['height']
            beta_val = soltab.obj._v_attrs['beta']
            r_0 = soltab.obj._v_attrs['r_0']
            pp = soltab.obj.piercepoint[:]
            midRA = soltab.obj._v_attrs['midra']
            midDec = soltab.obj._v_attrs['middec']
            prestr = os.path.basename(prefix) + 'screen'

            # Define various parameters
            N_stations = 1 # screens are single-station screens
            N_sources = len(source_names)
            N_times = len(times)
            N_piercepoints = N_sources * N_stations
            xp, yp, zp = station_positions[0, :] # use first station
            east = np.array([-yp, xp, 0])
            east = east / norm(east)

            north = np.array([-xp, -yp, (xp*xp + yp*yp)/zp])
            north = north / norm(north)

            up = np.array([xp, yp, zp])
            up = up / norm(up)

            T = concatenate([east[:, newaxis], north[:, newaxis]], axis=1)

            # Use pierce point locations of first and last time slots to estimate
            # required size of plot in meters
            is_image_plane = True # pierce points are image plane coords
            pp1_0 = pp[:, 0:2]
            pp1_1 = pp[:, 0:2]

            max_xy = np.amax(pp1_0, axis=0) - np.amin(pp1_0, axis=0)
            max_xy_1 = np.amax(pp1_1, axis=0) - np.amin(pp1_1, axis=0)
            if max_xy_1[0] > max_xy[0]:
                max_xy[0] = max_xy_1[0]
            if max_xy_1[1] > max_xy[1]:
                max_xy[1] = max_xy_1[1]

            min_xy = np.array([0.0, 0.0])
            extent = max_xy - min_xy
            lower = min_xy - 0.1 * extent
            upper = max_xy + 0.1 * extent
            im_extent_m = upper - lower
            fitted_phase1 = np.zeros((N_piercepoints, N_times))

            Nx = 40 # set approximate number of pixels in screen
            pix_per_m = Nx / im_extent_m[0]
            m_per_pix = 1.0 / pix_per_m
            xr = np.arange(lower[0], upper[0], m_per_pix)
            yr = np.arange(lower[1], upper[1], m_per_pix)
            Nx = len(xr)
            Ny = len(yr)
            lower = np.array([xr[0], yr[0]])
            upper = np.array([xr[-1], yr[-1]])

            x = np.zeros((N_times, N_piercepoints)) # plot x pos of piercepoints
            y = np.zeros((N_times, N_piercepoints)) # plot y pos of piercepoints
            screen = np.zeros((Nx, Ny, N_times))

            for sindx in range(station_positions.shape[0]):
                residuals = inresiduals[:, :, sindx, newaxis].transpose([0, 2, 1]).reshape(N_piercepoints, N_times)
                mpm = misc.multiprocManager(ncpu, calculate_kl_screen)
                for k in range(N_times):
                    mpm.put([inscreen[:, k, sindx], residuals[:, k], pp,
                        N_piercepoints, k, east, north, up, T, Nx, Ny, sindx, height,
                        beta_val, r_0, is_phase])
                mpm.wait()
                for (k, ft, scr, xa, ya) in mpm.get():
                    screen[:, :, k] = scr
                    fitted_phase1[:, k] = ft
                    x[k, :] = xa
                    y[k, :] = ya

                val_amp_xx = vals[t+g_start, f, sindx, poly.index, 0]
                val_amp_yy = vals[t+g_start, f, s, poly.index, 1]
                val_phase = vals_ph[t+g_start, f, s, poly.index]

                # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
                data[:, freq, sindx, 0, :, :] = val_amp_xx * np.cos(val_phase)
                data[:, f, s, 2, ind[0], ind[1]] = val_amp_yy * np.cos(val_phase)
                data[t, f, s, 1, ind[0], ind[1]] = val_amp_xx * np.sin(val_phase)
                data[t, f, s, 3, ind[0], ind[1]] = val_amp_yy * np.sin(val_phase)



    # Save to FITS image
