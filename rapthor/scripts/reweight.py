#!/usr/bin/env python3
"""
Script to reweight uv data

Based on https://github.com/ebonnassieux/Scripts/blob/master/QualityWeightsLOFAR.py
"""
from casacore.tables import table
import numpy as np
import sys
import warnings
from rapthor.lib import miscellaneous as misc


class CovWeights:
    def __init__(self, MSName, solint_sec, solint_hz, uvcut=[0, 2000], gainfile=None,
                 phaseonly=False, dirname=None, quiet=True):
        if MSName[-1] == "/":
            self.MSName = MSName[0:-1]
        else:
            self.MSName = MSName
        tab = table(self.MSName, ack=False)
        self.timepersample = tab.getcell('EXPOSURE', 0)
        self.ntSol = max(1, int(round(solint_sec / self.timepersample)))
        tab.close()
        sw = table(self.MSName+'::SPECTRAL_WINDOW', ack=False)
        self.referencefreq = sw.col('REF_FREQUENCY')[0]
        self.channelwidth = sw.col('CHAN_WIDTH')[0][0]
        self.numchannels = sw.col('NUM_CHAN')[0]
        sw.close()
        self.nchanSol = max(1, self.get_nearest_frequstep(solint_hz / self.channelwidth))
        self.uvcut = uvcut
        self.gainfile = gainfile
        self.phaseonly = phaseonly
        self.dirname = dirname
        self.quiet = quiet

    def FindWeights(self, colname=""):
        ms = table(self.MSName, ack=False)
        ants = table(ms.getkeyword("ANTENNA"), ack=False)
        antnames = ants.getcol("NAME")
        ants.close()
        nAnt = len(antnames)

        u, v, _ = ms.getcol("UVW").T
        A0 = ms.getcol("ANTENNA1")
        A1 = ms.getcol("ANTENNA2")
        tarray = ms.getcol("TIME")
        nbl = np.where(tarray == tarray[0])[0].size
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default")
        if "RESIDUAL_DATA" not in ms.colnames():
            print('reweight: RESIDUAL_DATA not found. Exiting...')
            sys.exit(1)
        residualdata = ms.getcol("RESIDUAL_DATA")
        flags = ms.getcol("FLAG")
        ms.close()

        # apply uvcut
        c_m_s = 2.99792458e8
        uvlen = np.sqrt(u**2 + v**2) / c_m_s * self.referencefreq
        flags[uvlen > self.uvcut[1], :, :] = True
        flags[uvlen < self.uvcut[0], :, :] = True
        residualdata[flags] = np.nan
        residualdata[residualdata == 0] = np.nan

        # initialise
        nChan = residualdata.shape[1]
        nPola = residualdata.shape[2]
        nt = residualdata.shape[0] / nbl
        residualdata = residualdata.reshape((nt, nbl, nChan, nPola))
        A0 = A0.reshape((nt, nbl))[0, :]
        A1 = A1.reshape((nt, nbl))[0, :]

        # make rms and residuals arrays
        rmsarray = np.zeros((nt, nbl, nChan, 2), dtype=np.complex64)
        residuals = np.zeros_like(rmsarray, dtype=np.complex64)
        rmsarray[:, :, :, 0] = residualdata[:, :, :, 1]
        rmsarray[:, :, :, 1] = residualdata[:, :, :, 2]
        residuals[:, :, :, 0] = residualdata[:, :, :, 0]
        residuals[:, :, :, 1] = residualdata[:, :, :, 3]

        # start calculating the weights
        CoeffArray = np.zeros((nt, nChan, nAnt))
        ant1 = np.arange(nAnt)
        num_sols_time = int(np.ceil(float(nt) / self.ntSol))
        num_sols_freq = int(np.ceil(float(nChan) / self.nchanSol))
        tcellsize = self.ntSol
        for t_i in range(0, nt, self.ntSol):
            if (t_i == nt - self.ntSol) and (nt % self.ntSol > 0):
                tcellsize = nt % self.ntSol
            t_e = t_i + tcellsize
            fcellsize = self.nchanSol
            for f_i in range(0, nChan, self.nchanSol):
                if (f_i == nChan - self.nchanSol) and (nChan % self.nchanSol > 0):
                    fcellsize = nChan % self.nchanSol
                f_e = f_i + fcellsize

                # build weights for each antenna in the current time-frequency block
                for ant in ant1:
                    # set of vis for baselines ant-ant_i
                    set1 = np.where(A0 == ant)[0]
                    # set of vis for baselines ant_i-ant
                    set2 = np.where(A1 == ant)[0]
                    CoeffArray[t_i:t_e, f_i:f_e, ant] = np.sqrt(
                        np.nanmean(
                            np.append(residuals[t_i:t_e, set1, f_i:f_e, :],
                                      residuals[t_i:t_e, set2, f_i:f_e, :]) *
                            np.append(residuals[t_i:t_e, set1, f_i:f_e, :],
                                      residuals[t_i:t_e, set2, f_i:f_e, :]).conj()
                            ) -
                        np.nanstd(
                            np.append(rmsarray[t_i:t_e, set1, f_i:f_e, :],
                                      rmsarray[t_i:t_e, set2, f_i:f_e, :])
                            )
                        )
            if not self.quiet:
                PrintProgress(t_i, nt)

        # plot
#         Nr = 8
#         Nc = 8
#         xvals = range(nt)
#         yvals = range(nChan)
#         figSize = [10+3*Nc, 8+2*Nr]
#         figgrid, axa = plt.subplots(Nr, Nc, sharex=True, sharey=True, figsize=figSize)
#         for i in range(nAnt):
#             ax = axa.flatten()[i]
#             bbox = ax.get_window_extent().transformed(figgrid.dpi_scale_trans.inverted())
#             aspect = ((xvals[-1]-xvals[0])*bbox.height)/((yvals[-1]-yvals[0])*bbox.width)
#             im = ax.imshow(CoeffArray[:, :, i].transpose(1,0), origin='lower', interpolation="none", cmap=plt.cm.rainbow, norm=None,
#                             extent=[xvals[0],xvals[-1],yvals[0],yvals[-1]], aspect=str(aspect))
#         figgrid.colorbar(im, ax=axa.ravel().tolist(), use_gridspec=True, fraction=0.02, pad=0.005, aspect=35)
#         figgrid.savefig(self.MSName+'_coef1.png', bbox_inches='tight')
#         plt.close()

        # get rid of NaNs and low values
        CoeffArray[np.isnan(CoeffArray)] = np.inf
        for i in range(nAnt):
            tempars = CoeffArray[:, :, i]
            thres = 0.25 * np.median(tempars[np.where(np.isfinite(tempars))])
            CoeffArray[:, :, i][tempars < thres] = thres
        return CoeffArray

    def SaveWeights(self, CoeffArray, colname=None):
        ms = table(self.MSName, readonly=False, ack=False)
        ants = table(ms.getkeyword("ANTENNA"), ack=False)
        antnames = ants.getcol("NAME")
        nAnt = len(antnames)
        tarray = ms.getcol("TIME")
        darray = ms.getcol("DATA")
        tvalues = np.array(sorted(list(set(tarray))))
        nt = tvalues.shape[0]
        nbl = tarray.shape[0]/nt
        nchan = darray.shape[1]
        npol = darray.shape[2]
        A0 = np.array(ms.getcol("ANTENNA1").reshape((nt, nbl)))
        A1 = np.array(ms.getcol("ANTENNA2").reshape((nt, nbl)))
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default")

        # initialize weight array
        w = np.zeros((nt, nbl, nchan, npol))
        A0ind = A0[0, :]
        A1ind = A1[0, :]

        # do gains stuff
        ant1gainarray, ant2gainarray = readGainFile(self.gainfile, ms, nt, nchan, nbl,
                                                    tarray, nAnt, self.MSName, self.phaseonly,
                                                    self.dirname)
        ant1gainarray = ant1gainarray.reshape((nt, nbl, nchan))
        ant2gainarray = ant2gainarray.reshape((nt, nbl, nchan))
        for t in range(nt):
            for i in range(nbl):
                for j in range(nchan):
                    w[t, i, j, :] = 1.0 / (CoeffArray[t, j, A0ind[i]] * ant1gainarray[t, i, j] +
                                           CoeffArray[t, j, A1ind[i]] * ant2gainarray[t, i, j] +
                                           CoeffArray[t, j, A0ind[i]] * CoeffArray[t, j, A1ind[i]] +
                                           0.1)
            if not self.quiet:
                PrintProgress(t, nt)

        # normalize
        w = w.reshape(nt*nbl, nchan, npol)
        w[np.isinf(w)] = np.nan
        w = w / np.nanmean(w)
        w[np.isnan(w)] = 0

        # save in weights column
        if colname is not None:
            ms.putcol(colname, w)
        ants.close()
        ms.close()


    def get_nearest_frequstep(self, freqstep):
        """
        Gets the nearest frequstep

        Parameters
        ----------
        freqstep : int
            Target frequency step

        Returns
        -------
        optimum_step : int
            Optimum frequency step nearest to target step
        """
        # Generate a list of possible values for freqstep
        if not hasattr(self, 'freq_divisors'):
            tmp_divisors = []
            for step in range(self.numchannels, 0, -1):
                if (self.numchannels % step) == 0:
                    tmp_divisors.append(step)
            self.freq_divisors = np.array(tmp_divisors)

        # Find nearest
        idx = np.argmin(np.abs(self.freq_divisors - freqstep))

        return self.freq_divisors[idx]


def readGainFile(gainfile, ms, nt, nchan, nbl, tarray, nAnt, msname, phaseonly, dirname):
    if phaseonly:
        ant1gainarray1 = np.ones((nt*nbl, nchan))
        ant2gainarray1 = np.ones((nt*nbl, nchan))
    else:
        import losoto.h5parm
        solsetName = "sol000"
        soltabName = "screenamplitude000"
        try:
            gfile = losoto.h5parm.openSoltab(gainfile, solsetName=solsetName, soltabName=soltabName)
        except:
            print("Could not find amplitude gains in h5parm. Assuming gains of 1 everywhere.")
            ant1gainarray1 = np.ones((nt*nbl, nchan))
            ant2gainarray1 = np.ones((nt*nbl, nchan))
            return ant1gainarray1, ant2gainarray1

        freqs = table(msname+"/SPECTRAL_WINDOW").getcol("CHAN_FREQ")
        gains = gfile.val  # axes: times, freqs, ants, dirs, pols
        flagged = np.where(gains == 0.0)
        gains[flagged] = np.nan
        gfreqs = gfile.freq
        times = gfile.time
        dindx = gfile.dir.tolist().index(dirname)
        ant1gainarray = np.zeros((nt*nbl, nchan))
        ant2gainarray = np.zeros((nt*nbl, nchan))
        A0arr = ms.getcol("ANTENNA1")
        A1arr = ms.getcol("ANTENNA2")
        deltime = (times[1] - times[0]) / 2.0
        delfreq = (gfreqs[1] - gfreqs[0]) / 2.0
        for i in range(len(times)):
            timemask = (tarray >= times[i]-deltime) * (tarray < times[i]+deltime)
            if np.all(~timemask):
                continue
            for j in range(nAnt):
                mask1 = timemask * (A0arr == j)
                mask2 = timemask * (A1arr == j)
                for k in range(nchan):
                    chan_freq = freqs[0, k]
                    freqmask = np.logical_and(gfreqs >= chan_freq-delfreq,
                                              gfreqs < chan_freq+delfreq)
                    if chan_freq < gfreqs[0]:
                        freqmask[0] = True
                    if chan_freq > gfreqs[-1]:
                        freqmask[-1] = True
                    ant1gainarray[mask1, k] = np.nanmean(gains[i, freqmask, j, dindx, :], axis=(0, 1))
                    ant2gainarray[mask2, k] = np.nanmean(gains[i, freqmask, j, dindx, :], axis=(0, 1))
        ant1gainarray1 = ant1gainarray**2
        ant2gainarray1 = ant2gainarray**2

    return ant1gainarray1, ant2gainarray1


def PrintProgress(currentIter, maxIter, msg=""):
    sys.stdout.flush()
    if msg == "":
        msg = "Progress:"
    sys.stdout.write("\r%s %5.1f %% " % (msg, 100*(currentIter+1.)/maxIter))
    if currentIter == (maxIter-1):
        sys.stdout.write("\n")


def main(msname, solint_sec, solint_hz, colname="CAL_WEIGHT", gainfile="", uvcut_min=80.0,
         uvcut_max=1e6, phaseonly=True, dirname=None, quiet=True):
    """
    Reweight visibilities

    Parameters
    ----------
    filename : str
        Name of the input measurement set
    solint_sec : float
        Solution interval in seconds of calibration
    solint_hz : float
        Solution interval in Hz of calibration
    colname : str, optional
        Name of the weights column name you want to save the weights to
    gainfile : str, optional
        Name of the gain file you want to read to rebuild the calibration quality weights.
        If no file is given, equivalent to rebuilding weights for phase-only calibration
    uvcut_min : float, optional
        Min uvcut in lambda used during calibration
    uvcut_max : float, optional
        Max uvcut in lambda used during calibration
    phaseonly : bool, optional
        Use if calibration was phase-only; this means that gain information doesn't need
        to be read
    dirname : str, optional
        Name of calibration patch
    """
    solint_sec = float(solint_sec)
    solint_hz = float(solint_hz)
    uvcut_min = float(uvcut_min)
    uvcut_max = float(uvcut_max)
    uvcut = [uvcut_min, uvcut_max]
    phaseonly = misc.string2bool(phaseonly)

    covweights = CovWeights(MSName=msname, solint_sec=solint_sec, solint_hz=solint_hz,
                            gainfile=gainfile, uvcut=uvcut, phaseonly=phaseonly,
                            dirname=dirname, quiet=quiet)
    coefficients = covweights.FindWeights(colname=colname)
    covweights.SaveWeights(coefficients, colname=colname)
