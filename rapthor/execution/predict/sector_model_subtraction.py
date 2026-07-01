"""Measurement Set helpers for subtracting sector model data."""

import logging
import os
from typing import Optional

import casacore.tables as pt
import numpy as np

from rapthor.execution.outputs import output_path
from rapthor.execution.predict.measurement_sets import (
    copy_measurement_set,
    input_rows_for_models,
    modeldata_output_stem,
    plan_row_chunks,
    predict_chunk_count,
    read_model_data,
    select_models_for_frequency,
    select_models_for_starttime,
    sum_model_data,
)
from rapthor.lib import miscellaneous as misc

log = logging.getLogger("rapthor:predict:sector_model_subtraction")


def get_nchunks(msin, nsectors, fraction=1.0, reweight=False, compressed=False):
    """
    Determines number of chunks for available memory of node

    Parameters
    ----------
    msin : str
        Input MS file name
    nsectors : int
        Number of imaging sectors
    fraction : float
        Fraction of MS file to be read
    reweight: bool
        True if reweighting is to be done
    compressed: bool
        True if data are compressed (by Dysco)

    Returns
    -------
    nchunks : int
        Number of chunks
    """
    scale_factor = 10.0 if reweight else 4.0
    return predict_chunk_count(
        msin,
        nsectors,
        fraction=fraction,
        scale_factor=scale_factor,
        compressed=compressed,
    )


def subtract_sector_models(
    msin,
    model_list,
    msin_column="DATA",
    model_column="DATA",
    out_column="DATA",
    nr_outliers=0,
    nr_bright=0,
    use_compression=False,
    peel_outliers=False,
    peel_bright=False,
    reweight=True,
    starttime=None,
    solint_sec=None,
    solint_hz=None,
    weights_colname="CAL_WEIGHT",
    gainfile="",
    uvcut_min=80.0,
    uvcut_max=1e6,
    phaseonly=True,
    dirname=None,
    quiet=True,
    infix="",
    output_dir: Optional[str] = None,
):
    """
    Subtract sector model data.

    Parameters
    ----------
    msin : str
        Name of MS file from which subtraction will be done
    model_list: list
        List of model data MS filenames
    msin_column : str, optional
        Name of input column
    model_column : str, optional
        Name of model column
    out_column : str, optional
        Name of output column
    nr_outliers : int, optional
        Number of outlier sectors. Outlier sectors must be given after normal sectors
        and bright-source sectors in msmod_list
    nr_bright : int, optional
        Number of bright-source sectors. Bright-source sectors must be given after normal
        sectors but before outlier sectors in msmod_list
    use_compression : bool, optional
        If True, use Dysco compression
    peel_outliers : bool, optional
        If True, outliers are peeled before sector models are subtracted
    peel_bright : bool, optional
        If True, bright sources are peeled before sector models are subtracted
    reweight : bool, optional
        If True, reweight using the residuals
    starttime : str, optional
        Start time in JD seconds
    solint_sec : float
        Solution interval in s
    solint_hz : float
        Solution interval in Hz
    weights_colname : str
        Name of weight column
    gainfile : str
        Filename of gain file
    uvcut_min : float
        Min uv cut in lambda
    uvcut_max : float
        Max uv cut in lambda
    phaseonly : bool
        Reweight with phases only
    dirname : str
        Name of gain file directory
    quiet : bool
        If True, suppress (most) output
    infix : str, optional
        Infix string used in filenames
    output_dir : str, optional
        Directory for generated Measurement Set outputs. If omitted, outputs
        are written relative to the current working directory for CLI parity.
    """
    use_compression = misc.string2bool(use_compression)
    peel_outliers = misc.string2bool(peel_outliers)
    peel_bright = misc.string2bool(peel_bright)
    nr_outliers = int(nr_outliers)
    nr_bright = int(nr_bright)
    solint_sec = float(solint_sec)
    solint_hz = float(solint_hz)
    uvcut_min = float(uvcut_min)
    uvcut_max = float(uvcut_max)
    uvcut = [uvcut_min, uvcut_max]
    phaseonly = misc.string2bool(phaseonly)
    reweight = misc.string2bool(reweight)

    # Get the model data filenames, filtering any that do not have the right start time.
    model_selection = select_models_for_starttime(model_list, starttime)
    model_list = select_models_for_frequency(
        msin,
        model_selection.paths,
        spectral_window_separator="/",
    )
    nsectors = len(model_list)
    if nsectors == 0:
        raise ValueError("No model data found.")
    log.info("Found %s model data files", nsectors)

    # Define the template MS file. This file is copied to one or more files
    # to be filled with new data
    ms_template = model_list[0]

    # If starttime is given, figure out startrow and nrows for input MS file
    tin = pt.table(msin, readonly=True, ack=False)
    input_rows = input_rows_for_models(
        tin,
        starttime=starttime,
        starttime_exact=model_selection.starttime_exact,
        model_nrows=model_selection.nrows,
    )
    startrow_in = input_rows.startrow
    nrows_in = input_rows.nrows
    nbl = input_rows.baseline_rows
    tin.close()

    # If outliers are to be peeled, do that first
    if peel_outliers and nr_outliers > 0:
        # Open input and output table
        tin = pt.table(msin, readonly=True, ack=False)
        root_filename = os.path.basename(msin)
        msout = output_path(output_dir, f"{root_filename}{infix}_field")

        copy_measurement_set(ms_template, msout)
        tout = pt.table(msout, readonly=False, ack=False)

        # Define chunks based on available memory
        fraction = float(nrows_in) / float(tin.nrows())
        nchunks = get_nchunks(msin, nr_outliers, fraction, compressed=True)
        chunks = plan_row_chunks(
            nrows=nrows_in,
            nchunks=nchunks,
            input_startrow=startrow_in,
        )
        log.info("Using %s chunk(s) for peeling of outliers", len(chunks))

        for chunk in chunks:
            # For each chunk, load data
            datain = tin.getcol(msin_column, startrow=chunk.input_startrow, nrow=chunk.nrows)
            if use_compression:
                # Replace flagged values with NaNs before compression
                flags = tin.getcol("FLAG", startrow=chunk.input_startrow, nrow=chunk.nrows)
                flagged = np.where(flags)
                datain[flagged] = np.NaN
            datamod_list = read_model_data(
                model_list[nsectors - nr_outliers :],
                model_column,
                startrow=chunk.model_startrow,
                nrows=chunk.nrows,
            )

            # Subtract sum of model data for this chunk
            datamod_all = sum_model_data(datamod_list)
            tout.putcol(
                out_column,
                datain - datamod_all,
                startrow=chunk.model_startrow,
                nrow=chunk.nrows,
            )
            tout.flush()
        tout.close()
        tin.close()

        # Now reset things for the imaging sectors
        msin = msout
        model_list = model_list[:-nr_outliers]
        nsectors = len(model_list)
        nr_outliers = 0
        startrow_in = 0
        datain = None
        datamod_all = None
        datamod_list = None

    # Next, peel the bright sources if desired
    if peel_bright and nr_bright > 0:
        # Open input and output table
        tin = pt.table(msin, readonly=True, ack=False)
        root_filename = os.path.basename(msin)
        msout = output_path(output_dir, f"{root_filename}{infix}_field_no_bright")

        copy_measurement_set(ms_template, msout)
        tout = pt.table(msout, readonly=False, ack=False)

        # Define chunks based on available memory
        fraction = float(nrows_in) / float(tin.nrows())
        nchunks = get_nchunks(msin, nr_bright, fraction, compressed=True)
        chunks = plan_row_chunks(
            nrows=nrows_in,
            nchunks=nchunks,
            input_startrow=startrow_in,
        )
        log.info("Using %s chunk(s) for peeling of bright sources", len(chunks))

        for chunk in chunks:
            # For each chunk, load data
            datain = tin.getcol(msin_column, startrow=chunk.input_startrow, nrow=chunk.nrows)
            if use_compression:
                # Replace flagged values with NaNs before compression
                flags = tin.getcol("FLAG", startrow=chunk.input_startrow, nrow=chunk.nrows)
                flagged = np.where(flags)
                datain[flagged] = np.NaN
            datamod_list = read_model_data(
                model_list[nsectors - nr_bright :],
                model_column,
                startrow=chunk.model_startrow,
                nrows=chunk.nrows,
            )

            # Subtract sum of model data for this chunk
            datamod_all = sum_model_data(datamod_list)
            tout.putcol(
                out_column,
                datain - datamod_all,
                startrow=chunk.model_startrow,
                nrow=chunk.nrows,
            )
            tout.flush()
        tout.close()
        tin.close()

        # Now reset things for the imaging sectors
        msin = msout
        model_list = model_list[:-nr_bright]
        nsectors = len(model_list)
        nr_bright = 0
        startrow_in = 0
        datain = None
        datamod_all = None
        datamod_list = None

    if len(model_list) == 0:
        # This means there is just a single sector and no reweighting is to be done,
        # so use the template MS filename as a basis for the output MS filename
        msout = output_path(
            output_dir,
            os.path.splitext(os.path.basename(ms_template))[0] + ".sector_1",
        )
        copy_measurement_set(msin, msout)
        return

    # Open input table and define chunks based on available memory, making sure each
    # chunk gives a full timeslot (needed for reweighting)
    tin = pt.table(msin, readonly=True, ack=False)
    fraction = float(nrows_in) / float(tin.nrows())
    nchunks = get_nchunks(msin, nsectors, fraction, reweight=reweight)
    chunks = plan_row_chunks(
        nrows=nrows_in,
        nchunks=nchunks,
        input_startrow=startrow_in,
        baseline_rows=nbl,
    )
    log.info("Using %s chunk(s) for peeling of sector sources", len(chunks))

    # Open output tables
    tout_list = []
    for i, msmod in enumerate(model_list):
        if nr_bright > 0 and nr_outliers > 0 and i == len(model_list) - nr_outliers - nr_bright:
            # Break so we don't open output tables for the outliers or bright sources
            break
        elif nr_outliers > 0 and i == len(model_list) - nr_outliers:
            # Break so we don't open output tables for the outliers
            break
        elif nr_bright > 0 and i == len(model_list) - nr_bright:
            # Break so we don't open output tables for the bright sources
            break
        msout = output_path(output_dir, modeldata_output_stem(msmod))

        copy_measurement_set(ms_template, msout)
        tout_list.append(pt.table(msout, readonly=False, ack=False))

    # Process the data chunk by chunk
    for chunk in chunks:
        # For each chunk, load data
        datain = tin.getcol(msin_column, startrow=chunk.input_startrow, nrow=chunk.nrows)
        flags = tin.getcol("FLAG", startrow=chunk.input_startrow, nrow=chunk.nrows)
        datamod_list = read_model_data(
            model_list,
            model_column,
            startrow=chunk.model_startrow,
            nrows=chunk.nrows,
        )

        # For each sector, subtract sum of model data of all other sectors
        weights = None
        for i, tout in enumerate(tout_list):
            other_sectors_ind = list(range(nsectors))
            other_sectors_ind.pop(i)
            datamod_all = None
            for sector_ind in other_sectors_ind:
                if datamod_all is None:
                    datamod_all = datamod_list[sector_ind].copy()
                else:
                    datamod_all += datamod_list[sector_ind]
            if datamod_all is not None:
                tout.putcol(
                    out_column,
                    datain - datamod_all,
                    startrow=chunk.model_startrow,
                    nrow=chunk.nrows,
                )
            else:
                tout.putcol(
                    out_column,
                    datain,
                    startrow=chunk.model_startrow,
                    nrow=chunk.nrows,
                )
            if reweight:
                # Also subtract sector's own model to make residual data for reweighting
                if weights is None:
                    if datamod_all is None:
                        datamod_all = datamod_list[i]
                    else:
                        datamod_all += datamod_list[i]
                    covweights = CovWeights(
                        model_list[0],
                        solint_sec,
                        solint_hz,
                        chunk.model_startrow,
                        chunk.nrows,
                        gainfile=gainfile,
                        uvcut=uvcut,
                        phaseonly=phaseonly,
                        dirname=dirname,
                        quiet=quiet,
                    )
                    coefficients = covweights.FindWeights(datain - datamod_all, flags)
                    weights = covweights.calcWeights(coefficients)
                    covweights = None
                tout.putcol(
                    weights_colname,
                    weights,
                    startrow=chunk.model_startrow,
                    nrow=chunk.nrows,
                )
            tout.flush()
    for tout in tout_list:
        tout.close()
    tin.close()


"""
The following reweighting code is based on that of
https://github.com/ebonnassieux/Scripts/blob/master/QualityWeightsLOFAR.py
"""


class CovWeights:
    def __init__(
        self,
        MSName,
        solint_sec,
        solint_hz,
        startrow,
        nrow,
        uvcut=[0, 2000],
        gainfile=None,
        phaseonly=False,
        dirname=None,
        quiet=True,
    ):
        if MSName[-1] == "/":
            self.MSName = MSName[0:-1]
        else:
            self.MSName = MSName
        tab = pt.table(self.MSName, ack=False)
        self.timepersample = tab.getcell("EXPOSURE", 0)
        self.ntSol = max(1, int(round(solint_sec / self.timepersample)))
        tab.close()
        sw = pt.table(self.MSName + "::SPECTRAL_WINDOW", ack=False)
        self.referencefreq = sw.col("REF_FREQUENCY")[0]
        self.channelwidth = sw.col("CHAN_WIDTH")[0][0]
        self.numchannels = sw.col("NUM_CHAN")[0]
        sw.close()
        self.nchanSol = max(1, self.get_nearest_frequstep(solint_hz / self.channelwidth))
        self.uvcut = uvcut
        self.gainfile = gainfile
        self.phaseonly = phaseonly
        self.dirname = dirname
        self.quiet = quiet
        self.startrow = startrow
        self.nrow = nrow

    def FindWeights(self, residualdata, flags):
        ms = pt.table(self.MSName, ack=False)
        ants = pt.table(ms.getkeyword("ANTENNA"), ack=False)
        antnames = ants.getcol("NAME")
        ants.close()
        nAnt = len(antnames)

        u, v, _ = ms.getcol("UVW", startrow=self.startrow, nrow=self.nrow).T
        A0 = ms.getcol("ANTENNA1", startrow=self.startrow, nrow=self.nrow)
        A1 = ms.getcol("ANTENNA2", startrow=self.startrow, nrow=self.nrow)
        tarray = ms.getcol("TIME", startrow=self.startrow, nrow=self.nrow)
        nbl = np.where(tarray == tarray[0])[0].size
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
        nt = int(residualdata.shape[0] / nbl)
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
                            np.append(
                                residuals[t_i:t_e, set1, f_i:f_e, :],
                                residuals[t_i:t_e, set2, f_i:f_e, :],
                            )
                            * np.append(
                                residuals[t_i:t_e, set1, f_i:f_e, :],
                                residuals[t_i:t_e, set2, f_i:f_e, :],
                            ).conj()
                        )
                        - np.nanstd(
                            np.append(
                                rmsarray[t_i:t_e, set1, f_i:f_e, :],
                                rmsarray[t_i:t_e, set2, f_i:f_e, :],
                            )
                        )
                    )

        # get rid of NaNs and low values
        CoeffArray[~np.isfinite(CoeffArray)] = np.inf
        for i in range(nAnt):
            tempars = CoeffArray[:, :, i]
            thres = 0.25 * np.median(tempars[np.where(np.isfinite(tempars))])
            CoeffArray[:, :, i][tempars < thres] = thres
        return CoeffArray

    def calcWeights(self, CoeffArray, max_radius=5e3):
        ms = pt.table(self.MSName, readonly=True, ack=False)
        ants = pt.table(ms.getkeyword("ANTENNA"), ack=False)
        antnames = ants.getcol("NAME")
        nAnt = len(antnames)
        tarray = ms.getcol("TIME", startrow=self.startrow, nrow=self.nrow)
        darray = ms.getcol("DATA", startrow=self.startrow, nrow=self.nrow)
        tvalues = np.array(sorted(list(set(tarray))))
        nt = tvalues.shape[0]
        nbl = int(tarray.shape[0] / nt)
        nchan = darray.shape[1]
        npol = darray.shape[2]
        A0 = np.array(
            ms.getcol("ANTENNA1", startrow=self.startrow, nrow=self.nrow).reshape((nt, nbl))
        )
        A1 = np.array(
            ms.getcol("ANTENNA2", startrow=self.startrow, nrow=self.nrow).reshape((nt, nbl))
        )

        # initialize weight array
        w = np.zeros((nt, nbl, nchan, npol))
        A0ind = A0[0, :]
        A1ind = A1[0, :]

        # do gains stuff
        ant1gainarray, ant2gainarray = readGainFile(
            self.gainfile,
            ms,
            nt,
            nchan,
            nbl,
            tarray,
            nAnt,
            self.MSName,
            self.phaseonly,
            self.dirname,
            self.startrow,
            self.nrow,
        )
        ant1gainarray = ant1gainarray.reshape((nt, nbl, nchan))
        ant2gainarray = ant2gainarray.reshape((nt, nbl, nchan))
        for t in range(nt):
            for i in range(nbl):
                for j in range(nchan):
                    w[t, i, j, :] = 1.0 / (
                        CoeffArray[t, j, A0ind[i]] * ant1gainarray[t, i, j]
                        + CoeffArray[t, j, A1ind[i]] * ant2gainarray[t, i, j]
                        + CoeffArray[t, j, A0ind[i]] * CoeffArray[t, j, A1ind[i]]
                        + 0.1
                    )

        # If desired, force the weights to be equal for the short baselines (this ensures
        # that shorter baselines are not downweighted due to, e.g., residual flux from poor
        # subtraction of the field)
        if max_radius is not None:
            u, v, _ = ms.getcol("UVW", startrow=self.startrow, nrow=self.nrow).T
            uvlen = np.sqrt(u**2 + v**2).reshape(nt, nbl)
            for t in range(nt):
                for p in range(npol):
                    for j in range(nchan):
                        core_bl_ind = np.where(uvlen[t, :] < max_radius)
                        w_core = w[t, core_bl_ind, j, p]
                        w_core[np.isinf(w_core)] = np.nan
                        w_core[:] = np.nanmean(w_core)
                        w[t, core_bl_ind, j, p] = w_core

        # normalize
        w = w.reshape(nt * nbl, nchan, npol)
        w[np.isinf(w)] = np.nan
        w = w / np.nanmean(w)
        w[~np.isfinite(w)] = 0

        return w

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
        if not hasattr(self, "freq_divisors"):
            tmp_divisors = []
            for step in range(self.numchannels, 0, -1):
                if (self.numchannels % step) == 0:
                    tmp_divisors.append(step)
            self.freq_divisors = np.array(tmp_divisors)

        # Find nearest
        idx = np.argmin(np.abs(self.freq_divisors - freqstep))

        return self.freq_divisors[idx]


def readGainFile(
    gainfile,
    ms,
    nt,
    nchan,
    nbl,
    tarray,
    nAnt,
    msname,
    phaseonly,
    dirname,
    startrow,
    nrow,
):
    if phaseonly:
        ant1gainarray1 = np.ones((nt * nbl, nchan))
        ant2gainarray1 = np.ones((nt * nbl, nchan))
    else:
        import losoto.h5parm

        solsetName = "sol000"
        soltabName = "screenamplitude000"
        try:
            gfile = losoto.h5parm.openSoltab(gainfile, solsetName=solsetName, soltabName=soltabName)
        except Exception:
            log.warning("Could not find amplitude gains in h5parm. Assuming gains of 1 everywhere.")
            ant1gainarray1 = np.ones((nt * nbl, nchan))
            ant2gainarray1 = np.ones((nt * nbl, nchan))
            return ant1gainarray1, ant2gainarray1

        freqs = pt.table(msname + "/SPECTRAL_WINDOW").getcol("CHAN_FREQ")
        gains = gfile.val  # axes: times, freqs, ants, dirs, pols
        flagged = np.where(gains == 0.0)
        gains[flagged] = np.nan
        gfreqs = gfile.freq
        times = gfile.time
        dindx = gfile.dir.tolist().index(dirname)
        ant1gainarray = np.zeros((nt * nbl, nchan))
        ant2gainarray = np.zeros((nt * nbl, nchan))
        A0arr = ms.getcol("ANTENNA1", startrow=startrow, nrow=nrow)
        A1arr = ms.getcol("ANTENNA2", startrow=startrow, nrow=nrow)
        deltime = (times[1] - times[0]) / 2.0
        delfreq = (gfreqs[1] - gfreqs[0]) / 2.0
        for i in range(len(times)):
            timemask = (tarray >= times[i] - deltime) * (tarray < times[i] + deltime)
            if np.all(~timemask):
                continue
            for j in range(nAnt):
                mask1 = timemask * (A0arr == j)
                mask2 = timemask * (A1arr == j)
                for k in range(nchan):
                    chan_freq = freqs[0, k]
                    freqmask = np.logical_and(
                        gfreqs >= chan_freq - delfreq, gfreqs < chan_freq + delfreq
                    )
                    if chan_freq < gfreqs[0]:
                        freqmask[0] = True
                    if chan_freq > gfreqs[-1]:
                        freqmask[-1] = True
                    ant1gainarray[mask1, k] = np.nanmean(
                        gains[i, freqmask, j, dindx, :], axis=(0, 1)
                    )
                    ant2gainarray[mask2, k] = np.nanmean(
                        gains[i, freqmask, j, dindx, :], axis=(0, 1)
                    )
        ant1gainarray1 = ant1gainarray**2
        ant2gainarray1 = ant2gainarray**2

    return ant1gainarray1, ant2gainarray1
