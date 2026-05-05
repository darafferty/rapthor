"""
Module that holds the Predict classes
"""
import os
import logging
from rapthor.lib.operation import Operation
from rapthor.lib.cwl import CWLFile, CWLDir
from rapthor.lib import miscellaneous as misc

log = logging.getLogger('rapthor:predict')


def _collect_sector_parameters(sectors, observations):
    """
    Collect sector-dependent CWL input parameters for a list of sectors.

    Returns a dict with keys: sector_skymodel, sector_filename,
    sector_model_filename, sector_patches, sector_starttime, sector_ntimes.
    """
    sector_skymodel = []
    sector_filename = []
    sector_starttime = []
    sector_ntimes = []
    sector_model_filename = []
    sector_patches = []
    for sector in sectors:
        sector.set_prediction_parameters()
        sector_skymodel.extend([sector.predict_skymodel_file] * len(observations))
        sector_filename.extend(sector.get_obs_parameters('ms_filename'))
        sector_model_filename.extend(
            [os.path.basename(f) for f in sector.get_obs_parameters('ms_model_filename')]
        )
        sector_patches.extend(sector.get_obs_parameters('patch_names'))
        sector_starttime.extend(sector.get_obs_parameters('predict_starttime'))
        sector_ntimes.extend(sector.get_obs_parameters('predict_ntimes'))
    return {
        'sector_skymodel': sector_skymodel,
        'sector_filename': sector_filename,
        'sector_model_filename': sector_model_filename,
        'sector_patches': sector_patches,
        'sector_starttime': sector_starttime,
        'sector_ntimes': sector_ntimes,
    }


def _collect_obs_parameters(observations, include_solint=False):
    """
    Collect observation-specific CWL input parameters.

    Returns a dict with keys: obs_filename, obs_starttime, obs_infix, and
    optionally obs_solint_sec, obs_solint_hz (when include_solint=True).
    """
    obs_filename = []
    obs_starttime = []
    obs_infix = []
    obs_solint_sec = []
    obs_solint_hz = []
    for obs in observations:
        obs_filename.append(obs.ms_filename)
        obs_starttime.append(misc.convert_mjd2mvt(obs.starttime))
        obs_infix.append(obs.infix)
        if include_solint:
            if ('solint_fast_timestep' in obs.parameters and
                    'solint_slow_freqstep_separate' in obs.parameters):
                obs_solint_sec.append(obs.parameters['solint_fast_timestep'][0] * obs.timepersample)
                obs_solint_hz.append(obs.parameters['solint_slow_freqstep_separate'][0] * obs.channelwidth)
            else:
                obs_solint_sec.append(0)
                obs_solint_hz.append(0)
    result = {
        'obs_filename': obs_filename,
        'obs_starttime': obs_starttime,
        'obs_infix': obs_infix,
    }
    if include_solint:
        result['obs_solint_sec'] = obs_solint_sec
        result['obs_solint_hz'] = obs_solint_hz
    return result


def _get_dp3_applycal_steps(field):
    """
    Return the DP3 applycal steps and normalize_h5parm based on field settings.

    Returns a tuple of (dp3_applycal_steps, normalize_h5parm).
    """
    dp3_applycal_steps = ['fastphase']
    if field.apply_amplitudes:
        dp3_applycal_steps.append('slowgain')
    if field.apply_normalizations:
        normalize_h5parm = CWLFile(field.normalize_h5parm).to_json()
        dp3_applycal_steps.append('normalization')
    else:
        normalize_h5parm = None
    return dp3_applycal_steps, normalize_h5parm


class Predict(Operation):
    """
    Operation to predict model data for further direction-dependent (DD)
    processing
    """
    def __init__(self, mode, field, index):
        if mode not in ["di", "dd"]:
            raise ValueError(f"Only di and dd mode are supported, chosen: {mode}")
        super().__init__(field, index=index, name="predict_di" if mode == "di" else "predict")
        self.mode = mode

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system.startswith('slurm'):
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'max_cores': max_cores}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Make list of sectors for which prediction needs to be done. Any imaging
        # sectors should come first, followed by bright-source, then outlier sectors
        # (as required by the rapthor/scripts/subtract_sector_models.py script)
        sectors = []
        if self.mode == "dd":
            include_solint = True
            if len(self.field.imaging_sectors) > 1 or self.field.reweight:
                # If we have more than one imaging sector or reweighting is desired,
                # predict the imaging sector models. (If we have a single imaging
                # sector, we don't need to predict its model data, just that of any
                # outlier or bright-source sectors)
                sectors.extend(self.field.imaging_sectors)
            sectors.extend(self.field.bright_source_sectors)
            sectors.extend(self.field.outlier_sectors)
        else: # if mode = di
            sectors = self.field.predict_sectors

        # shared logic
        sector_parms = _collect_sector_parameters(sectors, self.field.observations)
        # Set observation-specific parameters (input filenames, solution intervals, etc.)
        obs_parms = _collect_obs_parameters(self.field.observations, include_solint=include_solint)

        # Set the DP3 applycal steps depending on what solutions need to be
        # applied
        dp3_applycal_steps, normalize_h5parm = _get_dp3_applycal_steps(self.field)

        dd_params = {}
        if self.mode == "dd":
            dd_params = {
                'obs_solint_sec': obs_parms['obs_solint_sec'],
                'obs_solint_hz': obs_parms['obs_solint_hz'],
                'min_uv_lambda': self.field.parset['imaging_specific']['min_uv_lambda'],
                'max_uv_lambda': self.field.parset['imaging_specific']['max_uv_lambda'],
                'nr_outliers': len(self.field.outlier_sectors),
                'peel_outliers': self.field.peel_outliers,
                'nr_bright': len(self.field.bright_source_sectors),
                'peel_bright': self.field.peel_bright_sources,
                'reweight': self.field.reweight

            }
   
        common_params = {
            'sector_filename': CWLDir(sector_parms['sector_filename']).to_json(),
            'data_colname': self.field.data_colname,
            'sector_starttime': sector_parms['sector_starttime'],
            'sector_ntimes': sector_parms['sector_ntimes'],
            'sector_model_filename': sector_parms['sector_model_filename'],
            'sector_skymodel': CWLFile(sector_parms['sector_skymodel']).to_json(),
            'sector_patches': sector_parms['sector_patches'],

            'h5parm': CWLFile(self.field.h5parm_filename).to_json(),
            'normalize_h5parm': normalize_h5parm,
            'dp3_applycal_steps': f"[{','.join(dp3_applycal_steps)}]",

            'onebeamperpatch': self.field.onebeamperpatch,
            'sagecalpredict': self.field.sagecalpredict,
            'obs_filename': CWLDir(obs_parms['obs_filename']).to_json(),
            'obs_starttime': obs_parms['obs_starttime'],
            'obs_infix': obs_parms['obs_infix'],

            'correctfreqsmearing': self.field.correct_smearing_in_calibration,
            'correcttimesmearing': self.field.correct_smearing_in_calibration,
            'max_threads': self.field.parset['cluster_specific']['max_threads']}
        
        self.input_parms = {**common_params, **dd_params}
        

    def finalize(self):
        """
        Finalize this operation
        """
        self.field.data_colname = 'DATA'
        if self.field.peel_outliers:
            # Update the observations to use the new peeled datasets and remove the
            # outlier sectors (since, once peeled, they are no longer needed)
            self.field.sectors = [sector for sector in self.field.sectors if not sector.is_outlier]
            self.field.outlier_sectors = []

            # From now on, use imaged sources only in the sky models for selfcal, since
            # sources outside of imaged areas have been peeled
            self.field.imaged_sources_only = True

            for sector in self.field.sectors:
                for obs in sector.observations:
                    # Use new peeled datasets in future
                    obs.ms_filename = os.path.join(self.pipeline_working_dir, obs.ms_field)

                    # Remove infix for the sector observations, otherwise future predict
                    # operations will add it to the filenames multiple times
                    obs.infix = ''

                    # Update MS filename and infix of the field's observations to match
                    # those of the sector's observations. This is required because the
                    # sector's observations are distinct copies of the field ones
                    for field_obs in self.field.observations:
                        if (field_obs.name == obs.name) and (field_obs.starttime == obs.starttime):
                            field_obs.infix = ''
                            field_obs.ms_filename = obs.ms_filename

        # Update filenames of datasets used for imaging
        if (len(self.field.imaging_sectors) > 1 or self.field.reweight or
                (len(self.field.outlier_sectors) > 0 and self.field.peel_outliers) or
                (len(self.field.bright_source_sectors) > 0 and self.field.peel_bright_sources)):
            for sector in self.field.sectors:
                for obs in sector.observations:
                    obs.ms_imaging_filename = os.path.join(self.pipeline_working_dir,
                                                           obs.ms_subtracted_filename)
        else:
            for sector in self.field.sectors:
                for obs in sector.observations:
                    obs.ms_imaging_filename = obs.ms_filename

        # Finally call finalize() in the parent class
        super().finalize()


class PredictDI(Operation):
    """
    Operation to predict model data for direction-independent calibration
    """
    def __init__(self, field, index):
        super().__init__(field, index=index, name='predict_di')

    
    def finalize(self):
        """
        Finalize this operation
        """
        # Set the filenames of datasets used for direction-independent calibration
        for obs in self.field.predict_sectors[0].observations:
            # Transfer the filenames from the first sector to the field. This is required
            # because the sector's observations are distinct copies of the field ones
            for field_obs in self.field.observations:
                if (field_obs.name == obs.name) and (field_obs.starttime == obs.starttime):
                    field_obs.ms_predict_di_filename = os.path.join(self.pipeline_working_dir,
                                                                    obs.ms_predict_di)

        # Finally call finalize() in the parent class
        super().finalize()
