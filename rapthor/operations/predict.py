"""
Module that holds the Predict class
"""
import os
import logging
from rapthor.lib.operation import Operation
from rapthor.lib.cwl import CWLFile, CWLDir

log = logging.getLogger('rapthor:predict')


class Predict(Operation):
    """
    Operation to predict model data
    """
    def __init__(self, field, index):
        super(Predict, self).__init__(field, name='predict', index=index)

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'max_cores': max_cores,
                             'do_slowgain_solve': self.field.do_slowgain_solve}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Make list of sectors for which prediction needs to be done. Any imaging
        # sectors should come first, followed by bright-source, then outlier sectors
        # (as required by the rapthor/scripts/subtract_sector_models.py script)
        sectors = []
        if len(self.field.imaging_sectors) > 1 or self.field.reweight:
            # If we have more than one imaging sector or reweighting is desired,
            # predict the imaging sector models. (If we have a single imaging
            # sector, we don't need to predict its model data, just that of any
            # outlier or birght-source sectors)
            sectors.extend(self.field.imaging_sectors)
        sectors.extend(self.field.bright_source_sectors)
        sectors.extend(self.field.outlier_sectors)

        # Set sector-dependent parameters (input and output filenames, patch names, etc.)
        sector_skymodel = []
        sector_filename = []
        sector_starttime = []
        sector_ntimes = []
        sector_model_filename = []
        sector_patches = []
        for sector in sectors:
            sector.set_prediction_parameters()
            sector_skymodel.extend(
                [sector.predict_skymodel_file] * len(self.field.observations)
            )
            sector_filename.extend(sector.get_obs_parameters('ms_filename'))
            sector_model_filename.extend(
                [os.path.basename(f) for f in sector.get_obs_parameters('ms_model_filename')]
            )
            sector_patches.extend(sector.get_obs_parameters('patch_names'))
            sector_starttime.extend(sector.get_obs_parameters('predict_starttime'))
            sector_ntimes.extend(sector.get_obs_parameters('predict_ntimes'))

        # Set observation-specific parameters (input filenames, solution intervals, etc.)
        obs_filename = []
        obs_starttime = []
        obs_infix = []
        obs_solint_sec = []
        obs_solint_hz = []
        for obs in self.field.observations:
            obs_filename.append(obs.ms_filename)
            obs_starttime.append(obs.convert_mjd(obs.starttime))
            obs_infix.append(obs.infix)
            obs_solint_sec.append(obs.parameters['solint_fast_timestep'][0] * obs.timepersample)
            obs_solint_hz.append(obs.parameters['solint_slow_freqstep_separate'][0] * obs.channelwidth)

        # Set other parameters
        nr_outliers = len(self.field.outlier_sectors)
        peel_outliers = self.field.peel_outliers
        nr_bright = len(self.field.bright_source_sectors)
        peel_bright = self.field.peel_bright_sources
        reweight = self.field.reweight
        min_uv_lambda = self.field.parset['imaging_specific']['min_uv_lambda']
        max_uv_lambda = self.field.parset['imaging_specific']['max_uv_lambda']
        onebeamperpatch = self.field.onebeamperpatch
        sagecalpredict = self.field.sagecalpredict

        self.input_parms = {'sector_filename': CWLDir(sector_filename).to_json(),
                            'sector_starttime': sector_starttime,
                            'sector_ntimes': sector_ntimes,
                            'sector_model_filename': sector_model_filename,
                            'sector_skymodel': CWLFile(sector_skymodel).to_json(),
                            'sector_patches': sector_patches,
                            'h5parm': CWLFile(self.field.h5parm_filename).to_json(),
                            'obs_solint_sec': obs_solint_sec,
                            'obs_solint_hz': obs_solint_hz,
                            'min_uv_lambda': min_uv_lambda,
                            'max_uv_lambda': max_uv_lambda,
                            'onebeamperpatch': onebeamperpatch,
                            'sagecalpredict' : sagecalpredict,
                            'obs_filename': CWLDir(obs_filename).to_json(),
                            'obs_starttime': obs_starttime,
                            'obs_infix': obs_infix,
                            'nr_outliers': nr_outliers,
                            'peel_outliers': peel_outliers,
                            'nr_bright': nr_bright,
                            'peel_bright': peel_bright,
                            'reweight': reweight,
                            'max_threads': self.field.parset['cluster_specific']['max_threads']}

    def finalize(self):
        """
        Finalize this operation
        """
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
