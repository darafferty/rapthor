"""
Module that holds the Predict class
"""
import os
import logging
from rapthor.lib.operation import Operation

log = logging.getLogger('rapthor:predict')


class Predict(Operation):
    """
    Operation to predict model data
    """
    def __init__(self, field, index):
        super(Predict, self).__init__(field, name='predict', index=index)

    def set_parset_parameters(self):
        """
        Define parameters needed for the pipeline parset template
        """
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'do_slowgain_solve': self.field.do_slowgain_solve}

    def set_input_parameters(self):
        """
        Define the pipeline inputs
        """
        # Make list of sectors for which prediction needs to be done. Any imaging
        # sectors should come first, followed by bright-source, then outlier sectors
        # (as required by the rapthor/scripts/subtract_sector_models.py script)
        sectors = []
        if len(self.field.imaging_sectors) > 1 or self.field.reweight:
            # If we have more than one imaging sector, reweighting is desired,
            # predict the imaging sector models. (If we have a single imaging
            # sector, we don't need to predict its model data, just that of any
            # outlier or birght-source sectors)
            sectors.extend(self.field.imaging_sectors)
        sectors.extend(self.field.bright_source_sectors)
        sectors.extend(self.field.outlier_sectors)

        # Set sector-dependent parameters
        sector_skymodel = []
        sector_sourcedb = []
        sector_obs_sourcedb = []
        sector_filename = []
        sector_starttime = []
        sector_ntimes = []
        sector_model_filename = []
        sector_patches = []
        for sector in sectors:
            sector.set_prediction_parameters()
            sector_skymodel.append(sector.predict_skymodel_file)
            sdb = os.path.splitext(sector.predict_skymodel_file)[0]+'.sourcedb'
            sector_sourcedb.append(sdb)
            sector_obs_sourcedb.extend([sdb]*len(self.field.observations))
            sector_filename.extend(sector.get_obs_parameters('ms_filename'))
            sector_model_filename.extend(sector.get_obs_parameters('ms_model_filename'))
            sector_patches.extend(sector.get_obs_parameters('patch_names'))
            sector_starttime.extend(sector.get_obs_parameters('predict_starttime'))
            sector_ntimes.extend(sector.get_obs_parameters('predict_ntimes'))

        # Set observation-specific parameters
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
            obs_solint_hz.append(obs.parameters['solint_slow_freqstep'][0] * obs.channelwidth)

        nr_outliers = len(self.field.outlier_sectors)
        peel_outliers = self.field.peel_outliers
        nr_bright = len(self.field.bright_source_sectors)
        peel_bright = self.field.peel_bright_sources
        reweight = self.field.reweight
        min_uv_lambda = self.field.parset['imaging_specific']['min_uv_lambda']
        max_uv_lambda = self.field.parset['imaging_specific']['max_uv_lambda']

        self.input_parms = {'sector_filename': sector_filename,
                            'sector_starttime': sector_starttime,
                            'sector_ntimes': sector_ntimes,
                            'sector_model_filename': sector_model_filename,
                            'sector_skymodel': sector_skymodel,
                            'sector_sourcedb': sector_sourcedb,
                            'sector_obs_sourcedb': sector_obs_sourcedb,
                            'sector_patches': sector_patches,
                            'h5parm': self.field.h5parm_filename,
                            'obs_solint_sec': obs_solint_sec,
                            'obs_solint_hz': obs_solint_hz,
                            'min_uv_lambda': min_uv_lambda,
                            'max_uv_lambda': max_uv_lambda,
                            'obs_filename': obs_filename,
                            'obs_starttime': obs_starttime,
                            'obs_infix': obs_infix,
                            'nr_outliers': nr_outliers,
                            'peel_outliers': peel_outliers,
                            'nr_bright': nr_bright,
                            'peel_bright': peel_bright,
                            'reweight': reweight}

    def finalize(self):
        """
        Finalize this operation
        """
        if self.field.peel_outliers and len(self.field.outlier_sectors) > 0:
            # Update the observations to use the new peeled datasets and remove the
            # outlier sectors (since, once peeled, they are no longer needed)
            self.field.sectors = [sector for sector in self.field.sectors if not sector.is_outlier]
            self.field.outlier_sectors = []

            for sector in self.field.sectors:
                for obs in sector.observations:
                    obs.ms_filename = obs.ms_field  # use new peeled datasets in future

                    # Update MS filename of the field's observation object to match those of
                    # the sector's observation objects. This is required because the sector's
                    # observation objects are distinct copies of the field ones
                    for field_obs in self.field.observations:
                        if (field_obs.name == obs.name) and (field_obs.starttime == obs.starttime):
                            field_obs.ms_field = obs.ms_field
