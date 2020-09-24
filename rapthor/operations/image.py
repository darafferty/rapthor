"""
Module that holds the Image class
"""
import os
import logging
from rapthor.lib.operation import Operation
from rapthor.lib import miscellaneous as misc

log = logging.getLogger('rapthor:image')


class Image(Operation):
    """
    Operation to image a field sector
    """
    def __init__(self, field, index):
        super(Image, self).__init__(field, name='image', index=index)

        # For imaging we use a subworkflow, so we set the template filename for that here
        self.subpipeline_parset_template = '{0}_sector_pipeline.cwl'.format(self.rootname)

    def set_parset_parameters(self):
        """
        Define parameters needed for the pipeline parset template
        """
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'pipeline_working_dir': self.pipeline_working_dir,
                             'do_slowgain_solve': self.field.do_slowgain_solve,
                             'use_screens': self.field.use_screens,
                             'peel_bright_sources': self.field.peel_bright_sources,
                             'use_mpi': self.field.use_mpi}

    def set_input_parameters(self):
        """
        Define the pipeline inputs
        """
        nsectors = len(self.field.imaging_sectors)
        obs_filename = []
        prepare_filename = []
        previous_mask_filename = []
        mask_filename = []
        aterms_config_file = []
        starttime = []
        ntimes = []
        aterm_image_filenames = []
        image_freqstep = []
        image_timestep = []
        multiscale_scales_pixel = []
        local_dir = []
        phasecenter = []
        image_root = []
        central_patch_name = []
        for i, sector in enumerate(self.field.imaging_sectors):
            # Set the imaging parameters for each imaging sector. Note the we do not
            # let the imsize be recalcuated, as otherwise it may change from the previous
            # iteration and the mask made in that iteration can not be used in this one.
            # Generally, this should work fine, since we do not expect large changes in
            # the size of the sector from iteration to iteration (small changes are OK,
            # given the padding we use during imaging)
            sector_do_multiscale_list = self.field.parset['imaging_specific']['sector_do_multiscale_list']
            if len(sector_do_multiscale_list) == nsectors:
                do_multiscale = sector_do_multiscale_list[i]
            else:
                do_multiscale = None
            sector.set_imaging_parameters(do_multiscale=do_multiscale, recalculate_imsize=False)

            if self.field.do_predict:
                # If predict was done, use the model-subtracted/reweighted data
                sector_obs_filename = sector.get_obs_parameters('ms_subtracted_filename')
            else:
                sector_obs_filename = sector.get_obs_parameters('ms_filename')
            obs_filename.append(sector_obs_filename)

            # Each image job must have its own directory, so we create it here
            image_dir = os.path.join(self.pipeline_working_dir, sector.name)
            misc.create_directory(image_dir)
            image_root.append(os.path.join(image_dir, sector.name))

            prepare_filename.append([os.path.join(image_dir, os.path.basename(of)+'.prep')
                                     for of in sector_obs_filename])
            if sector.I_mask_file is not None:
                # Use the existing mask
                previous_mask_filename.append(sector.I_mask_file)
            else:
                # Use a dummy mask
                previous_mask_filename.append(image_root[-1] + '_dummy.fits')
            mask_filename.append(image_root[-1] + '_mask.fits')
            aterms_config_file.append(image_root[-1] + '_aterm.cfg')
            image_freqstep.append(sector.get_obs_parameters('image_freqstep'))
            image_timestep.append(sector.get_obs_parameters('image_timestep'))
            sector_starttime = []
            sector_ntimes = []
            for obs in self.field.observations:
                sector_starttime.append(obs.convert_mjd(obs.starttime))
                sector_ntimes.append(obs.numsamples)
            starttime.append(sector_starttime)
            ntimes.append(sector_ntimes)
            phasecenter.append("'[{0}deg, {1}deg]'".format(sector.ra, sector.dec))
            if self.temp_dir is None:
                local_dir.append(image_dir)
            else:
                local_dir.append(self.temp_dir)
            multiscale_scales_pixel.append("'{}'".format(sector.multiscale_scales_pixel))
            central_patch_name.append(sector.central_patch)

            # The following attribute was set by the preceding calibrate operation
            aterm_image_filenames.append("'[{}]'".format(','.join(self.field.aterm_image_filenames)))

        self.input_parms = {'obs_filename': obs_filename,
                            'prepare_filename': prepare_filename,
                            'previous_mask_filename': previous_mask_filename,
                            'mask_filename': mask_filename,
                            'starttime': starttime,
                            'ntimes': ntimes,
                            'image_freqstep': image_freqstep,
                            'image_timestep': image_timestep,
                            'phasecenter': phasecenter,
                            'image_name': image_root,
                            'multiscale_scales_pixel': multiscale_scales_pixel,
                            'local_dir': local_dir,
                            'do_slowgain_solve': [self.field.do_slowgain_solve] * nsectors,
                            'channels_out': [sector.wsclean_nchannels for sector in self.field.imaging_sectors],
                            'deconvolution_channels': [sector.wsclean_nchannels for sector in self.field.imaging_sectors],
                            'ra': [sector.ra for sector in self.field.imaging_sectors],
                            'dec': [sector.dec for sector in self.field.imaging_sectors],
                            'wsclean_imsize': [sector.imsize for sector in self.field.imaging_sectors],
                            'vertices_file': [sector.vertices_file for sector in self.field.imaging_sectors],
                            'region_file': [sector.region_file for sector in self.field.imaging_sectors],
                            'wsclean_niter': [sector.wsclean_niter for sector in self.field.imaging_sectors],
                            'wsclean_nmiter': [sector.wsclean_nmiter for sector in self.field.imaging_sectors],
                            'robust': [sector.robust for sector in self.field.imaging_sectors],
                            'wsclean_image_padding': [sector.wsclean_image_padding for sector in self.field.imaging_sectors],
                            'cellsize_deg': [sector.cellsize_deg for sector in self.field.imaging_sectors],
                            'min_uv_lambda': [sector.min_uv_lambda for sector in self.field.imaging_sectors],
                            'max_uv_lambda': [sector.max_uv_lambda for sector in self.field.imaging_sectors],
                            'taper_arcsec': [sector.taper_arcsec for sector in self.field.imaging_sectors],
                            'auto_mask': [sector.auto_mask for sector in self.field.imaging_sectors],
                            'idg_mode': [sector.idg_mode for sector in self.field.imaging_sectors],
                            'threshisl': [sector.threshisl for sector in self.field.imaging_sectors],
                            'threshpix': [sector.threshpix for sector in self.field.imaging_sectors],
                            'bright_skymodel_pb': [self.field.bright_source_skymodel_file] * nsectors,
                            'peel_bright': [self.field.peel_bright_sources] * nsectors}
        if self.field.use_screens:
            self.input_parms.update({'aterms_config_file': aterms_config_file,
                                     'aterm_image_filenames': aterm_image_filenames})

            if self.field.use_mpi:
                # Set number of nodes to allocate to each imaging subpipeline
                nnodes =  self.parset['cluster_specific']['max_nodes']
                nsubpipes = min(nsectors, nnodes)
                nnodes_per_subpipeline = int(nnodes / nsubpipes)

                self.input_parms.update({'mpi_ntasks_per_node': [self.parset['cluster_specific']['ncpu']] * nsectors,
                                         'mpi_nnodes': [nnodes_per_subpipeline] * nsectors})
        else:
            self.input_parms.update({'h5parm': [self.field.h5parm_filename] * nsectors})
            self.input_parms.update({'central_patch_name': central_patch_name})

    def finalize(self):
        """
        Finalize this operation
        """
        # Save output FITS image and model for each sector
        # NOTE: currently, -save-source-list only works with pol=I -- when it works with other
        # pols, save them all
        for sector in self.field.imaging_sectors:
            image_root = os.path.join(self.pipeline_working_dir, sector.name, sector.name)
            sector.I_image_file_true_sky = image_root + '-MFS-image-pb.fits'
            sector.I_image_file_apparent_sky = image_root + '-MFS-image.fits'
            sector.I_model_file_true_sky = image_root + '-MFS-model-pb.fits'

            # Check to see if a clean mask image was made (only made when at least one
            # island is found in the Stokes I image). The filename is defined
            # in the rapthor/scripts/filter_skymodel.py file
            mask_filename = sector.I_image_file_apparent_sky + '.mask'
            if os.path.exists(mask_filename):
                sector.I_mask_file = mask_filename
            else:
                sector.I_mask_file = None

            # The sky models, both true sky and apparent sky (the filenames are defined
            # in the rapthor/scripts/filter_skymodel.py file)
            sector.image_skymodel_file_true_sky = image_root + '.true_sky'
            sector.image_skymodel_file_apparent_sky = image_root + '.apparent_sky'

        # Symlink to datasets and remove old ones
#         dst_dir = os.path.join(self.parset['dir_working'], 'datasets', self.direction.name)
#         misc.create_directory(dst_dir)
#         ms_map = DataMap.load(os.path.join(self.pipeline_mapfile_dir,
#                                            'prepare_imaging_data.mapfile'))
#         for ms in ms_map:
#             dst = os.path.join(dst_dir, os.path.basename(ms.file))
#             os.system('ln -fs {0} {1}'.format(ms.file, dst))
#         if self.index > 1:
#             prev_iter_mapfile_dir = self.pipeline_mapfile_dir.replace('image_{}'.format(self.index),
#                                                                       'image_{}'.format(self.index-1))
#             self.cleanup_mapfiles = [os.path.join(prev_iter_mapfile_dir,
#                                      'prepare_imaging_data.mapfile')]
#         self.cleanup()
