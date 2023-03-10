"""
Module that holds the Image class
"""
import os
import json
import logging
import shutil
from rapthor.lib import miscellaneous as misc
from rapthor.lib.operation import Operation
from rapthor.lib.cwl import CWLFile, CWLDir

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
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        if self.field.dde_method == 'facets':
            use_facets = True
        else:
            use_facets = False
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'pipeline_working_dir': self.pipeline_working_dir,
                             'do_slowgain_solve': self.field.do_slowgain_solve,
                             'use_screens': self.field.use_screens,
                             'use_facets': use_facets,
                             'peel_bright_sources': self.field.peel_bright_sources,
                             'max_cores': max_cores,
                             'use_mpi': self.field.use_mpi,
                             'toil_version': self.toil_major_version}

    def set_input_parameters(self):
        """
        Define the pipeline inputs
        """
        nsectors = len(self.field.imaging_sectors)
        obs_filename = []
        prepare_filename = []
        previous_mask_filename = []
        mask_filename = []
        starttime = []
        ntimes = []
        image_freqstep = []
        image_timestep = []
        dir_local = []
        phasecenter = []
        image_root = []
        central_patch_name = []
        for i, sector in enumerate(self.field.imaging_sectors):
            image_root.append(sector.name)

            # Set the imaging parameters for each imaging sector. Note the we do not
            # let the imsize be recalcuated, as otherwise it may change from the previous
            # iteration and the mask made in that iteration can not be used in this one.
            # Generally, this should work fine, since we do not expect large changes in
            # the size of the sector from iteration to iteration (small changes are OK,
            # given the padding we use during imaging)
            sector.set_imaging_parameters(do_multiscale=self.field.do_multiscale_clean,
                                          recalculate_imsize=False)

            # Set input MS filenames
            if self.field.do_predict:
                sector_obs_filename = [obs.ms_imaging_filename for obs in sector.observations]
            else:
                sector_obs_filename = sector.get_obs_parameters('ms_filename')
            obs_filename.append(sector_obs_filename)

            # Set output MS filenames for step that prepares the data for WSClean
            prepare_filename.append(sector.get_obs_parameters('ms_prep_filename'))

            # Set other parameters
            if sector.I_mask_file is not None:
                # Use the existing mask
                previous_mask_filename.append(sector.I_mask_file)
            else:
                # Use a dummy mask
                previous_mask_filename.append(None)
            mask_filename.append(image_root[-1] + '_mask.fits')
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
            if self.scratch_dir is None:
                dir_local.append(self.pipeline_working_dir)
            else:
                dir_local.append(self.scratch_dir)
            central_patch_name.append(sector.central_patch)

        self.input_parms = {'obs_filename': [CWLDir(name).to_json() for name in obs_filename],
                            'prepare_filename': prepare_filename,
                            'previous_mask_filename': [None if name is None else CWLFile(name).to_json() for name in previous_mask_filename],
                            'mask_filename': mask_filename,
                            'starttime': starttime,
                            'ntimes': ntimes,
                            'image_freqstep': image_freqstep,
                            'image_timestep': image_timestep,
                            'phasecenter': phasecenter,
                            'image_name': image_root,
                            'dir_local': dir_local,
                            'do_slowgain_solve': [self.field.do_slowgain_solve] * nsectors,
                            'channels_out': [sector.wsclean_nchannels for sector in self.field.imaging_sectors],
                            'deconvolution_channels': [sector.wsclean_deconvolution_channels for sector in self.field.imaging_sectors],
                            'ra': [sector.ra for sector in self.field.imaging_sectors],
                            'dec': [sector.dec for sector in self.field.imaging_sectors],
                            'wsclean_imsize': [sector.imsize for sector in self.field.imaging_sectors],
                            'vertices_file': [CWLFile(sector.vertices_file).to_json() for sector in self.field.imaging_sectors],
                            'region_file': [None if sector.region_file is None else CWLFile(sector.region_file).to_json() for sector in self.field.imaging_sectors],
                            'wsclean_niter': [sector.wsclean_niter for sector in self.field.imaging_sectors],
                            'wsclean_nmiter': [sector.wsclean_nmiter for sector in self.field.imaging_sectors],
                            'robust': [sector.robust for sector in self.field.imaging_sectors],
                            'cellsize_deg': [sector.cellsize_deg for sector in self.field.imaging_sectors],
                            'min_uv_lambda': [sector.min_uv_lambda for sector in self.field.imaging_sectors],
                            'max_uv_lambda': [sector.max_uv_lambda for sector in self.field.imaging_sectors],
                            'taper_arcsec': [sector.taper_arcsec for sector in self.field.imaging_sectors],
                            'auto_mask': [sector.auto_mask for sector in self.field.imaging_sectors],
                            'idg_mode': [sector.idg_mode for sector in self.field.imaging_sectors],
                            'wsclean_mem': [sector.mem_percent for sector in self.field.imaging_sectors],
                            'threshisl': [sector.threshisl for sector in self.field.imaging_sectors],
                            'threshpix': [sector.threshpix for sector in self.field.imaging_sectors],
                            'do_multiscale': [sector.multiscale for sector in self.field.imaging_sectors],
                            'max_threads': self.field.parset['cluster_specific']['max_threads'],
                            'deconvolution_threads': self.field.parset['cluster_specific']['deconvolution_threads']}

        if self.field.peel_bright_sources:
            self.input_parms.update({'bright_skymodel_pb': CWLFile(self.field.bright_source_skymodel_file).to_json()})
        if self.field.use_screens:
            # The following parameters were set by the preceding calibrate operation, where
            # aterm image files were generated. They do not need to be set separately for
            # each sector
            self.input_parms.update({'aterm_image_filenames': CWLFile(self.field.aterm_image_filenames).to_json()})

            if self.field.use_mpi:
                # Set number of nodes to allocate to each imaging subpipeline. We subtract
                # one node because Toil must use one node for its job, which in turn calls
                # salloc to reserve the nodes for the MPI job
                nnodes = self.parset['cluster_specific']['max_nodes']
                nsubpipes = min(nsectors, nnodes)
                nnodes_per_subpipeline = max(1, int(nnodes / nsubpipes) - 1)
                self.input_parms.update({'mpi_nnodes': [nnodes_per_subpipeline] * nsectors})
                self.input_parms.update({'mpi_cpus_per_task': [self.parset['cluster_specific']['cpus_per_task']] * nsectors})
        else:
            self.input_parms.update({'h5parm': CWLFile(self.field.h5parm_filename).to_json()})
            if self.field.dde_method == 'facets':
                # For faceting, we need inputs for making the ds9 facet region files
                self.input_parms.update({'skymodel': CWLFile(self.field.calibration_skymodel_file).to_json()})
                ra_mid = []
                dec_mid = []
                width_ra = []
                width_dec = []
                facet_region_file = []
                for sector in self.field.imaging_sectors:
                    # Note: WSClean requires that all sources in the h5parm must have
                    # corresponding regions in the facets region file. We ensure this
                    # requirement is met by making the region file very large so that
                    # it covers the full field (10x10 deg)
                    ra_mid.append(self.field.ra)
                    dec_mid.append(self.field.dec)
                    width_ra.append(10.0)
                    width_dec.append(10.0)
                    facet_region_file.append('{}_facets_ds9.reg'.format(sector.name))
                self.input_parms.update({'ra_mid': ra_mid})
                self.input_parms.update({'dec_mid': dec_mid})
                self.input_parms.update({'width_ra': width_ra})
                self.input_parms.update({'width_dec': width_dec})
                self.input_parms.update({'facet_region_file': facet_region_file})
                if self.field.do_slowgain_solve:
                    self.input_parms.update({'soltabs': 'amplitude000,phase000'})
                else:
                    self.input_parms.update({'soltabs': 'phase000'})
                self.input_parms.update({'parallel_gridding_threads':
                                         self.field.parset['cluster_specific']['parallel_gridding_threads']})
            else:
                self.input_parms.update({'central_patch_name': central_patch_name})

    def finalize(self):
        """
        Finalize this operation
        """
        # Copy the output FITS image, the clean mask, sky models, and ds9 facet
        # region file for each sector. Also read the image diagnostics (rms noise,
        # etc.) derived by PyBDSF and print them to the log.
        # NOTE: currently, -save-source-list only works with pol=I -- when it works with other
        # pols, copy them all
        for sector in self.field.imaging_sectors:
            image_root = os.path.join(self.pipeline_working_dir, sector.name)
            sector.I_image_file_true_sky = image_root + '-MFS-image-pb.fits'
            sector.I_image_file_apparent_sky = image_root + '-MFS-image.fits'
            sector.I_model_file_true_sky = image_root + '-MFS-model-pb.fits'

            # Check to see if a clean mask image was made (only made when at least one
            # island is found in the Stokes I image). The filename is defined
            # in the rapthor/scripts/filter_skymodel.py file
            #
            # Note: for now, the clean mask is not used as it has not been found to
            # be necessary (WSClean automasking is used on its own)
            use_clean_mask = False
            mask_filename = sector.I_image_file_apparent_sky + '.mask'
            if use_clean_mask and os.path.exists(mask_filename):
                sector.I_mask_file = mask_filename
            else:
                sector.I_mask_file = None

            # The sky models, both true sky and apparent sky (the filenames are defined
            # in the rapthor/scripts/filter_skymodel.py file)
            sector.image_skymodel_file_true_sky = image_root + '.true_sky.txt'
            sector.image_skymodel_file_apparent_sky = image_root + '.apparent_sky.txt'

            # The ds9 region file, if made
            if self.field.dde_method == 'facets':
                dst_dir = os.path.join(self.parset['dir_working'], 'regions', 'image_{}'.format(self.index))
                misc.create_directory(dst_dir)
                region_filename = '{}_facets_ds9.reg'.format(sector.name)
                src_filename = os.path.join(self.pipeline_working_dir, region_filename)
                dst_filename = os.path.join(dst_dir, region_filename)
                if os.path.exists(dst_filename):
                    os.remove(dst_filename)
                shutil.copy(src_filename, dst_filename)

            # Read in the image diagnostics and log a summary of them
            diagnostics_file = image_root + '.image_diagnostics.json'
            with open(diagnostics_file, 'r') as f:
                diagnostics_dict = json.load(f)
            sector.diagnostics.append(diagnostics_dict)
            try:
                theoretical_rms = '{0:.1f} uJy/beam'.format(diagnostics_dict['theoretical_rms']*1e6)
                min_rms = '{0:.1f} uJy/beam'.format(diagnostics_dict['min_rms']*1e6)
                mean_rms = '{0:.1f} uJy/beam'.format(diagnostics_dict['mean_rms']*1e6)
                dynr = '{0:.2g}'.format(diagnostics_dict['dynamic_range_global'])
                freq = '{0:.1f} MHz'.format(diagnostics_dict['freq']/1e6)
                beam = '{0:.1f}" x {1:.1f}", PA = {2:.1f} deg'.format(diagnostics_dict['beam_fwhm'][0]*3600,
                                                                      diagnostics_dict['beam_fwhm'][1]*3600,
                                                                      diagnostics_dict['beam_fwhm'][2])
                unflagged_data_fraction = '{0:.2f}'.format(diagnostics_dict['unflagged_data_fraction'])
                self.log.info('Diagnostics for {}:'.format(sector.name))
                self.log.info('    Min RMS noise = {0} (theoretical = {1})'.format(min_rms, theoretical_rms))
                self.log.info('    Mean RMS noise = {}'.format(mean_rms))
                self.log.info('    Dynamic range = {}'.format(dynr))
                self.log.info('    Reference frequency = {}'.format(freq))
                self.log.info('    Beam = {}'.format(beam))
                self.log.info('    Fraction of unflagged data = {}'.format(unflagged_data_fraction))
                if 'meanClippedRatio' in diagnostics_dict:
                    # If 'meanClippedRatio' is present, assume all of the LSMTool-generated
                    # comparison diagnostics are available (these are only generated if there
                    # is a sufficient number of appropriate sources in the image to make the
                    # comparison)
                    #
                    # Note: the reported error is not allowed to fall below
                    # 10% for the flux ratio and 0.5" for the astrometry, as these
                    # are the realistic minimum uncertainties in these values
                    ratio = '{0:.1f}'.format(diagnostics_dict['meanClippedRatio'])
                    stdratio = '{0:.1f}'.format(max(0.1, diagnostics_dict['stdClippedRatio']))
                    self.log.info('    LOFAR/TGSS flux ratio = {0} +/- {1}'.format(ratio, stdratio))
                    raoff = '{0:.1f}"'.format(diagnostics_dict['meanClippedRAOffsetDeg']*3600)
                    stdraoff = '{0:.1f}"'.format(max(0.5, diagnostics_dict['stdClippedRAOffsetDeg']*3600))
                    self.log.info('    LOFAR-TGSS RA offset = {0} +/- {1}'.format(raoff, stdraoff))
                    decoff = '{0:.1f}"'.format(diagnostics_dict['meanClippedDecOffsetDeg']*3600)
                    stddecoff = '{0:.1f}"'.format(max(0.5, diagnostics_dict['stdClippedDecOffsetDeg']*3600))
                    self.log.info('    LOFAR-TGSS Dec offset = {0} +/- {1}'.format(decoff, stddecoff))
                else:
                    self.log.info('    LOFAR/TGSS flux ratio = N/A')
                    self.log.info('    LOFAR-TGSS RA offset = N/A')
                    self.log.info('    LOFAR-TGSS Dec offset = N/A')
            except KeyError:
                self.log.warn('One or more of the expected image diagnostics unavailable '
                              'for {}. Logging of diagnostics skipped.'.format(sector.name))
                req_keys = ['theoretical_rms', 'min_rms', 'mean_rms', 'dynamic_range_global',
                            'freq', 'beam_fwhm', 'unflagged_data_fraction', 'meanClippedRatio',
                            'stdClippedRatio', 'meanClippedRAOffsetDeg', 'stdClippedRAOffsetDeg',
                            'meanClippedDecOffsetDeg', 'stdClippedDecOffsetDeg']
                missing_keys = []
                for key in req_keys:
                    if key not in diagnostics_dict:
                        missing_keys.append(key)
                self.log.debug('Keys missing from the diagnostics dict: {}.'.format(', '.join(missing_keys)))

        # Finally call finalize() in the parent class
        super().finalize()
