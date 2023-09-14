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
        Define parameters needed for the CWL workflow template
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
                             'apply_fulljones': self.field.do_fulljones_solve,
                             'use_facets': use_facets,
                             'peel_bright_sources': self.field.peel_bright_sources,
                             'max_cores': max_cores,
                             'use_mpi': self.field.use_mpi,
                             'toil_version': self.toil_major_version}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        nsectors = len(self.field.imaging_sectors)
        obs_filename = []
        prepare_filename = []
        concat_filename = []
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
            concat_filename.append(image_root[-1] + '_concat.ms')

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
                            'concat_filename': concat_filename,
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
                            'fit_spectral_pol': [sector.wsclean_spectral_poly_order for sector in self.field.imaging_sectors],
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
                            'wsclean_mem': [sector.mem_limit_gb for sector in self.field.imaging_sectors],
                            'threshisl': [sector.threshisl for sector in self.field.imaging_sectors],
                            'threshpix': [sector.threshpix for sector in self.field.imaging_sectors],
                            'do_multiscale': [sector.multiscale for sector in self.field.imaging_sectors],
                            'dd_psf_grid': [sector.dd_psf_grid for sector in self.field.imaging_sectors],
                            'max_threads': self.field.parset['cluster_specific']['max_threads'],
                            'deconvolution_threads': self.field.parset['cluster_specific']['deconvolution_threads']}

        if self.field.peel_bright_sources:
            self.input_parms.update({'bright_skymodel_pb': CWLFile(self.field.bright_source_skymodel_file).to_json()})
        if self.field.use_mpi:
            # Set number of nodes to allocate to each imaging subworkflow. We subtract
            # one node because Toil must use one node for its job, which in turn calls
            # salloc to reserve the nodes for the MPI job
            self.use_mpi = True
            nnodes = self.parset['cluster_specific']['max_nodes']
            nsubpipes = min(nsectors, nnodes)
            nnodes_per_subpipeline = max(1, int(nnodes / nsubpipes) - 1)
            self.input_parms.update({'mpi_nnodes': [nnodes_per_subpipeline] * nsectors})
            self.input_parms.update({'mpi_cpus_per_task': [self.parset['cluster_specific']['cpus_per_task']] * nsectors})
        if self.field.use_screens:
            # The following parameters were set by the preceding calibrate operation, where
            # aterm image files were generated. They do not need to be set separately for
            # each sector
            self.input_parms.update({'aterm_image_filenames': CWLFile(self.field.aterm_image_filenames).to_json()})
        else:
            self.input_parms.update({'h5parm': CWLFile(self.field.h5parm_filename).to_json()})
            if self.field.do_fulljones_solve:
                self.input_parms.update({'fulljones_h5parm': CWLFile(self.field.fulljones_h5parm_filename).to_json()})
            if self.field.dde_method == 'facets':
                # For faceting, we need inputs for making the ds9 facet region files
                self.input_parms.update({'skymodel': CWLFile(self.field.calibration_skymodel_file).to_json()})
                ra_mid = []
                dec_mid = []
                width_ra = []
                width_dec = []
                facet_region_file = []
                min_width = 2 * self.field.get_calibration_radius() * 1.2
                for sector in self.field.imaging_sectors:
                    # Note: WSClean requires that all sources in the h5parm must have
                    # corresponding regions in the facets region file. We ensure this
                    # requirement is met by extending the regions to cover the larger of
                    # the calibration region and the sector region, plus a 20% padding
                    ra_mid.append(self.field.ra)
                    dec_mid.append(self.field.dec)
                    width_ra.append(max(min_width, sector.width_ra*1.2))
                    width_dec.append(max(min_width, sector.width_dec*1.2))
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
                if self.field.do_slowgain_solve and self.field.apply_diagonal_solutions:
                    # Diagonal solutions generated and should be applied
                    self.input_parms.update({'apply_diagonal_solutions': True})
                else:
                    self.input_parms.update({'apply_diagonal_solutions': False})
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
            sector.I_model_file_true_sky = image_root + '-MFS-model.fits'
            sector.I_residual_file_apparent_sky = image_root + '-MFS-residual.fits'

            # The sky models, both true sky and apparent sky (the filenames are defined
            # in the rapthor/scripts/filter_skymodel.py file)
            sector.image_skymodel_file_true_sky = image_root + '.true_sky.txt'
            sector.image_skymodel_file_apparent_sky = image_root + '.apparent_sky.txt'
            dst_dir = os.path.join(self.parset['dir_working'], 'skymodels', 'image_{}'.format(self.index))
            misc.create_directory(dst_dir)
            for src_filename in [sector.image_skymodel_file_true_sky, sector.image_skymodel_file_apparent_sky]:
                dst_filename = os.path.join(dst_dir, os.path.basename(src_filename))
                if os.path.exists(dst_filename):
                    os.remove(dst_filename)
                shutil.copy(src_filename, dst_filename)

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
                min_rms_true_sky = '{0:.1f} uJy/beam'.format(diagnostics_dict['min_rms_true_sky']*1e6)
                median_rms_true_sky = '{0:.1f} uJy/beam'.format(diagnostics_dict['median_rms_true_sky']*1e6)
                dynr_true_sky = '{0:.2g}'.format(diagnostics_dict['dynamic_range_global_true_sky'])
                min_rms_flat_noise = '{0:.1f} uJy/beam'.format(diagnostics_dict['min_rms_flat_noise']*1e6)
                median_rms_flat_noise = '{0:.1f} uJy/beam'.format(diagnostics_dict['median_rms_flat_noise']*1e6)
                dynr_flat_noise = '{0:.2g}'.format(diagnostics_dict['dynamic_range_global_flat_noise'])
                nsources = '{0}'.format(diagnostics_dict['nsources'])
                freq = '{0:.1f} MHz'.format(diagnostics_dict['freq']/1e6)
                beam = '{0:.1f}" x {1:.1f}", PA = {2:.1f} deg'.format(diagnostics_dict['beam_fwhm'][0]*3600,
                                                                      diagnostics_dict['beam_fwhm'][1]*3600,
                                                                      diagnostics_dict['beam_fwhm'][2])
                unflagged_data_fraction = '{0:.2f}'.format(diagnostics_dict['unflagged_data_fraction'])
                self.log.info('Diagnostics for {}:'.format(sector.name))
                self.log.info('    Min RMS noise = {0} (non-PB-corrected), '
                              '{1} (PB-corrected), {2} (theoretical)'.format(min_rms_flat_noise, min_rms_true_sky,
                                                                             theoretical_rms))
                self.log.info('    Median RMS noise = {0} (non-PB-corrected), '
                              '{1} (PB-corrected)'.format(median_rms_flat_noise, median_rms_true_sky))
                self.log.info('    Dynamic range = {0} (non-PB-corrected), '
                              '{1} (PB-corrected)'.format(dynr_flat_noise, dynr_true_sky))
                self.log.info('    Number of sources found by PyBDSF = {}'.format(nsources))
                self.log.info('    Reference frequency = {}'.format(freq))
                self.log.info('    Beam = {}'.format(beam))
                self.log.info('    Fraction of unflagged data = {}'.format(unflagged_data_fraction))

                # Log the estimates of the global flux ratio and astrometry offsets.
                # If the required keys are not present, then there were not enough
                # sources for a reliable estimate to be made so report 'N/A' (not
                # available)
                #
                # Note: the reported error is not allowed to fall below 10% for
                # the flux ratio and 0.5" for the astrometry, as these are the
                # realistic minimum uncertainties in these values
                if 'meanClippedRatio_pybdsf' in diagnostics_dict and 'stdClippedRatio_pybdsf' in diagnostics_dict:
                    ratio = '{0:.1f}'.format(diagnostics_dict['meanClippedRatio_pybdsf'])
                    self.field.lofar_to_true_flux_ratio = diagnostics_dict['meanClippedRatio_pybdsf']
                    stdratio = '{0:.1f}'.format(max(0.1, diagnostics_dict['stdClippedRatio_pybdsf']))
                    self.field.lofar_to_true_flux_std = max(0.1, diagnostics_dict['stdClippedRatio_pybdsf'])
                    self.log.info('    LOFAR/TGSS flux ratio = {0} +/- {1}'.format(ratio, stdratio))
                else:
                    self.field.lofar_to_true_flux_ratio = 1.0
                    self.field.lofar_to_true_flux_std = 0.0
                    self.log.info('    LOFAR/TGSS flux ratio = N/A')
                if 'meanClippedRAOffsetDeg' in diagnostics_dict and 'stdClippedRAOffsetDeg' in diagnostics_dict:
                    raoff = '{0:.1f}"'.format(diagnostics_dict['meanClippedRAOffsetDeg']*3600)
                    stdraoff = '{0:.1f}"'.format(max(0.5, diagnostics_dict['stdClippedRAOffsetDeg']*3600))
                    self.log.info('    LOFAR-TGSS RA offset = {0} +/- {1}'.format(raoff, stdraoff))
                else:
                    self.log.info('    LOFAR-TGSS RA offset = N/A')
                if 'meanClippedDecOffsetDeg' in diagnostics_dict and 'stdClippedDecOffsetDeg' in diagnostics_dict:
                    decoff = '{0:.1f}"'.format(diagnostics_dict['meanClippedDecOffsetDeg']*3600)
                    stddecoff = '{0:.1f}"'.format(max(0.5, diagnostics_dict['stdClippedDecOffsetDeg']*3600))
                    self.log.info('    LOFAR-TGSS Dec offset = {0} +/- {1}'.format(decoff, stddecoff))
                else:
                    self.log.info('    LOFAR-TGSS Dec offset = N/A')
            except KeyError:
                self.log.warn('One or more of the expected image diagnostics is unavailable '
                              'for {}. Logging of diagnostics skipped.'.format(sector.name))
                req_keys = ['theoretical_rms', 'min_rms_flat_noise', 'median_rms_flat_noise',
                            'dynamic_range_global_flat_noise', 'min_rms_true_sky',
                            'median_rms_true_sky', 'dynamic_range_global_true_sky',
                            'nsources', 'freq', 'beam_fwhm', 'unflagged_data_fraction',
                            'meanClippedRatio_pybdsf', 'stdClippedRatio_pybdsf',
                            'meanClippedRAOffsetDeg', 'stdClippedRAOffsetDeg',
                            'meanClippedDecOffsetDeg', 'stdClippedDecOffsetDeg']
                missing_keys = []
                for key in req_keys:
                    if key not in diagnostics_dict:
                        missing_keys.append(key)
                self.log.debug('Keys missing from the diagnostics dict: {}.'.format(', '.join(missing_keys)))

        # Finally call finalize() in the parent class
        super().finalize()
