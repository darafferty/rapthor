"""
Module that holds the Image classes
"""
import os
import json
import logging
import shutil
import glob
from rapthor.lib import miscellaneous as misc
from rapthor.lib.operation import Operation
from rapthor.lib.cwl import CWLFile, CWLDir

log = logging.getLogger('rapthor:image')


class Image(Operation):
    """
    Operation to image a field sector
    """
    def __init__(self, field, index, name='image'):
        super().__init__(field, name=name, index=index)

        # For imaging we use a subworkflow, so we set the template filename for that here
        self.subpipeline_parset_template = '{0}_sector_pipeline.cwl'.format(self.rootname)

        # Initialize various parameters
        self.apply_none = None
        self.apply_amplitudes = None
        self.use_screens = None
        self.apply_fulljones = None
        self.make_image_cube = None
        self.normalize_flux_scale = None
        self.dde_method = None
        self.use_facets = None
        self.save_source_list = None
        self.image_pol = None
        self.peel_bright_sources = None
        self.imaging_sectors = None
        self.imaging_parameters = None
        self.do_predict = None
        self.do_multiscale_clean = None
        self.pol_combine_method = None

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        # Set parameters as needed
        if self.apply_none is None:
            self.apply_none = False
        if self.apply_amplitudes is None:
            self.apply_amplitudes = self.field.apply_amplitudes
        if self.use_screens is None:
            self.use_screens = self.field.use_screens
        if self.apply_fulljones is None:
            self.apply_fulljones = self.field.apply_fulljones
        if self.make_image_cube is None:
            self.make_image_cube = False
        if self.normalize_flux_scale is None:
            self.normalize_flux_scale = False
        if self.dde_method is None:
            self.dde_method = self.field.dde_method
        if self.use_facets is None:
            self.use_facets = True if self.dde_method == 'facets' else False
        if self.image_pol is None:
            self.image_pol = self.field.image_pol
        if self.save_source_list is None:
            self.save_source_list = True if self.image_pol.lower() == 'i' else False
        if self.peel_bright_sources is None:
            self.peel_bright_sources = self.field.peel_bright_sources
        if self.batch_system == 'slurm':
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']

        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'pipeline_working_dir': self.pipeline_working_dir,
                             'apply_none': self.apply_none,
                             'apply_amplitudes': self.apply_amplitudes,
                             'use_screens': self.use_screens,
                             'apply_fulljones': self.apply_fulljones,
                             'make_image_cube': self.make_image_cube,
                             'normalize_flux_scale': self.normalize_flux_scale,
                             'use_facets': self.use_facets,
                             'save_source_list': self.save_source_list,
                             'peel_bright_sources': self.peel_bright_sources,
                             'max_cores': max_cores,
                             'use_mpi': self.field.use_mpi,
                             'toil_version': self.toil_major_version}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Set parameters as needed
        if self.imaging_sectors is None:
            self.imaging_sectors = self.field.imaging_sectors
        if self.imaging_parameters is None:
            self.imaging_parameters = self.field.parset['imaging_specific'].copy()
        if self.do_predict is None:
            self.do_predict = self.field.do_predict
        if self.do_multiscale_clean is None:
            self.do_multiscale_clean = self.field.do_multiscale_clean
        if self.pol_combine_method is None:
            self.pol_combine_method = self.field.pol_combine_method

        nsectors = len(self.imaging_sectors)
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
        for i, sector in enumerate(self.imaging_sectors):
            image_root.append(sector.name)

            # Set the imaging parameters for each imaging sector. Note the we do not
            # let the imsize be recalcuated, as otherwise it may change from the previous
            # iteration and the mask made in that iteration can not be used in this one.
            # Generally, this should work fine, since we do not expect large changes in
            # the size of the sector from iteration to iteration (small changes are OK,
            # given the padding we use during imaging)
            sector.set_imaging_parameters(self.do_multiscale_clean, recalculate_imsize=False,
                                          imaging_parameters=self.imaging_parameters)

            # Set input MS filenames
            if self.do_predict:
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
                sector_starttime.append(misc.convert_mjd2mvt(obs.starttime))
                sector_ntimes.append(obs.numsamples)
            starttime.append(sector_starttime)
            ntimes.append(sector_ntimes)
            phasecenter.append("'[{0}deg, {1}deg]'".format(sector.ra, sector.dec))
            if self.scratch_dir is None:
                dir_local.append(self.pipeline_working_dir)
            else:
                dir_local.append(self.scratch_dir)
            if not self.apply_none:
                central_patch_name.append(sector.central_patch)

        # Handle the polarization-related options
        link_polarizations = False
        join_polarizations = False
        if self.image_pol.lower() != 'i':
            if self.pol_combine_method == 'link':
                # Note: link_polarizations can be of CWL type boolean or string
                link_polarizations = 'I'
            else:
                join_polarizations = True

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
                            'pol': self.image_pol,
                            'save_source_list': self.save_source_list,
                            'link_polarizations': link_polarizations,
                            'join_polarizations': join_polarizations,
                            'apply_amplitudes': [self.apply_amplitudes] * nsectors,
                            'channels_out': [sector.wsclean_nchannels for sector in self.imaging_sectors],
                            'deconvolution_channels': [sector.wsclean_deconvolution_channels for sector in self.imaging_sectors],
                            'fit_spectral_pol': [sector.wsclean_spectral_poly_order for sector in self.imaging_sectors],
                            'ra': [sector.ra for sector in self.imaging_sectors],
                            'dec': [sector.dec for sector in self.imaging_sectors],
                            'wsclean_imsize': [sector.imsize for sector in self.imaging_sectors],
                            'vertices_file': [CWLFile(sector.vertices_file).to_json() for sector in self.imaging_sectors],
                            'region_file': [None if sector.region_file is None else CWLFile(sector.region_file).to_json() for sector in self.imaging_sectors],
                            'wsclean_niter': [sector.wsclean_niter for sector in self.imaging_sectors],
                            'wsclean_nmiter': [sector.wsclean_nmiter for sector in self.imaging_sectors],
                            'robust': [sector.robust for sector in self.imaging_sectors],
                            'cellsize_deg': [sector.cellsize_deg for sector in self.imaging_sectors],
                            'min_uv_lambda': [sector.min_uv_lambda for sector in self.imaging_sectors],
                            'max_uv_lambda': [sector.max_uv_lambda for sector in self.imaging_sectors],
                            'taper_arcsec': [sector.taper_arcsec for sector in self.imaging_sectors],
                            'auto_mask': [sector.auto_mask for sector in self.imaging_sectors],
                            'idg_mode': [sector.idg_mode for sector in self.imaging_sectors],
                            'wsclean_mem': [sector.mem_limit_gb for sector in self.imaging_sectors],
                            'threshisl': [sector.threshisl for sector in self.imaging_sectors],
                            'threshpix': [sector.threshpix for sector in self.imaging_sectors],
                            'do_multiscale': [sector.multiscale for sector in self.imaging_sectors],
                            'dd_psf_grid': [sector.dd_psf_grid for sector in self.imaging_sectors],
                            'max_threads': self.field.parset['cluster_specific']['max_threads'],
                            'deconvolution_threads': self.field.parset['cluster_specific']['deconvolution_threads']}

        if self.peel_bright_sources:
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
        if self.use_screens:
            # The following parameters were set by the preceding calibrate operation, where
            # aterm image files were generated. They do not need to be set separately for
            # each sector
            self.input_parms.update({'aterm_image_filenames': CWLFile(self.field.aterm_image_filenames).to_json()})
        elif not self.apply_none:
            self.input_parms.update({'h5parm': CWLFile(self.field.h5parm_filename).to_json()})
            if self.field.fulljones_h5parm_filename is not None:
                self.input_parms.update({'fulljones_h5parm': CWLFile(self.field.fulljones_h5parm_filename).to_json()})
            if self.use_facets:
                # For faceting, we need inputs for making the ds9 facet region files
                self.input_parms.update({'skymodel': CWLFile(self.field.calibration_skymodel_file).to_json()})
                ra_mid = []
                dec_mid = []
                width_ra = []
                width_dec = []
                facet_region_file = []
                min_width = 2 * self.field.get_calibration_radius() * 1.2
                for sector in self.imaging_sectors:
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
                if self.apply_amplitudes:
                    self.input_parms.update({'soltabs': 'amplitude000,phase000'})
                else:
                    self.input_parms.update({'soltabs': 'phase000'})
                self.input_parms.update({'parallel_gridding_threads':
                                         self.field.parset['cluster_specific']['parallel_gridding_threads']})
                if self.image_pol.lower() == 'i':
                    # For Stokes-I-only imaging, we can take advantage of the scalar or
                    # diagonal visibilities options in WSClean (saving I/O)
                    if self.apply_amplitudes:
                        # Diagonal solutions generated during calibration
                        if self.field.apply_diagonal_solutions:
                            # Diagonal solutions should be used during imaging
                            self.input_parms.update({'diagonal_visibilities': True})
                            self.input_parms.update({'scalar_visibilities': False})
                        else:
                            # Diagonal solutions should not be used during imaging (they
                            # were in fact converted to scalar solutions at the end of
                            # calibration)
                            self.input_parms.update({'diagonal_visibilities': False})
                            self.input_parms.update({'scalar_visibilities': True})
                    else:
                        # Diagonal solutions not generated; only scalar solutions are
                        # available
                        self.input_parms.update({'diagonal_visibilities': False})
                        self.input_parms.update({'scalar_visibilities': True})
                else:
                    # This case is of full-Stokes (IQUV) imaging, so do not use diagonal
                    # or scalar visibilities (we need all four)
                    self.input_parms.update({'diagonal_visibilities': False})
                    self.input_parms.update({'scalar_visibilities': False})
            else:
                self.input_parms.update({'central_patch_name': central_patch_name})

    def finalize(self):
        """
        Finalize this operation
        """
        # Save the output FITS image filenames, sky models, ds9 facet region file, and
        # visibilities (if desired) for each sector. Also read the image diagnostics (rms
        # noise, etc.) derived by PyBDSF and print them to the log. The images are not
        # copied to the final location here, as this is done after mosaicking (if needed)
        # by the mosaic operation
        self.field.lofar_to_true_flux_ratio = 1.0  # reset values for this cycle
        self.field.lofar_to_true_flux_std = 0.0
        for sector in self.field.imaging_sectors:
            # The output image filenames
            image_root = os.path.join(self.pipeline_working_dir, sector.name)
            if self.field.image_pol.lower() == 'i':
                # When making only Stokes I images, WSClean does not include the
                # Stokes parameter name in the output filenames
                setattr(sector, "I_image_file_true_sky", f'{image_root}-MFS-image-pb.fits')
                setattr(sector, "I_image_file_apparent_sky", f'{image_root}-MFS-image.fits')
                setattr(sector, "I_model_file_true_sky", f'{image_root}-MFS-model-pb.fits')
                setattr(sector, "I_residual_file_apparent_sky", f'{image_root}-MFS-residual.fits')
            else:
                # When making all Stokes images, WSClean includes the Stokes parameter
                # name in the output filenames
                for pol in self.field.image_pol:
                    polup = pol.upper()
                    setattr(sector, f"{polup}_image_file_true_sky", f'{image_root}-MFS-{polup}-image-pb.fits')
                    setattr(sector, f"{polup}_image_file_apparent_sky", f'{image_root}-MFS-{polup}-image.fits')
                    setattr(sector, f"{polup}_model_file_true_sky", f'{image_root}-MFS-{polup}-model-pb.fits')
                    setattr(sector, f"{polup}_residual_file_apparent_sky", f'{image_root}-MFS-{polup}-residual.fits')

            # The output sky models, both true sky and apparent sky (the filenames are
            # defined in the rapthor/scripts/filter_skymodel.py file)
            #
            # Note: these are not generated when QUV images are made (WSClean does not
            # currently support writing a source list in this mode)
            if self.field.image_pol.lower() == 'i':
                sector.image_skymodel_file_true_sky = image_root + '.true_sky.txt'
                sector.image_skymodel_file_apparent_sky = image_root + '.apparent_sky.txt'
                dst_dir = os.path.join(self.parset['dir_working'], 'skymodels', 'image_{}'.format(self.index))
                misc.create_directory(dst_dir)
                for src_filename in [sector.image_skymodel_file_true_sky, sector.image_skymodel_file_apparent_sky]:
                    dst_filename = os.path.join(dst_dir, os.path.basename(src_filename))
                    shutil.copy(src_filename, dst_filename)

            # The output ds9 region file, if made
            if self.field.dde_method == 'facets':
                dst_dir = os.path.join(self.parset['dir_working'], 'regions', 'image_{}'.format(self.index))
                misc.create_directory(dst_dir)
                region_filename = '{}_facets_ds9.reg'.format(sector.name)
                src_filename = os.path.join(self.pipeline_working_dir, region_filename)
                dst_filename = os.path.join(dst_dir, region_filename)
                shutil.copy(src_filename, dst_filename)

            # The imaging visibilities
            if self.field.save_visibilities:
                dst_dir = os.path.join(self.parset['dir_working'], 'visibilities',
                                       'image_{}'.format(self.index), sector.name)
                misc.create_directory(dst_dir)
                ms_filenames = sector.get_obs_parameters('ms_prep_filename')
                for ms_filename in ms_filenames:
                    src_filename = os.path.join(self.pipeline_working_dir, ms_filename)
                    dst_filename = os.path.join(dst_dir, ms_filename)
                    if os.path.exists(dst_filename):
                        shutil.rmtree(dst_filename)
                    shutil.copytree(src_filename, dst_filename)

            # The astrometry and photometry plots
            dst_dir = os.path.join(self.parset['dir_working'], 'plots', 'image_{}'.format(self.index))
            misc.create_directory(dst_dir)
            diagnostic_plots = glob.glob(os.path.join(self.pipeline_working_dir, f'{sector.name}*.pdf'))
            for src_filename in diagnostic_plots:
                dst_filename = os.path.join(dst_dir, os.path.basename(src_filename))
                shutil.copy(src_filename, dst_filename)

            # Read in the image diagnostics and log a summary of them
            diagnostics_file = image_root + '.image_diagnostics.json'
            with open(diagnostics_file, 'r') as f:
                diagnostics_dict = json.load(f)
            sector.diagnostics.append(diagnostics_dict)
            ratio, std = report_sector_diagnostics(sector.name, diagnostics_dict, self.log)
            if self.field.lofar_to_true_flux_std == 0.0 or std < self.field.lofar_to_true_flux_std:
                # Save the ratio with the lowest scatter for later use
                self.field.lofar_to_true_flux_ratio = ratio
                self.field.lofar_to_true_flux_std = std

        # Finally call finalize() in the parent class
        super().finalize()


class ImageInitial(Image):
    """
    Operation to image the field to generate an initial sky model
    """
    def __init__(self, field):
        super().__init__(field, None, name='initial_image')

        # Set the template filenames
        self.pipeline_parset_template = 'image_pipeline.cwl'
        self.subpipeline_parset_template = 'image_sector_pipeline.cwl'

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        # Set parameters as needed
        self.apply_none = True
        self.apply_amplitudes = False
        self.use_screens = False
        self.apply_fulljones = False
        self.use_facets = False
        self.save_source_list = True
        self.peel_bright_sources = False
        self.image_pol = 'I'
        super().set_parset_parameters()

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Set the imaging parameters that are optimal for the initial sky
        # model generation
        self.field.full_field_sector.auto_mask = 5.0
        self.field.full_field_sector.threshisl = 4.0
        self.field.full_field_sector.threshpix = 5.0
        self.imaging_sectors = [self.field.full_field_sector]
        self.imaging_parameters = self.field.parset['imaging_specific'].copy()
        self.imaging_parameters['cellsize_arcsec'] = 1.5
        self.imaging_parameters['robust'] = -1.5
        self.imaging_parameters['taper_arcsec'] = 0.0
        self.imaging_parameters['min_uv_lambda'] = 0.0
        self.imaging_parameters['max_uv_lambda'] = 1e6
        self.imaging_parameters['reweight'] = False
        self.imaging_parameters['dd_psf_grid'] = [1, 1]
        self.do_predict = False
        self.do_multiscale_clean = True
        self.field.full_field_sector.max_nmiter = 8
        self.field.full_field_sector.max_wsclean_nchannels = 8
        super().set_input_parameters()

    def finalize(self):
        """
        Finalize this operation
        """
        # Save the output FITS images and sky models. Also read the image diagnostics (rms
        # noise, etc.) derived by PyBDSF and print them to the log
        sector = self.field.full_field_sector

        # The output image filenames
        image_root = os.path.join(self.pipeline_working_dir, sector.name)
        image_names = [f'{image_root}-MFS-image-pb.fits',
                       f'{image_root}-MFS-image.fits',
                       f'{image_root}-MFS-model-pb.fits',
                       f'{image_root}-MFS-residual.fits']
        dst_dir = os.path.join(self.parset['dir_working'], 'images', self.name)
        misc.create_directory(dst_dir)
        for src_filename in image_names:
            dst_filename = os.path.join(dst_dir, os.path.basename(src_filename))
            shutil.copy(src_filename, dst_filename)

        # The output sky models, both true sky and apparent sky (the filenames are
        # defined in the rapthor/scripts/filter_skymodel.py file)
        sector.image_skymodel_file_true_sky = image_root + '.true_sky.txt'
        sector.image_skymodel_file_apparent_sky = image_root + '.apparent_sky.txt'
        dst_dir = os.path.join(self.parset['dir_working'], 'skymodels', self.name)
        misc.create_directory(dst_dir)
        for src_filename in [sector.image_skymodel_file_true_sky, sector.image_skymodel_file_apparent_sky]:
            dst_filename = os.path.join(dst_dir, os.path.basename(src_filename))
            shutil.copy(src_filename, dst_filename)

        # The astrometry and photometry plots
        dst_dir = os.path.join(self.parset['dir_working'], 'plots', self.name)
        misc.create_directory(dst_dir)
        diagnostic_plots = glob.glob(os.path.join(self.pipeline_working_dir, f'{sector.name}*.pdf'))
        for src_filename in diagnostic_plots:
            dst_filename = os.path.join(dst_dir, os.path.basename(src_filename))
            shutil.copy(src_filename, dst_filename)

        # Read in the image diagnostics and log a summary of them
        diagnostics_file = image_root + '.image_diagnostics.json'
        with open(diagnostics_file, 'r') as f:
            diagnostics_dict = json.load(f)
        sector.diagnostics.append(diagnostics_dict)
        ratio, std = report_sector_diagnostics(sector.name, diagnostics_dict, self.log)
        self.field.lofar_to_true_flux_ratio = ratio
        self.field.lofar_to_true_flux_std = std

        # Finally call finalize() of the Operation class
        super(Image, self).finalize()


class ImageNormalize(Image):
    """
    Operation to image for flux-scale normalization
    """
    def __init__(self, field, index):
        super().__init__(field, index, name='normalize')

        # Set the template filenames
        self.pipeline_parset_template = 'image_pipeline.cwl'
        self.subpipeline_parset_template = 'image_sector_pipeline.cwl'

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        # Set parameters as needed
        self.save_source_list = False
        self.peel_bright_sources = False
        self.make_image_cube = True
        self.normalize_flux_scale = False  # False for testing only; must be True in final version
        super().set_parset_parameters()

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Set the imaging parameters that are optimal for the flux-scale
        # normalization
        self.imaging_sectors = self.field.imaging_sectors
        for sector in self.imaging_sectors:
            sector.auto_mask = 5.0
            sector.threshisl = 4.0
            sector.threshpix = 5.0
            sector.max_nmiter = 8
            sector.max_wsclean_nchannels = 8

        self.imaging_parameters = self.field.parset['imaging_specific'].copy()
        self.imaging_parameters['cellsize_arcsec'] = 6.0
        self.imaging_parameters['robust'] = -0.5
        self.imaging_parameters['taper_arcsec'] = 24.0
        self.do_predict = False
        self.do_multiscale_clean = False
        super().set_input_parameters()

    def finalize(self):
        """
        Finalize this operation
        """
        # Save the output FITS cubes and flux-scale corrections h5parms
        # When multiple sectors are used, average the flux-scale corrections
        # over the sectors, weighted by stdev?

        # Finally call finalize() of the Operation class
        super(Image, self).finalize()


def report_sector_diagnostics(sector_name, diagnostics_dict, log):
    """
    Report the sector's image diagnostics

    Parameters
    ----------
    sector_name : str
        The name of the sector.
    diagnostics_dict : dict
        The dict containing the diagnostics. A check is mode for the required keys;
        if any is not present, the report is skipped.
    log : logging.Logger object
        The logger to use for the report.

    Returns
    -------
    lofar_to_true_flux_ratio : float
        Mean ratio of the LOFAR flux densities to the "true" ones. The true flux
        densities are assumed to be from one of the TGSS, NVSS, or LoTSS surveys.
        If ratios from multiple surveys are present, the one with the lowest scatter
        is returned
    lofar_to_true_flux_std : float
        Stdev of the ratio of the LOFAR flux densities to the "true" ones
    """
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
        log.info('Diagnostics for {}:'.format(sector_name))
        log.info('    Min RMS noise = {0} (non-PB-corrected), '
                 '{1} (PB-corrected), {2} (theoretical)'.format(min_rms_flat_noise, min_rms_true_sky,
                                                                theoretical_rms))
        log.info('    Median RMS noise = {0} (non-PB-corrected), '
                 '{1} (PB-corrected)'.format(median_rms_flat_noise, median_rms_true_sky))
        log.info('    Dynamic range = {0} (non-PB-corrected), '
                 '{1} (PB-corrected)'.format(dynr_flat_noise, dynr_true_sky))
        log.info('    Number of sources found by PyBDSF = {}'.format(nsources))
        log.info('    Reference frequency = {}'.format(freq))
        log.info('    Beam = {}'.format(beam))
        log.info('    Fraction of unflagged data = {}'.format(unflagged_data_fraction))

        # Log the estimates of the global flux ratio and astrometry offsets.
        # If the required keys are not present, then there were not enough
        # sources for a reliable estimate to be made so report 'N/A' (not
        # available)
        #
        # Note: the reported error is not allowed to fall below 10% for
        # the flux ratio and 0.5" for the astrometry, as these are the
        # realistic minimum uncertainties in these values
        lofar_to_true_flux_ratio = 1.0
        lofar_to_true_flux_std = 0.0
        missing_surveys = []
        for survey in ['TGSS', 'LOTSS', 'NVSS']:
            if survey in ['TGSS', 'LOTSS'] or (survey == 'NVSS' and missing_surveys == ['TGSS', 'LOTSS']):
                # Always report TGSS and LoTSS values when available, but only
                # report NVSS values if both the TGSS and LoTSS comparisons failed (the
                # NVSS ones can be highly uncertain due to the large extrapolation needed).
                # We add the warning below for NVSS
                warn_text = ' (warning: may be highly uncertain due to large extrapolation)' if survey == 'NVSS' else ''
                if f'meanClippedRatio_{survey}' in diagnostics_dict and f'stdClippedRatio_{survey}' in diagnostics_dict:
                    ratio = '{0:.1f}'.format(diagnostics_dict[f'meanClippedRatio_{survey}'])
                    stdratio = '{0:.1f}'.format(max(0.1, diagnostics_dict[f'stdClippedRatio_{survey}']))
                    log.info(f'    LOFAR/{survey} flux ratio = {ratio} +/- {stdratio}{warn_text}')

                    if ((lofar_to_true_flux_std == 0.0 or
                            diagnostics_dict[f'stdClippedRatio_{survey}'] < lofar_to_true_flux_std) and
                            survey != 'NVSS'):
                        # Save the ratio with the lowest scatter (excluding NVSS
                        # estimate) for later use
                        lofar_to_true_flux_ratio = diagnostics_dict[f'meanClippedRatio_{survey}']
                        lofar_to_true_flux_std = max(0.1, diagnostics_dict[f'stdClippedRatio_{survey}'])
                else:
                    missing_surveys.append(survey)
                    log.info(f'    LOFAR/{survey} flux ratio = N/A')
        if 'meanClippedRAOffsetDeg' in diagnostics_dict and 'stdClippedRAOffsetDeg' in diagnostics_dict:
            raoff = '{0:.1f}"'.format(diagnostics_dict['meanClippedRAOffsetDeg']*3600)
            stdraoff = '{0:.1f}"'.format(max(0.5, diagnostics_dict['stdClippedRAOffsetDeg']*3600))
            log.info('    LOFAR-PanSTARRS RA offset = {0} +/- {1}'.format(raoff, stdraoff))
        else:
            log.info('    LOFAR-PanSTARRS RA offset = N/A')
        if 'meanClippedDecOffsetDeg' in diagnostics_dict and 'stdClippedDecOffsetDeg' in diagnostics_dict:
            decoff = '{0:.1f}"'.format(diagnostics_dict['meanClippedDecOffsetDeg']*3600)
            stddecoff = '{0:.1f}"'.format(max(0.5, diagnostics_dict['stdClippedDecOffsetDeg']*3600))
            log.info('    LOFAR-PanSTARRS Dec offset = {0} +/- {1}'.format(decoff, stddecoff))
        else:
            log.info('    LOFAR-PanSTARRS Dec offset = N/A')

        return (lofar_to_true_flux_ratio, lofar_to_true_flux_std)

    except KeyError:
        log.warn('One or more of the expected image diagnostics is unavailable '
                 'for {}. Logging of diagnostics skipped.'.format(sector_name))
        req_keys = ['theoretical_rms', 'min_rms_flat_noise', 'median_rms_flat_noise',
                    'dynamic_range_global_flat_noise', 'min_rms_true_sky',
                    'median_rms_true_sky', 'dynamic_range_global_true_sky',
                    'nsources', 'freq', 'beam_fwhm', 'unflagged_data_fraction',
                    'meanClippedRatio_TGSS', 'stdClippedRatio_TGSS',
                    'meanClippedRAOffsetDeg', 'stdClippedRAOffsetDeg',
                    'meanClippedDecOffsetDeg', 'stdClippedDecOffsetDeg']
        missing_keys = []
        for key in req_keys:
            if key not in diagnostics_dict:
                missing_keys.append(key)
        log.debug('Keys missing from the diagnostics dict: {}.'.format(', '.join(missing_keys)))

        return (1.0, 0.0)
