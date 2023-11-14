cwlVersion: v1.2
class: Workflow
label: Rapthor imaging subworkflow
doc: |
  This subworkflow performs imaging with direction-dependent corrections. The
  imaging data are generated (and averaged if possible) and WSClean is
  used to perform the imaging. Masking and sky model filtering is then done
  using PyBDSF.

requirements:
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: {{ max_cores }}
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: obs_filename
    label: Filenames of input MS
    doc: |
      The filenames of input MS files for which imaging will be done (length =
      n_obs).
    type: Directory[]

  - id: prepare_filename
    label: Filenames of preparatory MSs
    doc: |
      The filenames of the preparatory MS files used as input to concatenation
      (length = n_obs).
    type: string[]

  - id: concat_filename
    label: Filename of imaging MS
    doc: |
      The filename of the MS file resulting from concatenation of the preparatory
      files and used for imaging (length = 1).
    type: string

  - id: starttime
    label: Start times of each obs
    doc: |
      The start time (in casacore MVTime) for each observation (length = n_obs).
    type: string[]

  - id: ntimes
    label: Number of times of each obs
    doc: |
      The number of timeslots for each observation (length = n_obs).
    type: int[]

  - id: image_freqstep
    label: Averaging interval in frequency
    doc: |
      The averaging interval in number of frequency channels (length = n_obs).
    type: int[]

  - id: image_timestep
    label: Averaging interval in time
    doc: |
      The averaging interval in number of timeslots (length = n_obs).
    type: int[]

  - id: previous_mask_filename
    label: Filename of previous mask
    doc: |
      The filename of the image mask from the previous iteration (length = 1).
    type: File?

  - id: mask_filename
    label: Filename of current mask
    doc: |
      The filename of the current image mask (length = 1).
    type: string

  - id: phasecenter
    label: Phase center of image
    doc: |
      The phase center of the image in deg (length = 1).
    type: string

  - id: ra
    label: RA of center of image
    doc: |
      The RA of the center of the image in deg (length = 1).
    type: float

  - id: dec
    label: Dec of center of image
    doc: |
      The Dec of the center of the image in deg (length = 1).
    type: float

  - id: image_name
    label: Filename of output image
    doc: |
      The root filename of the output image (length = 1).
    type: string

  - id: cellsize_deg
    label: Pixel size
    doc: |
      The size of one pixel of the image in deg (length = 1).
    type: float

  - id: wsclean_imsize
    label: Image size
    doc: |
      The size of the image in pixels (length = 2).
    type: int[]

  - id: vertices_file
    label: Filename of vertices file
    doc: |
      The filename of the file containing sector vertices (length = 1).
    type: File

  - id: region_file
    label: Filename of region file
    doc: |
      The filename of the region file (length = 1).
    type: File?

{% if use_mpi %}
  - id: mpi_cpus_per_task
    label: Number of CPUs per task
    doc: |
      The number of CPUs per task to request from Slurm for MPI jobs (length = 1).
    type: int

  - id: mpi_nnodes
    label: Number of nodes
    doc: |
      The number of nodes for MPI jobs (length = 1).
    type: int
{% endif %}

{% if use_screens %}
# start use_screens
  - id: aterm_image_filenames
    label: Filenames of a-terms
    doc: |
      The filenames of the a-term images (length = 1, with n_aterms subelements).
    type: File[]

{% else %}
# start not use_screens

  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the direction-dependent calibration
      solutions (length = 1).
    type: File

{% if apply_fulljones %}
  - id: fulljones_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the full-Jones calibration solutions
      (length = 1).
    type: File
{% endif %}

{% if use_facets %}
# start use_facets
  - id: skymodel
    label: Filename of sky model
    doc: |
      The filename of the sky model file with the calibration patches (length =
      1).
    type: File

  - id: ra_mid
    label: RA of the midpoint
    doc: |
        The RA in degrees of the middle of the region to be imaged (length = 1).
    type: float

  - id: dec_mid
    label: Dec of the midpoint
    doc: |
        The Dec in degrees of the middle of the region to be imaged (length = 1).
    type: float

  - id: width_ra
    label: Width along RA
    doc: |
      The width along RA in degrees (corrected to Dec = 0) of the region to be
      imaged (length = 1).
    type: float

  - id: width_dec
    label:  Width along Dec
    doc: |
      The width along Dec in degrees of the region to be imaged (length = 1).
    type: float

  - id: facet_region_file
    label: Filename of output region file
    doc: |
      The filename of the output ds9 region file (length = 1).
    type: string

  - id: soltabs
    label: Names of calibration soltabs
    doc: |
      The names of the calibration solution tables (length = 1).
    type: string

  - id: apply_diagonal_solutions
    label: Apply diagonal solutions
    doc: |
      Apply diagonal (separate XX and YY) solutions (length = 1).
    type: boolean

  - id: parallel_gridding_threads
    label: Max number of gridding threads
    doc: |
      The maximum number of threads to use during parallel gridding (length = 1).
    type: int

{% else %}
# start not use_facets

  - id: central_patch_name
    label: Name of central patch
    doc: |
      The name of the central-most patch of the sector (length = 1).
    type: string
{% endif %}
# end use_facets / not use_facets

{% endif %}
# end use_screens / not use_screens

  - id: channels_out
    label: Number of channels
    doc: |
      The number of WSClean output channels (length = 1).
    type: int

  - id: deconvolution_channels
    label: Number of deconvolution channels
    doc: |
      The number of WSClean deconvolution channels (length = 1).
    type: int

  - id: fit_spectral_pol
    label: Spectral poly order
    doc: |
      The order of WSClean spectral polynomial (length = 1).
    type: int

  - id: wsclean_niter
    label: Number of iterations
    doc: |
      The number of WSClean iterations (length = 1).
    type: int

  - id: wsclean_nmiter
    label: Number of major iterations
    doc: |
      The number of WSClean major iterations (length = 1).
    type: int

  - id: robust
    label: Robust weighting
    doc: |
      The value of the WSClean robust weighting parameter (length = 1).
    type: float

  - id: min_uv_lambda
    label: Minimum uv distance
    doc: |
      The WSClean minimum uv distance in lambda (length = 1).
    type: float

  - id: max_uv_lambda
    label: Maximum uv distance
    doc: |
      The WSClean maximum uv distance in lambda (length = 1).
    type: float

  - id: do_multiscale
    label: Activate multiscale
    doc: |
      Activate multiscale clean (length = 1).
    type: boolean

  - id: pol
    label: Pol list
    doc: |
      List of polarizations to image; e.g. "i" or "iquv" (length = 1).
    type: string

  - id: save_source_list
    label: Save source list
    doc: |
      Save list of clean components (length = 1).
    type: boolean

  - id: link_polarizations
    label: Link polarizations
    doc: |
      Link polarizations during clean (length = 1).
    type:
      - boolean?
      - string?

  - id: join_polarizations
    label: Join polarizations
    doc: |
      Join polarizations during clean (length = 1).
    type: boolean

  - id: taper_arcsec
    label: Taper value
    doc: |
      The WSClean taper value in arcsec (length = 1).
    type: float

  - id: wsclean_mem
    label: Memory in GB
    doc: |
      The memory limit for WSClean in GB (length = 1).
    type: float

  - id: auto_mask
    label: Auto mask value
    doc: |
      The WSClean auto mask value (length = 1).
    type: float

  - id: idg_mode
    label: IDG mode
    doc: |
      The WSClean IDG mode (length = 1).
    type: string

  - id: threshisl
    label: Island threshold
    doc: |
      The PyBDSF island threshold (length = 1).
    type: float

  - id: threshpix
    label: Pixel threshold
    doc: |
      The PyBDSF pixel threshold (length = 1).
    type: float

{% if peel_bright_sources %}
  - id: bright_skymodel_pb
    label: Bright-source sky model
    doc: |
      The primary-beam-corrected bright-source sky model (length = 1).
    type: File
{% endif %}

  - id: max_threads
    label: Max number of threads
    doc: |
      The maximum number of threads to use for a job (length = 1).
    type: int

  - id: deconvolution_threads
    label: Max number of deconvolution threads
    doc: |
      The maximum number of threads to use during deconvolution (length = 1).
    type: int

  - id: dd_psf_grid
    label: Direction-dependent PSF grid
    doc: |
      The number of direction-dependent PSFs which should be fit horizontally and
      vertically in the image (length = 2).
    type: int[]


outputs:
  - id: filtered_skymodel_true_sky
    outputSource:
      - filter/filtered_skymodel_true_sky
    type: File
  - id: filtered_skymodel_apparent_sky
    outputSource:
      - filter/filtered_skymodel_apparent_sky
    type: File
  - id: sector_diagnostics
    outputSource:
      - find_diagnostics/diagnostics
    type: File
  - id: sector_I_images
    outputSource:
{% if peel_bright_sources %}
      - restore_nonpb/restored_image
      - restore_pb/restored_image
{% else %}
      - image/image_I_nonpb_name
      - image/image_I_pb_name
{% endif %}
    type: File[]
  - id: sector_extra_images
    outputSource:
      - image/images_extra
    type: File[]
{% if save_source_list %}
  - id: sector_skymodels
    outputSource:
      - image/skymodel_nonpb
      - image/skymodel_pb
    type: File[]
{% endif %}
{% if use_facets %}
  - id: sector_region_file
    outputSource:
      - make_region_file/region_file
    type: File
{% endif %}

steps:
  - id: prepare_imaging_data
    label: Prepare imaging data
    doc: |
      This step uses DP3 to prepare the input data for imaging. This involves
      averaging, phase shifting, and optionally the application of the
      calibration solutions.
{% if use_screens or use_facets %}
# start use_screens or use_facets
{% if apply_fulljones %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data_fulljones.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data.cwl
{% endif %}

{% else %}
# start not use_screens and not use_facets
{% if do_slowgain_solve %}
{% if apply_fulljones %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data_no_dde_fulljones.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data_no_dde.cwl
{% endif %}
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data_no_dde_phase_only.cwl
{% endif %}

{% endif %}
# end use_screens or use_facets / not use_screens and not use_facets

{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: obs_filename
      - id: msout
        source: prepare_filename
      - id: starttime
        source: starttime
      - id: ntimes
        source: ntimes
      - id: phasecenter
        source: phasecenter
      - id: freqstep
        source: image_freqstep
      - id: timestep
        source: image_timestep
      - id: beamdir
        source: phasecenter
      - id: numthreads
        source: max_threads
{% if use_screens or use_facets %}
{% if apply_fulljones %}
      - id: h5parm
        source: fulljones_h5parm
{% endif %}
    scatter: [msin, msout, starttime, ntimes, freqstep, timestep]
{% else %}
      - id: h5parm
        source: h5parm
{% if apply_fulljones %}
      - id: fulljones_h5parm
        source: fulljones_h5parm
{% endif %}
      - id: central_patch_name
        source: central_patch_name
    scatter: [msin, msout, starttime, ntimes, freqstep, timestep]
{% endif %}
    scatterMethod: dotproduct
    out:
      - id: msimg

  - id: concat_in_time
    label: Concatenate MS file in time
    doc: |
      This step concatenates the imaging MS files in time.
    run: {{ rapthor_pipeline_dir }}/steps/concat_ms_files.cwl
    in:
      - id: mslist
        source: prepare_imaging_data/msimg
      - id: msout
        source: concat_filename
      - id: concat_property
        valueFrom: 'time'
    out:
      - id: msconcat

  - id: premask
    label: Make an image mask
    doc: |
      This step makes a FITS mask for the imaging.
    run: {{ rapthor_pipeline_dir }}/steps/blank_image.cwl
    in:
      - id: imagefile
        source: previous_mask_filename
      - id: maskfile
        source: mask_filename
      - id: wsclean_imsize
        source: wsclean_imsize
      - id: vertices_file
        source: vertices_file
      - id: ra
        source: ra
      - id: dec
        source: dec
      - id: cellsize_deg
        source: cellsize_deg
      - id: region_file
        source: region_file
    out:
      - id: maskimg

{% if use_facets %}
  - id: make_region_file
    label: Make a ds9 region file
    doc: |
      This step makes a ds9 region file for the imaging.
    run: {{ rapthor_pipeline_dir }}/steps/make_region_file.cwl
    in:
      - id: skymodel
        source: skymodel
      - id: ra_mid
        source: ra_mid
      - id: dec_mid
        source: dec_mid
      - id: width_ra
        source: width_ra
      - id: width_dec
        source: width_dec
      - id: outfile
        source: facet_region_file
    out:
      - id: region_file
{% endif %}

  - id: image
    label: Make an image
    doc: |
      This step makes an image using WSClean. Direction-dependent effects
      can be corrected for using a-term images or facet-based corrections.
{% if use_screens %}
# start use_screens

{% if use_mpi %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_screens.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_screens.cwl
{% endif %}

{% else %}
# start not use_screens

{% if use_facets %}
# start use_facets

{% if use_mpi %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_facets.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_facets.cwl
{% endif %}

{% else %}
# start not use_facets and not use_screens (i.e., use no_dde)

{% if use_mpi %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_no_dde.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_no_dde.cwl
{% endif %}

{% endif %}
# end use no_dde

{% endif %}
# end use_screens / not use_screens

{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: concat_in_time/msconcat
      - id: name
        source: image_name
      - id: mask
        source: premask/maskimg
{% if use_mpi %}
      - id: nnodes
        source: mpi_nnodes
{% endif %}
{% if use_screens %}
      - id: aterm_images
        source: aterm_image_filenames
{% endif %}
{% if use_facets %}
      - id: h5parm
        source: h5parm
      - id: soltabs
        source: soltabs
      - id: region_file
        source: make_region_file/region_file
      - id: apply_diagonal_solutions
        source: apply_diagonal_solutions
{% if not use_mpi %}
      - id: num_gridding_threads
        source: parallel_gridding_threads
{% endif %}
{% endif %}
      - id: wsclean_imsize
        source: wsclean_imsize
      - id: wsclean_niter
        source: wsclean_niter
      - id: wsclean_nmiter
        source: wsclean_nmiter
      - id: robust
        source: robust
      - id: min_uv_lambda
        source: min_uv_lambda
      - id: max_uv_lambda
        source: max_uv_lambda
      - id: multiscale
        source: do_multiscale
      - id: pol
        source: pol
      - id: save_source_list
        source: save_source_list
      - id: link_polarizations
        source: link_polarizations
      - id: join_polarizations
        source: join_polarizations
      - id: cellsize_deg
        source: cellsize_deg
      - id: channels_out
        source: channels_out
      - id: deconvolution_channels
        source: deconvolution_channels
      - id: fit_spectral_pol
        source: fit_spectral_pol
      - id: taper_arcsec
        source: taper_arcsec
      - id: wsclean_mem
        source: wsclean_mem
      - id: auto_mask
        source: auto_mask
      - id: idg_mode
        source: idg_mode
      - id: num_threads
        source: max_threads
      - id: num_deconvolution_threads
        source: deconvolution_threads
      - id: dd_psf_grid
        source: dd_psf_grid
    out:
      - id: image_I_nonpb_name
      - id: image_I_pb_name
      - id: images_extra
{% if save_source_list %}
      - id: skymodel_nonpb
      - id: skymodel_pb
{% endif %}

{% if peel_bright_sources %}
# start peel_bright_sources
  - id: restore_pb
    label: Restore sources to PB image
    doc: |
      This step uses WSClean to restore the bright sources to the primary-beam-
      corrected image.
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_restore.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: residual_image
        source: image/image_I_pb_name
      - id: source_list
        source: bright_skymodel_pb
      - id: output_image
        source: image/image_I_pb_name
        valueFrom: $(self.basename)
      - id: numthreads
        source: max_threads
    out:
      - id: restored_image

  - id: restore_nonpb
    label: Restore sources to non-PB image
    doc: |
      This step uses WSClean to restore the bright sources to the non-primary-beam-
      corrected image.
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_restore.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: residual_image
        source: image/image_I_nonpb_name
      - id: source_list
        source: bright_skymodel_pb
      - id: output_image
        source: image/image_I_nonpb_name
        valueFrom: $(self.basename)
      - id: numthreads
        source: max_threads
    out:
      - id: restored_image
{% endif %}
# end peel_bright_sources

  - id: filter
    label: Filter sources
    doc: |
      This step uses PyBDSF to filter artifacts from the sky model.
    run: {{ rapthor_pipeline_dir }}/steps/filter_skymodel.cwl
    in:
{% if peel_bright_sources %}
      - id: true_sky_image
        source: restore_pb/restored_image
      - id: flat_noise_image
        source: restore_nonpb/restored_image
      - id: bright_true_sky_skymodel
        source: bright_skymodel_pb
{% else %}
      - id: true_sky_image
        source: image/image_I_pb_name
      - id: flat_noise_image
        source: image/image_I_nonpb_name
{% endif %}
      - id: true_sky_skymodel
{% if save_source_list %}
        source: image/skymodel_pb
{% else %}
        valueFrom: 'none'
{% endif %}
      - id: output_root
        source: image_name
      - id: vertices_file
        source: vertices_file
      - id: threshisl
        source: threshisl
      - id: threshpix
        source: threshpix
      - id: beamMS
        source: obs_filename
      - id: ncores
        source: max_threads
    out:
      - id: filtered_skymodel_true_sky
      - id: filtered_skymodel_apparent_sky
      - id: diagnostics
      - id: flat_noise_rms_image
      - id: true_sky_rms_image
      - id: source_catalog

  - id: find_diagnostics
    label: Find image diagnostics
    doc: |
      This step derives various image diagnostics.
    run: {{ rapthor_pipeline_dir }}/steps/calculate_image_diagnostics.cwl
    in:
      - id: flat_noise_image
        source: image/image_I_nonpb_name
      - id: flat_noise_rms_image
        source: filter/flat_noise_rms_image
      - id: true_sky_image
        source: image/image_I_pb_name
      - id: true_sky_rms_image
        source: filter/true_sky_rms_image
      - id: input_catalog
        source: filter/source_catalog
      - id: input_skymodel
        source: filter/filtered_skymodel_true_sky
      - id: output_root
        source: image_name
      - id: obs_ms
        source: obs_filename
      - id: diagnostics_file
        source: filter/diagnostics
    out:
      - id: diagnostics
