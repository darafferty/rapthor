cwlVersion: v1.2
class: Workflow
label: Rapthor imaging workflow
doc: |
  This workflow performs imaging with direction-dependent corrections. The
  imaging data are generated (and averaged if possible) and WSClean is
  used to perform the imaging. Masking and sky model filtering is then done
  using either PyBDSF or SoFiA-2 as source finder.

requirements:
  MultipleInputFeatureRequirement: {}
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}
  SubworkflowFeatureRequirement: {}

inputs:
  - id: obs_filename
    label: Filename of input MS
    doc: |
      The filenames of input MS files for which imaging will be done (length =
      n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: Directory

  - id: data_colname
    label: Input MS data column
    doc: |
      The data column to be read from the MS files for imaging (length=1).
    type: string

  - id: prepare_filename
    label: Filename of preparatory MS
    doc: |
      The filenames of the preparatory MS files used as input to concatenation
      (length = n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: string

  - id: concat_filename
    label: Filename of imaging MS
    doc: |
      The filename of the MS file resulting from concatenation of the preparatory
      files and used for imaging (length = n_sectors).
    type: string[]

  - id: starttime
    label: Start time of each obs
    doc: |
      The start time (in casacore MVTime) for each observation (length = n_obs *
      n_sectors).
    type:
      type: array
      items:
        type: array
        items: string

  - id: ntimes
    label: Number of times of each obs
    doc: |
      The number of timeslots for each observation (length = n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: int

  - id: image_freqstep
    label: Averaging interval in frequency
    doc: |
      The averaging interval in number of frequency channels (length = n_obs *
      n_sectors).
    type:
      type: array
      items:
        type: array
        items: int

  - id: image_timestep
    label: Fast solution interval in time
    doc: |
      The averaging interval in number of timeslots (length = n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: int

  - id: image_timebase
    label: BDA timebase
    doc: |
      The baseline length (in meters) below which BDA time averaging is done
      (length = n_sectors).
    type: float[]

  - id: image_maxinterval
    label: BDA maxinterval
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is
      done (length = n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: int

  - id: previous_mask_filename
    label: Filename of previous mask
    doc: |
      The filename of the image mask from the previous iteration (length = n_sectors).
    type:
      type: array
      items:
        - File
        - "null"

  - id: mask_filename
    label: Filename of current mask
    doc: |
      The filename of the current image mask (length = n_sectors).
    type: string[]

  - id: phasecenter
    label: Phase center of image
    doc: |
      The phase center of the image in deg (length = n_sectors).
    type: string[]

  - id: ra
    label: RA of center of image
    doc: |
      The RA of the center of the image in deg (length = n_sectors).
    type: float[]

  - id: dec
    label: Dec of center of image
    doc: |
      The Dec of the center of the image in deg (length = n_sectors).
    type: float[]

  - id: image_name
    label: Filename of output image
    doc: |
      The root filename of the output image (length = n_sectors).
    type: string[]

  - id: cellsize_deg
    label: Pixel size
    doc: |
      The size of one pixel of the image in deg (length = n_sectors).
    type: float[]

  - id: wsclean_imsize
    label: Image size
    doc: |
      The size of the image in pixels (length = 2 * n_sectors).
    type:
      type: array
      items:
        type: array
        items: int

  - id: vertices_file
    label: Filename of vertices file
    doc: |
      The filename of the file containing sector vertices (length = n_sectors).
    type: File[]

  - id: region_file
    label: Filename of region file
    doc: |
      The filename of the region file (length = n_sectors).
    type:
      type: array
      items:
        - File
        - "null"

{% if use_mpi %}
  - id: mpi_cpus_per_task
    label: Number of CPUs per task
    doc: |
      The number of CPUs per task to request from Slurm for MPI jobs (length = n_sectors).
    type: int[]

  - id: mpi_nnodes
    label: Number of nodes
    doc: |
      The number of nodes for MPI jobs (length = n_sectors).
    type: int[]
{% endif %}

  - id: prepare_data_steps
    label: DP3 steps
    doc: |
      The steps to perform in the prepare data DP3 step (length = 1).
    type: string

  - id: prepare_data_applycal_steps
    label: DP3 steps
    doc: |
      The steps to perform in the applycal part of the prepare data DP3 step (length = 1).
    type: string?

  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the calibration solutions (length =
      1).
    type: File?

  - id: fulljones_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the full-Jones calibration solutions
      (length = 1).
    type: File?

  - id: input_normalize_h5parm
    label: Filename of normalize h5parm
    doc: |
      The filename of the input h5parm file with the flux-scale normalizations
      (length = n_sectors).
    type: File?

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
        The RA in degrees of the middle of the region to be imaged (length =
        n_sectors).
    type: float[]

  - id: dec_mid
    label: Dec of the midpoint
    doc: |
        The Dec in degrees of the middle of the region to be imaged (length =
        n_sectors).
    type: float[]

  - id: width_ra
    label: Width along RA
    doc: |
      The width along RA in degrees (corrected to Dec = 0) of the region to be
      imaged (length = n_sectors).
    type: float[]

  - id: width_dec
    label:  Width along Dec
    doc: |
      The width along Dec in degrees of the region to be imaged (length =
      n_sectors).
    type: float[]

  - id: facet_region_file
    label: Filename of output region file
    doc: |
      The filename of the output ds9 region file (length = n_sectors).
    type: string[]

  - id: soltabs
    label: Names of calibration soltabs
    doc: |
      The name of the calibration solution table (length = 1).
    type: string

  - id: scalar_visibilities
    label: Use scalar visibilities
    doc: |
      Use only scalar (Stokes I) visibilities (length = 1).
    type: boolean

  - id: diagonal_visibilities
    label: Use diagonal visibilities
    doc: |
      Use only diagonal (XX and YY) visibilities (length = 1).
    type: boolean

  - id: parallel_gridding_threads
    label: Max number of gridding threads
    doc: |
      The maximum number of threads to use during parallel gridding (length = 1).
    type: int

{% else %}
# start not use_facets

{% if preapply_dde_solutions %}
  - id: central_patch_name
    label: Name of central patch
    doc: |
      The name of the central patch of the sector (length = n_sectors).
    type: string[]?
{% endif %}

{% endif %}
# end use_facets / not use_facets

  - id: channels_out
    label: Number of channels
    doc: |
      The number of WSClean output channels (length = n_sectors).
    type: int[]

  - id: deconvolution_channels
    label: Number of deconvolution channels
    doc: |
      The number of WSClean deconvolution channels (length = n_sectors).
    type: int[]

  - id: fit_spectral_pol
    label: Spectral poly order
    doc: |
      The order of WSClean spectral polynomial (length = n_sectors).
    type: int[]

  - id: wsclean_niter
    label: Number of iterations
    doc: |
      The number of WSClean iterations (length = n_sectors).
    type: int[]

  - id: wsclean_nmiter
    label: Number of major iterations
    doc: |
      The number of WSClean major iterations (length = n_sectors).
    type: int[]

  - id: robust
    label: Robust weighting
    doc: |
      The value of the WSClean robust weighting parameter (length = n_sectors).
    type: float[]

  - id: min_uv_lambda
    label: Minimum uv distance
    doc: |
      The WSClean minimum uv distance in lambda (length = n_sectors).
    type: float[]

  - id: max_uv_lambda
    label: Maximum uv distance
    doc: |
      The WSClean maximum uv distance in lambda (length = n_sectors).
    type: float[]

  - id: mgain
    label: Cleaning gain
    doc: |
      The WSClean cleaning gain for major iterations (length = n_sectors).
    type: float[]

  - id: do_multiscale
    label: Activate multiscale
    doc: |
      Activate multiscale clean (length = n_sectors).
    type: boolean[]

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

  - id: skip_final_iteration
    label: Skip final major iteration
    doc: |
      Skip the final WSClean major iteration at the end of clean (length = 1).
    type: boolean

  - id: taper_arcsec
    label: Taper value
    doc: |
      The WSClean taper value in arcsec (length = n_sectors).
    type: float[]

  - id: local_rms_strength
    label: RMS strength value
    doc: |
      The WSClean local RMS strength value (length = n_sectors).
    type: float[]

  - id: local_rms_window
    label: RMS window size
    doc: |
      The WSClean local RMS window size (length = n_sectors).
    type: float[]

  - id: local_rms_method
    label: RMS method
    doc: |
      The WSClean local RMS method (length = n_sectors).
    type: string[]

  - id: wsclean_mem
    label: Memory in GB
    doc: |
      The memory limit for WSClean in GB (length = n_sectors).
    type: float[]

  - id: auto_mask
    label: Auto mask value
    doc: |
      The WSClean auto mask value (length = n_sectors).
    type: float[]

  - id: auto_mask_nmiter
    label: Auto mask nmiter value
    doc: |
      The WSClean auto mask nmiter value (length = n_sectors).
    type: int[]

  - id: idg_mode
    label: IDG mode
    doc: |
      The WSClean IDG mode (length = n_sectors).
    type: string[]

  - id: threshisl
    label: Island threshold
    doc: |
      The PyBDSF island threshold (length = n_sectors).
    type: float[]

  - id: threshpix
    label: Pixel threshold
    doc: |
      The PyBDSF pixel threshold (length = n_sectors).
    type: float[]

  - id: filter_by_mask
    label: Filter source list by mask
    doc: |
      Filter the source list by the PyBDSF mask (length = 1).
    type: boolean

  - id: source_finder
    label: Source finder
    doc: |
      Name of the source finder to use.
    type:
      type: enum
      symbols: ["bdsf", "sofia"]

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
      vertically in the image (length = n_sectors).
    type:
      type: array
      items:
        type: array
        items: int

  - id: interval
    label: Input data interval
    doc: |
      The interval to use for the input data, as [start_timeslot, end_timeslot] (length = 2).
    type: int[]

  - id: apply_time_frequency_smearing
    label: Apply smearing corrections
    doc: |
      Apply corrections for time and frequency smearing (length = 1).
    type: boolean

{% if make_image_cube %}
  - id: image_cube_name
    label: Filename of output image cube
    doc: |
      The filename of the output image cube (length = n_sectors).
    type: string[]
{% endif %}

{% if normalize_flux_scale %}
  - id: output_source_catalog
    label: Filename of FITS source catalog
    doc: |
      The filename of the FITS source catalog to use for flux-scale normalizations
      (length = n_sectors).
    type: string[]
  - id: output_normalize_h5parm
    label: Filename of normalize h5parm
    doc: |
      The filename of the output h5parm file with the flux-scale normalizations
      (length = n_sectors).
    type: string[]
{% endif %}

outputs:
  - id: filtered_skymodel_true_sky
    outputSource:
      - image_sector/filtered_skymodel_true_sky
    type: File[]
  - id: filtered_skymodel_apparent_sky
    outputSource:
      - image_sector/filtered_skymodel_apparent_sky
    type: File[]
  - id: pybdsf_catalog
    outputSource:
      - image_sector/pybdsf_catalog
    type: File[]
  - id: sector_diagnostics
    outputSource:
      - image_sector/sector_diagnostics
    type: File[]
  - id: sector_offsets
    outputSource:
      - image_sector/sector_offsets
    type: File[]
  - id: sector_diagnostic_plots
    outputSource:
      - image_sector/sector_diagnostic_plots
    type:
      type: array
      items:
        type: array
        items: File
  - id: visibilities
    outputSource:
      - image_sector/visibilities
    type:
      type: array
      items:
        type: array
        items: Directory
  - id: sector_I_images
    outputSource:
      - image_sector/sector_I_images
    type:
      type: array
      items:
        type: array
        items: File
  - id: sector_extra_images
    outputSource:
      - image_sector/sector_extra_images
    type:
      type: array
      items:
        type: array
        items: File
  - id: source_filtering_mask
    outputSource:
      - image_sector/source_filtering_mask
    type: File[]
{% if save_source_list %}
  - id: sector_skymodels
    outputSource:
      - image_sector/sector_skymodels
    type:
      type: array
      items:
        - type: array
          items: File
        - "null"
{% endif %}
{% if use_facets %}
  - id: sector_region_file
    outputSource:
      - image_sector/sector_region_file
    type:
      type: array
      items: File
{% endif %}
{% if make_image_cube %}
{% if "i" in image_cube_stokes_list %}
  - id: sector_image_I_cube
    outputSource:
      - image_sector/sector_image_I_cube
    type: File[]
  - id: sector_image_I_cube_beams
    outputSource:
      - image_sector/sector_image_I_cube_beams
    type: File[]
  - id: sector_image_I_cube_frequencies
    outputSource:
      - image_sector/sector_image_cube_I_frequencies
    type: File[]
{% endif %}
{% if "q" in image_cube_stokes_list %}
  - id: sector_image_Q_cube
    outputSource:
      - image_sector/sector_image_Q_cube
    type: File[]
  - id: sector_image_Q_cube_beams
    outputSource:
      - image_sector/sector_image_Q_cube_beams
    type: File[]
  - id: sector_image_Q_cube_frequencies
    outputSource:
      - image_sector/sector_image_cube_Q_frequencies
    type: File[]
{% endif %}
{% if "u" in image_cube_stokes_list %}
  - id: sector_image_U_cube
    outputSource:
      - image_sector/sector_image_U_cube
    type: File[]
  - id: sector_image_U_cube_beams
    outputSource:
      - image_sector/sector_image_U_cube_beams
    type: File[]
  - id: sector_image_U_cube_frequencies
    outputSource:
      - image_sector/sector_image_cube_U_frequencies
    type: File[]
{% endif %}
{% if "v" in image_cube_stokes_list %}
  - id: sector_image_V_cube
    outputSource:
      - image_sector/sector_image_V_cube
    type: File[]
  - id: sector_image_V_cube_beams
    outputSource:
      - image_sector/sector_image_V_cube_beams
    type: File[]
  - id: sector_image_V_cube_frequencies
    outputSource:
      - image_sector/sector_image_cube_V_frequencies
    type: File[]
{% endif %}
{% endif %}
{% if normalize_flux_scale %}
  - id: sector_source_catalog
    outputSource:
      - image_sector/sector_source_catalog
    type: File[]
  - id: sector_normalize_h5parm
    outputSource:
      - image_sector/sector_normalize_h5parm
    type: File[]
{% endif %}


steps:
  - id: image_sector
    label: Image a sector
    doc: |
      This step is a subworkflow that performs the processing (imaging, etc) for
      each sector.
    run: {{ pipeline_working_dir }}/subpipeline_parset.cwl
    in:
      - id: obs_filename
        source: obs_filename
      - id: data_colname
        source: data_colname
      - id: prepare_filename
        source: prepare_filename
      - id: concat_filename
        source: concat_filename
      - id: starttime
        source: starttime
      - id: ntimes
        source: ntimes
      - id: image_freqstep
        source: image_freqstep
      - id: image_timestep
        source: image_timestep
      - id: image_maxinterval
        source: image_maxinterval
      - id: image_timebase
        source: image_timebase
      - id: previous_mask_filename
        source: previous_mask_filename
      - id: mask_filename
        source: mask_filename
      - id: phasecenter
        source: phasecenter
      - id: ra
        source: ra
      - id: dec
        source: dec
      - id: image_name
        source: image_name
      - id: cellsize_deg
        source: cellsize_deg
      - id: wsclean_imsize
        source: wsclean_imsize
      - id: vertices_file
        source: vertices_file
      - id: region_file
        source: region_file
{% if use_mpi %}
      - id: mpi_cpus_per_task
        source: mpi_cpus_per_task
      - id: mpi_nnodes
        source: mpi_nnodes
{% endif %}
      - id: prepare_data_steps
        source: prepare_data_steps
      - id: prepare_data_applycal_steps
        source: prepare_data_applycal_steps
      - id: h5parm
        source: h5parm
      - id: fulljones_h5parm
        source: fulljones_h5parm
      - id: input_normalize_h5parm
        source: input_normalize_h5parm
{% if use_facets %}
# start use_facets
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
      - id: facet_region_file
        source: facet_region_file
      - id: soltabs
        source: soltabs
      - id: scalar_visibilities
        source: scalar_visibilities
      - id: diagonal_visibilities
        source: diagonal_visibilities
{% else %}
# start not use_facets
{% if preapply_dde_solutions %}
      - id: central_patch_name
        source: central_patch_name
{% endif %}
{% endif %}
# end use_facets / not use_facets
      - id: channels_out
        source: channels_out
      - id: deconvolution_channels
        source: deconvolution_channels
      - id: fit_spectral_pol
        source: fit_spectral_pol
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
      - id: mgain
        source: mgain
      - id: do_multiscale
        source: do_multiscale
      - id: pol
        source: pol
      - id: save_source_list
        source: save_source_list
      - id: link_polarizations
        source: link_polarizations
      - id: join_polarizations
        source: join_polarizations
      - id: skip_final_iteration
        source: skip_final_iteration
      - id: taper_arcsec
        source: taper_arcsec
      - id: local_rms_strength
        source: local_rms_strength
      - id: local_rms_window
        source: local_rms_window
      - id: local_rms_method
        source: local_rms_method
      - id: wsclean_mem
        source: wsclean_mem
      - id: auto_mask
        source: auto_mask
      - id: auto_mask_nmiter
        source: auto_mask_nmiter
      - id: idg_mode
        source: idg_mode
      - id: threshisl
        source: threshisl
      - id: threshpix
        source: threshpix
      - id: filter_by_mask
        source: filter_by_mask
      - id: source_finder
        source: source_finder
      - id: max_threads
        source: max_threads
      - id: deconvolution_threads
        source: deconvolution_threads
      - id: dd_psf_grid
        source: dd_psf_grid
      - id: interval
        source: interval
      - id: apply_time_frequency_smearing
        source: apply_time_frequency_smearing
{% if use_facets %}
      - id: parallel_gridding_threads
        source: parallel_gridding_threads
{% endif %}
{% if peel_bright_sources %}
      - id: bright_skymodel_pb
        source: bright_skymodel_pb
{% endif %}
{% if make_image_cube %}
      - id: image_cube_name
        source: image_cube_name
{% endif %}
{% if normalize_flux_scale %}
      - id: output_source_catalog
        source: output_source_catalog
      - id: output_normalize_h5parm
        source: output_normalize_h5parm
{% endif %}
    scatter: [obs_filename, prepare_filename, concat_filename, starttime, ntimes,
              image_freqstep, image_timestep, image_maxinterval, image_timebase,
              previous_mask_filename, mask_filename, phasecenter, ra, dec,
              image_name, cellsize_deg, wsclean_imsize, vertices_file, region_file,
{% if use_mpi %}
              mpi_cpus_per_task, mpi_nnodes,
{% endif %}
{% if use_facets %}
              ra_mid, dec_mid, width_ra, width_dec, facet_region_file,
{% else %}
{% if preapply_dde_solutions %}
              central_patch_name,
{% endif %}
{% endif %}
{% if make_image_cube %}
              image_cube_name,
{% endif %}
{% if normalize_flux_scale %}
              output_source_catalog,
              output_normalize_h5parm,
{% endif %}
              channels_out, deconvolution_channels, fit_spectral_pol, wsclean_niter,
              wsclean_nmiter, robust, min_uv_lambda, max_uv_lambda, mgain, do_multiscale,
              taper_arcsec, local_rms_strength, local_rms_window, local_rms_method,
              wsclean_mem, auto_mask, auto_mask_nmiter, idg_mode, threshisl, threshpix,
              dd_psf_grid]

    scatterMethod: dotproduct

    out:
      - id: filtered_skymodel_true_sky
      - id: filtered_skymodel_apparent_sky
      - id: pybdsf_catalog
      - id: sector_I_images
      - id: sector_extra_images
      - id: visibilities
      - id: source_filtering_mask
{% if save_source_list %}
      - id: sector_skymodels
{% endif %}
      - id: sector_diagnostics
      - id: sector_offsets
      - id: sector_diagnostic_plots
{% if use_facets %}
      - id: sector_region_file
{% endif %}
{% if make_image_cube %}
{% if "i" in image_cube_stokes_list %}
      - id: sector_image_I_cube
      - id: sector_image_I_cube_beams
      - id: sector_image_I_cube_frequencies
{% endif %}
{% if "q" in image_cube_stokes_list %}
      - id: sector_image_Q_cube
      - id: sector_image_Q_cube_beams
      - id: sector_image_Q_cube_frequencies
{% endif %}
{% if "u" in image_cube_stokes_list %}
      - id: sector_image_U_cube
      - id: sector_image_U_cube_beams
      - id: sector_image_U_cube_frequencies
{% endif %}
{% if "v" in image_cube_stokes_list %}
      - id: sector_image_V_cube
      - id: sector_image_V_cube_beams
      - id: sector_image_V_cube_frequencies
{% endif %}
{% endif %}
{% if normalize_flux_scale %}
      - id: sector_source_catalog
      - id: sector_normalize_h5parm
{% endif %}
