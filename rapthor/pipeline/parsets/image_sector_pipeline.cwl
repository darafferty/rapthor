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

  - id: data_colname
    label: Input MS data column
    doc: |
      The data column to be read from the MS files for imaging (length=1).
    type: string

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

  - id: image_timebase
    label: BDA timebase
    doc: |
      The baseline length (in meters) below which BDA time averaging is done
      (length = 1).
    type: float

  - id: image_maxinterval
    label: BDA maxinterval
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is
      done (length = n_obs).
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

  - id: prepare_data_steps
    label: DP3 steps
    doc: |
      The steps to perform in the prepare data DP3 step (length = 1).
    type: string

  - id: prepare_data_applycal_steps
    label: DP3 steps
    doc: |
      The steps to perform in the applycal part of the prepare data DP3 step
      (length = 1).
    type: string?

  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the direction-dependent calibration
      solutions (length = 1).
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
      (length = 1).
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
      The name of the central-most patch of the sector (length = 1).
    type: string?
{% endif %}

{% endif %}
# end use_facets / not use_facets

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

  - id: mgain
    label: Cleaning gain
    doc: |
      The WSClean cleaning gain for major iterations (length = 1).
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

  - id: skip_final_iteration
    label: Skip final major iteration
    doc: |
      Skip the final major iteration at the end of clean (length = 1).
    type: boolean

  - id: taper_arcsec
    label: Taper value
    doc: |
      The WSClean taper value in arcsec (length = 1).
    type: float

  - id: local_rms_strength
    label: Local RMS strength value
    doc: |
      The WSClean local RMS strength value (length = 1).
    type: float

  - id: local_rms_window
    label: RMS window size
    doc: |
      The WSClean local RMS window size (length = 1).
    type: float

  - id: local_rms_method
    label: RMS method
    doc: |
      The WSClean local RMS method (length = 1).
    type: string

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

  - id: auto_mask_nmiter
    label: Auto mask nmiter value
    doc: |
      The WSClean auto mask nmiter value (length = 1).
    type: int

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
      vertically in the image (length = 2).
    type: int[]

  - id: interval
    label: Input data interval
    doc: |
      The interval to use for the input data, as [start_timeslot, end_timeslot] (length =
      2).
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
      The filename of the output image cube (length = 1).
    type: string
{% endif %}

{% if normalize_flux_scale %}
  - id: output_source_catalog
    label: Filename of FITS source catalog
    doc: |
      The filename of the FITS source catalog to use for flux-scale normalizations
      (length = 1).
    type: string

  - id: output_normalize_h5parm
    label: Filename of normalize h5parm
    doc: |
      The filename of the output h5parm file with the flux-scale normalizations
      (length = 1).
    type: string
{% endif %}

outputs:
  - id: filtered_skymodel_true_sky
    outputSource:
      - filter/filtered_skymodel_true_sky
    type: File
  - id: filtered_skymodel_apparent_sky
    outputSource:
      - filter/filtered_skymodel_apparent_sky
    type: File
  - id: pybdsf_catalog
    outputSource:
      - filter/source_catalog
    type: File
  - id: sector_diagnostics
    outputSource:
      - find_diagnostics/diagnostics
    type: File
  - id: sector_offsets
    outputSource:
      - find_diagnostics/offsets
    type: File
  - id: sector_diagnostic_plots
    outputSource:
      - find_diagnostics/plots
    type: File[]
  - id: visibilities
    outputSource:
      - prepare_imaging_data/msimg
    type: Directory[]
  - id: source_filtering_mask
    outputSource:
      - filter/source_filtering_mask
    type: File
{% if compress_images %}
  - id: sector_I_images
    outputSource:
      - compress/image_I_nonpb_name
      - compress/image_I_pb_name
    type: File[]
  - id: sector_extra_images
    outputSource:
      - compress/images_extra
    type: File[]
{% else %}
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
{% endif %}
{% if save_source_list %}
  - id: sector_skymodels
    outputSource:
      - image/skymodel_nonpb
      - image/skymodel_pb
    type:
      - "null"
      - File[]
{% endif %}
{% if use_facets %}
  - id: sector_region_file
    outputSource:
      - make_region_file/region_file
    type: File
{% endif %}
{% if make_image_cube %}
  - id: sector_image_cube
    outputSource:
      - make_image_cube/image_cube
    type: File
  - id: sector_image_cube_beams
    outputSource:
      - make_image_cube/image_cube_beams
    type: File
  - id: sector_image_cube_frequencies
    outputSource:
      - make_image_cube/image_cube_frequencies
    type: File
{% endif %}
{% if normalize_flux_scale %}
  - id: sector_source_catalog
    outputSource:
      - make_catalog_from_image_cube/source_catalog
    type: File
  - id: sector_normalize_h5parm
    outputSource:
      - normalize_flux_scale/output_h5parm
    type: File
{% endif %}

steps:
  - id: prepare_imaging_data
    label: Prepare imaging data
    doc: |
      This step uses DP3 to prepare the input data for imaging. This involves
      averaging, phase shifting, and optionally the application of the
      calibration solutions.
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: obs_filename
      - id: data_colname
        source: data_colname
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
      - id: maxinterval
        source: image_maxinterval
      - id: timebase
        source: image_timebase
      - id: beamdir
        source: phasecenter
      - id: numthreads
        source: max_threads
{% if preapply_dde_solutions %}
      - id: central_patch_name
        source: central_patch_name
{% endif %}
      - id: h5parm
        source: h5parm
      - id: fulljones_h5parm
        source: fulljones_h5parm
      - id: normalize_h5parm
        source: input_normalize_h5parm
      - id: steps
        source: prepare_data_steps
      - id: applycal_steps
        source: prepare_data_applycal_steps
    scatter: [msin, msout, starttime, ntimes, freqstep, timestep, maxinterval]
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
      - id: data_colname
        source: data_colname
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
      - id: enclose_names
        valueFrom: 'True'
    out:
      - id: region_file
{% endif %}

  - id: image
    label: Make an image
    doc: |
      This step makes an image using WSClean. Direction-dependent effects
      can be corrected for using a-term images or facet-based corrections.
{% if apply_screens %}
# start apply_screens

{% if use_mpi %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_screens.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_screens.cwl
{% endif %}

{% else %}
# start not apply_screens

{% if use_facets %}
# start use_facets

{% if use_mpi %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_facets.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_facets.cwl
{% endif %}

{% else %}
# start not use_facets and not apply_screens (i.e., use no_dde)

{% if use_mpi %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_no_dde.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_no_dde.cwl
{% endif %}

{% endif %}
# end use no_dde

{% endif %}
# end apply_screens / not apply_screens

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
{% if apply_screens or use_facets %}
      - id: h5parm
        source: h5parm
{% endif %}
{% if use_mpi %}
      - id: nnodes
        source: mpi_nnodes
{% endif %}
{% if use_facets %}
      - id: soltabs
        source: soltabs
      - id: region_file
        source: make_region_file/region_file
      - id: scalar_visibilities
        source: scalar_visibilities
      - id: diagonal_visibilities
        source: diagonal_visibilities
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
      - id: mgain
        source: mgain
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
      - id: skip_final_iteration
        source: skip_final_iteration
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
      - id: num_threads
        source: max_threads
      - id: num_deconvolution_threads
        source: deconvolution_threads
      - id: dd_psf_grid
        source: dd_psf_grid
{% if apply_screens %}
      - id: interval
        source: interval
{% endif %}
      - id: apply_time_frequency_smearing
        source: apply_time_frequency_smearing
    out:
      - id: image_I_nonpb_name
      - id: image_I_pb_name
      - id: image_I_pb_channels
      - id: images_extra
{% if save_source_list %}
      - id: skymodel_nonpb
      - id: skymodel_pb
{% endif %}

{% if compress_images %}
# start compress_images
  - id: compress
    label: Compress wsclean FITS images
    doc: |
      This step uses cfitsio fpack to compress all images produced by wsclean
    run: {{ rapthor_pipeline_dir }}/steps/compress_sector_images.cwl
    in:
      - id: images
        source:
{% if peel_bright_sources %}
          - restore_nonpb/restored_image
          - restore_pb/restored_image
{% else %}
          - image/image_I_nonpb_name
          - image/image_I_pb_name
{% endif %}
          - image/images_extra
        linkMerge: merge_flattened
      - id: name
        source: image_name
    out:
      - id: image_I_nonpb_name
      - id: image_I_pb_name
      - id: images_extra

{% endif %}
# end compress_images

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

{% if make_image_cube %}
# start make_image_cube
  - id: make_image_cube
    label: Make an image cube
    doc: |
      This step uses combines the channel images from WSClean
      into an image cube.
    run: {{ rapthor_pipeline_dir }}/steps/make_image_cube.cwl
    in:
      - id: input_image_list
        source: image/image_I_pb_channels
      - id: output_image
        source: image_cube_name
    out:
      - id: image_cube
      - id: image_cube_beams
      - id: image_cube_frequencies
{% endif %}
# end make_image_cube

  - id: check_beam_true_sky_image
    label: Check beam
    doc: |
      This step checks that the restoring beam in the true-sky image is valid.
    run: {{ rapthor_pipeline_dir }}/steps/check_image_beam.cwl
    in:
{% if peel_bright_sources %}
      - id: input_image
        source: restore_pb/restored_image
{% else %}
      - id: input_image
        source: image/image_I_pb_name
{% endif %}
      - id: beam_size_arcsec
        source: taper_arcsec
    out:
      - id: validated_image

  - id: check_beam_flat_noise_image
    label: Check beam
    doc: |
      This step checks that the restoring beam in the flat-noise image is valid.
    run: {{ rapthor_pipeline_dir }}/steps/check_image_beam.cwl
    in:
{% if peel_bright_sources %}
      - id: input_image
        source: restore_nonpb/restored_image
{% else %}
      - id: input_image
        source: image/image_I_nonpb_name
{% endif %}
      - id: beam_size_arcsec
        source: taper_arcsec
    out:
      - id: validated_image

  - id: filter
    label: Filter sources
    doc: |
      This step uses PyBDSF to filter artifacts from the sky model.
    run: {{ rapthor_pipeline_dir }}/steps/filter_skymodel.cwl
    in:
      - id: true_sky_image
        source: check_beam_true_sky_image/validated_image
      - id: flat_noise_image
        source: check_beam_flat_noise_image/validated_image
{% if peel_bright_sources %}
      - id: bright_true_sky_skymodel
        source: bright_skymodel_pb
{% endif %}
      - id: true_sky_skymodel
{% if save_source_list %}
        source: image/skymodel_pb
{% else %}
        valueFrom: 'none'
{% endif %}
      - id: apparent_sky_skymodel
{% if save_source_list %}
        source: image/skymodel_nonpb
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
      - id: filter_by_mask
        source: filter_by_mask
      - id: source_finder
        source: source_finder
      - id: ncores
        source: max_threads
    out:
      - id: filtered_skymodel_true_sky
      - id: filtered_skymodel_apparent_sky
      - id: diagnostics
      - id: flat_noise_rms_image
      - id: true_sky_rms_image
      - id: source_catalog
      - id: source_filtering_mask

  - id: find_diagnostics
    label: Find image diagnostics
    doc: |
      This step derives various image diagnostics.
    run: {{ rapthor_pipeline_dir }}/steps/calculate_image_diagnostics.cwl
    in:
      - id: flat_noise_image
        source: check_beam_flat_noise_image/validated_image
      - id: flat_noise_rms_image
        source: filter/flat_noise_rms_image
      - id: true_sky_image
        source: check_beam_true_sky_image/validated_image
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
      - id: obs_starttime
        source: starttime
      - id: obs_ntimes
        source: ntimes
      - id: diagnostics_file
        source: filter/diagnostics
      - id: facet_region_file
{% if use_facets %}
        source: make_region_file/region_file
{% else %}
        valueFrom: 'none'
{% endif %}
    out:
      - id: diagnostics
      - id: offsets
      - id: plots

{% if normalize_flux_scale %}
# start normalize_flux_scale
  - id: make_catalog_from_image_cube
    label: Make a source catalog from an image cube
    doc: |
      This step makes a source catalog from a FITS image cube
    run: {{ rapthor_pipeline_dir }}/steps/make_catalog_from_image_cube.cwl
    in:
      - id: cube
        source: make_image_cube/image_cube
      - id: cube_beams
        source: make_image_cube/image_cube_beams
      - id: cube_frequencies
        source: make_image_cube/image_cube_frequencies
      - id: output_catalog
        source: output_source_catalog
      - id: threshisl
        source: threshisl
      - id: threshpix
        source: threshpix
      - id: ncores
        source: max_threads
    out:
      - id: source_catalog

  - id: normalize_flux_scale
    label: Normalize the flux scale
    doc: |
      This step determines the corrections necessary to
      normalize the flux scale
    run: {{ rapthor_pipeline_dir }}/steps/normalize_flux_scale.cwl
    in:
      - id: source_catalog
        source: make_catalog_from_image_cube/source_catalog
      - id: ms_file
        source: concat_in_time/msconcat
      - id: normalize_h5parm
        source: output_normalize_h5parm
    out:
      - id: output_h5parm

{% endif %}
# end normalize_flux_scale
