cwlVersion: v1.0
class: Workflow
label: Rapthor imaging pipeline
doc: |
  This workflow performs imaging with direction-dependent corrections. The
  imaging data are generated (and averaged if possible) and WSClean+IDG is
  used to perform the imaging. Masking and sky model filtering is then done
  using PyBDSF.

requirements:
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
        items: string

  - id: prepare_filename
    label: Filename of imaging MS
    doc: |
      The filenames of output MS files used for imaging (length = n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: string

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

  - id: previous_mask_filename
    label: Filename of previous mask
    doc: |
      The filename of the image mask from the previous iteration (length = n_sectors).
    type: string[]

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
    type: string[]

  - id: region_file
    label: Filename of region file
    doc: |
      The filename of the region file (length = n_sectors).
    type: string[]

{% if use_screens %}
  - id: aterms_config_file
    label: Filename of config file
    doc: |
      The filename of the a-term config file (length = n_sectors).
    type: string[]

  - id: aterm_image_filenames
    label: Filenames of a-terms
    doc: |
      The filenames of the a-term images (length = n_sectors).
    type: string[]

{% if use_mpi %}
  - id: mpi_nnodes
    label: Number of nodes
    doc: |
      The number of nodes for MPI jobs (length = n_sectors).
    type: int[]

{% endif %}
{% else %}
  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the calibration solutions (length =
      n_sectors).
    type: string[]

  - id: central_patch_name
    label: Name of central patch
    doc: |
      The name of the central patch of the sector (length = n_sectors).
    type: string[]

{% endif %}
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
    label: Minimum us distance
    doc: |
      The WSClean minimum uv distance in lambda (length = n_sectors).
    type: float[]

  - id: max_uv_lambda
    label: Maximum us distance
    doc: |
      The WSClean maximum uv distance in lambda (length = n_sectors).
    type: float[]

{% if do_multiscale_clean %}
  - id: multiscale_scales_pixel
    label: Multiscale scales
    doc: |
      The WSClean multiscale scales in pixels (length = n_sectors).
    type: string[]

{% endif %}
  - id: dir_local
    label: Scratch directory
    doc: |
      The path to a (node-local) scratch directory (length = n_sectors).
    type: string[]

  - id: taper_arcsec
    label: Taper value
    doc: |
      The WSClean taper value in arcsec (length = n_sectors).
    type: float[]

  - id: wsclean_mem
    label: Memory percentage
    doc: |
      The memory limit for WSClean in percent of total (length = n_sectors).
    type: float[]

  - id: auto_mask
    label: Auto mask value
    doc: |
      The WSClean auto mask value (length = n_sectors).
    type: float[]

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

  - id: bright_skymodel_pb
    label: Bright-source sky model
    doc: |
      The primary-beam-corrected bright-source sky model (length = n_sectors).
    type: string[]

  - id: peel_bright
    label: Peeling flag
    doc: |
      The flag that sets whether peeling of bright sources was done in the predict
      pipeline (length = n_sectors).
    type: string[]

outputs: []

steps:
  - id: image_sector
    label: Image a sector
    doc: |
      This step is a subpipeline that performs the processing (imaging, etc) for
      each sector.
    run: {{ pipeline_working_dir }}/subpipeline_parset.cwl
    in:
      - id: obs_filename
        source: obs_filename
      - id: prepare_filename
        source: prepare_filename
      - id: starttime
        source: starttime
      - id: ntimes
        source: ntimes
      - id: image_freqstep
        source: image_freqstep
      - id: image_timestep
        source: image_timestep
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
{% if use_screens %}
      - id: aterms_config_file
        source: aterms_config_file
      - id: aterm_image_filenames
        source: aterm_image_filenames
{% if use_mpi %}
      - id: mpi_nnodes
        source: mpi_nnodes
{% endif %}
{% else %}
      - id: h5parm
        source: h5parm
      - id: central_patch_name
        source: central_patch_name
{% endif %}
      - id: channels_out
        source: channels_out
      - id: deconvolution_channels
        source: deconvolution_channels
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
{% if do_multiscale_clean %}
      - id: multiscale_scales_pixel
        source: multiscale_scales_pixel
{% endif %}
      - id: dir_local
        source: dir_local
      - id: taper_arcsec
        source: taper_arcsec
      - id: wsclean_mem
        source: wsclean_mem
      - id: auto_mask
        source: auto_mask
      - id: idg_mode
        source: idg_mode
      - id: threshisl
        source: threshisl
      - id: threshpix
        source: threshpix
      - id: bright_skymodel_pb
        source: bright_skymodel_pb
      - id: peel_bright
        source: peel_bright
{% if use_screens %}
    scatter: [obs_filename, prepare_filename, starttime, ntimes, image_freqstep,
              image_timestep, previous_mask_filename, mask_filename,
              phasecenter, ra, dec, image_name, cellsize_deg, wsclean_imsize,
              vertices_file, region_file, aterms_config_file, aterm_image_filenames,
{% if use_mpi %}
              mpi_nnodes,
{% endif %}
{% if do_multiscale_clean %}
              multiscale_scales_pixel,
{% endif %}
              channels_out, deconvolution_channels, wsclean_niter,
              wsclean_nmiter, robust, min_uv_lambda,
              max_uv_lambda, dir_local, taper_arcsec, wsclean_mem,
              auto_mask, idg_mode, threshisl, threshpix, bright_skymodel_pb,
              peel_bright]
{% else %}
    scatter: [obs_filename, prepare_filename, starttime, ntimes, image_freqstep,
              image_timestep, previous_mask_filename, mask_filename,
              phasecenter, ra, dec, image_name, cellsize_deg, wsclean_imsize,
              vertices_file, region_file, h5parm, central_patch_name,
{% if use_mpi %}
              mpi_nnodes,
{% endif %}
{% if do_multiscale_clean %}
              multiscale_scales_pixel,
{% endif %}
              channels_out, deconvolution_channels, wsclean_niter,
              wsclean_nmiter, robust, min_uv_lambda,
              max_uv_lambda, dir_local, taper_arcsec, wsclean_mem,
              auto_mask, idg_mode, threshisl, threshpix, bright_skymodel_pb,
              peel_bright]
{% endif %}
    scatterMethod: dotproduct
    out: []
