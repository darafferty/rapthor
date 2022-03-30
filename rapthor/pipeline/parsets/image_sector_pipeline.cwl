cwlVersion: v1.2
class: Workflow
label: Rapthor imaging subpipeline
doc: |
  This subworkflow performs imaging with direction-dependent corrections. The
  imaging data are generated (and averaged if possible) and WSClean+IDG is
  used to perform the imaging. Masking and sky model filtering is then done
  using PyBDSF.

requirements:
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: 1
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: obs_filename
    label: Filenames of input MS
    doc: |
      The filenames of input MS files for which imaging will be done (length =
      n_obs).
    type: string[]

  - id: prepare_filename
    label: Filenames of imaging MS
    doc: |
      The filenames of output MS files used for imaging (length = n_obs).
    type: string[]

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
    type: string

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
    type: string

  - id: region_file
    label: Filename of region file
    doc: |
      The filename of the region file (length = 1).
    type: string

{% if use_screens %}
  - id: aterms_config_file
    label: Filename of config file
    doc: |
      The filename of the a-term config file (length = 1).
    type: string

  - id: aterm_image_filenames
    label: Filenames of a-terms
    doc: |
      The filenames of the a-term images (length = 1).
    type: string
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
{% else %}
  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the calibration solutions (length =
      1).
    type: string

  - id: central_patch_name
    label: Name of central patch
    doc: |
      The name of the central-most patch of the sector (length = 1).
    type: string

{% endif %}
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
    label: Minimum us distance
    doc: |
      The WSClean minimum uv distance in lambda (length = 1).
    type: float

  - id: max_uv_lambda
    label: Maximum us distance
    doc: |
      The WSClean maximum uv distance in lambda (length = 1).
    type: float

{% if do_multiscale_clean %}
  - id: multiscale_scales_pixel
    label: Multiscale scales
    doc: |
      The WSClean multiscale scales in pixels (length = 1).
    type: string

{% endif %}
  - id: dir_local
    label: Scratch directory
    doc: |
      The path to a (node-local) scratch directory (length = 1).
    type: string

  - id: taper_arcsec
    label: Taper value
    doc: |
      The WSClean taper value in arcsec (length = 1).
    type: float

  - id: wsclean_mem
    label: Memory percentage
    doc: |
      The memory limit for WSClean in percent of total (length = 1).
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

  - id: bright_skymodel_pb
    label: Bright-source sky model
    doc: |
      The primary-beam-corrected bright-source sky model (length = 1).
    type: string

  - id: peel_bright
    label: Peeling flag
    doc: |
      The flag that sets whether peeling of bright sources was done in the predict
      pipeline (length = 1).
    type: boolean

outputs: []

steps:
  - id: prepare_imaging_data
    label: Prepare imaging data
    doc: |
      This step uses DPPP to prepare the input data for imaging. This involves
      averaging, phase shifting, and optionally the application of the
      calibration solutions at the center.
{% if use_screens %}

    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data.cwl

{% else %}

{% if do_slowgain_solve %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data_no_screens.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/prepare_imaging_data_no_screens_phase_only.cwl
{% endif %}

{% endif %}
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
        valueFrom: '{{ max_threads }}'
{% if use_screens %}
    scatter: [msin, msout, starttime, ntimes, freqstep, timestep]
{% else %}
      - id: h5parm
        source: h5parm
      - id: central_patch_name
        source: central_patch_name
    scatter: [msin, msout, starttime, ntimes, freqstep, timestep]
{% endif %}
    scatterMethod: dotproduct
    out:
      - id: msimg

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

{% if use_screens %}
  - id: make_aterm_config
    label: Make a-term config file
    doc: |
      This step makes the a-term configuration file needed for WSClean+IDG.
    run: {{ rapthor_pipeline_dir }}/steps/make_aterm_config.cwl
    in:
      - id: outfile
        source: aterms_config_file
      - id: gain_filenames
        source: aterm_image_filenames
    out:
      - id: aterms_config
{% endif %}

  - id: image
    label: Make an image
    doc: |
      This step makes an image using WSClean. Direction-dependent effects
      can be corrected for using a-term images.
{% if use_screens %}
{% if use_mpi %}
{% if do_multiscale_clean %}
{% if toil_version < 5 %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_multiscale_toil4.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_multiscale.cwl
{% endif %}
{% else %}
{% if toil_version < 5 %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_toil4.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image.cwl
{% endif %}
{% endif %}
{% else %}
{% if do_multiscale_clean %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_multiscale.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image.cwl
{% endif %}
{% endif %}
{% else %}
{% if do_multiscale_clean %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_no_screens_multiscale.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_no_screens.cwl
{% endif %}
{% endif %}
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: prepare_imaging_data/msimg
      - id: name
        source: image_name
      - id: mask
        source: premask/maskimg
{% if use_screens %}
      - id: config
        source: make_aterm_config/aterms_config
{% if use_mpi %}
      - id: ntasks
        source: mpi_cpus_per_task
      - id: nnodes
        source: mpi_nnodes
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
      - id: cellsize_deg
        source: cellsize_deg
{% if do_multiscale_clean %}
      - id: multiscale_scales_pixel
        source: multiscale_scales_pixel
{% endif %}
      - id: dir_local
        source: dir_local
      - id: channels_out
        source: channels_out
      - id: deconvolution_channels
        source: deconvolution_channels
      - id: taper_arcsec
        source: taper_arcsec
      - id: wsclean_mem
        source: wsclean_mem
      - id: auto_mask
        source: auto_mask
      - id: idg_mode
        source: idg_mode
      - id: num_threads
        valueFrom: '{{ max_threads }}'
      - id: num_deconvolution_threads
        valueFrom: '{{ deconvolution_threads }}'
    out:
      - id: image_nonpb_name
      - id: image_pb_name
      - id: skymodel_nonpb
      - id: skymodel_pb

{% if peel_bright_sources %}
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
        source: image/image_pb_name
      - id: source_list
        source: bright_skymodel_pb
      - id: output_image
        source: image/image_pb_name
      - id: numthreads
        valueFrom: '{{ max_threads }}'
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
        source: image/image_nonpb_name
      - id: source_list
        source: bright_skymodel_pb
      - id: output_image
        source: image/image_nonpb_name
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    out:
      - id: restored_image
{% endif %}

  - id: filter
    label: Filter sources
    doc: |
      This step uses PyBDSF to filter artifacts from the sky model and make
      a clean mask for the next iteration.
    run: {{ rapthor_pipeline_dir }}/steps/filter_skymodel.cwl
    in:
      - id: input_image
{% if peel_bright_sources %}
        source: restore_nonpb/restored_image
{% else %}
        source: image/image_nonpb_name
{% endif %}
      - id: input_skymodel_pb
        source: image/skymodel_pb
      - id: input_bright_skymodel_pb
        source: bright_skymodel_pb
      - id: output_root
        source: image_name
      - id: vertices_file
        source: vertices_file
      - id: threshisl
        source: threshisl
      - id: threshpix
        source: threshpix
      - id: beamMS
        source: prepare_imaging_data/msimg
      - id: peel_bright
        source: peel_bright
    out: []
