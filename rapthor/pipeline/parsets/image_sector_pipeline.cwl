cwlVersion: v1.0
class: Workflow

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
    type: string[]
  - id: prepare_filename
    type: string[]
  - id: starttime
    type: string[]
  - id: ntimes
    type: int[]
  - id: image_freqstep
    type: int[]
  - id: image_timestep
    type: int[]
  - id: previous_mask_filename
    type: string
  - id: mask_filename
    type: string
  - id: phasecenter
    type: string
  - id: ra
    type: float
  - id: dec
    type: float
  - id: image_name
    type: string
  - id: cellsize_deg
    type: float
  - id: wsclean_imsize
    type: int[]
  - id: vertices_file
    type: string
  - id: region_file
    type: string
{% if use_screens %}
  - id: aterms_config_file
    type: string
  - id: aterm_image_filenames
    type: string
{% if use_mpi %}
  - id: mpi_ntasks_per_node
    type: int
  - id: mpi_nnodes
    type: int
{% endif %}
{% else %}
  - id: h5parm
    type: string
  - id: central_patch_name
    type: string
{% endif %}
  - id: channels_out
    type: int
  - id: deconvolution_channels
    type: int
  - id: wsclean_niter
    type: int
  - id: wsclean_nmiter
    type: int
  - id: robust
    type: float
  - id: min_uv_lambda
    type: float
  - id: max_uv_lambda
    type: float
{% if do_multiscale_clean %}
  - id: multiscale_scales_pixel
    type: string
{% endif %}
  - id: dir_local
    type: string
  - id: taper_arcsec
    type: float
  - id: wsclean_mem
    type: float
  - id: auto_mask
    type: float
  - id: idg_mode
    type: string
  - id: threshisl
    type: float
  - id: threshpix
    type: float
  - id: bright_skymodel_pb
    type: string
  - id: peel_bright
    type: string

outputs: []

steps:
  - id: prepare_imaging_data
    label: prepare_imaging_data
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
    label: premask
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
    label: make_aterm_config
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
    label: image
{% if use_screens %}
{% if use_mpi %}
{% if do_multiscale_clean %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image_multiscale.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_mpi_image.cwl
{% endif %}
{% else %}
{% if do_multiscale_clean %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_multiscale.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image.cwl
{% endif %}
{% endif %}
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/wsclean_image_no_screens.cwl
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
        source: mpi_ntasks_per_node
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
    label: restore_pb
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
    label: restore_nonpb
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
    label: filter
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
