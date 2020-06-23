cwlVersion: v1.0
class: Workflow

requirements:
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}
  SubworkflowFeatureRequirement: {}

inputs:
  - id: obs_filename
    type:
      type: array
      items:
        type: array
        items: string
  - id: prepare_filename
    type:
      type: array
      items:
        type: array
        items: string
  - id: starttime
    type:
      type: array
      items:
        type: array
        items: string
  - id: ntimes
    type:
      type: array
      items:
        type: array
        items: int
  - id: image_freqstep
    type:
      type: array
      items:
        type: array
        items: int
  - id: image_timestep
    type:
      type: array
      items:
        type: array
        items: int
  - id: previous_mask_filename
    type: string[]
  - id: mask_filename
    type: string[]
  - id: phasecenter
    type: string[]
  - id: ra
    type: float[]
  - id: dec
    type: float[]
  - id: image_name
    type: string[]
  - id: cellsize_deg
    type: float[]
  - id: wsclean_imsize
    type:
      type: array
      items:
        type: array
        items: int
  - id: vertices_file
    type: string[]
  - id: region_file
    type: string[]
{% if use_screens %}
  - id: aterms_config_file
    type: string[]
  - id: aterm_image_filenames
    type: string[]
{% else %}
  - id: h5parm
    type: string[]
  - id: central_patch_name
    type: string[]
{% endif %}
  - id: channels_out
    type: int[]
  - id: deconvolution_channels
    type: int[]
  - id: wsclean_niter
    type: int[]
  - id: robust
    type: float[]
  - id: wsclean_image_padding
    type: float[]
  - id: min_uv_lambda
    type: float[]
  - id: max_uv_lambda
    type: float[]
  - id: multiscale_scales_pixel
    type: string[]
  - id: local_dir
    type: string[]
  - id: taper_arcsec
    type: float[]
  - id: auto_mask
    type: float[]
  - id: idg_mode
    type: string[]
  - id: threshisl
    type: float[]
  - id: threshpix
    type: float[]

outputs: []

steps:
  - id: image_sector
    label: image_sector
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
      - id: robust
        source: robust
      - id: wsclean_image_padding
        source: wsclean_image_padding
      - id: min_uv_lambda
        source: min_uv_lambda
      - id: max_uv_lambda
        source: max_uv_lambda
      - id: multiscale_scales_pixel
        source: multiscale_scales_pixel
      - id: local_dir
        source: local_dir
      - id: taper_arcsec
        source: taper_arcsec
      - id: auto_mask
        source: auto_mask
      - id: idg_mode
        source: idg_mode
      - id: threshisl
        source: threshisl
      - id: threshpix
        source: threshpix
{% if use_screens %}
    scatter: [obs_filename, prepare_filename, starttime, ntimes, image_freqstep,
              image_timestep, previous_mask_filename, mask_filename,
              phasecenter, ra, dec, image_name, cellsize_deg, wsclean_imsize,
              vertices_file, region_file, aterms_config_file,
              aterm_image_filenames, channels_out, deconvolution_channels,
              wsclean_niter, robust, wsclean_image_padding, min_uv_lambda,
              max_uv_lambda, multiscale_scales_pixel, local_dir, taper_arcsec,
              auto_mask, idg_mode, threshisl, threshpix]
{% else %}
    scatter: [obs_filename, prepare_filename, starttime, ntimes, image_freqstep,
              image_timestep, previous_mask_filename, mask_filename,
              phasecenter, ra, dec, image_name, cellsize_deg, wsclean_imsize,
              vertices_file, region_file, h5parm, central_patch_name,
              channels_out, deconvolution_channels, wsclean_niter, robust,
              wsclean_image_padding, min_uv_lambda, max_uv_lambda,
              multiscale_scales_pixel, local_dir, taper_arcsec, auto_mask,
              idg_mode, threshisl, threshpix]
{% endif %}
    scatterMethod: dotproduct
    out: []
