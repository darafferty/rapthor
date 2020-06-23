cwlVersion: v1.0
class: CommandLineTool
baseCommand: [wsclean]
label: "Images a dataset using WSClean+IDG"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - -no-update-model-required
  - -multiscale
  - -fit-beam
  - -reorder
  - -save-source-list
  - -local-rms
  - -join-channels
  - -use-idg
  - valueFrom: 'I'
    prefix: -pol
  - valueFrom: '0.6'
    prefix: -mgain
  - valueFrom: '3'
    prefix: -fit-spectral-pol
  - valueFrom: 'gaussian'
    prefix: -multiscale-shape
  - valueFrom: '3'
    prefix: -weighting-rank-filter
  - valueFrom: '1.0'
    prefix: -auto-threshold
  - valueFrom: '50'
    prefix: -local-rms-window
  - valueFrom: 'rms-with-min'
    prefix: -local-rms-method
  - valueFrom: '32'
    prefix: -aterm-kernel-size
  - valueFrom: '12'
    prefix: -nmiter
  - valueFrom: 'briggs'
    # Note: we have to set part of the 'weight' argument here and part below, as it has
    # three parts (e.g., '-weight briggs -0.5'), and WSClean will not parse the value
    # correctly if it's given together with 'briggs'. We force the parts to be adjacent
    # using the position arg here and below
    prefix: -weight
    position: 1

inputs:
  - id: msin
    type: string[]
    inputBinding:
      position: 3
  - id: name
    type: string
    inputBinding:
      prefix: -name
  - id: mask
    type: string
    inputBinding:
      prefix: -fits-mask
  - id: config
    type: string
    inputBinding:
      prefix: -aterm-config
  - id: wsclean_imsize
    type: int[]
    inputBinding:
      prefix: -size
  - id: wsclean_niter
    type: int
    inputBinding:
      prefix: -niter
  - id: robust
    type: float
    inputBinding:
      position: 2
  - id: wsclean_image_padding
    type: float
    inputBinding:
      prefix: -padding
  - id: min_uv_lambda
    type: float
    inputBinding:
      prefix: -minuv-l
  - id: max_uv_lambda
    type: float
    inputBinding:
      prefix: -maxuv-l
  - id: cellsize_deg
    type: float
    inputBinding:
      prefix: -scale
  - id: multiscale_scales_pixel
    type: string
    inputBinding:
      prefix: -multiscale-scales
  - id: local_dir
    type: string
    inputBinding:
      prefix: -temp-dir
  - id: channels_out
    type: int
    inputBinding:
      prefix: -channels-out
  - id: deconvolution_channels
    type: int
    inputBinding:
      prefix: -deconvolution-channels
  - id: taper_arcsec
    type: float
    inputBinding:
      prefix: -taper-gaussian
  - id: auto_mask
    type: float
    inputBinding:
      prefix: -auto-mask
  - id: idg_mode
    type: string
    inputBinding:
      prefix: -idg-mode

outputs:
  - id: image_nonpb_name
    type: string
    outputBinding:
      outputEval: $(inputs.name)-MFS-image.fits
  - id: skymodel_nonpb
    type: string
    outputBinding:
      outputEval: $(inputs.name)-sources.txt
  - id: skymodel_pb
    type: string
    outputBinding:
      outputEval: $(inputs.name)-sources-pb.txt
