cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean]
label: Make an image
doc: |
  This tool makes an image using WSClean with facet-based corrections. See
  wsclean_image_screens.cwl for a detailed description of the inputs and outputs.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - -no-update-model-required
  - -save-source-list
  - -local-rms
  - -join-channels
  - -apply-facet-beam
  - -log-time
  - valueFrom: 'wgridder'
    prefix: -gridder
  - valueFrom: '$(runtime.tmpdir)'
    prefix: -temp-dir
  - valueFrom: '2048'
    prefix: -parallel-deconvolution
  - valueFrom: 'I'
    prefix: -pol
  - valueFrom: '0.85'
    prefix: -mgain
  - valueFrom: '0.8'
    prefix: -multiscale-scale-bias
  - valueFrom: '3'
    prefix: -fit-spectral-pol
  - valueFrom: '1.0'
    prefix: -auto-threshold
  - valueFrom: '50'
    prefix: -local-rms-window
  - valueFrom: 'rms-with-min'
    prefix: -local-rms-method
  - valueFrom: '120'
    prefix: -facet-beam-update
  - valueFrom: 'briggs'
    # Note: we have to set part of the 'weight' argument here and part below, as it has
    # three parts (e.g., '-weight briggs -0.5'), and WSClean will not parse the value
    # correctly if it's given together with 'briggs'. We force the parts to be adjacent
    # using the position arg here and below
    prefix: -weight
    position: 1

inputs:
  - id: msin
    type: Directory[]
    inputBinding:
      position: 5
  - id: name
    type: string
    inputBinding:
      prefix: -name
  - id: mask
    type: File
    inputBinding:
      prefix: -fits-mask
  - id: wsclean_imsize
    type: int[]
    inputBinding:
      prefix: -size
  - id: wsclean_niter
    type: int
    inputBinding:
      prefix: -niter
  - id: wsclean_nmiter
    type: int
    inputBinding:
      prefix: -nmiter
  - id: robust
    type: float
    inputBinding:
      position: 2
  - id: min_uv_lambda
    type: float
    inputBinding:
      prefix: -minuv-l
  - id: max_uv_lambda
    type: float
    inputBinding:
      prefix: -maxuv-l
  - id: multiscale
    type: boolean
    inputBinding:
      prefix: -multiscale
  - id: apply_diagonal_solutions
    label: Apply diagonal solutions
    doc: |
      Apply diagonal (separate XX and YY) solutions.
    type: boolean
    inputBinding:
      prefix: -diagonal-solutions
  - id: cellsize_deg
    type: float
    inputBinding:
      prefix: -scale
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
  - id: wsclean_mem
    type: float
    inputBinding:
      prefix: -abs-mem
  - id: auto_mask
    type: float
    inputBinding:
      prefix: -auto-mask
  - id: idg_mode
    type: string
    inputBinding:
      prefix: -idg-mode
  - id: num_threads
    type: int
    inputBinding:
      prefix: -j
  - id: num_deconvolution_threads
    type: int
    inputBinding:
      prefix: -deconvolution-threads
  - id: num_gridding_threads
    label: Number of gridding threads
    doc: |
      The number of threads to use during gridding.
    type: int
    inputBinding:
      prefix: -parallel-gridding
  - id: h5parm
    label: h5parm filename
    doc: |
      The filename of the h5parm containing the solutions to apply to correct for DDEs.
    type: File
    inputBinding:
      prefix: -apply-facet-solutions
      position: 3
  - id: soltabs
    label: Solution tables
    doc: |
      The solution table names to apply to correct for DDEs.
    type: string
    inputBinding:
      position: 4
  - id: region_file
    label: ds9 region file
    doc: |
      The ds9 region file that defines the facets.
    type: File
    inputBinding:
      prefix: -facet-regions

outputs:
  - id: image_nonpb_name
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-image.fits
  - id: image_pb_name
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-image-pb.fits
  - id: residual_name
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-residual.fits
  - id: model_name
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-model.fits
  - id: skymodel_nonpb
    type: File
    outputBinding:
      glob: $(inputs.name)-sources.txt
  - id: skymodel_pb
    type: File
    outputBinding:
      glob: $(inputs.name)-sources-pb.txt

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'
