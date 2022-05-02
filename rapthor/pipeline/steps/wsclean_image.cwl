cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean]
label: Make an image
doc: |
  This tool makes an image using WSClean+IDG with a-term corrections.

requirements:
  - class: InitialWorkDirRequirement
    listing:
      - entryname: aterm_plus_beam.cfg
        entry: |
          aterms = [diagonal, beam]
          diagonal.images = [$(inputs.aterm_images.map( (e,i) => (e.path) ).join(' '))]
          beam.differential = true
          beam.update_interval = 120
          beam.usechannelfreq = true
        writable: false
  - class: InlineJavascriptRequirement

arguments:
  - -no-update-model-required
  - -save-source-list
  - -local-rms
  - -join-channels
  - -use-idg
  - -log-time
  - valueFrom: 'I'
    prefix: -pol
  - valueFrom: '0.85'
    prefix: -mgain
  - valueFrom: '3'
    prefix: -fit-spectral-pol
  - valueFrom: '2048'
    prefix: -parallel-deconvolution
  - valueFrom: '1.0'
    prefix: -auto-threshold
  - valueFrom: '50'
    prefix: -local-rms-window
  - valueFrom: 'rms-with-min'
    prefix: -local-rms-method
  - valueFrom: '32'
    prefix: -aterm-kernel-size
  - valueFrom: 'aterm_plus_beam.cfg'
    prefix: -aterm-config
  - valueFrom: 'briggs'
    # Note: we have to set part of the 'weight' argument here and part below, as it has
    # three parts (e.g., '-weight briggs -0.5'), and WSClean will not parse the value
    # correctly if it's given together with 'briggs'. We force the parts to be adjacent
    # using the position arg here and below
    prefix: -weight
    position: 1

inputs:
  - id: msin
    label: Filenames of input MS
    doc: |
      The filenames of input MS files for which imaging will be done.
    type: Directory[]
    inputBinding:
      position: 3
  - id: name
    label: Filename of output image
    doc: |
      The root filename of the output image.
    type: string
    inputBinding:
      prefix: -name
  - id: mask
    label: Filename of mask
    doc: |
      The filename of the clean mask.
    type: File
    inputBinding:
      prefix: -fits-mask
  - id: aterm_images
    label: Filenames of aterm files
    doc: |
      The filenames of the a-term image files.
    type: File[]
    inputBinding:
      valueFrom: null
  - id: wsclean_imsize
    label: Image size
    doc: |
      The size of the image in pixels.
    type: int[]
    inputBinding:
      prefix: -size
  - id: wsclean_niter
    label: Number of iterations
    doc: |
      The maximum number of iterations.
    type: int
    inputBinding:
      prefix: -niter
  - id: wsclean_nmiter
    label: Number of major iterations
    doc: |
      The maximum number of major iterations.
    type: int
    inputBinding:
      prefix: -nmiter
  - id: robust
    label: Robust weighting
    doc: |
      The value of the robust weighting parameter.
    type: float
    inputBinding:
      position: 2
  - id: min_uv_lambda
    label: Minimum us distance
    doc: |
      The minimum uv distance in lambda.
    type: float
    inputBinding:
      prefix: -minuv-l
  - id: max_uv_lambda
    label: Maximum us distance
    doc: |
      The maximum uv distance in lambda.
    type: float
    inputBinding:
      prefix: -maxuv-l
  - id: cellsize_deg
    label: Pixel size
    doc: |
      The size of one pixel of the image in deg.
    type: float
    inputBinding:
      prefix: -scale
  - id: dir_local
    label: Scratch directory
    doc: |
      The path to a (node-local) scratch directory.
    type: string
    inputBinding:
      prefix: -temp-dir
  - id: channels_out
    label: Number of channels
    doc: |
      The number of output channels.
    type: int
    inputBinding:
      prefix: -channels-out
  - id: deconvolution_channels
    label: Number of deconvolution channels
    doc: |
      The number of deconvolution channels.
    type: int
    inputBinding:
      prefix: -deconvolution-channels
  - id: taper_arcsec
    label: Taper value
    doc: |
      The taper value in arcsec.
    type: float
    inputBinding:
      prefix: -taper-gaussian
  - id: wsclean_mem
    label: Memory percentage
    doc: |
      The memory limit in percent of total.
    type: float
    inputBinding:
      prefix: -mem
  - id: auto_mask
    label: Auto mask value
    doc: |
      The auto mask value.
    type: float
    inputBinding:
      prefix: -auto-mask
  - id: idg_mode
    label: IDG mode
    doc: |
      The IDG mode.
    type: string
    inputBinding:
      prefix: -idg-mode
  - id: num_threads
    type: string
    inputBinding:
      prefix: -j
  - id: num_deconvolution_threads
    label: Number of threads
    doc: |
      The number of threads to use.
    type: string
    inputBinding:
      prefix: -deconvolution-threads

outputs:
  - id: image_nonpb_name
    label: Output non-PB-corrected image
    doc: |
      The filename of the output non-primary-beam-corrected image. The value is
      constructed from the input parameter "name"
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-image.fits
  - id: image_pb_name
    label: Output PB-corrected image
    doc: |
      The filename of the output primary-beam-corrected image. The value is
      constructed from the input parameter "name"
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-image-pb.fits
  - id: skymodel_nonpb
    label: Output non-PB-corrected sky model
    doc: |
      The filename of the output primary beam-corrected image. The value is
      constructed from the input parameter "name"
    type: File
    outputBinding:
      glob: $(inputs.name)-sources.txt
  - id: skymodel_pb
    label: Output PB-corrected image
    doc: |
      The filename of the output primary beam-corrected image. The value is
      constructed from the input parameter "name"
    type: File
    outputBinding:
      glob: $(inputs.name)-sources-pb.txt

hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
