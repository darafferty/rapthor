#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
baseCommand: wsclean-mp
label: Make an image
doc: |
  This tool makes an image using WSClean with a-term corrections, distributed
  over multiple nodes with MPI. See wsclean_image.cwl for a detailed description
  of the inputs and outputs.

requirements:
  InlineJavascriptRequirement: {}
  cwltool:MPIRequirement:
    processes: $(inputs.nnodes)

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
  - id: cellsize_deg
    type: float
    inputBinding:
      prefix: -scale
  - id: dir_local
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
  - id: wsclean_mem
    type: float
    inputBinding:
      prefix: -mem
  - id: auto_mask
    type: float
    inputBinding:
      prefix: -auto-mask
  - id: idg_mode
    type: string
    inputBinding:
      prefix: -idg-mode
  - id: num_threads
    type: string
    inputBinding:
      prefix: -j
  - id: num_deconvolution_threads
    type: string
    inputBinding:
      prefix: -deconvolution-threads
  - id: nnodes
    label: Number of nodes
    doc: |
      The number of nodes to use for the MPI job.
    type: int

outputs:
  - id: image_nonpb_name
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-image.fits
  - id: image_pb_name
    type: File
    outputBinding:
      glob: $(inputs.name)-MFS-image-pb.fits
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
    dockerPull: 'loose/rapthor'
