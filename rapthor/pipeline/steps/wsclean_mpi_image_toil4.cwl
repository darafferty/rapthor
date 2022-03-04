cwlVersion: v1.0
class: CommandLineTool
baseCommand: [run_wsclean_mpi.sh]
label: Make an image
doc: |
  This tool makes an image using WSClean with a-term corrections,
  distributed over multiple nodes with MPI. See wsclean_image.cwl
  for a detailed description of the inputs and outputs.

requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}

inputs:
  - id: msin
    type: string[]
    inputBinding:
      prefix: -m
      itemSeparator: " "
  - id: name
    type: string
    inputBinding:
      prefix: -n
  - id: mask
    type: string
    inputBinding:
      prefix: -k
  - id: config
    type: string
    inputBinding:
      prefix: -c
  - id: wsclean_imsize
    type: int[]
    inputBinding:
      prefix: -z
      itemSeparator: " "
  - id: wsclean_niter
    type: int
    inputBinding:
      prefix: -i
  - id: wsclean_nmiter
    type: int
    inputBinding:
      prefix: -j
  - id: robust
    type: float
    inputBinding:
      prefix: -r
  - id: min_uv_lambda
    type: float
    inputBinding:
      prefix: -u
  - id: max_uv_lambda
    type: float
    inputBinding:
      prefix: -v
  - id: cellsize_deg
    type: float
    inputBinding:
      prefix: -x
  - id: dir_local
    type: string
    inputBinding:
      prefix: -l
  - id: channels_out
    type: int
    inputBinding:
      prefix: -o
  - id: deconvolution_channels
    type: int
    inputBinding:
      prefix: -d
  - id: taper_arcsec
    type: float
    inputBinding:
      prefix: -t
  - id: wsclean_mem
    type: float
    inputBinding:
      prefix: -p
  - id: auto_mask
    type: float
    inputBinding:
      prefix: -a
  - id: idg_mode
    type: string
    inputBinding:
      prefix: -g
  - id: ntasks
    label: Number of tasks
    doc: |
      The number of tasks per node for MPI jobs.
    type: int
    inputBinding:
      prefix: -y
  - id: nnodes
    label: Number of nodes
    doc: |
      The number of nodes for MPI jobs.
    type: int
    inputBinding:
      prefix: -q
  - id: num_threads
    type: string
    inputBinding:
      prefix: -b
  - id: num_deconvolution_threads
    type: string
    inputBinding:
      prefix: -h

outputs:
  - id: image_nonpb_name
    type: string
    outputBinding:
      outputEval: $(inputs.name)-MFS-image.fits
  - id: image_pb_name
    type: string
    outputBinding:
      outputEval: $(inputs.name)-MFS-image-pb.fits
  - id: skymodel_nonpb
    type: string
    outputBinding:
      outputEval: $(inputs.name)-sources.txt
  - id: skymodel_pb
    type: string
    outputBinding:
      outputEval: $(inputs.name)-sources-pb.txt

hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
