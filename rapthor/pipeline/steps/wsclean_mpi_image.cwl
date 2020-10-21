cwlVersion: v1.0
class: CommandLineTool
baseCommand: [run_wsclean_mpi.sh]
label: "Images a dataset using WSClean+IDG, distributed over multiple nodes with MPI"

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
  - id: auto_mask
    type: float
    inputBinding:
      prefix: -a
  - id: idg_mode
    type: string
    inputBinding:
      prefix: -g
  - id: ntasks
    type: int
    inputBinding:
      prefix: -y
  - id: nnodes
    type: int
    inputBinding:
      prefix: -q
  - id: numthreads
    type: string
    inputBinding:
      prefix: -j

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
