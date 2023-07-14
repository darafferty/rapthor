#!/usr/bin/env cwl-runner
cwlVersion: v1.1  # Note: MPIRequirement does not currently work with v1.2
class: CommandLineTool
$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
baseCommand: [wsclean-mp]
label: Make an image
doc: |
  This tool makes an image using WSClean with a-term corrections, distributed
  over multiple nodes with MPI. See wsclean_image_screens.cwl for a detailed
  description of the inputs and outputs.

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
requirements:
  - class: InitialWorkDirRequirement
    listing:
      - entryname: aterm_plus_beam.cfg
        # Note: WSClean requires that the aterm image filenames be input as part of an
        # aterm config file (and not directly on the command line). Therefore, a config
        # file is made here that contains the filenames defined in the aterm_images
        # input parameter. Also, the required beam parameters are set here
        entry: |
          aterms = [diagonal, beam]
          diagonal.images = [$(inputs.aterm_images.map( (e,i) => (e.path) ).join(' '))]
          beam.differential = true
          beam.update_interval = 120
          beam.usechannelfreq = true
        writable: false
  - class: InlineJavascriptRequirement
  - class: cwltool:MPIRequirement
    processes: $(inputs.nnodes)

arguments:
  - -no-update-model-required
  - -save-source-list
  - -local-rms
  - -join-channels
  - -use-idg
  - -log-time
  - valueFrom: '$(runtime.tmpdir)'
    prefix: -temp-dir
  - valueFrom: 'I'
    prefix: -pol
  - valueFrom: '0.85'
    prefix: -mgain
  - valueFrom: '0.8'
    prefix: -multiscale-scale-bias
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
    # Note: this file is generated on the fly in the requirements section above
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
    type: Directory
    inputBinding:
      position: 3
  - id: name
    type: string
    inputBinding:
      prefix: -name
  - id: mask
    type: File
    inputBinding:
      prefix: -fits-mask
  - id: aterm_images
    label: Filenames of aterm files
    doc: |
      The filenames of the a-term image files. These filenames are not used directly in the
      WSClean call (they are read by WSClean from the aterm config file, defined in the
      requirements section above), hence the value is set to "null" (which results in
      nothing being added to the command for this input).
    type: File[]
    inputBinding:
      valueFrom: null
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
  - id: fit_spectral_pol
    type: int
    inputBinding:
      prefix: -fit-spectral-pol
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
  - id: dd_psf_grid
    type: int[]
    inputBinding:
      prefix: -dd-psf-grid
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
