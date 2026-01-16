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
  - -local-rms
  - -join-channels
  - -apply-facet-beam
  - -log-time
  - valueFrom: 'wgridder'
    prefix: -gridder
  - valueFrom: 'single'
    prefix: -major-iteration-mode
  - valueFrom: '$(runtime.tmpdir)'
    prefix: -temp-dir
  - valueFrom: '2048'
    prefix: -parallel-deconvolution
  - valueFrom: '0.8'
    prefix: -multiscale-scale-bias
  - valueFrom: '1.0'
    prefix: -auto-threshold
  - valueFrom: '1.3'
    prefix: -mgain-boosting
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
    type: Directory
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
  - id: mgain
    type: float
    inputBinding:
      prefix: -mgain
  - id: multiscale
    type: boolean
    inputBinding:
      prefix: -multiscale
  - id: scalar_visibilities
    label: Use scalar visibilities
    doc: |
      Use only scalar (Stokes I) visibilities.
    type: boolean
    inputBinding:
      prefix: -scalar-visibilities
  - id: diagonal_visibilities
    label: Use diagonal visibilities
    doc: |
      Use only diagonal (XX and YY) visibilities.
    type: boolean
    inputBinding:
      prefix: -diagonal-visibilities
  - id: save_source_list
    type: boolean
    inputBinding:
      prefix: -save-source-list
  - id: pol
    type: string
    inputBinding:
      prefix: -pol
  - id: link_polarizations
    type:
      - boolean?
      - string?
    inputBinding:
      prefix: -link-polarizations
  - id: join_polarizations
    type: boolean
    inputBinding:
      prefix: -join-polarizations
  - id: skip_final_iteration
    type: boolean
    inputBinding:
      prefix: -skip-final-iteration
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
  - id: local_rms_strength
    type: float
    inputBinding:
      prefix: -local-rms-strength
  - id: local_rms_window
    type: float
    inputBinding:
      prefix: -local-rms-window
  - id: local_rms_method
    type: string
    inputBinding:
      prefix: -local-rms-method
  - id: wsclean_mem
    type: float
    inputBinding:
      prefix: -abs-mem
  - id: auto_mask
    type: float
    inputBinding:
      prefix: -auto-mask
  - id: auto_mask_nmiter
    type: int
    inputBinding:
      prefix: -auto-mask-nmiter
  - id: idg_mode
    type: string
    inputBinding:
      prefix: -idg-mode
  - id: num_threads
    type: int
    inputBinding:
      prefix: -j
      valueFrom: $(runtime.cores)
  - id: num_deconvolution_threads
    type: int
    inputBinding:
      prefix: -deconvolution-threads
  - id: dd_psf_grid
    type: int[]
    inputBinding:
      prefix: -dd-psf-grid
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
  - id: apply_time_frequency_smearing
    type: boolean
    inputBinding:
      prefix: -apply-time-frequency-smearing

outputs:
  - id: image_I_nonpb_name
    type: File
    outputBinding:
      glob: [$(inputs.name)-MFS-image.fits, $(inputs.name)-MFS-I-image.fits]
  - id: image_I_pb_name
    type: File
    outputBinding:
      glob: [$(inputs.name)-MFS-image-pb.fits, $(inputs.name)-MFS-I-image-pb.fits]
  - id: image_I_pb_channels
    type: File[]
    outputBinding:
      glob: [$(inputs.name)-0???-image-pb.fits, $(inputs.name)-0???-I-image-pb.fits]
  - id: image_Q_pb_channels
    type: File[]?
    outputBinding:
      glob: [$(inputs.name)-0???-Q-image-pb.fits]
  - id: image_U_pb_channels
    type: File[]?
    outputBinding:
      glob: [$(inputs.name)-0???-U-image-pb.fits]
  - id: image_V_pb_channels
    type: File[]?
    outputBinding:
      glob: [$(inputs.name)-0???-V-image-pb.fits]
  - id: images_extra
    type: File[]
    outputBinding:
      glob: ['$(inputs.name)-MFS-[QUV]-image.fits', '$(inputs.name)-MFS-[QUV]-image-pb.fits', '$(inputs.name)-MFS-*residual.fits', '$(inputs.name)-MFS-*model-pb.fits', '$(inputs.name)-MFS-*dirty.fits']
  - id: skymodel_nonpb
    type: File?
    outputBinding:
      glob: $(inputs.name)-sources.txt
  - id: skymodel_pb
    type: File?
    outputBinding:
      glob: $(inputs.name)-sources-pb.txt

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
