cwlVersion: v1.2
class: CommandLineTool
baseCommand: [filter_skymodel.py]
label: Filter a sky model
doc: |
  This tool uses PyBDSF to filter artifacts from the sky model and make
  a clean mask for the next iteration.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.input_image)
        writable: true


inputs:
  - id: input_image
    label: Input image
    doc: |
      The filename of the input FITS image.
    type: File
    inputBinding:
      position: 1
  - id: input_skymodel_pb
    label: PB-corrected model
    doc: |
      The filename of the input primary-beam-corrected sky model.
    type: File
    inputBinding:
      position: 2
  - id: input_bright_skymodel_pb
    label: Bright-source PB-corrected model
    doc: |
      The filename of the input bright-source primary-beam-corrected sky model.
    type: File
    inputBinding:
      position: 3
  - id: output_root
    label: Output root name
    doc: |
      The root of the filenames of the output filtered sky models.
    type: string
    inputBinding:
      position: 4
  - id: vertices_file
    label: Filename of vertices file
    doc: |
      The filename of the file containing sector vertices.
    type: File
    inputBinding:
      position: 5
  - id: threshisl
    label: Island threshold
    doc: |
      The PyBDSF island threshold.
    type: float
    inputBinding:
      prefix: --threshisl=
      separate: false
  - id: threshpix
    label: Pixel threshold
    doc: |
      The PyBDSF pixel threshold.
    type: float
    inputBinding:
      prefix: --threshpix=
      separate: false
  - id: beamMS
    label: Filename of MS file for beam
    doc: |
      The filenames of the MS files to use for beam calculations.
    type: Directory[]
    inputBinding:
      prefix: --beamMS=
      itemSeparator: ","
      separate: false
  - id: peel_bright
    label: Peeling flag
    doc: |
      The flag that sets whether peeling of bright sources was done in the predict
      pipeline.
    type: boolean
    inputBinding:
      prefix: --peel_bright=
      valueFrom: "$(self ? 'True': 'False')"
      separate: false

outputs:
  - id: skymodels
    type: File[]
    outputBinding:
      glob: ['$(inputs.output_root)-MFS-*.fits', '$(inputs.output_root)-MFS-*.mask', '$(inputs.output_root).*_sky.txt']

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'
