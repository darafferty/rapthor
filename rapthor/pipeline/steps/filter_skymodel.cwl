cwlVersion: v1.0
class: CommandLineTool
baseCommand: [filter_skymodel.py]
label: Filter a sky model
doc: |
  This tool uses PyBDSF to filter artifacts from the sky model and make
  a clean mask for the next iteration.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image
    label: Input image
    doc: |
      The filename of the input FITS image.
    type: string
    inputBinding:
      position: 1
  - id: input_skymodel_pb
    label: PB-corrected model
    doc: |
      The filename of the input primary-beam-corrected sky model.
    type: string
    inputBinding:
      position: 2
  - id: input_bright_skymodel_pb
    label: Bright-source PB-corrected model
    doc: |
      The filename of the input bright-source primary-beam-corrected sky model.
    type: string
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
    type: string
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
      The filename of the MS file to use for beam calculations.
    type: string[]
    inputBinding:
      prefix: --beamMS=
      itemSeparator: ","
      separate: false
  - id: peel_bright
    label: Peeling flag
    doc: |
      The flag that sets whether peeling of bright sources was done in the predict
      pipeline.
    type: string
    inputBinding:
      prefix: --peel_bright=
      separate: false

outputs: []
