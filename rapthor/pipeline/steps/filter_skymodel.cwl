cwlVersion: v1.0
class: CommandLineTool
baseCommand: [filter_skymodel.py]
label: "Filter a sky model"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image
    type: string
    inputBinding:
      position: 1
  - id: input_skymodel_pb
    type: string
    inputBinding:
      position: 2
  - id: input_bright_skymodel_pb
    type: string
    inputBinding:
      position: 3
  - id: output_root
    type: string
    inputBinding:
      position: 4
  - id: threshisl
    type: float
    inputBinding:
      prefix: --threshisl=
      separate: false
  - id: threshpix
    type: float
    inputBinding:
      prefix: --threshpix=
      separate: false
  - id: beamMS
    type: string[]
    inputBinding:
      prefix: --beamMS=
      itemSeparator: ","
      separate: false
  - id: peel_bright
    type: string
    inputBinding:
      prefix: --peel_bright=
      separate: false

outputs: []
