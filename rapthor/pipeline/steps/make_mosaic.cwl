cwlVersion: v1.0
class: CommandLineTool
baseCommand: [make_mosaic.py]
label: "Make a mosaic from the input images"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image_list
    type: string[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: template_image
    type: string
    inputBinding:
      position: 2
  - id: output_image
    type: string
    inputBinding:
      position: 3
  - id: skip
    type: string
    inputBinding:
      prefix: --skip=
      separate: false

outputs: []
