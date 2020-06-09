cwlVersion: v1.0
class: CommandLineTool
baseCommand: [make_mosaic_template.py]
label: "Make FITS template image for mosaicking"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image_list
    type: string[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: vertices_file_list
    type: string[]
    inputBinding:
      position: 2
      itemSeparator: ","
  - id: output_image
    type: string
    inputBinding:
      position: 3
  - id: skip
    type: string
    inputBinding:
      prefix: --skip=
      separate: false

outputs:
  - id: template_image
    type: string
    outputBinding:
      outputEval: $(inputs.output_image)
