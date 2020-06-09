cwlVersion: v1.0
class: CommandLineTool
baseCommand: [regrid_image.py]
label: "Regrid FITS images to match a template image"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image
    type: string
    inputBinding:
      position: 1
  - id: template_image
    type: string
    inputBinding:
      position: 2
  - id: vertices_file
    type: string
    inputBinding:
      position: 3
  - id: output_image
    type: string
    inputBinding:
      position: 4
  - id: skip
    type: string
    inputBinding:
      prefix: --skip=
      separate: false

outputs:
  - id: regridded_image
    type: string
    outputBinding:
      outputEval: $(inputs.output_image)
