cwlVersion: v1.2
class: CommandLineTool
baseCommand: [regrid_image.py]
label: Regrid an image
doc: |
  This tool regrids a FITS image to the grid of the given template FITS
  image.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image
    label: Input image
    doc: |
      The filename of the input FITS image.
    type: File
    inputBinding:
      position: 1
  - id: template_image
    label: Input template
    doc: |
      The filename of the input template FITS image.
    type: File
    inputBinding:
      position: 2
  - id: vertices_file
    label: Filename of vertices file
    doc: |
      The filename of the sector vertices file.
    type: File
    inputBinding:
      position: 3
  - id: output_image
    label: Filename of output image
    doc: |
      The filename of the regridded FITS image.
    type: string
    inputBinding:
      position: 4
  - id: skip
    label: Flag to skip processing
    doc: |
      The flag that sets whether processing is skipped or not.
    type: boolean
    inputBinding:
      prefix: --skip=
      valueFrom: "$(self ? 'True': 'False')"
      separate: false

outputs:
  - id: regridded_image
    label: Output image
    doc: |
      The filename of the regridded FITS image. The value is taken from the input
      parameter "output_image".
    type: File
    outputBinding:
      glob: $(inputs.output_image)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
