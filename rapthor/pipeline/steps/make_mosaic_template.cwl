cwlVersion: v1.2
class: CommandLineTool
baseCommand: [make_mosaic_template.py]
label: Make template image for mosaicking
doc: |
  This tool makes a FITS image that can be used as a template for
  regridding and mosaicking. The template image is an intermediate product
  and can be deleted after the final mosaic is made.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image_list
    label: Filenames of images
    doc: |
      The filenames of the FITS images to be mosaicked.
    type: File[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: vertices_file_list
    label: Filenames of vertices files
    doc: |
      The filenames of the sector vertices files.
    type: File[]
    inputBinding:
      position: 2
      itemSeparator: ","
  - id: output_image
    label: Filename of output image
    doc: |
      The filename of the output template FITS image.
    type: string
    inputBinding:
      position: 3
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
  - id: template_image
    label: Output image
    doc: |
      The filename of the output FITS image. The value is taken from the input
      parameter "output_image".
    type: File
    outputBinding:
      glob: $(inputs.output_image)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
