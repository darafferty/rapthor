cwlVersion: v1.0
class: CommandLineTool
baseCommand: [make_mosaic.py]
label: Make a mosaic
doc: |
  This tool makes a FITS mosiac from the input FITS images. The input
  images must all have been regridded to the same grid.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image_list
    label: Filenames of images
    doc: |
      The filenames of the regridded FITS images to be mosaicked.
    type: string[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: template_image
    label: Filename of template image
    doc: |
      The filename of the template mosaic FITS image.
    type: string
    inputBinding:
      position: 2
  - id: output_image
    label: Filename of output image
    doc: |
      The filename of the output mosaic FITS image.
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
      separate: false

outputs: []
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
