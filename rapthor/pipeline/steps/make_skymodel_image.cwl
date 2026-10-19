cwlVersion: v1.2
class: CommandLineTool
baseCommand: [restore_skymodel.py]
label: Restore a skymodel into an image
doc: |
  This tool restores a skymodel text file into an image.
  To compute the image dimensions it uses a reference image.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: source_catalog
    label: Filename of source catalog
    doc: |
      The filename of the source catalog text file.
    type: File
    inputBinding:
      position: 1
  - id: reference_image
    label: Reference image
    doc: |
      The reference FITS image file used to get the image dimensions and header.
    type: File
    inputBinding:
      position: 2
  - id: output_image_name
    label: Filename of output restored image
    doc: |
      The filename of the output restored FITS image.
    type: string
    inputBinding:
      position: 3

outputs:
  - id: output_image
    label: Output restored image
    doc: |
      The filename of the output restored FITS image. The value is taken from the input
      parameter "output_image_name".
    type: File
    outputBinding:
      glob: $(inputs.output_image_name)
hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
