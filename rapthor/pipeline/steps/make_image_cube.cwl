cwlVersion: v1.2
class: CommandLineTool
baseCommand: [make_image_cube.py]
label: Make image cube
doc: |
  This tool makes a FITS image cube from the input images.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: input_image_list
    label: Filenames of images
    doc: |
      The filenames of the FITS channel images.
    type: File[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: output_image
    label: Filename of output image
    doc: |
      The filename of the output FITS image cube.
    type: string
    inputBinding:
      position: 2

outputs:
  - id: image_cube
    label: Output image
    doc: |
      The filename of the output FITS image cube. The value is taken from the input
      parameter "output_image".
    type: File
    outputBinding:
      glob: $(inputs.output_image)
  - id: image_cube_beams
    label: Output image beams
    doc: |
      The filename of the output image cube beams. The value is taken from the input
      parameter "output_image".
    type: File
    outputBinding:
      glob: $(inputs.output_image)_beams.txt
  - id: image_cube_frequencies
    label: Output image frequencies
    doc: |
      The filename of the output FITS image cube. The value is taken from the input
      parameter "output_image".
    type: File
    outputBinding:
      glob: $(inputs.output_image)_frequencies.txt

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
