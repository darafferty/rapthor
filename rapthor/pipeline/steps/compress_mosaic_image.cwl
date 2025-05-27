cwlVersion: v1.2
class: CommandLineTool
baseCommand: [fpack]
label: Compress fits format image (Single Image)
doc: |
  This tool compresses a single FITS image in the FITS data format.

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: mosaic_image
    label: Input image FITS file
    doc: |
      The FITS format filename of the input image.
    type: File
    inputBinding:
      position: 1

outputs:
  - id: compressed_mosaic_image
    type: File
    outputBinding:
      glob: $(inputs.mosaic_image).fz

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
