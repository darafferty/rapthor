cwlVersion: v1.2
class: CommandLineTool
baseCommand: [blank_image.py]
label: Make an image mask
doc: |
  This tool either modifies an existing FITS image or makes a blank FITS image
  for use as a clean mask during imaging.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: maskfile
    label: Filename of output mask
    doc: |
      The filename of the output FITS image mask.
    type: string
    inputBinding:
      position: 1
  - id: imagefile
    label: Filename of input image
    doc: |
      The filenames of the input FITS image. If the file does not exist, a blank
      FITS image will be made.
    type: File?
    inputBinding:
      position: 2
  - id: wsclean_imsize
    label: Image size
    doc: |
      The size of the image in pixels.
    type: int[]
    inputBinding:
      prefix: --imsize=
      valueFrom: $(self[0]),$(self[1])
      separate: false
  - id: vertices_file
    label: Filename of vertices file
    doc: |
      The filename of the file containing sector vertices.
    type: File
    inputBinding:
      prefix: --vertices_file=
      separate: false
  - id: ra
    label: RA of center of image
    doc: |
      The RA of the center of the image in deg.
    type: float
    inputBinding:
      prefix: --reference_ra_deg=
      separate: false
  - id: dec
    label: Dec of center of image
    doc: |
      The Dec of the center of the image in deg.
    type: float
    inputBinding:
      prefix: --reference_dec_deg=
      separate: false
  - id: cellsize_deg
    label: Pixel size
    doc: |
      The size of one pixel of the image in deg.
    type: float
    inputBinding:
      prefix: --cellsize_deg=
      separate: false
  - id: region_file
    label: Filename of region file
    doc: |
      The filename of a user-supplied region file.
    type: File?
    inputBinding:
      prefix: --region_file=
      separate: false

outputs:
  - id: maskimg
    label: Output mask image
    doc: |
      The filename of the output FITS mask. The value is taken from the input
      parameter "maskfile".
    type: File
    outputBinding:
      glob: $(inputs.maskfile)
hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
