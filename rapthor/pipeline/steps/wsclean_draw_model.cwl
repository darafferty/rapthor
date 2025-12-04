cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean]
label: Draw model images
doc: |
  This tool generates model images (one per spectral term) from a sky model using image-
  based predict in WSClean.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads to use.
    type: int
    inputBinding:
      prefix: -j

  - id: skymodel
    label: Filename of input sky model
    doc: |
      The filename of the input sky model in makesourcedb format used to
      generate the output model images.
    type: File
    inputBinding:
      prefix: -draw-model

  - id: numterms
    label: Number of spectral terms
    doc: |
      The number of spectral terms to generate (one output image per term).
    type: int
    inputBinding:
      prefix: -draw-spectral-terms

  - id: name
    label: Root name of output images
    doc: |
      The root name of the output images. The images will be named "[name]-term-0.fits",
      "[name]-term-1.fits", etc.
    type: string
    inputBinding:
      prefix: -name

  - id: ra_dec
    label: RA and Dec of center
    doc: |
      The RA and Dec in hmsdms of the center of the output images.
    type: string[]
    inputBinding:
      prefix: -draw-centre

  - id: frequency_bandwidth
    label: Frequency and bandwidth of image
    doc: |
      The central frequency and bandwidth in Hz of the output images.
    type: float[]
    inputBinding:
      prefix: -draw-frequencies

  - id: imsize
    label: Image size
    doc: |
      The size of the image in pixels.
    type: int[]
    inputBinding:
      prefix: -size

  - id: cellsize_deg
    label: Pixel size
    doc: |
      The size of one pixel of the image in deg.
    type: float
    inputBinding:
      prefix: -scale

outputs:
  - id: model_images
    label: Output model images
    doc: |
      The filenames of the output model images (one per spectral term). The value is
      constructed from the input parameter "name".
    type: File[]
    outputBinding:
      glob: $(inputs.name)-term-*.fits

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1
