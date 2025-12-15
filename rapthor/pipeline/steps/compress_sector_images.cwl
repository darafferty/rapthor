cwlVersion: v1.2
class: CommandLineTool
baseCommand: [fpack]
label: Compress fits format image (Multiple Images)
doc: |
  This tool compresses multiple FITS images in the FITS data format. 
  Works specifically for the naming convention of images from a wsclean
  imaging run as given in image_sector_pipeline.cwl. For single images 
  compression use compress_mosaic_image.cwl

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - $(inputs.images)

inputs:
  - id: images
    label: Input image FITS files
    doc: |
      The FITS format filenames of the input images.
    type: File[]
    inputBinding:
      position: 1

  - id: name
    label: root name of input image
    doc: |
      The root filename of the input image as given to wsclean (length = 1).
    type: string?

outputs:
  - id: image_I_nonpb_name
    type: File
    outputBinding:
      glob: [$(inputs.name)-MFS-image.fits.fz, $(inputs.name)-MFS-I-image.fits.fz]
  - id: image_I_pb_name
    type: File
    outputBinding:
      glob: [$(inputs.name)-MFS-image-pb.fits.fz, $(inputs.name)-MFS-I-image-pb.fits.fz]
  - id: images_extra
    type: File[]
    outputBinding:
      glob: ['$(inputs.name)-MFS-[QUV]-image.fits.fz', '$(inputs.name)-MFS-[QUV]-image-pb.fits.fz', '$(inputs.name)-MFS-*residual.fits.fz', '$(inputs.name)-MFS-*model-pb.fits.fz', '$(inputs.name)-MFS-*dirty.fits.fz']

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
