cwlVersion: v1.2
class: CommandLineTool
baseCommand: [correct_astrometry.py]
label: Correct image astrometry
doc: |
  This tool applies astrometry corrections to a FITS image made using faceting.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - --overwrite

inputs:
  - id: input_image
    label: Input image
    doc: |
      The filename of the input uncorrected FITS image.
    type: File
    inputBinding:
      position: 0
  - id: region_file
    label: Input region file
    doc: |
      The filename of the input ds9 region file that defines the facets.
    type: File
    inputBinding:
      position: 1
  - id: corrections_file
    label: Input corrections file
    doc: |
      The filename of the input json file that defines the astrometry corrections per facet.
      If not available, no correction is done.
    type: File?
    inputBinding:
      prefix: --corrections_file=
      separate: false

outputs:
  - id: corrected_image
    label: Corrected image
    doc: |
      The filename of the output corrected FITS image.
    type: File
    outputBinding:
      glob: ['*.fits', '*.fits.fz']

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
