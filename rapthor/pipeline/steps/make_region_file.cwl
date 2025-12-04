cwlVersion: v1.2
class: CommandLineTool
baseCommand: [make_region_file.py]
label: Make ds9 region file
doc: |
  This tool makes a ds9 region file needed for imaging
  using faceting in WSClean+IDG.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: skymodel
    label: Filename of sky model
    doc: |
      The filename of the input ds9 sky model file that defines the calibration patches.
    type: File
    inputBinding:
      position: 1
  - id: ra_mid
    label: RA of the midpoint
    doc: |
      The RA in degrees of the middle of the region to be imaged.
    type: float
    inputBinding:
      position: 2
  - id: dec_mid
    label: Dec of the midpoint
    doc: |
      The Dec in degrees of the middle of the region to be imaged.
    type: float
    inputBinding:
      position: 3
  - id: width_ra
    label: Width along RA
    doc: |
      The width along RA in degrees (corrected to Dec = 0) of the region to be imaged.
    type: float
    inputBinding:
      position: 4
  - id: width_dec
    label:  Width along Dec
    doc: |
      The width along Dec in degrees of the region to be imaged.
    type: float
    inputBinding:
      position: 5
  - id: outfile
    label: Filename of output region file
    doc: |
      The filename of the output ds9 region file.
    type: string
    inputBinding:
      position: 6
  - id: enclose_names
    label: Enclose name in braces
    doc: |
      Enclose the patch names in curly braces.
    type: string
    inputBinding:
      prefix: --enclose_names=
      separate: False

outputs:
  - id: region_file
    label: Output filename
    doc: |
      The filename of the output region file. The value is taken from the input
      parameter "outfile".
    type:
      - File
    outputBinding:
      glob: $(inputs.outfile)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1
