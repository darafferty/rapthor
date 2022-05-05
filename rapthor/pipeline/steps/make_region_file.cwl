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
  - id: outfile
    label: Filename of output file
    doc: |
      The filename of the output config file.
    type: string
    inputBinding:
      position: 1
  - id: gain_filenames
    label: Filenames of a-term images
    doc: |
      The filenames of the a-term gain images.
    type: File[]
    inputBinding:
      prefix: --gain_filenames=
      separate: false
      itemSeparator: ","

outputs:
  - id: aterms_config
    label: Output filename
    doc: |
      The filename of the output config file. The value is taken from the input
      parameter "outfile".
    type:
      - File
      - string
    outputBinding:
      glob: $(inputs.outfile)

hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
