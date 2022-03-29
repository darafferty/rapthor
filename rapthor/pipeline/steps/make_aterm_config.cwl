cwlVersion: v1.2
class: CommandLineTool
baseCommand: [make_aterm_config.py]
label: Make a-term config file
doc: |
  This tool makes the a-term configuration file needed for imaging
  with a-term screens in WSClean+IDG.

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
    type: string
    inputBinding:
      prefix: --gain_filenames=
      separate: false

outputs:
  - id: aterms_config
    label: Output filename
    doc: |
      The filename of the output config file. The value is taken from the input
      parameter "outfile".
    type: string
    outputBinding:
      outputEval: $(inputs.outfile)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
