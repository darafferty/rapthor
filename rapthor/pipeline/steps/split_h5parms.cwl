cwlVersion: v1.0
class: CommandLineTool
baseCommand: [split_h5parms.py]
label: Splits an h5parm
doc: |
  This tool splits the input h5parm in time to produce multiple
  output h5parms.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: inh5parm
    label: Input solution table
    doc: |
      The filename of the input h5parm file.
    type: string
    inputBinding:
      position: 0
  - id: outh5parms
    label: Output solution tables
    doc: |
      The filenames of the output h5parm files.
    type: string[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: soltabname
    label: Solution table name
    doc: |
      The name of the solution table to split.
    type: string
    inputBinding:
      prefix: --soltabname=
      separate: false

outputs:
  - id: splith5parms
    label: Output solution tables
    doc: |
      The filenames of the output h5parm files. The value is taken from the input
      parameter "outh5parms"
    type: string[]
    outputBinding:
      outputEval: $(inputs.outh5parms)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
