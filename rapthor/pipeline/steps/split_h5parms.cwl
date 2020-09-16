cwlVersion: v1.0
class: CommandLineTool
baseCommand: [split_h5parms.py]
label: "Splits an h5parm into multiple h5parms"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: inh5parm
    type: string
    inputBinding:
      position: 0
  - id: outh5parms
    type: string[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: soltabname
    type: string
    inputBinding:
      prefix: --soltabname=
      separate: false

outputs:
  - id: splith5parms
    type: string[]
    outputBinding:
      outputEval: $(inputs.outh5parms)
