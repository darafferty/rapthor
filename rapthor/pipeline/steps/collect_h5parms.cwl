cwlVersion: v1.0
class: CommandLineTool
baseCommand: [H5parm_collector.py, -c]
label: "Collects multiple h5parms into one"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: inh5parms
    type: string[]
    inputBinding:
      position: 0
      itemSeparator: ","
  - id: outputh5parm
    type: string
    inputBinding:
      prefix: --outh5parm=
      separate: false

outputs:
  - id: outh5parm
    type: string
    outputBinding:
      outputEval: $(inputs.outputh5parm)
