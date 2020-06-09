cwlVersion: v1.0
class: CommandLineTool
baseCommand: [combine_h5parms.py]
label: "Combines multiple h5parms into one"

requirements:
  InlineJavascriptRequirement: {}


arguments:
  - '--rename_from=phase000'
  - '--rename_to=phase001'

inputs:
  - id: inh5parm1
    type: string
    inputBinding:
      position: 0
  - id: inh5parm2
    type: string
    inputBinding:
      position: 1
  - id: outh5parm
    type: string
    inputBinding:
      position: 2

outputs:
  - id: combinedh5parm
    type: string
    outputBinding:
      outputEval: $(inputs.outh5parm)
