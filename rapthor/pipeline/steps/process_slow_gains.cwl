cwlVersion: v1.0
class: CommandLineTool
baseCommand: [process_slow_gains.py]
label: "Process slow gain solutions"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - '--normalize=True'
  - '--smooth=True'

inputs:
  - id: slowh5parm
    type: string
    inputBinding:
      position: 1

outputs:
  - id: outh5parm
    type: string
    outputBinding:
      outputEval: $(inputs.slowh5parm)
