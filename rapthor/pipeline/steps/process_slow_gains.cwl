cwlVersion: v1.0
class: CommandLineTool
baseCommand: [process_slow_gains.py]
label: "Process slow gain solutions"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - '--solsetname=sol000'
  - '--ampsoltabname=amplitude000'
  - '--phsoltabname=phase000'
  - '--normalize=True'
  - '--find_bandpass=False'
  - '--smooth_phases=False'

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
