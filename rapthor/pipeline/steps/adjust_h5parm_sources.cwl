cwlVersion: v1.2
class: CommandLineTool
baseCommand: [adjust_h5parm_sources.py]
label: Adjust h5parm source coordinates
doc: |
  This tool adjusts the h5parm source coordinates to match those in the sky model.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: skymodel
    label: Filename of sky model
    doc: |
      The filename of the input sky model file that defines the calibration patches.
    type: File
    inputBinding:
      position: 1
  - id: h5parm
    label: Filename of the h5parm file
    doc: |
      The filename of the input h5parm file that contains the calibration solutions.
    type: string
    inputBinding:
      position: 6

outputs:
  - id: adjustedh5parm
    label: Output filename
    doc: |
      The filename of the adjusted h5parm file. The value is taken from the input
      parameter "h5parm".
    type:
      - File
      - string
    outputBinding:
      glob: $(inputs.h5parm)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'
