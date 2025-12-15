cwlVersion: v1.2
class: CommandLineTool
baseCommand: [adjust_h5parm_sources.py]
label: Adjust h5parm source coordinates
doc: |
  This tool adjusts the h5parm source coordinates to match those in the sky model.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.h5parm)
        writable: true

inputs:
  - id: skymodel
    label: Filename of sky model
    doc: |
      The filename of the input sky model file that defines the calibration patches.
    type: File
    inputBinding:
      position: 0
  - id: h5parm
    label: Filename of h5parm file
    doc: |
      The filename of the input h5parm file that contains the calibration solutions.
    type: File?
    inputBinding:
      position: 1

outputs:
  - id: adjustedh5parm
    label: Output filename
    doc: |
      The filename of the adjusted h5parm file. The value is taken from the input
      parameter "h5parm".
    type:
      - File
    outputBinding:
      glob: $(inputs.h5parm.basename)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
