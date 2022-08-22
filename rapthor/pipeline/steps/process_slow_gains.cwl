cwlVersion: v1.2
class: CommandLineTool
baseCommand: [process_slow_gains.py]
label: Process slow gain solutions
doc: |
  This tool processes the slow-gain solutions, smoothing and normalizing
  them.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.slowh5parm)
        writable: true

arguments:
  - '--normalize=True'

inputs:
  - id: slowh5parm
    label: Input solution table
    doc: |
      The filename of the input h5parm file.
    type: File
    inputBinding:
      position: 1
  - id: smooth
    label: Smooth flag
    doc: |
      The flag that determines whether smoothing of the solutions will be done.
    type: string
    inputBinding:
      prefix: --smooth=
      separate: false

outputs:
  - id: outh5parm
    label: Output solution table
    doc: |
      The filename of the output h5parm file. The value is taken from the input
      parameter "slowh5parm".
    type: File
    outputBinding:
      glob: $(inputs.slowh5parm.basename)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'
