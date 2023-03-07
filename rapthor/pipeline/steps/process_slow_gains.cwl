cwlVersion: v1.2
class: CommandLineTool
baseCommand: [process_slow_gains.py]
label: Process slow gain solutions
doc: |
  This tool processes the slow-gain solutions, smoothing, flagging, and normalizing
  them. Note: normalization is always done, but smoothing and flagging are optional

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
  - id: flag
    label: Flagging flag
    doc: |
      The flag that determines whether flagging of the solutions will be done.
    type: string
    inputBinding:
      prefix: --flag=
      separate: false
  - id: max_station_delta
    label: Max station delta
    doc: |
      The maximum allowed difference of the median of the amplitudes from unity (per
      station).
    type: float
    inputBinding:
      prefix: --max_station_delta=
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
