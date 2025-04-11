cwlVersion: v1.2
class: CommandLineTool
baseCommand: [process_gains.py]
label: Process gain solutions
doc: |
  This tool processes gain solutions, smoothing, flagging, and normalizing
  them. Note: normalization is always done, but smoothing and flagging are optional

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.h5parm)
        writable: true

arguments:
  - '--normalize=True'

inputs:
  - id: h5parm
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
  - id: scale_station_delta
    label: Scale station delta flag
    doc: |
      Flag that enables scaling (with distance from the phase center) of the
      maximum allowed difference in the median of the amplitudes from unity (per
      station)
    type: string
    inputBinding:
      prefix: --scale_delta_with_dist=
      separate: false
  - id: phase_center_ra
    label: Phase center RA
    doc: |
      The RA in degrees of the phase center.
    type:
      - float
      - string
    inputBinding:
      prefix: --phase_center_ra=
      separate: false
  - id: phase_center_dec
    label: Phase center Dec
    doc: |
      The Dec in degrees of the phase center.
    type:
      - float
      - string
    inputBinding:
      prefix: --phase_center_dec=
      separate: false

outputs:
  - id: outh5parm
    label: Output solution table
    doc: |
      The filename of the output h5parm file. The value is taken from the input
      parameter "h5parm".
    type: File
    outputBinding:
      glob: $(inputs.h5parm.basename)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
