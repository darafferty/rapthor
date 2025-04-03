cwlVersion: v1.2
class: CommandLineTool
baseCommand: [combine_h5parms.py]
label: Combines multiple h5parms
doc: |
  This tool combines the solution tables from multiple h5parm files
  into a single output h5parm. Unlike collect_h5parms.cwl, which
  simply collects solutions (none of which overlap in type, time, or
  frequency), this tool can combine solutions (either by adding or
  multipling them) that overlap. It can also interpolate and reweight
  the solutions if needed.

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: inh5parm1
    label: Input solution table 1
    doc: |
      The filename of the first input h5parm file.
    type: File
    inputBinding:
      position: 0
  - id: inh5parm2
    label: Input solution table 2
    doc: |
      The filename of the second input h5parm file.
    type: File
    inputBinding:
      position: 1
  - id: outh5parm
    label: Output solution table
    doc: |
      The filename of the output h5parm file.
    type: string
    inputBinding:
      position: 2
  - id: mode
    label: Combine mode
    doc: |
      The mode to use when combining: 'p1a2' - phases from 1 and amplitudes from
      2; 'p1a1a2' - phases and amplitudes from 1 and amplitudes from 2
      (amplitudes 1 and 2 are multiplied to create combined amplitudes); 'p1p2a2'
      - phases from 1 and phases and amplitudes from 2 (phases 2 are averaged
      over XX and YY, then interpolated to time grid of 1 and summed); 'separate'
      - solutions from 1 and 2 are kept completely separate, no combination by
      addition or multiplication is done.
    type: string
    inputBinding:
      position: 3
  - id: reweight
    label: Reweight flag
    doc: |
      Flag that determines whether reweighting of the solutions will be done.
    type: string
    inputBinding:
      prefix: --reweight=
      separate: false
  - id: calibrator_names
    label: Calibrator names
    doc: |
      Comma-separated list of calibrator names, used for reweighting.
    type: string[]
    inputBinding:
      prefix: --cal_names=
      itemSeparator: ','
      separate: false
  - id: calibrator_fluxes
    label: Calibrator flux densities
    doc: |
      Comma-separated list of calibrator total flux densities, used for reweighting.
    type: float[]
    inputBinding:
      prefix: --cal_fluxes=
      itemSeparator: ','
      separate: false

outputs:
  - id: combinedh5parm
    label: Output solution table
    doc: |
      The filename of the output h5parm file. The value is taken from the input
      parameter "outh5parm".
    type: File
    outputBinding:
      glob: $(inputs.outh5parm)
hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
