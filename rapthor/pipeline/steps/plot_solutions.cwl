cwlVersion: v1.2
class: CommandLineTool
baseCommand: [plotrapthor]
label: Plots calibration solutions
doc: |
  This tool plots the calibration solutions.

inputs:
  - id: h5parm
    label: Input solution table
    doc: |
      The filename of the input h5parm file.
    type: File
    inputBinding:
      position: 0
  - id: soltype
    label: Solution type
    doc: |
      The type of solution to plot (one of phase, amplitude, phasescreen, or ampscreen).
    type: string
    inputBinding:
      position: 1
  - id: root
    label: Output filename root
    doc: |
      The root to use for the output files.
    type: string?
    inputBinding:
      prefix: --root=
      separate: False

outputs:
  - id: plots
    label: Output plots
    doc: |
      The filenames of the output plots.
    type: File[]
    outputBinding:
      glob: '*.png'

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
