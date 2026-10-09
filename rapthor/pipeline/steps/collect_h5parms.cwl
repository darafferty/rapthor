cwlVersion: v1.2
class: CommandLineTool
baseCommand: [H5parm_collector.py, -c]
label: Collects multiple h5parms
doc: |
  This tool collects the solution tables from multiple h5parm files
  into a single output h5parm using LoSoTo's H5parm_collector.py script.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: inh5parms
    label: Input solution tables
    doc: |
      A list of input h5parm files.
    type: File[]
    inputBinding:
      position: 0
      itemSeparator: ","
  - id: outputh5parm
    label: Output solution table
    doc: |
      The filename of the output h5parm file.
    type: string
    inputBinding:
      prefix: --outh5parm=
      separate: false
  - id: insolset
    label: Input solution set
    doc: |
      The name of the input solution set.
    type: string?
    inputBinding:
      prefix: --insolset=
      separate: false

outputs:
  - id: outh5parm
    label: Output solution table
    doc: |
      The filename of the output h5parm file. The value is taken from the input
      parameter "outputh5parm".
    type: File
    outputBinding:
      glob: $(inputs.outputh5parm)
hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
