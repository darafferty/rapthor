cwlVersion: v1.2
class: CommandLineTool
baseCommand: [makesourcedb]
label: Makes a sourcedb
doc: |
  This tool makes a sourcedb from an input sky model.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - format=<
  - append=False  # overwrite existing sourcedb
  - outtype=blob  # suitable for parallel reading by multiple DPPPs

inputs:
  - id: in
    label: Input sky model
    doc: |
      The filename of the input sky model.
    type: File
    inputBinding:
      prefix: in=
      separate: false

  - id: out
    label: Output sourcedb
    doc: |
      The filename of the output sourcedb model.
    type: string
    inputBinding:
      prefix: out=
      separate: false

outputs:
  - id: sourcedb
    label: Output sourcedb
    doc: |
      The filename of the output sourcedb model.
    type: File
    outputBinding:
      glob: $(inputs.out)

hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
