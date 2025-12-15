cwlVersion: v1.2
class: CommandLineTool
baseCommand: [concat_ms.py]
label: Concatenate multiple MS files
doc: |
  This tool concatenates multiple MS files in time or frequency.

requirements:
  - class: InlineJavascriptRequirement
  - class: ShellCommandRequirement

inputs:
  - id: mslist
    label: List of input Measurement Sets
    doc: |
      The filenames of the input MS files.
    type: Directory[]
    inputBinding:
      position: 1
      itemSeparator: " "
      shellQuote: false
  - id: msout
    label: Output Measurement Set
    doc: |
      The filename of the output concatenated MS file.
    type: string
    inputBinding:
      position: 2
      prefix: --msout=
      separate: false
  - id: concat_property
    label: Property for concatenation
    doc: |
      The property over which concatenation is to be done ('time' or
      'frequency').
    type: string
    inputBinding:
      position: 3
      prefix: --concat_property=
      separate: false
  - id: data_colname
    label: Input MS data column
    doc: |
      The data column to be read from the MS files for concatenation.
    type: string
    inputBinding:
      position: 4
      prefix: --data_colname=
      separate: false

outputs:
  - id: msconcat
    label: Concatenated Measurement Set
    doc: |
      The filename of the output concatenated MS file. The value is taken
      from the input parameter "msout".
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
