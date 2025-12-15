cwlVersion: v1.2
class: CommandLineTool
baseCommand: [add_sector_models.py]
label: Add sector model data
doc: |
  This tool adds sector model uv data to the input MS files.

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: msobs
    label: Filename of data MS
    doc: |
      The filename of the input MS file for which addition will be done.
    type: Directory
    inputBinding:
      position: 0
  - id: msmod
    label: Filenames of model MS
    doc: |
      The filenames of the input model MS files which will be added.
    type: Directory[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: data_colname
    label: Input MS data column
    doc: |
      The input MS file data column for which addition will be done.
    type: string
    inputBinding:
      prefix: --msin_column=
      separate: false
  - id: obs_starttime
    label: Start time
    doc: |
      The start time (in casacore MVTime) for the time chunk for addition.
    type: string
    inputBinding:
      prefix: --starttime=
      separate: False
  - id: infix
    label: Output infix string
    doc: |
      The infix string to use when building the output MS filename.
    type: string
    inputBinding:
      prefix: --infix=
      separate: False

outputs:
  - id: output_ms
    type: Directory[]
    outputBinding:
      glob: ['$(inputs.msobs.basename)*_di.ms']

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
