class: CommandLineTool
cwlVersion: v1.2
id: make_sourcedb
baseCommand:
  - makesourcedb
inputs:
  - id: in
    type:
      - File
      - string
    inputBinding:
      position: 0
      prefix: in=
      separate: false
      shellQuote: false
  - id: out
    type: string
    inputBinding:
      position: 1
      prefix: out=
      separate: false
      valueFrom: $(inputs.out)
      shellQuote: false
  - default: blob
    id: outtype
    type: string
    inputBinding:
      position: 2
      prefix: outtype=
      separate: false
      shellQuote: false
  - default: '"<"'
    id: format
    type: string
    inputBinding:
      position: 3
      prefix: format=
      separate: false
      shellQuote: false
outputs:
  - id: sourcedb
    type:
      - File
    outputBinding:
      glob: $(inputs.out)
label: make_sourcedb_ateam
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
requirements:
  - class: InlineJavascriptRequirement
  - class: ShellCommandRequirement
