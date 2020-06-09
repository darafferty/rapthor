cwlVersion: v1.0
class: CommandLineTool
baseCommand: [makesourcedb]
label: "Makes a sourcedb file from a sky model file"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - format=<
  - append=False
  - outtype=blob

inputs:
  - id: in
    type: string
    inputBinding:
      prefix: in=
      separate: false
  - id: out
    type: string
    inputBinding:
      prefix: out=
      separate: false

outputs:
  - id: sourcedb
    type: string
    outputBinding:
      outputEval: $(inputs.out)
