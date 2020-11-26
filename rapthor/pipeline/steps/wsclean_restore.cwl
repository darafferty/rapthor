cwlVersion: v1.0
class: CommandLineTool
baseCommand: [wsclean]
label: "Restores a source list to an image using WSClean"

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: numthreads
    type: string
    inputBinding:
      prefix: -j
      position: 1
  - id: residual_image
    type: string
    inputBinding:
      prefix: -restore-list
      position: 2
  - id: source_list
    type: string
    inputBinding:
      position: 3
  - id: output_image
    type: string
    inputBinding:
      position: 4

outputs:
  - id: restored_image
    type: string
    outputBinding:
      outputEval: $(inputs.output_image)
