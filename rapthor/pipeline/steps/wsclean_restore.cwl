cwlVersion: v1.0
class: CommandLineTool
baseCommand: [wsclean]
label: "Restores a source list to an image using WSClean"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - -restore-list

inputs:
  - id: residual_image
    type: string
    inputBinding:
      position: 1
  - id: source_list
    type: string
    inputBinding:
      position: 2
  - id: output_image
    type: string
    inputBinding:
      position: 3
  - id: numthreads
    type: string
    inputBinding:
      prefix: -j

outputs:
  - id: restored_image
    type: string
    outputBinding:
      outputEval: $(inputs.output_image)
