cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean]
label: Restores a source list
doc: |
  This tool restores a source list to an image using WSClean.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads to use.
    type: int
    inputBinding:
      prefix: -j
      position: 1
  - id: residual_image
    label: Filename of input image
    doc: |
      The filename of the input residual image to which sources will
      be restored.
    type: File
    inputBinding:
      prefix: -restore-list
      position: 2
  - id: source_list
    label: Filename of input model
    doc: |
      The filename of the input sky model which will be restored.
    type: File
    inputBinding:
      position: 3
  - id: output_image
    label: Filename of output image
    doc: |
      The filename of the output image.
    type: string
    inputBinding:
      position: 4

outputs:
  - id: restored_image
    label: Output restored image
    doc: |
      The filename of the output restored image. The value is
      taken from the input parameter "output_image"
    type: File
    outputBinding:
      glob: $(inputs.output_image)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
