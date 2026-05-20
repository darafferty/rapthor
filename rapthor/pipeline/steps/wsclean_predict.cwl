cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean_predict.py]
label: Predict using WSClean
doc: |
  This tool uses WSClean to predict model data into separate columns.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: region_file
    label: DS9 region file
    doc: |
      The filename of the region file. 
    type:
      - File
    inputBinding:
      prefix: --region

  - id: msin
    label: Input MS directory name
    doc: |
      The name of the input MS directory.
    type: Directory
    inputBinding:
      prefix: --msin

  - id: model
    label: Filename of model FITS image
    doc: |
      The filename of the input model FITS image.
    type: File
    inputBinding:
      prefix: --model

outputs:
  - id: msout
    label: Output MS
    doc: |
      The directory name of the output MS. The input msin is returned if it
      is writable, otherwise a copy with temp name is made.
    type: Directory
    outputBinding:
      loadContents: true
      glob: '$(inputs.msin).wsclean_predict.json'
      outputEval: '$(JSON.parse(self[0].contents).msout)'

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
