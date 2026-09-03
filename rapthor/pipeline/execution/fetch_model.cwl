id: fetchmodel
label: fetch_model
class: CommandLineTool
cwlVersion: v1.2

doc: Download sky model needed for Rapthor.
baseCommand:
  - fetch_skymodel.py

inputs:
  - id: msin
    type: Directory
    inputBinding:
      position: 0
  - id: survey
    doc: Survey to query
    type: string
    inputBinding:
      position: 1
  - id: radius
    doc: Query radius in deg
    type: string
    inputBinding:
      position: 2
  - id: skymodel
    doc: Output sky model filename
    type: string
    inputBinding:
      position: 3

outputs:
  - id: model
    type: File
    outputBinding:
      glob: $(inputs.skymodel)

requirements:
  - class: InitialWorkDirRequirement
    listing:
     - entry: $(inputs.msin)
       writable: false

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
