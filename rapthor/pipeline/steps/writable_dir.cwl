cwlVersion: v1.2

class: CommandLineTool
baseCommand: touch
arguments: 
  - scratch

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entryname: scratch
        entry: "$({class: 'Directory', listing: []})"
        writable: true

inputs: []

outputs:
  - id: out
    type: Directory
    outputBinding:
      glob: scratch

