id: concat_ms
label: Concatenate MS files
class: CommandLineTool
cwlVersion: v1.2

doc: Concatenate multiple input Measurement Sets into one output Measurement Set.
baseCommand:
  - concat_ms.py

inputs:
- id: msin
  doc: List of input Measurement Sets
  type: Directory[]
  inputBinding:
    position: 0
- id: msout_name
  doc: Output Measurement Set
  type: string
  default: out.ms
  inputBinding:
    position: 1

outputs:
- id: msout
  type: Directory
  outputBinding:
    glob: $(inputs.msout_name)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor
