cwlVersion: v1.2
class: CommandLineTool
baseCommand: [normalize_flux_scale.py]
label: Calculate normalization corrections
doc: |
  This tool calculates the corrections needed to normalize the flux
  scale such that (true flux / observed flux) * correction = 1

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: source_catalog
    label: Source catalog filename
    doc: |
      The filename of the input FITS source catalog.
    type: File
    inputBinding:
      position: 1
  - id: ms_file
    label: MS file filename
    doc: |
      The filename of the imaging MS file.
    type: Directory
    inputBinding:
      position: 2
  - id: normalize_h5parm
    label: Output H5parm filename
    doc: |
      The filename of the output H5parm.
    type: string
    inputBinding:
      position: 3

outputs:
  - id: output_h5parm
    label: Output H5parm
    doc: |
      The filename of the output H5parm.
    type: File
    outputBinding:
      glob: '$(inputs.normalize_h5parm)'

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1
