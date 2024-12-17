cwlVersion: v1.2
class: CommandLineTool
baseCommand: [normalize_flux_scale.py]
label: Calculate normalization corrections
doc: |
  This tool calculates the corrections needed to normalize the flux
  scale such that Flux_rapthor / Flux_true = 1

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
  - id: ra
    label: RA of center of image
    doc: |
      The RA of the center of the image in deg.
    type: float
    inputBinding:
      position: 2
  - id: dec
    label: Dec of center of image
    doc: |
      The Dec of the center of the image in deg.
    type: float
    inputBinding:
      position: 3
  - id: normalize_h5parm
    label: Output H5parm filename
    doc: |
      The filename of the output H5parm.
    type: string
    inputBinding:
      position: 4

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
    dockerPull: 'astronrd/rapthor'