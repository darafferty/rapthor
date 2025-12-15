cwlVersion: v1.2
class: CommandLineTool
baseCommand: [make_catalog_from_image_cube.py]
label: Make a catalog from an image cube
doc: |
  This tool uses PyBDSF to make a source catalog from a FITS
  image cube.

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: cube
    label: Detection cube
    doc: |
      The filename of the input FITS image cube used for source detection.
    type: File
    inputBinding:
      position: 1
  - id: cube_beams
    label: Input image
    doc: |
      The filename of the input text file with the beam information.
    type: File
    inputBinding:
      position: 2
  - id: cube_frequencies
    label: PB-corrected model
    doc: |
      The filename of the input text file with the frequency information.
    type: File
    inputBinding:
      position: 3
  - id: output_catalog
    label: Output catalog name
    doc: |
      The filename of the output FITS source catalog.
    type: string
    inputBinding:
      position: 4
  - id: threshisl
    label: Island threshold
    doc: |
      The PyBDSF island threshold.
    type: float?
    inputBinding:
      prefix: --threshisl=
      separate: false
  - id: threshpix
    label: Pixel threshold
    doc: |
      The PyBDSF pixel threshold.
    type: float?
    inputBinding:
      prefix: --threshpix=
      separate: false
  - id: ncores
    type: int?
    inputBinding:
      prefix: --ncores=
      separate: false

outputs:
  - id: source_catalog
    label: Source catalog
    doc: |
      The filename of the FITS source catalog.
    type: File
    outputBinding:
      glob: '$(inputs.output_catalog)'

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
