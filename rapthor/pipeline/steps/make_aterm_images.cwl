cwlVersion: v1.2
class: CommandLineTool
baseCommand: [make_aterm_images.py]
label: Make a-term images
doc: |
  This tool makes FITS a-term images from the input solution table. The images
  are suitable for use with WSClean+IDG.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.h5parm)
        writable: true

arguments:
  - '--smooth_deg=0.1'

inputs:
  - id: h5parm
    label: Input solution table
    doc: |
      The filename of the input h5parm file.
    type: File
    inputBinding:
      position: 0
  - id: soltabname
    label: Input soltab name
    doc: |
      The name of the input soltab.
    type: string
    inputBinding:
      prefix: --soltabname=
      separate: false
  - id: screen_type
    label: Screen type
    doc: |
      The type of the screen to generate.
    type: string
    inputBinding:
      prefix: --screen_type=
      separate: false
  - id: outroot
    label: Output root name
    doc: |
      The root of the filenames of the output images.
    type: string
    inputBinding:
      prefix: --outroot=
      separate: false
  - id: skymodel
    label: Input sky model
    doc: |
      The filename of the input sky model file.
    type: File
    inputBinding:
      prefix: --skymodel=
      separate: false
  - id: sector_bounds_deg
    label: Bounds of imaging sectors
    doc: |
      The global bounds of the imaging sectors in deg.
    type: string
    inputBinding:
      prefix: --bounds_deg=
      separate: false
  - id: sector_bounds_mid_deg
    label: Midpoint of imaging bounds
    doc: |
      The global midpoint of the imaging sectors in deg.
    type: string
    inputBinding:
      prefix: --bounds_mid_deg=
      separate: false
  - id: ncpu
    label: Number of CPUs
    doc: |
      The number of CPUs / cores to use.
    type: string
    inputBinding:
      prefix: --ncpu=
      separate: false

outputs:
  - id: output_images
    type: File[]
    outputBinding:
      glob: '$(inputs.outroot)*'

hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
