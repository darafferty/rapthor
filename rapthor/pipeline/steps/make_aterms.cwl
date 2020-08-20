cwlVersion: v1.0
class: CommandLineTool
baseCommand: [make_aterm_images.py]
label: "Make FITS images of the aterms"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - '--smooth_deg=0.1'
  - '--gsize_deg=0.15'
  - '--time_avg_factor=1'

inputs:
  - id: h5parm
    type: string
    inputBinding:
      position: 0
  - id: soltabname
    type: string
    inputBinding:
      prefix: --soltabname=
      separate: false
  - id: outroot
    type: string
    inputBinding:
      prefix: --outroot=
      separate: false
  - id: skymodel
    type: string
    inputBinding:
      prefix: --skymodel=
      separate: false
  - id: sector_bounds_deg
    type: string
    inputBinding:
      prefix: --bounds_deg=
      separate: false
  - id: sector_bounds_mid_deg
    type: string
    inputBinding:
      prefix: --bounds_mid_deg=
      separate: false

outputs: []
