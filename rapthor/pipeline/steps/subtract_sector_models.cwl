cwlVersion: v1.0
class: CommandLineTool
baseCommand: [subtract_sector_models.py]
label: "Subtracts sector model data"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - '--weights_colname=WEIGHT_SPECTRUM'
  - '--phaseonly=True'

inputs:
  - id: msobs
    type: string
    inputBinding:
      position: 0
  - id: msmod
    type: string[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: obs_starttime
    type: string
    inputBinding:
      prefix: --starttime=
      separate: False
  - id: solint_sec
    type: float
    inputBinding:
      prefix: --solint_sec=
      separate: False
  - id: solint_hz
    type: float
    inputBinding:
      prefix: --solint_hz=
      separate: False
  - id: infix
    type: string
    inputBinding:
      prefix: --infix=
      separate: False
  - id: min_uv_lambda
    type: float
    inputBinding:
      prefix: --uvcut_min=
      separate: False
  - id: max_uv_lambda
    type: float
    inputBinding:
      prefix: --uvcut_max=
      separate: False
  - id: nr_outliers
    type: int
    inputBinding:
      prefix: --nr_outliers=
      separate: False
  - id: peel_outliers
    type: string
    inputBinding:
      prefix: --peel_outliers=
      separate: False
  - id: reweight
    type: string
    inputBinding:
      prefix: --reweight=
      separate: False

outputs: []
