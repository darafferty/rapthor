cwlVersion: v1.2
class: CommandLineTool
baseCommand: [subtract_sector_models.py]
label: Subtracts sector model data
doc: |
  This tool subtracts sector model uv data from the input MS files. For each
  sector, all sources that lie outside of the sector are subtracted (or
  peeled), generating data suitable for use as input to the imaging
  workflow. Reweighting by the residuals can also be done, by generating
  data in which all sources have been subtracted.

requirements:
  - class: InlineJavascriptRequirement

arguments:
  - '--weights_colname=WEIGHT_SPECTRUM'
  - '--phaseonly=True'

inputs:
  - id: msobs
    label: Filename of data MS
    doc: |
      The filename of the input MS file for which subtraction will be done.
    type: Directory
    inputBinding:
      position: 0
  - id: data_colname
    label: Input MS data column
    doc: |
      The data column to be read from the data MS.
    type: string
    inputBinding:
      prefix: --msin_column=
      separate: false
  - id: msmod
    label: Filenames of model MS
    doc: |
      The filenames of the input model MS files which will be subtracted.
    type: Directory[]
    inputBinding:
      position: 1
      itemSeparator: ","
  - id: obs_starttime
    label: Start time
    doc: |
      The start time (in casacore MVTime) for the time chunk for subtraction.
    type: string
    inputBinding:
      prefix: --starttime=
      separate: False
  - id: solint_sec
    label: Solution interval in sec
    doc: |
      The solution interval in sec used during the fast-phase calibration.
    type: float
    inputBinding:
      prefix: --solint_sec=
      separate: False
  - id: solint_hz
    label: Solution interval in Hz
    doc: |
      The solution interval in Hz used during the slow-gain calibration.
    type: float
    inputBinding:
      prefix: --solint_hz=
      separate: False
  - id: infix
    label: Output infix string
    doc: |
      The infix string to use when building the output MS filename.
    type: string
    inputBinding:
      prefix: --infix=
      separate: False
  - id: min_uv_lambda
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance used during the calibration.
    type: float
    inputBinding:
      prefix: --uvcut_min=
      separate: False
  - id: max_uv_lambda
    label: Maximum uv distance in lambda
    doc: |
      The maximum uv distance used during the calibration.
    type: float
    inputBinding:
      prefix: --uvcut_max=
      separate: False
  - id: nr_outliers
    label: Number outlier sectors
    doc: |
      The number of outlier sectors to process.
    type: int
    inputBinding:
      prefix: --nr_outliers=
      separate: False
  - id: peel_outliers
    label: Outlier flag
    doc: |
      The flag that sets peeling of outlier sources.
    type: boolean
    inputBinding:
      prefix: --peel_outliers=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: nr_bright
    label: Number bright-source sectors
    doc: |
      The number of bright-source sectors to process.
    type: int
    inputBinding:
      prefix: --nr_bright=
      separate: False
  - id: peel_bright
    label: Bright-source flag
    doc: |
      The flag that sets peeling of bright-source sources.
    type: boolean
    inputBinding:
      prefix: --peel_bright=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: reweight
    label: Reweight flag
    doc: |
      The flag that sets reweighting of uv data.
    type: boolean
    inputBinding:
      prefix: --reweight=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

outputs:
  - id: output_models
    type: Directory[]
    outputBinding:
      glob: ['$(inputs.msobs.basename)*_field', '$(inputs.msobs.basename)*.sector_*']

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor:2.1.post1
