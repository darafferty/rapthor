cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for full-Jones gains over a single direction for the given MS file,
  using the MODEL_DATA column with fast-phase and slow-gain corruptions (if available)
  preapplied. Output is the solution table in h5parm format. See
  ddecal_solve_scalarphase.cwl for a detailed description of any inputs and outputs not
  documented below.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout=
  - steps=[solve]
  - solve.type=ddecal
  - solve.mode=fulljones
  - solve.usebeammodel=True
  - solve.beammode=array_factor
  - solve.modeldatacolumns=[MODEL_DATA]

inputs:
  - id: msin
    type: Directory
    inputBinding:
      prefix: msin=
      separate: False
  - id: starttime
    type: string
    inputBinding:
      prefix: msin.starttime=
      separate: False
  - id: ntimes
    type: int
    inputBinding:
      prefix: msin.ntimes=
      separate: False
  - id: startchan
    type: int
    inputBinding:
      prefix: msin.startchan=
      separate: False
  - id: nchan
    type: int
    inputBinding:
      prefix: msin.nchan=
      separate: False
  - id: h5parm
    type: string
    inputBinding:
      prefix: solve.h5parm=
      separate: False
  - id: solint
    type: int
    inputBinding:
      prefix: solve.solint=
      separate: False
  - id: solve_nchan
    type: int
    inputBinding:
      prefix: solve.nchan=
      separate: False
  - id: llssolver
    type: string
    inputBinding:
      prefix: solve.llssolver=
      separate: False
  - id: maxiter
    type: int
    inputBinding:
      prefix: solve.maxiter=
      separate: False
  - id: propagatesolutions
    type: boolean
    inputBinding:
      prefix: solve.propagatesolutions=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: solveralgorithm
    type: string
    inputBinding:
      prefix: solve.solveralgorithm=
      separate: False
  - id: solverlbfgs_dof
    type: float
    inputBinding:
      prefix: solve.solverlbfgs.dof=
      separate: False
  - id: solverlbfgs_iter
    type: int
    inputBinding:
      prefix: solve.solverlbfgs.iter=
      separate: False
  - id: solverlbfgs_minibatches
    type: int
    inputBinding:
      prefix: solve.solverlbfgs.minibatches=
      separate: False
  - id: stepsize
    type: float
    inputBinding:
      prefix: solve.stepsize=
      separate: False
  - id: tolerance
    type: float
    inputBinding:
      prefix: solve.tolerance=
      separate: False
  - id: uvlambdamin
    type: float
    inputBinding:
      prefix: solve.uvlambdamin=
      separate: False
  - id: smoothnessconstraint
    type: float
    inputBinding:
      prefix: solve.smoothnessconstraint=
      separate: False
  - id: numthreads
    type: int
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: fulljonesh5parm
    type: File
    outputBinding:
      glob: $(inputs.h5parm)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'
