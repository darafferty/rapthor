cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for complex gains in multiple directions simultaneously for
  the given MS file with fast phase corrections preapplied, using the input
  sourcedb and h5parm. Output is the solution table in h5parm format. See
  ddecal_solve_scalarphase.cwl for a detailed description of any inputs and
  outputs not documented below.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout=
  - steps=[solve]
  - solve.type=ddecal
  - solve.mode=complexgain
  - solve.usebeammodel=True
  - solve.beammode=array_factor
  - solve.applycal.steps=[fastphase]
  - solve.applycal.fastphase.correction=phase000

inputs:
  - id: msin
    type: string
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
  - id: fast_h5parm
    label: Solution table
    doc: |
      The filename of the input solution table containing the fast phase solutions.
      These solutions are preapplied before the solve is done.
    type: string
    inputBinding:
      prefix: solve.applycal.parmdb=
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
  - id: sourcedb
    type: string
    inputBinding:
      prefix: solve.sourcedb=
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
    type: string
    inputBinding:
      prefix: solve.propagatesolutions=
      separate: False
  - id: solveralgorithm
    type: string
    inputBinding:
      prefix: solve.solveralgorithm=
      separate: False
  - id: onebeamperpatch
    type: string
    inputBinding:
      prefix: solve.onebeamperpatch=
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
  - id: llsstarttolerance
    type: float
    inputBinding:
      prefix: solve.llsstarttolerance=
      separate: False
  - id: llstolerance
    type: float
    inputBinding:
      prefix: solve.llstolerance=
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
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: slow_gains_h5parm
    type: string
    outputBinding:
      outputEval: $(inputs.h5parm)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
