cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for diagonal gains in multiple directions simultaneously for
  the given MS file with fast-phase only corrections preapplied, using
  the input sourcedb and h5parm and (optionally) baseline-dependent averaging.
  Output is the solution table in h5parm format. See
  ddecal_solve_scalarphase.cwl for a detailed description of any inputs and
  outputs not documented below.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout=
  - avg.type=bdaaverager
  - avg.minchannels=1
  - avg.frequencybase=0.0
  - solve.type=ddecal
  - solve.mode=diagonal
  - solve.usebeammodel=True
  - solve.beam_interval=120
  - solve.beammode=array_factor
  - solve.applycal.steps=[fastphase]
  - solve.applycal.fastphase.correction=phase000

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
  - id: steps
    type: string
    inputBinding:
      prefix: steps=
      separate: False
  - id: timebase
    type: float
    inputBinding:
      prefix: avg.timebase=
      separate: False
  - id: maxinterval
    type: int
    inputBinding:
      prefix: avg.maxinterval=
      separate: False
  - id: combined_h5parm
    label: Solution table
    doc: |
      The filename of the input solution table containing the fast-phase solutions.
      These solutions are preapplied before the solve is done.
    type: File
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
  - id: directions
    type: string[]
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve.directions=
      itemSeparator: ','
      separate: False
  - id: solutions_per_direction
    type: int[]
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve.solutions_per_direction=
      itemSeparator: ','
      separate: False
  - id: sourcedb
    type: File
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
  - id: onebeamperpatch
    type: boolean
    inputBinding:
      prefix: solve.onebeamperpatch=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: parallelbaselines
    type: boolean
    inputBinding:
      prefix: solve.parallelbaselines=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: sagecalpredict
    type: boolean
    inputBinding:
      prefix: solve.sagecalpredict=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: datause
    type: string
    inputBinding:
      prefix: solve.datause=
      separate: False
  - id: stepsize
    type: float
    inputBinding:
      prefix: solve.stepsize=
      separate: False
  - id: stepsigma
    type: float
    inputBinding:
      prefix: solve.stepsigma=
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
  - id: slow_gains_h5parm
    type: File
    outputBinding:
      glob: $(inputs.h5parm)
hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'
