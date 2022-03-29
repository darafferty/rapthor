cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
id: ddecal_solve_scalarphase
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for scalar phases in multiple directions simultaneously
  for the given MS file, using the input sourcedb. Output is the solution
  table in h5parm format.

requirements:
  - class: InlineJavascriptRequirement
#  - class: InitialWorkDirRequirement
#    listing:
#      - entry: $(inputs.msin)
#        writable: true
#  - class: InplaceUpdateRequirement
#    inplaceUpdate: true

arguments:
  - msin.datacolumn=DATA
  - msout=
  - steps=[solve]
  - solve.type=ddecal
  - solve.mode=scalarphase
  - solve.usebeammodel=True
  - solve.beammode=array_factor

inputs:
  - id: msin
    label: Input MS directory name
    doc: |
      The name of the input MS directory.
    type: Directory
    inputBinding:
      prefix: msin=
      separate: False

  - id: starttime
    label: Start time
    doc: |
      The start time (in casacore MVTime) for the time chunk to be processed.
    type: string
    inputBinding:
      prefix: msin.starttime=
      separate: False

  - id: ntimes
    label: Number of times
    doc: |
      The number of time slots for the time chunk to be processed.
    type: int
    inputBinding:
      prefix: msin.ntimes=
      separate: False

  - id: h5parm
    label: Solution table
    doc: |
      The filename of the output solution table.
    type: string
    inputBinding:
      prefix: solve.h5parm=
      separate: False

  - id: solint
    label: Solution interval
    doc: |
      The solution interval in timeslots for the solve.
    type: int
    inputBinding:
      prefix: solve.solint=
      separate: False

  - id: nchan
    label: Solution interval
    doc: |
      The solution interval in channels for the solve.
    type: int
    inputBinding:
      prefix: solve.nchan=
      separate: False

  - id: sourcedb
    label: Sky model
    doc: |
      The sourcedb sky model to use for the solve.
    type: File
    inputBinding:
      prefix: solve.sourcedb=
      separate: False

  - id: llssolver
    label: Linear least-squares solver
    doc: |
      The linear least-squares solver to use (one of 'qr', 'svd', or 'lsmr')
    type: string
    inputBinding:
      prefix: solve.llssolver=
      separate: False

  - id: maxiter
    label: Maximum iterations
    doc: |
      The maximum number of iterations in the solve.
    type: int
    inputBinding:
      prefix: solve.maxiter=
      separate: False

  - id: propagatesolutions
    label: Propagate solutions
    doc: |
      Flag that determines whether solutions are propagated as initial start values
      for the next solution interval.
    type: string
    inputBinding:
      prefix: solve.propagatesolutions=
      separate: False

  - id: solveralgorithm
    label: Solver algorithm
    doc: |
      The algorithm used for solving.
    type: string
    inputBinding:
      prefix: solve.solveralgorithm=
      separate: False

  - id: onebeamperpatch
    label: One beam per patch
    doc: |
      Flag that sets beam correction per patch or per source.
    type: string
    inputBinding:
      prefix: solve.onebeamperpatch=
      separate: False

  - id: stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations.
    type: float
    inputBinding:
      prefix: solve.stepsize=
      separate: False

  - id: tolerance
    label: Solver tolerance
    doc: |
      The solver tolerance used to define convergence.
    type: float
    inputBinding:
      prefix: solve.tolerance=
      separate: False

  - id: llsstarttolerance
    label: LLS solver starting tolerance
    doc: |
      The linear least-squares solver starting tolerance used to define convergence.
    type: float
    inputBinding:
      prefix: solve.llsstarttolerance=
      separate: False

  - id: llstolerance
    label: LLS solver tolerance
    doc: |
      The linear least-squares solver tolerance used to define convergence.
    type: float
    inputBinding:
      prefix: solve.llstolerance=
      separate: False

  - id: uvlambdamin
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance to use in the calibration.
    type: float
    inputBinding:
      prefix: solve.uvlambdamin=
      separate: False

  - id: smoothnessconstraint
    label: Smoothness constraint kernel size
    doc: |
      The smoothness constraint kernel size in Hz, used to enforce a smooth frequency
      dependence of the phase solutions.
    type: float
    inputBinding:
      prefix: solve.smoothnessconstraint=
      separate: False

  - id: antennaconstraint
    label: Antenna constraint
    doc: |
      A list of antennas that will be constrained to have the same solutions.
    type: string
    inputBinding:
      prefix: solve.antennaconstraint=
      separate: False

  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads to use during solve (0 = all).
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: fast_phases_h5parm
    label: Solution table
    doc: |
      The filename of the output solution table. The value is taken from the input
      parameter "h5parm"
    type: File
    outputBinding:
      outputEval: $(inputs.h5parm)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
