cwlVersion: v1.0
class: CommandLineTool
baseCommand: [DPPP]
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for scalar phases in multiple directions simultaneously
  for the given MS file, using the input sourcedb. Output is the solution
  table in h5parm format.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout=.
  - steps=[solve]
  - solve.type=ddecal
  - solve.mode=scalarphase
  - solve.usebeammodel=True
  - solve.beammode=array_factor  # element beam was removed at phase center
  - solve.onebeamperpatch=True  # better set to False, but slow

inputs:
  - id: msin
    label: Input MS filename
    doc: |
      The filename of the input MS file.
    type: string
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
    type: string
    inputBinding:
      prefix: solve.sourcedb=
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
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance to use in the calibration.
    type: float
    inputBinding:
      prefix: solve.uvlambdamin=
      separate: False

  - id: smoothnessconstraint
    type: float
    inputBinding:
      prefix: solve.smoothnessconstraint=
      separate: False

  - id: antennaconstraint
    type: string
    inputBinding:
      prefix: solve.antennaconstraint=
      separate: False

  - id: numthreads
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: fast_phases_h5parm
    type: string
    outputBinding:
      outputEval: $(inputs.h5parm)
