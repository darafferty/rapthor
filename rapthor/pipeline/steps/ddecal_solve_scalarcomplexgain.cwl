cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for scalar complex gains in multiple directions
  simultaneously for the given MS file, using the input sourcedb. Output is the
  solution table in h5parm format. See ddecal_solve_scalarphase.cwl for a
  detailed description of the inputs and outputs.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout=
  - steps=[solve]
  - solve.type=ddecal
  - solve.mode=scalarcomplexgain
  - solve.usebeammodel=True
  - solve.beammode=array_factor

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
  - id: nchan
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
  - id: onebeamperpatch
    type: boolean
    inputBinding:
      prefix: solve.onebeamperpatch=
      valueFrom: "$(self ? 'True': 'False')"
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
    type: File
    outputBinding:
      glob: $(inputs.h5parm)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'
