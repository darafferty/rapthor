cwlVersion: v1.0
class: CommandLineTool
baseCommand: [DPPP]
label: "Calibrates a dataset using DDECal"

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - numthreads=0
  - msin.datacolumn=DATA
  - msout=.
  - steps=[solve]
  - solve.type=ddecal
  - solve.mode=complexgain
  - solve.usebeammodel=True
  - solve.beammode=array_rapthor
  - solve.onebeamperpatch=True
  - solve.applycal.steps=[slowamp,slowphase,fastphase]
  - solve.applycal.slowamp.correction=amplitude000
  - solve.applycal.slowphase.correction=phase000
  - solve.applycal.fastphase.correction=phase001

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
  - id: combined_h5parm
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

outputs:
  - id: slow_gains_h5parm
    type: string
    outputBinding:
      outputEval: $(inputs.h5parm)
